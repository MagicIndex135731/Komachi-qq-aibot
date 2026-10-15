from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import logging
import math
import time

from sqlalchemy.engine import Engine

from app.adapters.onebot_models import GroupMessageEvent
from app.agent.contracts import (
    AgentActionResult, AgentBatch, AgentDecision, AgentStoppedError, GroupAgentOptions,
)
from app.storage.group_agent import GroupAgentRepository


logger = logging.getLogger(__name__)

DecisionCallback = Callable[[AgentBatch], Awaitable[AgentDecision]]
ExecutionCallback = Callable[[GroupMessageEvent, AgentBatch, AgentDecision, "AgentDispatchGuard"], Awaitable[AgentActionResult]]


class AgentDispatchGuard:
    """Check immediately before a non-idempotent API write, never just planning.

    A cancellation cannot terminate synchronous model work in a thread. This
    generation token prevents its late result from ever becoming a QQ write.
    The marker must commit before the gateway begins transmitting the frame.
    """

    def __init__(
        self, runtime: GroupAgentRuntime, *, group_id: int, turn_id: str, epoch: int,
        working_note_source_message_ids: tuple[int, ...] = (),
    ) -> None:
        self._runtime = runtime
        self.group_id = group_id
        self.turn_id = turn_id
        self.epoch = epoch
        self._working_note_source_message_ids = tuple(working_note_source_message_ids)
        self._connection_generation = runtime._connection_generation
        self._dispatched = False
        self._closed = False

    @property
    def dispatched(self) -> bool:
        return self._dispatched

    def valid(self) -> bool:
        runtime = self._runtime
        return bool(
            not self._closed and runtime.running and runtime.connected
            and runtime._connection_generation == self._connection_generation
            and runtime._epochs.get(self.group_id) == self.epoch
        )

    def require_valid(self) -> None:
        if not self.valid():
            raise AgentStoppedError("Agent turn is stopped or disconnected")

    def on_dispatch(self) -> None:
        self.require_valid()
        if not self._runtime.repository.mark_dispatching(
            turn_id=self.turn_id, group_id=self.group_id, epoch=self.epoch,
            working_note_source_message_ids=self._working_note_source_message_ids,
        ):
            raise AgentStoppedError("Agent turn no longer owns its persisted session")
        self._dispatched = True

    def close(self) -> None:
        self._closed = True


class GroupAgentRuntime:
    """One durable conversation consumer per group, with finite model turns.

    decide/execute must use bounded providers. The callbacks own persona and
    memory policy; this service owns inbox ordering, cancellation, dispatch
    fencing and crash recovery. It never starts a model because a group is idle.
    """

    def __init__(
        self, *, engine: Engine, bot_qq: int, bot_name: str,
        decide: DecisionCallback, execute: ExecutionCallback,
        options: GroupAgentOptions | None = None,
        repository: GroupAgentRepository | None = None,
    ) -> None:
        self.repository = repository or GroupAgentRepository(engine)
        self.bot_qq = int(bot_qq)
        self.bot_name = bot_name
        self.decide = decide
        self.execute = execute
        self.options = options or GroupAgentOptions()
        self.running = False
        self.connected = False
        self._connection_generation = 0
        self._connection_event = asyncio.Event()
        self._workers: dict[int, asyncio.Task] = {}
        self._wake: dict[int, asyncio.Event] = {}
        self._last_observed: dict[int, float] = {}
        self._epochs: dict[int, int] = {}
        self._model_slots = asyncio.Semaphore(self.options.model_concurrency)
        self._admission_lock = asyncio.Lock()
        self._worker_failures: dict[int, int] = {}
        self._worker_paused: set[int] = set()
        self._worker_reset_used: set[int] = set()
        self._retry_tasks: dict[int, asyncio.Task] = {}
        self._prune_task: asyncio.Task | None = None
        self._last_prune_started: float | None = None

    @property
    def active_group_ids(self) -> tuple[int, ...]:
        return tuple(sorted(group_id for group_id, task in self._workers.items() if not task.done()))

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        groups = self.repository.recover()
        self._worker_failures.clear()
        self._worker_paused.clear()
        self._worker_reset_used.clear()
        self._schedule_prune(force=True)
        for group_id in groups[:self.options.max_active_groups]:
            self._ensure_worker(group_id)
        logger.info("group_agent_started pending_groups=%s max_groups=%s", len(groups), self.options.max_active_groups)

    def set_connected(self, connected: bool) -> None:
        connected = bool(connected)
        if connected == self.connected:
            return
        self.connected = connected
        self._connection_generation += 1
        if connected:
            self._connection_event.set()
        else:
            self._connection_event.clear()
        for event in self._wake.values():
            event.set()
        logger.info("group_agent_connection connected=%s", connected)

    async def observe(self, event: GroupMessageEvent) -> bool:
        """Reference a committed canonical message; never ingest its body twice."""
        if not self.running or event.user_id == self.bot_qq:
            return False
        async with self._admission_lock:
            added = self.repository.enqueue(
                group_id=event.group_id, platform_msg_id=event.platform_msg_id,
                bot_qq=self.bot_qq, max_pending=self.options.max_pending_per_group,
            )
            if not added:
                return False
            if event.group_id in self._worker_paused and event.group_id not in self._worker_reset_used:
                self._worker_reset_used.add(event.group_id)
                self._worker_paused.discard(event.group_id)
                self._worker_failures[event.group_id] = 0
            self._last_observed[event.group_id] = time.monotonic()
            self._wake.setdefault(event.group_id, asyncio.Event())
            if len(self._workers) < self.options.max_active_groups or event.group_id in self._workers:
                self._ensure_worker(event.group_id)
            else:
                logger.info("group_agent_consumer_deferred group_id=%s reason=consumer_limit", event.group_id)
            self._wake[event.group_id].set()
            return True

    def _ensure_worker(self, group_id: int) -> None:
        if not self.running or group_id in self._worker_paused:
            return
        existing = self._workers.get(group_id)
        if existing is not None and not existing.done():
            return
        snapshot = self.repository.begin_session(
            group_id, working_note_max_age_seconds=self.options.working_note_max_age_seconds,
            working_note_max_sources=self.options.working_note_max_sources,
        )
        self._epochs[group_id] = snapshot.epoch
        self._wake.setdefault(group_id, asyncio.Event())
        task = asyncio.create_task(self._consume_group(group_id), name=f"group-agent-{group_id}")
        self._workers[group_id] = task
        task.add_done_callback(lambda completed, gid=group_id: self._worker_done(gid, completed))

    def _worker_done(self, group_id: int, task: asyncio.Task) -> None:
        if self._workers.get(group_id) is not task:
            return
        self._workers.pop(group_id, None)
        self._wake.pop(group_id, None)
        self._last_observed.pop(group_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("group_agent_worker_failed group_id=%s error_type=%s", group_id, type(error).__name__)
            failures = self._worker_failures.get(group_id, 0) + 1
            self._worker_failures[group_id] = failures
            if failures >= 3:
                self._worker_paused.add(group_id)
                logger.error("group_agent_worker_paused group_id=%s failures=%s", group_id, failures)
            elif self.running and group_id not in self._retry_tasks:
                retry = asyncio.create_task(
                    self._retry_group_after(group_id, failures), name=f"group-agent-retry-{group_id}",
                )
                self._retry_tasks[group_id] = retry
                retry.add_done_callback(lambda completed, gid=group_id: self._retry_done(gid, completed))
            return
        if self.running:
            for pending_group in self.repository.pending_group_ids():
                if len(self._workers) >= self.options.max_active_groups:
                    break
                self._ensure_worker(pending_group)

    async def _retry_group_after(self, group_id: int, failures: int) -> None:
        await asyncio.sleep(min(2.0, 0.25 * (2 ** max(0, failures - 1))))
        if self.running and group_id not in self._worker_paused and self.repository.has_pending(group_id):
            self._ensure_worker(group_id)

    def _retry_done(self, group_id: int, task: asyncio.Task) -> None:
        if self._retry_tasks.get(group_id) is task:
            self._retry_tasks.pop(group_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("group_agent_worker_retry_failed group_id=%s error_type=%s", group_id, type(task.exception()).__name__)

    async def cancel_group(self, group_id: int) -> None:
        """Invalidate the group immediately; the caller decides when to resume."""
        self._epochs[group_id] = self._epochs.get(group_id, 0) + 1
        task = self._workers.pop(group_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._wake.pop(group_id, None)
        self._last_observed.pop(group_id, None)

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._connection_generation += 1
        self._connection_event.set()
        worker_tasks = tuple(self._workers.values())
        for task in worker_tasks:
            task.cancel()
        tasks = worker_tasks
        if self._prune_task is not None and not self._prune_task.done():
            # Cancelling to_thread cannot stop SQLite. Keep its task alive and
            # retain the reference so a restart cannot overlap another prune.
            tasks += (self._prune_task,)
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.options.shutdown_timeout_seconds)
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                logger.warning("group_agent_shutdown_incomplete pending_tasks=%s", len(pending))
        for retry in self._retry_tasks.values():
            if not retry.done():
                retry.cancel()
        self._retry_tasks.clear()
        self._workers.clear()
        self._wake.clear()
        self._last_observed.clear()
        logger.info("group_agent_stopped")

    def _schedule_prune(self, *, force: bool = False) -> None:
        if not self.running or self._prune_task is not None and not self._prune_task.done():
            return
        now = time.monotonic()
        if not force and self._last_prune_started is not None and (
            now - self._last_prune_started < self.options.prune_interval_seconds
        ):
            return
        self._last_prune_started = now
        # Deduplication records must outlive every source that could still be
        # claimed as fresh, even when a caller configures a short retention.
        retention = max(self.options.completed_retention_seconds, self.options.max_message_age_seconds)
        completed_before = datetime.now(UTC) - timedelta(seconds=retention)
        self._prune_task = asyncio.create_task(
            self._prune_completed_records(completed_before), name="group-agent-prune",
        )

    async def _prune_completed_records(self, completed_before: datetime) -> None:
        try:
            await asyncio.to_thread(self.repository.prune, completed_before=completed_before)
        except Exception:
            logger.exception("group_agent_prune_failed")

    async def _collect_quiet_window(self, group_id: int) -> None:
        started = time.monotonic()
        wake = self._wake[group_id]
        while self.running and self.connected:
            now = time.monotonic()
            quiet_until = self._last_observed.get(group_id, started) + self.options.quiet_seconds
            deadline = min(quiet_until, started + self.options.max_batch_wait_seconds)
            delay = deadline - now
            if delay <= 0:
                return
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                return

    def _validated_decision(self, decision: AgentDecision, batch: AgentBatch) -> AgentDecision:
        if not isinstance(decision, AgentDecision) or decision.action not in {"silent", "wait", "reply"}:
            return AgentDecision("silent", reason="invalid_decision")
        source_ids = {event.platform_msg_id for event in batch.events}
        target = decision.target_message_id
        if decision.action == "reply" and target not in source_ids:
            return AgentDecision("silent", reason="target_outside_batch")
        wait_seconds = decision.wait_seconds
        if not isinstance(wait_seconds, (float, int)) or not math.isfinite(wait_seconds):
            wait_seconds = 0.0
        return replace(
            decision, reason=str(decision.reason)[:160],
            working_note=str(decision.working_note)[:self.options.working_note_max_chars],
            wait_seconds=max(0.0, min(float(wait_seconds), self.options.max_wait_seconds)),
        )

    async def _consume_group(self, group_id: int) -> None:
        wake = self._wake[group_id]
        while self.running:
            if not self.connected:
                await self._connection_event.wait()
                continue
            if not self.repository.has_pending(group_id):
                # An idle group owns no task/model slot. Its persisted session
                # resumes on the next observation, freeing capacity for others.
                return
            await self._collect_quiet_window(group_id)
            if not self.running or not self.connected:
                continue
            claimed = self.repository.claim(
                group_id=group_id, epoch=self._epochs[group_id],
                max_messages=self.options.max_batch_messages,
                max_age_seconds=self.options.max_message_age_seconds,
                bot_qq=self.bot_qq,
                working_note_max_age_seconds=self.options.working_note_max_age_seconds,
                working_note_max_sources=self.options.working_note_max_sources,
            )
            if claimed is None:
                self._schedule_prune()
                if not self.repository.epoch_is_current(group_id, self._epochs[group_id]):
                    logger.warning("group_agent_session_replaced group_id=%s", group_id)
                    return
                await asyncio.sleep(0)
                continue
            batch = AgentBatch(
                group_id, claimed.events, claimed.turn_id,
                claimed.session.working_note, claimed.session.last_sent_at,
            )
            guard = AgentDispatchGuard(
                self, group_id=group_id, turn_id=batch.turn_id, epoch=claimed.session.epoch,
                working_note_source_message_ids=claimed.session.working_note_source_message_ids,
            )
            await self._run_turn(batch, guard)
            self._worker_failures[group_id] = 0
            self._worker_paused.discard(group_id)
            self._worker_reset_used.discard(group_id)
            self._schedule_prune()

    async def _run_turn(self, batch: AgentBatch, guard: AgentDispatchGuard) -> None:
        decision = AgentDecision("silent")
        try:
            async with self._model_slots:
                guard.require_valid()
                decision = self._validated_decision(
                    await asyncio.wait_for(self.decide(batch), timeout=self.options.decision_timeout_seconds), batch,
                )
                guard.require_valid()
                if decision.action == 'reply' and decision.reason in {'direct_trigger', 'persona_switch'}:
                    boundary = self.repository.retain_through_target(
                        turn_id=batch.turn_id, group_id=batch.group_id, epoch=guard.epoch,
                        target_message_id=decision.target_message_id,
                    )
                    # The executor and its context receive the same prefix as
                    # the persistent turn. Later questions retain pending IDs.
                    batch = replace(batch, events=batch.events[:boundary])
                self.repository.record_decision(turn_id=batch.turn_id, action=decision.action, reason=decision.reason)
                if decision.action == "wait":
                    # Wait on arrivals once, without re-sampling the same batch.
                    # New messages belong to the next independently idempotent turn.
                    delay = decision.wait_seconds or min(self.options.quiet_seconds, self.options.max_wait_seconds)
                    if self.options.max_wait_rounds > 0 and delay > 0 and not self.repository.has_pending(batch.group_id):
                        self._wake[batch.group_id].clear()
                        try:
                            await asyncio.wait_for(self._wake[batch.group_id].wait(), timeout=delay)
                        except asyncio.TimeoutError:
                            pass
                    guard.require_valid()
                    self.repository.finish(
                        turn_id=batch.turn_id, status="silent", working_note=decision.working_note,
                        reason="wait_next_batch" if self.repository.has_pending(batch.group_id) else "wait_timeout",
                        epoch=guard.epoch,
                        working_note_max_age_seconds=self.options.working_note_max_age_seconds,
                        working_note_max_sources=self.options.working_note_max_sources,
                    )
                    return
                if decision.action == "silent":
                    self.repository.finish(
                        turn_id=batch.turn_id, status="silent", working_note=decision.working_note, epoch=guard.epoch,
                        working_note_max_age_seconds=self.options.working_note_max_age_seconds,
                        working_note_max_sources=self.options.working_note_max_sources,
                    )
                    return
                target = next(event for event in batch.events if event.platform_msg_id == decision.target_message_id)
                timeout = (self.options.task_execution_timeout_seconds
                    if decision.reason == 'direct_trigger'
                    else self.options.execution_timeout_seconds)
                result = await asyncio.wait_for(
                    self.execute(target, batch, decision, guard),
                    timeout=timeout,
                )
                if not isinstance(result, AgentActionResult) or result.status not in {
                    "sent", "uncertain", "blocked", "failed", "silent",
                }:
                    raise ValueError("invalid Agent execution result")
                if result.status == "sent" and not guard.dispatched:
                    raise ValueError("Agent execution bypassed its durable dispatch guard")
                self.repository.finish(
                    turn_id=batch.turn_id, status=result.status,
                    working_note=decision.working_note, receipt_ids=result.receipt_ids,
                    charge_proactive_cadence=result.charge_proactive_cadence,
                    epoch=guard.epoch,
                    working_note_max_age_seconds=self.options.working_note_max_age_seconds,
                    working_note_max_sources=self.options.working_note_max_sources,
                )
                logger.info("group_agent_turn_finished group_id=%s turn_id=%s status=%s sources=%s", batch.group_id, batch.turn_id, result.status, len(batch.events))
        except asyncio.CancelledError:
            self.repository.finish(
                turn_id=batch.turn_id, status="uncertain" if guard.dispatched else "interrupted",
                reason="cancelled_after_dispatch" if guard.dispatched else "cancelled_before_dispatch",
                epoch=guard.epoch,
            )
            raise
        except AgentStoppedError:
            self.repository.finish(
                turn_id=batch.turn_id, status="uncertain" if guard.dispatched else "interrupted",
                reason="connection_or_generation_changed",
                epoch=guard.epoch,
            )
        except Exception as exc:
            self.repository.finish(
                turn_id=batch.turn_id, status="uncertain" if guard.dispatched else "failed",
                reason=type(exc).__name__,
                epoch=guard.epoch,
            )
            logger.exception("group_agent_turn_failed group_id=%s turn_id=%s dispatched=%s", batch.group_id, batch.turn_id, guard.dispatched)
        finally:
            guard.close()
