"""Bounded autonomous account scheduler; no fabricated inbound messages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import logging
from uuid import uuid4
from random import SystemRandom

from sqlalchemy.engine import Engine

from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation, SocialResult, SocialRuntimeOptions
from app.agent.social_policy import in_quiet_hours, friend_authorized
from app.storage.social_agent import DueSocialJob, SocialAgentRepository


logger = logging.getLogger(__name__)


class SocialStoppedError(RuntimeError):
    """The scheduled action no longer has authority to perform an external write."""


class SocialDispatchGuard:
    def __init__(self, runtime: SocialRuntime, *, observation: SocialObservation, decision: SocialDecision,
                 action_id: str, epoch: int, clock: Callable[[], datetime]) -> None:
        self.runtime = runtime
        self.observation = observation
        self.decision = decision
        self.action_id = action_id
        self.epoch = epoch
        self.clock = clock
        self.generation = runtime._generation
        self.connection_generation = runtime._connection_generation
        self.dispatched = False
        self.closed = False

    def valid(self) -> bool:
        runtime = self.runtime
        if (self.closed or not runtime.running or not runtime.connected or self.generation != runtime._generation
                or self.connection_generation != runtime._connection_generation or runtime._epoch != self.epoch):
            return False
        job = runtime._current_job(self.observation.job.key)
        # Any policy revision revokes a decision made under its previous value,
        # including visibility, friend scope, pacing and source age changes.
        if job != self.observation.job or runtime._in_quiet_hours(job, self.clock()):
            return False
        if not runtime._decision_allowed(self.observation, self.decision, self.clock()):
            return False
        if not runtime.policy_check(self.observation, self.decision):
            return False
        return runtime.repository.owns(account_id=runtime.account_id, owner=runtime._owner, epoch=self.epoch, now=self.clock())

    def require_valid(self) -> None:
        if not self.valid():
            raise SocialStoppedError("scheduled social action was revoked")

    def on_dispatch(self) -> None:
        self.require_valid()
        if self.dispatched or not self.runtime.repository.mark_dispatching(
            account_id=self.runtime.account_id, owner=self.runtime._owner, epoch=self.epoch,
            action_id=self.action_id, job=self.observation.job, now=self.clock(),
        ):
            raise SocialStoppedError("scheduled social action has no durable write authority")
        # The database marker has committed before this returns to an adapter.
        self.dispatched = True

    def close(self) -> None:
        self.closed = True

    def record_tool_dispatch(self, action_id: str) -> None:
        """The mature runtime commits each tool's own ledger before a write."""
        self.require_valid()
        self.dispatched = True
        self.observation = replace(self.observation, own_output_action_ids=(
            *self.observation.own_output_action_ids, action_id))


class SocialRuntime:
    def __init__(
        self, *, engine: Engine, account_id: int, jobs: Callable[[], tuple[SocialJob, ...]],
        observe: Callable[[SocialJob, datetime], Awaitable[SocialObservation]],
        decide: Callable[[SocialObservation], Awaitable[SocialDecision]],
        execute: Callable[[SocialObservation, SocialDecision, SocialDispatchGuard], Awaitable[SocialResult]],
        policy_check: Callable[[SocialObservation, SocialDecision], bool],
        options: SocialRuntimeOptions | None = None, clock: Callable[[], datetime] | None = None,
        repository: SocialAgentRepository | None = None,
        intention_repository=None, scope_for_job=None, choose_patrol=None,
    ) -> None:
        if account_id <= 0:
            raise ValueError("social account identity must be positive")
        self.account_id = account_id
        self.jobs, self.observe, self.decide, self.execute, self.policy_check = jobs, observe, decide, execute, policy_check
        self.options = options or SocialRuntimeOptions()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.repository = repository or SocialAgentRepository(engine)
        self.intention_repository = intention_repository
        self.scope_for_job = scope_for_job or (lambda job: job.key)
        self.choose_patrol = choose_patrol or SystemRandom().choice
        self.running = False
        self.connected = False
        self._owner = uuid4().hex
        self._epoch: int | None = None
        self._generation = 0
        self._connection_generation = 0
        self._tick_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._active_ticks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if self.running:
            return
        lease = self.repository.acquire(account_id=self.account_id, owner=self._owner, now=self.clock(),
                                        lease_seconds=self.options.lease_seconds)
        self._epoch = lease.epoch if lease else None
        self.running = True
        self._generation += 1
        self._loop_task = asyncio.create_task(self.run(), name=f"social-agent-{self.account_id}")
        logger.info("social_agent_started account_id=%s owns_lease=%s", self.account_id, lease is not None)

    async def run(self) -> None:
        if not self.running:
            await self.start()
        if self._loop_task is not asyncio.current_task():
            await self._loop_task
            return
        generation = self._generation
        while self.running and generation == self._generation:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Log only the category: provider/database errors may embed
                # full payloads in their exception text or SQL parameters.
                logger.error("social_agent_tick_failed account_id=%s error_type=%s", self.account_id, type(exc).__name__)
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.options.poll_seconds)
            except asyncio.TimeoutError:
                pass

    def set_connected(self, connected: bool) -> None:
        connected = bool(connected)
        if connected != self.connected:
            self.connected = connected
            self._connection_generation += 1
            self._wake.set()

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._generation += 1
        self._connection_generation += 1
        self._wake.set()
        current = asyncio.current_task()
        tasks = set(self._active_ticks)
        if self._loop_task is not None:
            tasks.add(self._loop_task)
        tasks.discard(current)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.options.shutdown_timeout_seconds)
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                logger.warning("social_agent_shutdown_incomplete account_id=%s pending_tasks=%s", self.account_id, len(pending))
        if self._epoch is not None:
            self.repository.release(account_id=self.account_id, owner=self._owner, epoch=self._epoch, now=self.clock())
        self._epoch = None
        self._loop_task = None
        logger.info("social_agent_stopped account_id=%s", self.account_id)

    def _current_job(self, key: str) -> SocialJob | None:
        return next((job for job in self.jobs() if job.key == key), None)

    @staticmethod
    def _in_quiet_hours(job: SocialJob, now: datetime) -> bool:
        return in_quiet_hours(job.quiet_hours, now)

    def _decision_allowed(self, observation: SocialObservation, decision: SocialDecision, now: datetime) -> bool:
        if not isinstance(decision, SocialDecision):
            return False
        expected = {"group": "send_group", "qzone_publish": "publish_qzone", "qzone_comment": "comment_qzone"}
        job = observation.job
        allowed = decision.action == "run_tools" if job.autonomous_tools else decision.action == expected.get(job.kind)
        if not allowed or not isinstance(decision.text, str) or not decision.text.strip() or len(decision.text) > 2000:
            return False
        if (len(decision.source_message_ids) > self.repository.MAX_SOURCE_IDS
                or any(type(value) is not int or value <= 0 for value in decision.source_message_ids)
                or not set(decision.source_message_ids).issubset(observation.source_message_ids)):
            return False
        if job.kind != "group" and (job.scope_id != self.account_id or decision.source_message_ids):
            return False
        if job.kind == "qzone_comment" and not job.autonomous_tools:
            return any(post.owner_id == decision.owner_id and post.post_id == decision.post_id
                       and friend_authorized(job, post.owner_id, observation.authorized_friend_ids) and 0 <= (now - post.published_at).total_seconds() <= job.max_post_age_seconds
                       for post in observation.posts)
        return True

    async def tick(self, now: datetime | None = None) -> None:
        if not self.running or not self.connected:
            return
        generation, connection = self._generation, self._connection_generation
        task = asyncio.current_task()
        self._active_ticks.add(task)
        # Explicit now makes a deterministic test turn. Production uses the
        # injected live clock at each non-idempotent write.
        turn_clock = self.clock if now is None else (lambda: now)
        try:
            async with self._tick_lock:
                if not self.running or not self.connected or generation != self._generation or connection != self._connection_generation:
                    return
                current = turn_clock()
                lease = self.repository.acquire(account_id=self.account_id, owner=self._owner, now=current,
                                                lease_seconds=self.options.lease_seconds)
                if lease is None:
                    self._epoch = None
                    return
                self._epoch = lease.epoch
                due = self.repository.claim_due_jobs(account_id=self.account_id, owner=self._owner, epoch=lease.epoch,
                                                     jobs=tuple(job for job in self.jobs() if not job.event_goals_only), now=current)
                prepared = {}
                patrol = []
                for scheduled in due:
                    if scheduled.job.patrol_interval_seconds is None:
                        continue
                    observed = await asyncio.wait_for(self.observe(scheduled.job, turn_clock()),
                        timeout=self.options.observation_timeout_seconds)
                    if not isinstance(observed, SocialObservation) or observed.job != scheduled.job:
                        raise ValueError("invalid patrol observation")
                    if observed.suppressed_reason:
                        self.repository.note_job(account_id=self.account_id, owner=self._owner, epoch=lease.epoch,
                            job_key=scheduled.job.key, status="silent", reason=observed.suppressed_reason, now=turn_clock())
                    else:
                        patrol.append(scheduled)
                        prepared[scheduled.job.key] = observed
                selected = self.choose_patrol(patrol) if patrol else None
                due = (*[item for item in due if item.job.patrol_interval_seconds is None],
                    *((selected,) if selected else ()))
                due_keys = {item.job.key for item in due}
                goals = ()
                if self.intention_repository is not None:
                    goals = self.intention_repository.claim_due_intentions(owner=self._owner,
                        scopes=tuple(dict.fromkeys(self.scope_for_job(job) for job in self.jobs()
                            if job.patrol_interval_seconds is None or job.key in due_keys)),
                        lease_seconds=self.options.lease_seconds)
                combined = {scheduled.job.key: (scheduled, None) for scheduled in due}
                for claim in goals:
                    job = next((job for job in self.jobs() if self.scope_for_job(job) == claim.scope), None)
                    if job is not None:
                        combined[job.key] = (DueSocialJob(job, claim.due_at), claim)
                for scheduled, claim in combined.values():
                    if not self.running or not self.connected or generation != self._generation or connection != self._connection_generation:
                        return
                    lease = self.repository.acquire(account_id=self.account_id, owner=self._owner, now=turn_clock(),
                                                    lease_seconds=self.options.lease_seconds)
                    if lease is None:
                        self._epoch = None
                        return
                    self._epoch = lease.epoch
                    entered_agent = False
                    try:
                        if claim is not None and not self.intention_repository.renew_intention_claim(claim,
                                lease_seconds=self.options.lease_seconds):
                            continue
                        if claim is not None and not self.repository.sources_usable(scheduled.job, claim.source_message_ids):
                            self.intention_repository.schedule_intention(claim.scope, next_due=None, status="cancelled")
                            continue
                        entered_agent = bool(await self._run_job(scheduled, turn_clock, generation, connection,
                            intention_claim=claim, prepared_observation=prepared.get(scheduled.job.key)))
                    finally:
                        if claim is not None:
                            self.intention_repository.finish_intention_claim(claim, consume=entered_agent,
                                retry_at=None if entered_agent else turn_clock() + timedelta(seconds=max(60, self.options.poll_seconds)))
        finally:
            self._active_ticks.discard(task)

    async def _run_job(self, scheduled: DueSocialJob, clock, generation: int, connection: int,
                       intention_claim=None, prepared_observation=None) -> bool:
        job, epoch = scheduled.job, self._epoch
        action_id = None
        guard = None
        status, reason = "silent", ""
        entered_agent = False
        try:
            if self._current_job(job.key) != job or self._in_quiet_hours(job, clock()):
                reason = "policy_or_quiet_hours"
                return
            if not self.repository.budget_available(account_id=self.account_id, owner=self._owner, epoch=epoch, job=job, now=clock()):
                reason = "budget_or_cooldown"
                return
            observation = prepared_observation or await asyncio.wait_for(self.observe(job, clock()), timeout=self.options.observation_timeout_seconds)
            if (not isinstance(observation, SocialObservation) or observation.job != job
                    or len(observation.posts) > 100 or len(observation.source_message_ids) > self.repository.MAX_SOURCE_IDS
                    or not self.repository.sources_usable(job, observation.source_message_ids)):
                reason = "invalid_observation"
                return
            if observation.suppressed_reason:
                reason = str(observation.suppressed_reason)[:160]
                return
            if job.kind == "qzone_comment":
                posts = tuple(post for post in observation.posts if friend_authorized(job, post.owner_id, observation.authorized_friend_ids)
                              and post.post_id and len(post.post_id) <= 256
                              and 0 <= (clock() - post.published_at).total_seconds() <= job.max_post_age_seconds)
                posts = self.repository.unhandled_posts(account_id=self.account_id, posts=posts)
                observation = replace(observation, posts=posts)
                if not posts and not job.autonomous_tools:
                    reason = "no_unhandled_posts"
                    return
            decision = await asyncio.wait_for(self.decide(observation), timeout=self.options.decision_timeout_seconds)
            if (not self.running or not self.connected or generation != self._generation
                    or connection != self._connection_generation):
                reason = "turn_revoked"
                return
            if not isinstance(decision, SocialDecision) or decision.action == "silent":
                reason = "silent_decision"
                return
            if not self._decision_allowed(observation, decision, clock()) or not self.policy_check(observation, decision):
                reason = "decision_or_policy_rejected"
                return
            action_id = self.repository.reserve_action(account_id=self.account_id, owner=self._owner, epoch=epoch, job=job,
                                                        decision=decision, dedupe_key=(f"goal:{intention_claim.scope}:{intention_claim.schedule_revision}:{scheduled.due_at.isoformat()}"
                                                            if intention_claim is not None else f"wake:{job.key}:{scheduled.due_at.isoformat()}"), now=clock())
            if action_id is None:
                reason = "reservation_rejected"
                return
            guard = SocialDispatchGuard(self, observation=observation, decision=decision, action_id=action_id, epoch=epoch, clock=clock)
            entered_agent = True
            result = await asyncio.wait_for(self.execute(observation, decision, guard), timeout=self.options.execution_timeout_seconds)
            if not isinstance(result, SocialResult) or result.status not in {"sent", "uncertain", "failed", "blocked", "silent"}:
                raise ValueError("invalid social execution result")
            if result.status == "sent" and not guard.dispatched:
                raise ValueError("social execution bypassed its durable dispatch guard")
            # SnowLuma can explicitly acknowledge a comment with an empty
            # comment_id. Publish/group success still needs a real post/msg ID.
            acknowledged_without_id = decision.action == "comment_qzone" or result.reason.startswith("profile_ack:") or result.reason == "tool:comment_qzone"
            status = "uncertain" if result.status == "sent" and not result.receipt_id and not acknowledged_without_id else result.status
            reason = result.reason
            self.repository.finish(account_id=self.account_id, owner=self._owner, epoch=epoch, action_id=action_id,
                                   status=status, receipt_id=result.receipt_id, reason=reason, now=clock())
            if job.patrol_interval_seconds is not None and status in {"sent", "silent"}:
                self.repository.mark_posts_reviewed(account_id=self.account_id, posts=observation.posts, now=clock())
        except asyncio.CancelledError:
            status = "uncertain" if guard is not None and guard.dispatched else "blocked"
            reason = "cancelled_after_dispatch" if status == "uncertain" else "cancelled_before_dispatch"
            if action_id:
                self.repository.finish(account_id=self.account_id, owner=self._owner, epoch=epoch, action_id=action_id,
                                       status=status, reason=reason, now=clock())
            raise
        except Exception as exc:
            status = "uncertain" if guard is not None and guard.dispatched else "failed"
            reason = type(exc).__name__
            if action_id:
                self.repository.finish(account_id=self.account_id, owner=self._owner, epoch=epoch, action_id=action_id,
                                       status=status, reason=reason, now=clock())
            logger.warning("social_agent_job_failed account_id=%s job_key=%s error_type=%s dispatched=%s",
                           self.account_id, job.key, type(exc).__name__, guard is not None and guard.dispatched)
        finally:
            if guard is not None:
                guard.close()
            self.repository.note_job(account_id=self.account_id, owner=self._owner, epoch=epoch, job_key=job.key,
                                     status=status, reason=reason, now=clock())
        return entered_agent
