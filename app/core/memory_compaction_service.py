from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
import logging
from threading import Lock
from typing import Protocol

from app.core.memory_compaction import (
    build_memory_compaction_prompt,
    canonical_key,
    parse_memory_compaction_response,
    structured_digest,
)
from app.core.summarizer import summarize_window
from app.storage.db import session_scope
from app.storage.repositories import JobRepository, MemoryRepository, MessageRepository, SummaryRepository, UserRepository


logger = logging.getLogger(__name__)


class MemoryBackgroundLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def wake(self) -> None: ...

    def enqueue_message(
        self,
        *,
        group_id: int,
        message_id: int,
        now: datetime | None = None,
        backfill_run_id: int | None = None,
        watermark_message_id: int | None = None,
    ): ...

    def enqueue_late_arrival(
        self,
        *,
        group_id: int,
        message_id: int,
        message_timestamp: datetime,
        now: datetime | None = None,
    ): ...


class MemoryCompactionService:
    job_type = "memory_compaction"

    def __init__(
        self,
        *,
        engine,
        llm_client,
        batch_size: int = 50,
        max_facts: int = 24,
        retry_limit: int = 3,
        backfill_windows: int = 24,
        excluded_user_ids: set[int] | None = None,
        background_service: MemoryBackgroundLifecycle | None = None,
        legacy_enabled: bool = True,
        shadow_enabled: bool = False,
    ) -> None:
        self.engine = engine
        self.llm_client = llm_client
        self.batch_size = max(10, int(batch_size))
        self.max_facts = max(1, int(max_facts))
        self.retry_limit = max(1, int(retry_limit))
        self.backfill_windows = max(0, int(backfill_windows))
        self.excluded_user_ids = {int(user_id) for user_id in (excluded_user_ids or set())}
        self.background_service = background_service
        self.legacy_enabled = bool(legacy_enabled)
        self.shadow_enabled = bool(shadow_enabled)
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._shadow_loop: asyncio.AbstractEventLoop | None = None
        self._shadow_futures: set[Future] = set()
        self._shadow_futures_lock = Lock()
        self._accept_shadow_enqueues = False

    async def start(self) -> None:
        self._stop_event.clear()
        with self._shadow_futures_lock:
            self._shadow_loop = asyncio.get_running_loop()
            self._accept_shadow_enqueues = True
        if self.legacy_enabled:
            await asyncio.to_thread(self._prepare_jobs)
        if self.background_service is not None:
            try:
                await self.background_service.start()
            except Exception as exc:
                logger.exception(
                    "memory_background_start_failed error=%s",
                    type(exc).__name__,
                )
        await self.wake()

    async def stop(self) -> None:
        self._stop_event.set()
        with self._shadow_futures_lock:
            self._accept_shadow_enqueues = False
            shadow_futures = tuple(self._shadow_futures)
        task = self._worker_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if shadow_futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in shadow_futures),
                return_exceptions=True,
            )
        if self.background_service is not None:
            try:
                await self.background_service.stop()
            except Exception as exc:
                logger.exception(
                    "memory_background_stop_failed error=%s",
                    type(exc).__name__,
                )

    async def wake(self) -> None:
        if self.legacy_enabled:
            async with self._lock:
                if self._worker_task is None or self._worker_task.done():
                    self._worker_task = asyncio.create_task(self._worker_loop())
        if self.background_service is not None:
            try:
                await self.background_service.wake()
            except Exception as exc:
                logger.exception(
                    "memory_background_wake_failed error=%s",
                    type(exc).__name__,
                )

    def enqueue_episode_allocation(
        self,
        *,
        group_id: int,
        message_id: int,
        now: datetime | None = None,
        backfill_run_id: int | None = None,
        watermark_message_id: int | None = None,
        late_arrival: bool = False,
    ):
        """Best-effort V2 enqueue; V1/reply behavior never depends on it."""
        if self.background_service is None:
            return None
        try:
            if late_arrival:
                enqueue_late = getattr(
                    self.background_service,
                    "enqueue_late_arrival",
                    None,
                )
                if callable(enqueue_late):
                    queued = enqueue_late(
                        group_id=group_id,
                        message_id=message_id,
                        message_timestamp=now or datetime.now(UTC),
                        now=datetime.now(UTC),
                    )
                    if queued is not None:
                        return queued
            return self.background_service.enqueue_message(
                group_id=group_id,
                message_id=message_id,
                now=now,
                backfill_run_id=backfill_run_id,
                watermark_message_id=watermark_message_id,
            )
        except Exception as exc:
            logger.exception(
                "memory_episode_enqueue_failed group_id=%s message_id=%s error=%s",
                group_id,
                message_id,
                type(exc).__name__,
            )
            return None

    def submit_shadow_enqueue(self, callback: Callable[[], object]) -> bool:
        """Schedule shadow persistence off the reply path and track shutdown."""
        if not self.shadow_enabled or self.background_service is None:
            return False
        with self._shadow_futures_lock:
            loop = self._shadow_loop
            if (
                not self._accept_shadow_enqueues
                or loop is None
                or loop.is_closed()
            ):
                return False
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(callback),
                loop,
            )
            self._shadow_futures.add(future)
        future.add_done_callback(self._shadow_enqueue_done)
        return True

    def _shadow_enqueue_done(self, future: Future) -> None:
        with self._shadow_futures_lock:
            self._shadow_futures.discard(future)
        try:
            future.result()
        except Exception as exc:
            logger.warning(
                "memory_shadow_enqueue_failed error=%s",
                type(exc).__name__,
            )

    def _prepare_jobs(self) -> None:
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            jobs.requeue_stale_running_jobs(job_type=self.job_type, now=datetime.now(UTC))
            if self.backfill_windows <= 0:
                return
            messages = MessageRepository(session)
            for group_id in messages.list_group_ids():
                windows = messages.list_recent_group_message_windows(
                    group_id=group_id,
                    batch_size=self.batch_size,
                    limit_windows=self.backfill_windows,
                    excluded_user_ids=self.excluded_user_ids,
                )
                for window in windows:
                    if not window:
                        continue
                    start_id = window[0].id
                    end_id = window[-1].id
                    job_key = f"memory:{group_id}:{start_id}:{end_id}"
                    jobs.add_job(
                        job_type=self.job_type,
                        job_key=job_key,
                        payload_json={
                            "group_id": group_id,
                            "start_id": start_id,
                            "end_id": end_id,
                            "attempts": 0,
                        },
                        run_at=datetime.now(UTC),
                        status="queued",
                    )

    async def _worker_loop(self) -> None:
        while True:
            try:
                claimed = await asyncio.to_thread(self._claim_next_job)
            except Exception:
                logger.exception("memory_compaction_claim_failed")
                if self._stop_event.is_set():
                    return
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                return
            if claimed is None:
                if self._stop_event.is_set():
                    return
                next_run_at = await asyncio.to_thread(self._next_queued_job_at)
                if next_run_at is None:
                    return
                if next_run_at.tzinfo is None:
                    next_run_at = next_run_at.replace(tzinfo=UTC)
                else:
                    next_run_at = next_run_at.astimezone(UTC)
                delay = max(0.1, min(30.0, (next_run_at - datetime.now(UTC)).total_seconds()))
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    continue
                return
            try:
                await asyncio.to_thread(self._process_job, claimed.id, dict(claimed.payload_json or {}), claimed.job_key)
            except Exception as exc:
                logger.exception("memory_compaction_job_failed job_id=%s", claimed.id)
                await asyncio.to_thread(self._record_failure, claimed.id, dict(claimed.payload_json or {}), exc)
            if self._stop_event.is_set():
                return

    def _claim_next_job(self):
        with session_scope(self.engine) as session:
            return JobRepository(session).claim_oldest_queued_job(job_type=self.job_type, now=datetime.now(UTC))

    def _next_queued_job_at(self):
        with session_scope(self.engine) as session:
            return JobRepository(session).next_queued_job_at(job_type=self.job_type)

    def _process_job(self, job_id: int, payload: dict, job_key: str) -> None:
        group_id = int(payload["group_id"])
        start_id = int(payload["start_id"])
        end_id = int(payload["end_id"])
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            users = UserRepository(session)
            rows = messages.list_group_messages_by_id_range(
                group_id=group_id,
                start_id=start_id,
                end_id=end_id,
                limit=max(self.batch_size * 2, 100),
            )
            rows = [row for row in rows if row.user_id not in self.excluded_user_ids]
            if not rows:
                JobRepository(session).mark_job_status(job_id=job_id, status="completed", payload_json=payload)
                return
            users_by_id = users.get_users_by_ids([row.user_id for row in rows])
            source_lines = []
            prompt_messages = []
            for row in rows:
                user = users_by_id.get(row.user_id)
                label = str(getattr(user, "group_card", "") or getattr(user, "nickname", "") or row.user_id).strip()
                text = str(row.plain_text or "").strip()
                if not text:
                    continue
                source_lines.append(f"{label}: {text}")
                prompt_messages.append(
                    {
                        "source_msg_id": row.platform_msg_id,
                        "content": f"user_id={row.user_id}; name={label}; text={text}",
                    }
                )
            if not prompt_messages:
                JobRepository(session).mark_job_status(job_id=job_id, status="completed", payload_json=payload)
                return
            day_key = f"semantic-daily:{rows[-1].timestamp.date().isoformat()}"
            existing_daily = SummaryRepository(session).list_group_summaries(
                scope_id=str(group_id),
                limit=1,
                summary_levels=["semantic_daily"],
                summary_key=day_key,
            )
            previous_digest = existing_daily[-1].content if existing_daily else ""

        fallback = summarize_window(source_lines)
        prompt = build_memory_compaction_prompt(
            messages=prompt_messages,
            previous_digest=previous_digest,
            language="zh",
        )
        raw = self.llm_client.generate_text([prompt])
        compaction = parse_memory_compaction_response(
            raw,
            allowed_source_msg_ids={item["source_msg_id"] for item in prompt_messages},
            allowed_subject_ids={"group", *(str(row.user_id) for row in rows)},
            source_subject_ids={row.platform_msg_id: str(row.user_id) for row in rows},
            fallback_text=fallback,
            strict=True,
        )
        digest = structured_digest(compaction.summary, compaction.facts[: self.max_facts])

        with session_scope(self.engine) as session:
            summaries = SummaryRepository(session)
            memories = MemoryRepository(session)
            rows = MessageRepository(session).list_group_messages_by_id_range(
                group_id=group_id,
                start_id=start_id,
                end_id=end_id,
                limit=max(self.batch_size * 2, 100),
            )
            rows = [row for row in rows if row.user_id not in self.excluded_user_ids]
            if not rows:
                JobRepository(session).mark_job_status(job_id=job_id, status="completed", payload_json=payload)
                return
            start_at = rows[0].timestamp
            end_at = rows[-1].timestamp
            window_summary = summaries.upsert_summary(
                scope_type="group",
                scope_id=str(group_id),
                summary_level="semantic_window",
                summary_key=job_key,
                start_at=start_at,
                end_at=end_at,
                content=digest,
                source_count=len(rows),
                source_start_msg_id=rows[0].platform_msg_id,
                source_end_msg_id=rows[-1].platform_msg_id,
            )
            for fact in compaction.facts[: self.max_facts]:
                valid_until = _parse_timestamp(fact.valid_until)
                if fact.kind == "expired":
                    memories.supersede_current_memories(
                        scope_id=str(group_id),
                        subject_id=fact.subject_id,
                        predicate=fact.predicate,
                        object_text=fact.object_text,
                        valid_until=valid_until or end_at,
                    )
                    continue
                memories.upsert_canonical_memory(
                    scope_type="group",
                    scope_id=str(group_id),
                    subject_type="group" if fact.subject_id == "group" else "user",
                    subject_id=fact.subject_id,
                    memory_kind=fact.kind,
                    canonical_key=canonical_key(fact.kind, fact.subject_id, fact.predicate, fact.object_text),
                    predicate=fact.predicate,
                    object_text=fact.object_text,
                    content=fact.content,
                    importance=fact.importance,
                    confidence=fact.confidence,
                    source_msg_ids=list(fact.source_msg_ids),
                    valid_from=end_at,
                    valid_until=valid_until,
                )
            existing_daily = summaries.list_group_summaries(
                scope_id=str(group_id),
                limit=1,
                summary_levels=["semantic_daily"],
                summary_key=day_key,
            )
            previous_daily = existing_daily[-1] if existing_daily else None
            daily_rows = MessageRepository(session).list_group_messages_for_day(
                group_id=group_id,
                day=end_at.date(),
                excluded_user_ids=self.excluded_user_ids,
            )
            daily_rows = daily_rows or rows
            summaries.upsert_summary(
                scope_type="group",
                scope_id=str(group_id),
                summary_level="semantic_daily",
                summary_key=day_key,
                start_at=previous_daily.start_at if previous_daily else start_at,
                end_at=end_at,
                content=(
                    previous_daily.content
                    if previous_daily is not None and compaction.rejected_fact_count > 0
                    else digest
                ),
                source_count=len(daily_rows),
                source_start_msg_id=daily_rows[0].platform_msg_id,
                source_end_msg_id=daily_rows[-1].platform_msg_id,
                source_summary_ids=list(
                    dict.fromkeys([*(previous_daily.source_summary_ids if previous_daily else []), window_summary.id])
                ),
            )
            payload.update(
                {
                    "summary_id": window_summary.id,
                    "fact_count": len(compaction.facts),
                    "rejected_fact_count": compaction.rejected_fact_count,
                    "attempts": int(payload.get("attempts", 0)),
                }
            )
            JobRepository(session).mark_job_status(job_id=job_id, status="completed", payload_json=payload)
        logger.info(
            "memory_compaction_completed group_id=%s start_id=%s end_id=%s facts=%s rejected_facts=%s",
            group_id,
            start_id,
            end_id,
            len(compaction.facts),
            compaction.rejected_fact_count,
        )

    def _record_failure(self, job_id: int, payload: dict, exc: Exception) -> None:
        attempts = int(payload.get("attempts", 0)) + 1
        payload.update({"attempts": attempts, "last_error": str(exc)[:500]})
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            if attempts >= self.retry_limit:
                jobs.mark_job_status(job_id=job_id, status="failed", payload_json=payload)
                return
            jobs.retry_job(
                job_id=job_id,
                payload_json=payload,
                run_at=datetime.now(UTC) + timedelta(seconds=min(300, 5 * (2 ** (attempts - 1)))),
            )


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
