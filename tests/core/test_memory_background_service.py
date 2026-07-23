from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest
from sqlalchemy import text

from app.core.memory_background_service import (
    BackgroundEpisode,
    BackgroundJob,
    BackgroundMessage,
    DerivedFact,
    EpisodeDerivation,
    LateArrivalPlan,
    MemoryBackgroundService,
    ShadowEvaluation,
    ShadowJobRequest,
    SqlAlchemyMemoryBackgroundStore,
    _compatible_segmentation_generation,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    EpisodeRepository,
    GroupRepository,
    JobRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)


NOW = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)


class FakeStore:
    def __init__(self) -> None:
        self.jobs: list[BackgroundJob] = []
        self.messages: dict[int, list[BackgroundMessage]] = {}
        self.episodes: dict[int, BackgroundEpisode] = {}
        self.episode_messages: dict[int, list[BackgroundMessage]] = {}
        self.documents: list[tuple[int, tuple[int, ...], str]] = []
        self.compactions: list[tuple[int, EpisodeDerivation]] = []
        self.invalidated: list[tuple[int, str]] = []
        self.shadow_results: list[tuple[int, ShadowEvaluation]] = []
        self.recovered_at: datetime | None = None
        self.failures: list[tuple[int, str, int]] = []
        self.completed: list[tuple[int, int]] = []
        self._next_job_id = 1
        self._next_episode_id = 1

    def enqueue_allocator(
        self,
        *,
        group_id: int,
        latest_message_id: int,
        segmentation_generation: str,
        backfill_run_id: int | None,
        watermark_message_id: int | None,
        now: datetime,
    ) -> BackgroundJob:
        for index, job in enumerate(self.jobs):
            if (
                job.job_type == "episode_allocate"
                and job.group_id == group_id
                and job.target_generation == segmentation_generation
            ):
                updated = replace(
                    job,
                    payload={
                        **job.payload,
                        "latest_message_id": latest_message_id,
                        "watermark_message_id": watermark_message_id,
                    },
                    requested_generation=job.requested_generation + 1,
                    status="running" if job.status == "running" else "queued",
                )
                self.jobs[index] = updated
                return updated
        job = BackgroundJob(
            id=self._next_job_id,
            job_type="episode_allocate",
            group_id=group_id,
            payload={
                "latest_message_id": latest_message_id,
                "watermark_message_id": watermark_message_id,
            },
            status="queued",
            requested_generation=1,
            claimed_generation=0,
            attempt_count=0,
            max_attempts=3,
            target_generation=segmentation_generation,
            backfill_run_id=backfill_run_id,
        )
        self._next_job_id += 1
        self.jobs.append(job)
        return job

    def enqueue_episode_processing(
        self,
        *,
        episode_id: int,
        group_id: int,
        compaction_generation: str,
        backfill_run_id: int | None,
        now: datetime,
    ) -> BackgroundJob:
        job = BackgroundJob(
            id=self._next_job_id,
            job_type="memory_episode_process",
            group_id=group_id,
            payload={"episode_id": episode_id},
            status="queued",
            requested_generation=1,
            claimed_generation=0,
            attempt_count=0,
            max_attempts=3,
            target_generation=compaction_generation,
            backfill_run_id=backfill_run_id,
        )
        self._next_job_id += 1
        self.jobs.append(job)
        return job

    def enqueue_shadow(
        self,
        *,
        request: ShadowJobRequest,
        now: datetime,
    ) -> BackgroundJob:
        del now
        job = BackgroundJob(
            id=self._next_job_id,
            job_type="memory_shadow_evaluate",
            group_id=request.group_id,
            payload={
                "message_id": request.message_id,
                "config_generation": request.config_generation,
                "index_generation": request.index_generation,
            },
            status="queued",
            requested_generation=1,
            claimed_generation=0,
            attempt_count=0,
            max_attempts=3,
            target_generation=(
                f"{request.config_generation}:{request.index_generation}"
            ),
        )
        self._next_job_id += 1
        self.jobs.append(job)
        return job

    def recover_stale_jobs(self, *, now: datetime) -> int:
        self.recovered_at = now
        recovered = 0
        for index, job in enumerate(self.jobs):
            if job.status == "running":
                self.jobs[index] = replace(job, status="queued")
                recovered += 1
        return recovered

    def claim_next_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> BackgroundJob | None:
        del worker_id, now, lease_seconds
        for index, job in enumerate(self.jobs):
            if job.status != "queued":
                continue
            claimed = replace(
                job,
                status="running",
                claimed_generation=job.requested_generation,
            )
            self.jobs[index] = claimed
            return claimed
        return None

    def complete_job(
        self,
        *,
        job: BackgroundJob,
        worker_id: str,
        now: datetime,
    ) -> bool:
        del worker_id, now
        for index, current in enumerate(self.jobs):
            if current.id != job.id or current.status != "running":
                continue
            if current.requested_generation != job.claimed_generation:
                self.jobs[index] = replace(current, status="queued")
            else:
                self.jobs[index] = replace(current, status="completed")
            self.completed.append((job.id, job.claimed_generation))
            return True
        return False

    def record_failure(
        self,
        *,
        job: BackgroundJob,
        error_code: str,
        now: datetime,
        retry_delay_seconds: int,
    ) -> bool:
        del now, retry_delay_seconds
        for index, current in enumerate(self.jobs):
            if current.id != job.id:
                continue
            attempts = current.attempt_count + 1
            status = "failed" if attempts >= current.max_attempts else "queued"
            self.jobs[index] = replace(
                current,
                status=status,
                attempt_count=attempts,
            )
            self.failures.append((job.id, error_code, attempts))
            return status != "failed"
        return False

    def list_unassigned_messages(
        self,
        *,
        group_id: int,
        segmentation_generation: str,
        through_message_id: int,
        watermark_message_id: int | None,
    ) -> list[BackgroundMessage]:
        del segmentation_generation
        upper = min(
            through_message_id,
            watermark_message_id
            if watermark_message_id is not None
            else through_message_id,
        )
        assigned = {
            message.id
            for episode_id, messages in self.episode_messages.items()
            if self.episodes[episode_id].group_id == group_id
            for message in messages
        }
        return [
            message
            for message in self.messages.get(group_id, [])
            if message.id <= upper and message.id not in assigned and not message.is_reserved
        ]

    def get_open_episode(
        self,
        *,
        group_id: int,
        segmentation_generation: str,
    ) -> BackgroundEpisode | None:
        for episode in self.episodes.values():
            if (
                episode.group_id == group_id
                and episode.status == "open"
                and episode.segmentation_generation == segmentation_generation
            ):
                return episode
        return None

    def create_episode(
        self,
        *,
        group_id: int,
        first_message: BackgroundMessage,
        segmentation_generation: str,
        backfill_run_id: int | None,
    ) -> BackgroundEpisode:
        episode = BackgroundEpisode(
            id=self._next_episode_id,
            group_id=group_id,
            status="open",
            segmentation_generation=segmentation_generation,
            compaction_generation="",
            message_count=0,
            token_count=0,
            started_at=first_message.timestamp,
            ended_at=None,
            content_hash="",
            backfill_run_id=backfill_run_id,
        )
        self._next_episode_id += 1
        self.episodes[episode.id] = episode
        self.episode_messages[episode.id] = []
        return episode

    def append_message(
        self,
        *,
        episode_id: int,
        message: BackgroundMessage,
        estimated_tokens: int,
    ) -> BackgroundEpisode:
        rows = self.episode_messages[episode_id]
        if all(existing.id != message.id for existing in rows):
            rows.append(message)
        episode = self.episodes[episode_id]
        updated = replace(
            episode,
            message_count=len(rows),
            token_count=episode.token_count + estimated_tokens,
            ended_at=message.timestamp,
        )
        self.episodes[episode_id] = updated
        return updated

    def list_episode_messages(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> list[BackgroundMessage]:
        episode = self.episodes.get(episode_id)
        if episode is None or episode.group_id != group_id:
            return []
        return list(self.episode_messages.get(episode_id, []))

    def close_episode(
        self,
        *,
        episode_id: int,
        reason: str,
        content_hash: str,
        compaction_generation: str,
        now: datetime,
    ) -> BackgroundEpisode:
        del now
        episode = self.episodes[episode_id]
        rows = self.episode_messages[episode_id]
        closed = replace(
            episode,
            status="closed",
            ended_at=rows[-1].timestamp,
            content_hash=content_hash,
            compaction_generation=compaction_generation,
        )
        self.episodes[episode_id] = closed
        return closed

    def list_idle_open_episodes(
        self,
        *,
        idle_before: datetime,
        segmentation_generation: str,
    ) -> list[BackgroundEpisode]:
        return [
            episode
            for episode in self.episodes.values()
            if episode.status == "open"
            and episode.segmentation_generation == segmentation_generation
            and episode.ended_at is not None
            and episode.ended_at <= idle_before
        ]

    def load_episode(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> tuple[BackgroundEpisode, list[BackgroundMessage]] | None:
        episode = self.episodes.get(episode_id)
        if episode is None or episode.group_id != group_id:
            return None
        return episode, list(self.episode_messages.get(episode_id, []))

    def persist_episode_derivation(
        self,
        *,
        episode_id: int,
        expected_compaction_generation: str,
        derivation: EpisodeDerivation,
        now: datetime,
    ) -> bool:
        del now
        episode = self.episodes[episode_id]
        if episode.compaction_generation != expected_compaction_generation:
            return False
        self.compactions.append((episode_id, derivation))
        self.episodes[episode_id] = replace(episode, status="processed")
        return True

    def persist_shadow_result(
        self,
        *,
        job: BackgroundJob,
        result: ShadowEvaluation,
        now: datetime,
    ) -> bool:
        del now
        self.shadow_results.append((job.id, result))
        return True

    def invalidate_episode_derivations(
        self,
        *,
        episode_id: int,
        expected_compaction_generation: str,
        now: datetime,
    ) -> bool:
        del now
        episode = self.episodes[episode_id]
        if episode.compaction_generation != expected_compaction_generation:
            return False
        self.invalidated.append((episode_id, expected_compaction_generation))
        return True

    def prepare_late_arrival_resegment(
        self,
        *,
        group_id: int,
        message_id: int,
        message_timestamp: datetime,
        segmentation_generation: str,
        compaction_generation: str,
        now: datetime,
    ) -> LateArrivalPlan | None:
        del message_timestamp, now
        affected = [
            episode
            for episode in self.episodes.values()
            if episode.group_id == group_id
            and episode.segmentation_generation == segmentation_generation
            and episode.status != "superseded"
        ]
        if not affected:
            return None
        through_message_id = max(
            [message_id, *(message.id for message in self.messages[group_id])]
        )
        for episode in affected:
            self.episodes[episode.id] = replace(
                episode,
                status="superseded",
                compaction_generation=compaction_generation,
            )
            self.episode_messages[episode.id] = []
            self.invalidated.append((episode.id, compaction_generation))
        return LateArrivalPlan(
            group_id=group_id,
            through_message_id=through_message_id,
            watermark_message_id=None,
            backfill_run_id=affected[0].backfill_run_id,
            superseded_episode_ids=tuple(episode.id for episode in affected),
        )


class FakeDeriver:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[int, tuple[int, ...], tuple[str, ...]]] = []

    def derive(
        self,
        *,
        episode: BackgroundEpisode,
        messages: tuple[BackgroundMessage, ...],
        windows,
    ) -> EpisodeDerivation:
        self.calls.append(
            (
                episode.id,
                tuple(message.id for message in messages),
                tuple(window.content for window in windows),
            )
        )
        if self.fail is not None:
            raise self.fail
        return EpisodeDerivation(
            summary="safe summary",
            facts=(),
            events=(),
            windows=windows,
        )


class FakeEmbedder:
    available = True

    def __init__(self) -> None:
        self.payloads: list[list[str]] = []

    def embed_documents(self, documents) -> list[list[float]]:
        values = list(documents)
        self.payloads.append(values)
        return [[float(index), 1.0] for index, _document in enumerate(values)]


class FakeShadowEvaluator:
    def __init__(
        self,
        *,
        candidate_scores: tuple[tuple[int, float], ...] = (),
    ) -> None:
        self.requests: list[ShadowJobRequest] = []
        self.candidate_scores = candidate_scores

    def evaluate(self, request: ShadowJobRequest) -> ShadowEvaluation:
        self.requests.append(request)
        return ShadowEvaluation(
            source_message_ids=("m-1", "m-2"),
            route_counts={"fts": 2, "vector": 1},
            token_count=123,
            latency_ms=9,
            rewrite_used=False,
            fallback_used=False,
            candidate_scores=self.candidate_scores,
        )


def _message(
    message_id: int,
    *,
    minute: int,
    text: str = "普通聊天",
    user_id: int = 42,
    reply_to_msg_id: str | None = None,
    mentioned_bot: bool = False,
    blocked: bool = False,
    reserved: bool = False,
) -> BackgroundMessage:
    return BackgroundMessage(
        id=message_id,
        platform_msg_id=f"m-{message_id}",
        group_id=10001,
        user_id=user_id,
        timestamp=NOW + timedelta(minutes=minute),
        plain_text=text,
        reply_to_msg_id=reply_to_msg_id,
        mentioned_bot=mentioned_bot,
        is_blocked=blocked,
        is_reserved=reserved,
    )


def _service(
    store: FakeStore,
    deriver: FakeDeriver | None = None,
    **kwargs,
) -> MemoryBackgroundService:
    return MemoryBackgroundService(
        store=store,
        deriver=deriver or FakeDeriver(),
        worker_id="test-worker",
        segmentation_generation="segment-v2",
        compaction_generation="compact-v2",
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
        chunk_max_tokens=1800,
        chunk_overlap_messages=5,
        poll_interval_seconds=0.01,
        **kwargs,
    )


def test_enqueue_is_constant_scope_and_allocation_runs_only_in_worker() -> None:
    store = FakeStore()
    store.messages[10001] = [_message(1, minute=0)]
    service = _service(store)

    queued = service.enqueue_message(group_id=10001, message_id=1, now=NOW)

    assert queued.requested_generation == 1
    assert store.episodes == {}
    assert service.run_once(now=NOW)
    assert len(store.episodes) == 1
    assert [message.id for message in store.episode_messages[1]] == [1]


def test_allocator_closes_at_boundary_but_extends_continuous_bot_reply() -> None:
    store = FakeStore()
    store.messages[10001] = [
        _message(1, minute=0, user_id=999),
        _message(2, minute=31, mentioned_bot=True),
        _message(3, minute=62),
    ]
    service = _service(store, bot_user_id=999)
    service.enqueue_message(group_id=10001, message_id=3, now=NOW)

    assert service.run_once(now=NOW)

    assert len(store.episodes) == 2
    assert [message.id for message in store.episode_messages[1]] == [1, 2]
    assert [message.id for message in store.episode_messages[2]] == [3]
    assert store.episodes[1].status == "closed"
    assert any(job.job_type == "memory_episode_process" for job in store.jobs)


def test_blocked_and_reserved_text_never_reaches_windows_or_deriver() -> None:
    store = FakeStore()
    secret = "sensitive generated detail"
    store.messages[10001] = [
        _message(1, minute=0, text="safe one"),
        _message(2, minute=1, text=secret, blocked=True),
        _message(3, minute=2, text="not delivered", reserved=True),
        _message(4, minute=40, text="safe two"),
    ]
    deriver = FakeDeriver()
    embedder = FakeEmbedder()
    service = _service(store, deriver, embedder=embedder)
    service.enqueue_message(group_id=10001, message_id=4, now=NOW)

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    assert deriver.calls
    _, source_ids, window_contents = deriver.calls[0]
    assert source_ids == (1,)
    assert all(secret not in content for content in window_contents)
    assert all("not delivered" not in content for content in window_contents)
    assert all(secret not in content for content in embedder.payloads[0])
    assert all("not delivered" not in content for content in embedder.payloads[0])
    assert store.compactions[0][1].windows[0].source_message_ids == (1,)
    # The blocked row remains in the raw episode membership for audit/recent use.
    assert [message.id for message in store.episode_messages[1]] == [1, 2]


def test_stale_compaction_generation_cannot_publish_results() -> None:
    store = FakeStore()
    rows = [_message(1, minute=0)]
    episode = store.create_episode(
        group_id=10001,
        first_message=rows[0],
        segmentation_generation="segment-v2",
        backfill_run_id=7,
    )
    store.append_message(episode_id=episode.id, message=rows[0], estimated_tokens=1)
    store.close_episode(
        episode_id=episode.id,
        reason="idle",
        content_hash="hash",
        compaction_generation="compact-v3",
        now=NOW,
    )
    store.enqueue_episode_processing(
        episode_id=episode.id,
        group_id=10001,
        compaction_generation="compact-v2",
        backfill_run_id=7,
        now=NOW,
    )
    deriver = FakeDeriver()
    service = _service(store, deriver)

    assert service.run_once(now=NOW)

    assert deriver.calls == []
    assert store.compactions == []
    assert store.jobs[0].status == "completed"


def test_failure_retries_are_finite_and_record_only_error_code() -> None:
    store = FakeStore()
    row = _message(1, minute=0)
    episode = store.create_episode(
        group_id=10001,
        first_message=row,
        segmentation_generation="segment-v2",
        backfill_run_id=None,
    )
    store.append_message(episode_id=episode.id, message=row, estimated_tokens=1)
    store.close_episode(
        episode_id=episode.id,
        reason="idle",
        content_hash="hash",
        compaction_generation="compact-v2",
        now=NOW,
    )
    store.enqueue_episode_processing(
        episode_id=episode.id,
        group_id=10001,
        compaction_generation="compact-v2",
        backfill_run_id=None,
        now=NOW,
    )
    service = _service(store, FakeDeriver(fail=RuntimeError("secret payload")))

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    assert store.jobs[0].status == "failed"
    assert [failure[1] for failure in store.failures] == ["RuntimeError"] * 3
    assert all("secret payload" not in failure[1] for failure in store.failures)


def test_backfill_watermark_excludes_live_tail_and_late_arrival_rearms_generation() -> None:
    store = FakeStore()
    store.messages[10001] = [
        _message(1, minute=0),
        _message(2, minute=1),
        _message(3, minute=2),
    ]
    service = _service(store)
    first = service.enqueue_message(
        group_id=10001,
        message_id=3,
        now=NOW,
        backfill_run_id=9,
        watermark_message_id=2,
    )

    assert service.run_once(now=NOW)
    assert [message.id for message in store.episode_messages[1]] == [1, 2]

    rearmed = service.enqueue_message(
        group_id=10001,
        message_id=3,
        now=NOW,
        backfill_run_id=None,
    )
    assert rearmed.id == first.id
    assert rearmed.requested_generation == 2
    assert service.run_once(now=NOW)
    assert [message.id for message in store.episode_messages[1]] == [1, 2]
    assert [message.id for message in store.episode_messages[2]] == [3]


def test_late_arrival_invalidates_old_generation_and_resegments_in_time_order() -> None:
    store = FakeStore()
    first = _message(1, minute=0)
    second = _message(2, minute=40)
    late = _message(3, minute=20)
    store.messages[10001] = [first, second, late]
    old = store.create_episode(
        group_id=10001,
        first_message=first,
        segmentation_generation="segment-v2",
        backfill_run_id=9,
    )
    store.append_message(episode_id=old.id, message=first, estimated_tokens=1)
    store.append_message(episode_id=old.id, message=second, estimated_tokens=1)
    store.close_episode(
        episode_id=old.id,
        reason="idle",
        content_hash="old-hash",
        compaction_generation="compact-v2",
        now=NOW,
    )
    service = _service(store)

    queued = service.enqueue_late_arrival(
        group_id=10001,
        message_id=late.id,
        message_timestamp=late.timestamp,
        now=NOW,
    )

    assert queued is not None
    assert store.episodes[old.id].status == "superseded"
    assert store.invalidated == [(old.id, "compact-v2")]
    assert service.run_once(now=NOW)
    current = [
        episode
        for episode in store.episodes.values()
        if episode.status == "open"
    ]
    assert len(current) == 1
    assert [
        message.id for message in store.episode_messages[current[0].id]
    ] == [1, 3, 2]


def test_late_arrival_generations_from_the_same_base_can_share_an_open_episode() -> None:
    assert _compatible_segmentation_generation(
        "segment-v2:late:101",
        "segment-v2:late:102",
    )
    assert _compatible_segmentation_generation("segment-v2:late:101", "segment-v2")
    assert not _compatible_segmentation_generation(
        "segment-v2:late:101",
        "segment-v3:late:102",
    )


def test_fact_with_unknown_or_blocked_source_id_is_retried_not_persisted() -> None:
    class InvalidFactDeriver(FakeDeriver):
        def derive(self, *, episode, messages, windows) -> EpisodeDerivation:
            return EpisodeDerivation(
                summary="unsafe",
                facts=(
                    DerivedFact(
                        content="unsupported",
                        source_msg_ids=("blocked-source",),
                    ),
                ),
                events=(),
                windows=windows,
            )

    store = FakeStore()
    row = _message(1, minute=0)
    episode = store.create_episode(
        group_id=10001,
        first_message=row,
        segmentation_generation="segment-v2",
        backfill_run_id=None,
    )
    store.append_message(episode_id=episode.id, message=row, estimated_tokens=1)
    store.close_episode(
        episode_id=episode.id,
        reason="idle",
        content_hash="hash",
        compaction_generation="compact-v2",
        now=NOW,
    )
    store.enqueue_episode_processing(
        episode_id=episode.id,
        group_id=10001,
        compaction_generation="compact-v2",
        backfill_run_id=None,
        now=NOW,
    )
    service = _service(store, InvalidFactDeriver())

    assert service.run_once(now=NOW)

    assert store.compactions == []
    assert store.jobs[0].status == "queued"
    assert store.failures[0][1] == "ValueError"


def test_shadow_job_payload_contains_only_ids_and_safe_generations() -> None:
    store = FakeStore()
    evaluator = FakeShadowEvaluator()
    service = _service(store, shadow_evaluator=evaluator)
    request = ShadowJobRequest(
        group_id=10001,
        message_id=77,
        config_generation="config-2",
        index_generation="index-4",
    )

    queued = service.enqueue_shadow(request, now=NOW)

    assert queued is not None
    serialized_payload = repr(queued.payload)
    assert "secret query text" not in serialized_payload
    assert set(queued.payload) == {
        "message_id",
        "config_generation",
        "index_generation",
    }
    assert service.run_once(now=NOW)
    assert evaluator.requests == [request]
    assert store.shadow_results[0][1].source_message_ids == ("m-1", "m-2")


def test_shadow_candidate_scores_reject_non_finite_values() -> None:
    store = FakeStore()
    evaluator = FakeShadowEvaluator(candidate_scores=((7, float("nan")),))
    service = _service(store, shadow_evaluator=evaluator)
    service.enqueue_shadow(
        ShadowJobRequest(
            group_id=10001,
            message_id=77,
            config_generation="config-2",
            index_generation="index-4",
        ),
        now=NOW,
    )

    assert service.run_once(now=NOW)
    assert store.shadow_results == []
    assert store.failures[0][1] == "ValueError"


@pytest.mark.asyncio
async def test_start_recovers_leases_and_stop_is_graceful() -> None:
    store = FakeStore()
    service = _service(store)

    await service.start()
    await asyncio.sleep(0.03)
    await asyncio.wait_for(service.stop(), timeout=0.5)

    assert store.recovered_at is not None
    assert service.worker_task is None


def _seed_sqlite_messages(engine, rows: list[BackgroundMessage]) -> list[int]:
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        for user_id in sorted({row.user_id for row in rows}):
            UserRepository(session).upsert_user(
                user_id=user_id,
                nickname=str(user_id),
                group_card=str(user_id),
            )
        persisted_ids: list[int] = []
        messages = MessageRepository(session)
        for row in rows:
            raw_json = {}
            if row.is_blocked:
                raw_json = {
                    "direction": "outbound",
                    "delivery_state": "blocked",
                    "failure_kind": "qq_sensitive_content",
                }
            elif row.is_reserved:
                raw_json = {
                    "direction": "outbound",
                    "delivery_state": "reserved",
                }
            persisted = messages.add_group_message(
                platform_msg_id=row.platform_msg_id,
                group_id=row.group_id,
                user_id=row.user_id,
                timestamp=row.timestamp,
                plain_text=row.plain_text,
                raw_json=raw_json,
                msg_type="text",
                reply_to_msg_id=row.reply_to_msg_id,
                mentioned_bot=row.mentioned_bot,
            )
            session.flush()
            persisted_ids.append(persisted.id)
        return persisted_ids


def test_sqlalchemy_store_end_to_end_allocate_close_derive_complete(
    sqlite_engine,
) -> None:
    message_ids = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(1, minute=0, text="safe first"),
            _message(2, minute=40, text="safe second"),
        ],
    )
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(
        group_id=10001,
        message_id=message_ids[-1],
        now=NOW,
    )

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        allocator = jobs.list_jobs(
            job_type="episode_allocate",
            statuses=["completed"],
        )
        processors = jobs.list_jobs(
            job_type="memory_episode_process",
            statuses=["completed"],
        )
        assert len(allocator) == 1
        assert len(processors) == 1
        episode_id = int(processors[0].payload_json["episode_id"])
        episode = EpisodeRepository(session).get_episode(episode_id)
        assert episode is not None
        assert episode.status == "processed"
        assert episode.compaction_version == "compact-v2"
        documents = RetrievalDocumentRepository(
            session
        ).search_group_documents_fts(
            group_id=10001,
            query="safe first",
            limit=10,
        )
        assert documents
        assert all(document.group_id == 10001 for document in documents)


def test_sqlalchemy_worker_does_not_claim_another_generation(
    sqlite_engine,
) -> None:
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    store.enqueue_allocator(
        group_id=10001,
        latest_message_id=999,
        segmentation_generation="segment-v3",
        backfill_run_id=None,
        watermark_message_id=None,
        now=NOW,
    )

    assert service.run_once(now=NOW) is False

    with session_scope(sqlite_engine) as session:
        queued = JobRepository(session).list_jobs(
            job_type="episode_allocate",
            statuses=["queued"],
        )
    assert len(queued) == 1
    assert queued[0].target_generation == "segment-v3"


def test_sqlalchemy_store_rejects_completion_from_another_worker(
    sqlite_engine,
) -> None:
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(group_id=10001, message_id=999, now=NOW)
    claimed = store.claim_next_job(
        worker_id="worker-a",
        now=NOW,
        lease_seconds=30,
    )
    assert claimed is not None

    with pytest.raises(ValueError, match="worker identity"):
        store.complete_job(job=claimed, worker_id="worker-b", now=NOW)


@pytest.mark.asyncio
async def test_sqlalchemy_store_restart_recovers_expired_lease(
    sqlite_engine,
) -> None:
    message_id = _seed_sqlite_messages(
        sqlite_engine,
        [_message(1, minute=-60, text="recover me")],
    )[0]
    base = datetime.now(UTC) - timedelta(seconds=10)
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    store.enqueue_allocator(
        group_id=10001,
        latest_message_id=message_id,
        segmentation_generation="segment-v2",
        backfill_run_id=None,
        watermark_message_id=None,
        now=base,
    )
    with session_scope(sqlite_engine) as session:
        claimed = JobRepository(session).claim_coalescing_job(
            job_type="episode_allocate",
            worker_id="crashed-worker",
            now=base,
            lease_seconds=1,
        )
        assert claimed is not None

    service = _service(store)
    await service.start()
    for _ in range(50):
        with session_scope(sqlite_engine) as session:
            terminal = JobRepository(session).list_jobs(
                job_type="episode_allocate",
                statuses=["completed"],
            )
        if terminal:
            break
        await asyncio.sleep(0.01)
    await asyncio.wait_for(service.stop(), timeout=1.0)

    assert terminal


def test_sqlalchemy_store_backfill_watermark_and_generations_are_persisted(
    sqlite_engine,
) -> None:
    message_ids = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(1, minute=0),
            _message(2, minute=1),
            _message(3, minute=2),
        ],
    )
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(
        group_id=10001,
        message_id=message_ids[-1],
        now=NOW,
        backfill_run_id=9,
        watermark_message_id=message_ids[1],
    )

    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        unassigned = EpisodeRepository(session).list_unassigned_messages(
            group_id=10001,
            watermark_message_id=message_ids[-1],
        )
        assert [message.id for message in unassigned] == [message_ids[-1]]
        allocator = JobRepository(session).list_jobs(
            job_type="episode_allocate",
            statuses=["completed"],
        )[0]
        processor = JobRepository(session).list_jobs(
            job_type="memory_episode_process",
            statuses=["queued"],
        )[0]
        assert allocator.backfill_run_id == 9
        assert allocator.target_generation == "segment-v2"
        assert processor.backfill_run_id == 9
        assert processor.target_generation == "compact-v2"


def test_sqlalchemy_shadow_job_persists_only_safe_metrics(sqlite_engine) -> None:
    message_ids = _seed_sqlite_messages(
        sqlite_engine,
        [_message(1, minute=0), _message(2, minute=1)],
    )
    evaluator = FakeShadowEvaluator()
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store, shadow_evaluator=evaluator)
    service.enqueue_shadow(
        ShadowJobRequest(
            group_id=10001,
            message_id=message_ids[-1],
            config_generation="config-2",
            index_generation="index-4",
        ),
        now=NOW,
    )

    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session).list_jobs(
            job_type="memory_shadow_evaluate",
            statuses=["completed"],
        )
        assert len(jobs) == 1
        payload = jobs[0].payload_json
        assert set(payload) == {
            "group_id",
            "message_id",
            "config_generation",
            "index_generation",
            "result",
        }
        assert set(payload["result"]) == {
            "source_message_ids",
            "candidate_scores",
            "route_counts",
            "token_count",
            "latency_ms",
            "rewrite_used",
            "fallback_used",
            "error_category",
        }
        assert "query" not in repr(payload).lower()
        assert "text" not in repr(payload).lower()


def test_sqlalchemy_episode_failure_retries_are_finite(sqlite_engine) -> None:
    message_id = _seed_sqlite_messages(
        sqlite_engine,
        [_message(1, minute=0, text="safe source")],
    )[0]
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine, max_attempts=3)
    service = _service(
        store,
        FakeDeriver(fail=RuntimeError("provider detail")),
    )
    service.enqueue_message(
        group_id=10001,
        message_id=message_id,
        now=NOW,
        backfill_run_id=7,
        watermark_message_id=message_id,
    )
    assert service.run_once(now=NOW)

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW + timedelta(seconds=10))
    assert service.run_once(now=NOW + timedelta(seconds=30))

    with session_scope(sqlite_engine) as session:
        failed = JobRepository(session).list_jobs(
            job_type="memory_episode_process",
            statuses=["failed"],
        )
        assert len(failed) == 1
        assert failed[0].attempt_count == 3
        assert failed[0].last_error_code == "RuntimeError"
        episode = EpisodeRepository(session).get_episode(
            int(failed[0].payload_json["episode_id"])
        )
        assert episode is not None
        assert episode.status == "failed"


def test_sqlalchemy_blocked_only_episode_produces_no_derived_document(
    sqlite_engine,
) -> None:
    blocked_id = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(
                1,
                minute=0,
                text="blocked secret detail",
                blocked=True,
            )
        ],
    )[0]
    deriver = FakeDeriver()
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store, deriver)
    service.enqueue_message(
        group_id=10001,
        message_id=blocked_id,
        now=NOW,
        watermark_message_id=blocked_id,
    )

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    assert deriver.calls == []
    with session_scope(sqlite_engine) as session:
        assert (
            RetrievalDocumentRepository(session).search_group_documents_fts(
                group_id=10001,
                query="blocked secret detail",
                limit=10,
            )
            == []
        )
        raw = MessageRepository(session).get_by_platform_msg_id("m-1")
        assert raw is not None
        assert raw.plain_text == "blocked secret detail"


def test_sqlalchemy_mixed_episode_never_attaches_blocked_provenance(
    sqlite_engine,
) -> None:
    message_ids = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(1, minute=0, text="safe evidence"),
            _message(
                2,
                minute=1,
                text="blocked secret detail",
                blocked=True,
            ),
            _message(3, minute=40, text="next episode"),
        ],
    )
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(
        group_id=10001,
        message_id=message_ids[-1],
        now=NOW,
    )

    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(
            session
        ).search_group_documents_fts(
            group_id=10001,
            query="safe evidence",
            limit=10,
        )
        assert documents
        for document in documents:
            sources = RetrievalDocumentRepository(
                session
            ).list_source_message_ids(
                document_id=document.id,
                group_id=10001,
            )
            assert message_ids[1] not in sources
        blocked = MessageRepository(session).get_by_platform_msg_id("m-2")
        assert blocked is not None
        assert blocked.plain_text == "blocked secret detail"


def test_sqlalchemy_late_arrival_invalidates_and_resegments_without_mutating_raw(
    sqlite_engine,
) -> None:
    initial_ids = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(1, minute=0, text="first"),
            _message(2, minute=20, text="second"),
            _message(3, minute=60, text="third"),
        ],
    )
    store = SqlAlchemyMemoryBackgroundStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(
        group_id=10001,
        message_id=initial_ids[-1],
        now=NOW,
    )
    assert service.run_once(now=NOW)
    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        processed_jobs = JobRepository(session).list_jobs(
            job_type="memory_episode_process",
            statuses=["completed"],
        )
        old_episode_id = int(processed_jobs[0].payload_json["episode_id"])
        old_episode = EpisodeRepository(session).get_episode(old_episode_id)
        assert old_episode is not None
        assert old_episode.status == "processed"

    late_id = _seed_sqlite_messages(
        sqlite_engine,
        [_message(4, minute=10, text="late")],
    )[0]
    queued = service.enqueue_late_arrival(
        group_id=10001,
        message_id=late_id,
        message_timestamp=NOW + timedelta(minutes=10),
        now=NOW,
    )
    assert queued is not None
    assert service.run_once(now=NOW)

    with session_scope(sqlite_engine) as session:
        old_episode = EpisodeRepository(session).get_episode(old_episode_id)
        assert old_episode is not None
        assert old_episode.status == "superseded"
        assert not old_episode.is_current
        assert (
            RetrievalDocumentRepository(session).search_group_documents_fts(
                group_id=10001,
                query="first",
                limit=10,
            )
            == []
        )
        current_open = EpisodeRepository(session).get_open_episode(
            group_id=10001
        )
        assert current_open is not None
        current_rows = EpisodeRepository(session).list_episode_messages(
            episode_id=current_open.id,
            group_id=10001,
        )
        assert [row.platform_msg_id for row in current_rows] == ["m-3"]
        raw_rows = MessageRepository(session).list_group_messages_chronological(
            group_id=10001
        )
        assert [row.platform_msg_id for row in raw_rows] == [
            "m-1",
            "m-4",
            "m-2",
            "m-3",
        ]

    assert (
        service.enqueue_late_arrival(
            group_id=10001,
            message_id=late_id,
            message_timestamp=NOW + timedelta(minutes=10),
            now=NOW,
        )
        is None
    )
    with sqlite_engine.connect() as connection:
        prepared = connection.execute(
            text(
                "SELECT COUNT(*) FROM memory_late_arrival_preparations "
                "WHERE group_id = :group_id AND message_id = :message_id"
            ),
            {"group_id": 10001, "message_id": late_id},
        ).scalar_one()
    assert prepared == 1


def test_allocator_requeries_after_cached_episode_is_superseded(
    sqlite_engine,
) -> None:
    message_ids = _seed_sqlite_messages(
        sqlite_engine,
        [
            _message(1, minute=0, text="first"),
            _message(2, minute=20, text="second"),
            _message(3, minute=10, text="late"),
        ],
    )
    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        first = MessageRepository(session).get_by_platform_msg_id("m-1")
        assert first is not None
        stale = episodes.create_episode(
            group_id=10001,
            start_message_id=first.id,
            started_at=first.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        episodes.add_message(
            episode_id=stale.id,
            group_id=10001,
            message_id=first.id,
            ordinal=0,
            estimated_tokens=1,
        )
        stale_id = stale.id

    fetched = Event()
    resume = Event()

    class PausingStore(SqlAlchemyMemoryBackgroundStore):
        paused = False

        def list_unassigned_messages(self, **kwargs):
            rows = super().list_unassigned_messages(**kwargs)
            if rows and not self.paused:
                self.paused = True
                fetched.set()
                assert resume.wait(timeout=5)
            return rows

    store = PausingStore(sqlite_engine)
    service = _service(store)
    service.enqueue_message(
        group_id=10001,
        message_id=message_ids[1],
        now=NOW,
    )
    errors: list[BaseException] = []

    def run_allocator() -> None:
        try:
            assert service.run_once(now=NOW)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = Thread(target=run_allocator)
    worker.start()
    assert fetched.wait(timeout=5)
    with session_scope(sqlite_engine) as session:
        replay_ids = EpisodeRepository(session).prepare_late_arrival_resegment(
            group_id=10001,
            message_id=message_ids[2],
            timestamp=NOW + timedelta(minutes=10),
            segmentation_version="segment-v2",
            compaction_version="compact-v2",
        )
        assert message_ids[0] in replay_ids
    resume.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []

    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        stale = episodes.get_episode(stale_id)
        assert stale is not None
        assert stale.status == "superseded"
        assert episodes.list_episode_messages(
            episode_id=stale_id,
            group_id=10001,
        ) == []
        current = episodes.get_open_episode(group_id=10001)
        assert current is not None
        current_rows = episodes.list_episode_messages(
            episode_id=current.id,
            group_id=10001,
        )
        assert [row.platform_msg_id for row in current_rows] == ["m-1", "m-2"]
