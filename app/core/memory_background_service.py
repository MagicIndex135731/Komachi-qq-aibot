from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import logging
import math
import re
from typing import Protocol, Sequence

from app.core.episode_segmenter import (
    build_overlap_windows,
    decide_episode_boundary,
    estimate_message_tokens,
)
from app.core.memory_compaction import (
    build_memory_compaction_prompt,
    canonical_key,
    parse_memory_compaction_response,
)
from app.core.summarizer import summarize_window
from app.storage.db import (
    mark_retrieval_vector_embeddings_failed,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.repositories import (
    EpisodeRepository,
    JobRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    SummaryRepository,
)


logger = logging.getLogger(__name__)


class EpisodeAppendConflict(RuntimeError):
    """The cached open episode was superseded before its message append."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class BackgroundMessage:
    id: int
    platform_msg_id: str
    group_id: int
    user_id: int
    timestamp: datetime
    plain_text: str
    reply_to_msg_id: str | None
    mentioned_bot: bool
    is_blocked: bool = False
    is_reserved: bool = False


@dataclass(frozen=True, slots=True)
class BackgroundEpisode:
    id: int
    group_id: int
    status: str
    segmentation_generation: str
    compaction_generation: str
    message_count: int
    token_count: int
    started_at: datetime
    ended_at: datetime | None
    content_hash: str
    backfill_run_id: int | None = None


@dataclass(frozen=True, slots=True)
class BackgroundJob:
    id: int
    job_type: str
    group_id: int
    payload: dict
    status: str
    requested_generation: int
    claimed_generation: int
    attempt_count: int
    max_attempts: int
    target_generation: str
    backfill_run_id: int | None = None


@dataclass(frozen=True, slots=True)
class LateArrivalPlan:
    group_id: int
    through_message_id: int
    watermark_message_id: int | None
    backfill_run_id: int | None
    superseded_episode_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShadowJobRequest:
    group_id: int
    message_id: int
    config_generation: str
    index_generation: str


@dataclass(frozen=True, slots=True)
class ShadowEvaluation:
    source_message_ids: tuple[str, ...]
    route_counts: dict[str, int]
    token_count: int
    latency_ms: int
    rewrite_used: bool
    fallback_used: bool
    error_category: str = ""
    candidate_scores: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalWindow:
    source_message_ids: tuple[int, ...]
    source_platform_msg_ids: tuple[str, ...]
    content: str
    start_at: datetime
    end_at: datetime
    token_count: int
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class DerivedFact:
    content: str
    source_msg_ids: tuple[str, ...]
    kind: str = "fact"
    subject_id: str = "group"
    predicate: str = ""
    object_text: str = ""
    importance: int = 1
    confidence: float = 0.5
    valid_until: str | None = None


@dataclass(frozen=True, slots=True)
class DerivedEvent:
    content: str
    source_msg_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeDerivation:
    summary: str
    facts: tuple[DerivedFact, ...]
    events: tuple[DerivedEvent, ...]
    windows: tuple[RetrievalWindow, ...]


class EpisodeDeriver(Protocol):
    def derive(
        self,
        *,
        episode: BackgroundEpisode,
        messages: tuple[BackgroundMessage, ...],
        windows: tuple[RetrievalWindow, ...],
    ) -> EpisodeDerivation: ...


class TextGenerationClient(Protocol):
    def generate_text(self, prompt_lines: list[str]) -> str: ...


class CompactionEpisodeDeriver:
    """Reuse the strict V1 compaction parser for source-backed episode facts."""

    def __init__(
        self,
        *,
        llm_client: TextGenerationClient,
        max_facts: int = 24,
    ) -> None:
        self.llm_client = llm_client
        self.max_facts = max(1, int(max_facts))

    def derive(
        self,
        *,
        episode: BackgroundEpisode,
        messages: tuple[BackgroundMessage, ...],
        windows: tuple[RetrievalWindow, ...],
    ) -> EpisodeDerivation:
        del episode
        prompt_messages = [
            {
                "source_msg_id": message.platform_msg_id,
                "content": (
                    f"user_id={message.user_id}; name={message.user_id}; "
                    f"text={str(message.plain_text or '').strip()}"
                ),
            }
            for message in messages
            if str(message.plain_text or "").strip()
        ]
        source_lines = [
            f"{message.user_id}: {str(message.plain_text or '').strip()}"
            for message in messages
            if str(message.plain_text or "").strip()
        ]
        if not prompt_messages:
            return EpisodeDerivation("", (), (), windows)
        prompt = build_memory_compaction_prompt(
            messages=prompt_messages,
            previous_digest="",
            language="zh",
        )
        raw = self.llm_client.generate_text([prompt])
        compaction = parse_memory_compaction_response(
            raw,
            allowed_source_msg_ids={
                str(item["source_msg_id"]) for item in prompt_messages
            },
            allowed_subject_ids={
                "group",
                *(str(message.user_id) for message in messages),
            },
            source_subject_ids={
                message.platform_msg_id: str(message.user_id)
                for message in messages
            },
            fallback_text=summarize_window(source_lines),
            strict=True,
        )
        facts = tuple(
            DerivedFact(
                content=fact.content,
                source_msg_ids=fact.source_msg_ids,
                kind=fact.kind,
                subject_id=fact.subject_id,
                predicate=fact.predicate,
                object_text=fact.object_text,
                importance=fact.importance,
                confidence=fact.confidence,
                valid_until=fact.valid_until,
            )
            for fact in compaction.facts[: self.max_facts]
        )
        events = tuple(
            DerivedEvent(
                content=fact.content,
                source_msg_ids=fact.source_msg_ids,
            )
            for fact in compaction.facts[: self.max_facts]
            if fact.kind == "event"
        )
        return EpisodeDerivation(
            summary=compaction.summary,
            facts=facts,
            events=events,
            windows=windows,
        )


class DocumentEmbedder(Protocol):
    @property
    def available(self) -> bool: ...

    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]] | None: ...


class ShadowEvaluator(Protocol):
    def evaluate(self, request: ShadowJobRequest) -> ShadowEvaluation: ...


class MemoryBackgroundStore(Protocol):
    def enqueue_allocator(
        self,
        *,
        group_id: int,
        latest_message_id: int,
        segmentation_generation: str,
        backfill_run_id: int | None,
        watermark_message_id: int | None,
        now: datetime,
    ) -> BackgroundJob: ...

    def enqueue_episode_processing(
        self,
        *,
        episode_id: int,
        group_id: int,
        compaction_generation: str,
        backfill_run_id: int | None,
        now: datetime,
    ) -> BackgroundJob: ...

    def enqueue_shadow(
        self,
        *,
        request: ShadowJobRequest,
        now: datetime,
    ) -> BackgroundJob: ...

    def recover_stale_jobs(self, *, now: datetime) -> int: ...

    def claim_next_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> BackgroundJob | None: ...

    def complete_job(
        self,
        *,
        job: BackgroundJob,
        worker_id: str,
        now: datetime,
    ) -> bool: ...

    def record_failure(
        self,
        *,
        job: BackgroundJob,
        error_code: str,
        now: datetime,
        retry_delay_seconds: int,
    ) -> bool: ...

    def list_unassigned_messages(
        self,
        *,
        group_id: int,
        segmentation_generation: str,
        through_message_id: int,
        watermark_message_id: int | None,
    ) -> list[BackgroundMessage]: ...

    def get_open_episode(
        self,
        *,
        group_id: int,
        segmentation_generation: str,
    ) -> BackgroundEpisode | None: ...

    def create_episode(
        self,
        *,
        group_id: int,
        first_message: BackgroundMessage,
        segmentation_generation: str,
        backfill_run_id: int | None,
    ) -> BackgroundEpisode: ...

    def append_message(
        self,
        *,
        episode_id: int,
        message: BackgroundMessage,
        estimated_tokens: int,
    ) -> BackgroundEpisode: ...

    def list_episode_messages(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> list[BackgroundMessage]: ...

    def close_episode(
        self,
        *,
        episode_id: int,
        reason: str,
        content_hash: str,
        compaction_generation: str,
        now: datetime,
    ) -> BackgroundEpisode: ...

    def list_idle_open_episodes(
        self,
        *,
        idle_before: datetime,
        segmentation_generation: str,
    ) -> list[BackgroundEpisode]: ...

    def load_episode(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> tuple[BackgroundEpisode, list[BackgroundMessage]] | None: ...

    def persist_episode_derivation(
        self,
        *,
        episode_id: int,
        expected_compaction_generation: str,
        derivation: EpisodeDerivation,
        now: datetime,
    ) -> bool: ...

    def persist_shadow_result(
        self,
        *,
        job: BackgroundJob,
        result: ShadowEvaluation,
        now: datetime,
    ) -> bool: ...

    def prepare_late_arrival_resegment(
        self,
        *,
        group_id: int,
        message_id: int,
        message_timestamp: datetime,
        segmentation_generation: str,
        compaction_generation: str,
        now: datetime,
    ) -> LateArrivalPlan | None: ...


class SqlAlchemyMemoryBackgroundStore:
    """Repository-backed store used by the runtime composition root."""

    allocator_job_type = "episode_allocate"
    episode_job_type = "memory_episode_process"
    shadow_job_type = "memory_shadow_evaluate"

    def __init__(
        self,
        engine,
        *,
        batch_size: int = 500,
        max_attempts: int = 3,
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_version: str = "",
        embedding_dimensions: int | None = None,
        embedding_generation: int | None = None,
    ) -> None:
        self.engine = engine
        self.batch_size = max(1, int(batch_size))
        self.max_attempts = max(1, int(max_attempts))
        self.embedding_provider = str(embedding_provider)
        self.embedding_model = str(embedding_model)
        self.embedding_version = str(embedding_version)
        self.embedding_dimensions = (
            max(1, int(embedding_dimensions))
            if embedding_dimensions is not None
            else None
        )
        self.embedding_generation = (
            int(embedding_generation)
            if embedding_generation is not None
            else None
        )
        self.segmentation_generation: str | None = None
        self.compaction_generation: str | None = None

    def configure_generations(
        self,
        *,
        segmentation_generation: str,
        compaction_generation: str,
    ) -> None:
        self.segmentation_generation = str(segmentation_generation)
        self.compaction_generation = str(compaction_generation)

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
        payload = {
            "group_id": int(group_id),
            "latest_message_id": int(latest_message_id),
            "watermark_message_id": (
                int(watermark_message_id)
                if watermark_message_id is not None
                else None
            ),
            "segmentation_generation": str(segmentation_generation),
        }
        with session_scope(self.engine) as session:
            row = JobRepository(session).enqueue_coalescing_job(
                job_type=self.allocator_job_type,
                job_key=f"group:{int(group_id)}:{segmentation_generation}",
                payload_json=payload,
                run_at=now,
                backfill_run_id=backfill_run_id,
                target_generation=segmentation_generation,
                max_attempts=self.max_attempts,
            )
            return _background_job(row)

    def enqueue_episode_processing(
        self,
        *,
        episode_id: int,
        group_id: int,
        compaction_generation: str,
        backfill_run_id: int | None,
        now: datetime,
    ) -> BackgroundJob:
        payload = {
            "group_id": int(group_id),
            "episode_id": int(episode_id),
            "compaction_generation": str(compaction_generation),
        }
        with session_scope(self.engine) as session:
            row = JobRepository(session).enqueue_coalescing_job(
                job_type=self.episode_job_type,
                job_key=f"episode:{int(episode_id)}:{compaction_generation}",
                payload_json=payload,
                run_at=now,
                backfill_run_id=backfill_run_id,
                target_generation=compaction_generation,
                max_attempts=self.max_attempts,
            )
            return _background_job(row)

    def enqueue_shadow(
        self,
        *,
        request: ShadowJobRequest,
        now: datetime,
    ) -> BackgroundJob:
        payload = {
            "group_id": int(request.group_id),
            "message_id": int(request.message_id),
            "config_generation": str(request.config_generation),
            "index_generation": str(request.index_generation),
        }
        with session_scope(self.engine) as session:
            row = JobRepository(session).enqueue_coalescing_job(
                job_type=self.shadow_job_type,
                job_key=(
                    f"shadow:{request.group_id}:{request.message_id}:"
                    f"{request.config_generation}:{request.index_generation}"
                ),
                payload_json=payload,
                run_at=now,
                target_generation=(
                    f"{request.config_generation}:"
                    f"{request.index_generation}"
                ),
                max_attempts=self.max_attempts,
            )
            return _background_job(row)

    def recover_stale_jobs(self, *, now: datetime) -> int:
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            return sum(
                jobs.requeue_stale_coalescing_jobs(
                    job_type=job_type,
                    now=now,
                )
                for job_type in (
                    self.allocator_job_type,
                    self.episode_job_type,
                    self.shadow_job_type,
                )
            )

    def claim_next_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
    ) -> BackgroundJob | None:
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            for job_type in (
                self.allocator_job_type,
                self.episode_job_type,
                self.shadow_job_type,
            ):
                target_generation = None
                include_derived_generations = False
                if job_type == self.allocator_job_type:
                    target_generation = self.segmentation_generation
                    include_derived_generations = True
                elif job_type == self.episode_job_type:
                    target_generation = self.compaction_generation
                row = jobs.claim_coalescing_job(
                    job_type=job_type,
                    worker_id=worker_id,
                    now=now,
                    lease_seconds=lease_seconds,
                    target_generation=target_generation,
                    include_derived_generations=include_derived_generations,
                )
                if row is not None:
                    return _background_job(row)
        return None

    def complete_job(
        self,
        *,
        job: BackgroundJob,
        worker_id: str,
        now: datetime,
    ) -> bool:
        claimed_worker = str(job.payload.get("_worker_id") or "")
        if claimed_worker and claimed_worker != str(worker_id):
            raise ValueError("background job worker identity does not match its lease")
        with session_scope(self.engine) as session:
            completed = JobRepository(session).complete_coalescing_job(
                job_id=job.id,
                worker_id=worker_id,
                claimed_generation=job.claimed_generation,
                now=now,
            )
            return completed is not None

    def record_failure(
        self,
        *,
        job: BackgroundJob,
        error_code: str,
        now: datetime,
        retry_delay_seconds: int,
    ) -> bool:
        with session_scope(self.engine) as session:
            failed = JobRepository(session).fail_coalescing_job(
                job_id=job.id,
                worker_id=self._locked_worker(job),
                claimed_generation=job.claimed_generation,
                error_code=error_code,
                now=now,
                retry_at=now + timedelta(seconds=max(1, retry_delay_seconds)),
            )
            if failed is None:
                return False
            will_retry = failed.status == "queued"
            if not will_retry and job.job_type == self.episode_job_type:
                episode_id = int(job.payload.get("episode_id", 0))
                if episode_id:
                    EpisodeRepository(session).compare_and_set_status(
                        episode_id=episode_id,
                        group_id=job.group_id,
                        expected_statuses=("processing",),
                        new_status="failed",
                    )
            return will_retry

    def persist_shadow_result(
        self,
        *,
        job: BackgroundJob,
        result: ShadowEvaluation,
        now: datetime,
    ) -> bool:
        del now
        with session_scope(self.engine) as session:
            source_ids = list(dict.fromkeys(result.source_message_ids))
            candidate_scores = _validated_candidate_scores(result.candidate_scores)
            messages = MessageRepository(session)
            for source_id in source_ids:
                source = messages.get_by_platform_msg_id(str(source_id))
                if source is None or source.group_id != job.group_id:
                    raise ValueError(
                        "shadow source provenance is missing or cross-group"
                    )
            safe_payload = {
                "group_id": int(job.group_id),
                "message_id": int(job.payload["message_id"]),
                "config_generation": str(job.payload["config_generation"]),
                "index_generation": str(job.payload["index_generation"]),
                "result": {
                    "source_message_ids": source_ids,
                    "candidate_scores": [
                        {
                            "document_id": document_id,
                            "fused_score": fused_score,
                        }
                        for document_id, fused_score in candidate_scores
                    ],
                    "route_counts": {
                        _safe_metric_key(route): max(0, int(count))
                        for route, count in result.route_counts.items()
                    },
                    "token_count": max(0, int(result.token_count)),
                    "latency_ms": max(0, int(result.latency_ms)),
                    "rewrite_used": bool(result.rewrite_used),
                    "fallback_used": bool(result.fallback_used),
                    "error_category": _safe_metric_key(
                        result.error_category or "none"
                    ),
                },
            }
            current = JobRepository(session).update_coalescing_job_payload(
                job_id=job.id,
                worker_id=self._locked_worker(job),
                claimed_generation=job.claimed_generation,
                payload_json=safe_payload,
            )
            return current is not None

    @staticmethod
    def _locked_worker(job: BackgroundJob) -> str:
        worker_id = str(job.payload.get("_worker_id") or "")
        if not worker_id:
            raise ValueError("claimed background job is missing its worker id")
        return worker_id

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
            int(through_message_id),
            (
                int(watermark_message_id)
                if watermark_message_id is not None
                else int(through_message_id)
            ),
        )
        with session_scope(self.engine) as session:
            rows = EpisodeRepository(session).list_unassigned_messages(
                group_id=group_id,
                watermark_message_id=upper,
                limit=self.batch_size,
            )
            return [_background_message(row) for row in rows]

    def get_open_episode(
        self,
        *,
        group_id: int,
        segmentation_generation: str,
    ) -> BackgroundEpisode | None:
        with session_scope(self.engine) as session:
            row = EpisodeRepository(session).get_open_episode(group_id=group_id)
            if row is None:
                return None
            if not _compatible_segmentation_generation(
                str(row.segmentation_version or ""),
                segmentation_generation,
            ):
                raise RuntimeError("open episode belongs to another segmentation generation")
            return _background_episode(row)

    def create_episode(
        self,
        *,
        group_id: int,
        first_message: BackgroundMessage,
        segmentation_generation: str,
        backfill_run_id: int | None,
    ) -> BackgroundEpisode:
        del backfill_run_id
        with session_scope(self.engine) as session:
            row = EpisodeRepository(session).create_episode(
                group_id=group_id,
                start_message_id=first_message.id,
                started_at=first_message.timestamp,
                segmentation_version=segmentation_generation,
            )
            session.flush()
            return _background_episode(row)

    def append_message(
        self,
        *,
        episode_id: int,
        message: BackgroundMessage,
        estimated_tokens: int,
    ) -> BackgroundEpisode:
        with session_scope(self.engine) as session:
            episodes = EpisodeRepository(session)
            appended = episodes.add_message_if_current(
                episode_id=episode_id,
                group_id=message.group_id,
                message_id=message.id,
                estimated_tokens=estimated_tokens,
            )
            if not appended:
                raise EpisodeAppendConflict("open episode changed before append")
            session.flush()
            updated = episodes.get_episode(episode_id)
            if updated is None:
                raise RuntimeError("episode disappeared after message append")
            return _background_episode(updated)

    def list_episode_messages(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> list[BackgroundMessage]:
        with session_scope(self.engine) as session:
            rows = EpisodeRepository(session).list_episode_messages(
                episode_id=episode_id,
                group_id=group_id,
            )
            return [_background_message(row) for row in rows]

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
        with session_scope(self.engine) as session:
            episodes = EpisodeRepository(session)
            current = episodes.get_episode(episode_id)
            if current is None:
                raise RuntimeError("episode not found")
            rows = episodes.list_episode_messages(
                episode_id=episode_id,
                group_id=current.group_id,
            )
            if not rows:
                raise RuntimeError("cannot close empty episode")
            row = episodes.close_episode(
                episode_id=episode_id,
                ended_at=rows[-1].timestamp,
                end_message_id=rows[-1].id,
                boundary_reason=reason,
                content_hash=content_hash,
            )
            if row is None:
                raise RuntimeError("episode disappeared during close")
            # ``compare_and_set_status`` uses textual SQL and expires ORM
            # state. Flush the close first or the expiry would discard the
            # pending status/end-message mutation.
            session.flush()
            if not episodes.compare_and_set_status(
                episode_id=episode_id,
                group_id=row.group_id,
                expected_statuses=("closed",),
                new_status="closed",
                compaction_version=compaction_generation,
            ):
                raise RuntimeError("episode close generation CAS failed")
            session.flush()
            return _background_episode(row)

    def list_idle_open_episodes(
        self,
        *,
        idle_before: datetime,
        segmentation_generation: str,
    ) -> list[BackgroundEpisode]:
        with session_scope(self.engine) as session:
            rows = EpisodeRepository(session).list_idle_open_episodes(
                idle_before=idle_before,
                limit=100,
            )
            return [
                _background_episode(row)
                for row in rows
                if _compatible_segmentation_generation(
                    str(row.segmentation_version or ""),
                    segmentation_generation,
                )
            ]

    def load_episode(
        self,
        *,
        episode_id: int,
        group_id: int,
    ) -> tuple[BackgroundEpisode, list[BackgroundMessage]] | None:
        with session_scope(self.engine) as session:
            episodes = EpisodeRepository(session)
            row = episodes.get_episode(episode_id)
            if row is None or row.group_id != group_id or not row.is_current:
                return None
            if row.status in {"closed", "failed"}:
                if not episodes.compare_and_set_status(
                    episode_id=episode_id,
                    group_id=group_id,
                    expected_statuses=(row.status,),
                    new_status="processing",
                ):
                    return None
                session.flush()
                row = episodes.get_episode(episode_id)
                if row is None:
                    return None
            messages = episodes.list_episode_messages(
                episode_id=episode_id,
                group_id=group_id,
            )
            return (
                _background_episode(row),
                [_background_message(message) for message in messages],
            )

    def persist_episode_derivation(
        self,
        *,
        episode_id: int,
        expected_compaction_generation: str,
        derivation: EpisodeDerivation,
        now: datetime,
    ) -> bool:
        class _StaleDerivation(RuntimeError):
            pass

        vector_rows: list[tuple[int, int, Sequence[float]]] = []
        resolved_group_id: int | None = None
        try:
            with session_scope(self.engine) as session:
                episodes = EpisodeRepository(session)
                episode = episodes.get_episode(episode_id)
                if (
                    episode is None
                    or not episode.is_current
                    or episode.status != "processing"
                    or episode.compaction_version
                    != expected_compaction_generation
                ):
                    raise _StaleDerivation
                resolved_group_id = int(episode.group_id)
                messages = [
                    message
                    for message in episodes.list_episode_messages(
                        episode_id=episode.id,
                        group_id=episode.group_id,
                    )
                    if not MessageRepository.is_reserved_outbound(message)
                    and not MessageRepository.is_qq_blocked_outbound(message)
                ]
                platform_to_id = {
                    message.platform_msg_id: message.id for message in messages
                }
                documents = RetrievalDocumentRepository(session)
                for index, window in enumerate(derivation.windows):
                    content_hash = hashlib.sha256(
                        window.content.encode("utf-8")
                    ).hexdigest()
                    has_embedding = (
                        window.embedding is not None
                        and self.embedding_generation is not None
                    )
                    document = documents.upsert_document(
                        scope_type="group",
                        scope_id=str(episode.group_id),
                        group_id=episode.group_id,
                        episode_id=episode.id,
                        document_kind="episode",
                        source_table="conversation_episodes",
                        source_id=(
                            f"{episode.id}:window:{index}:"
                            f"{expected_compaction_generation}"
                        ),
                        start_at=window.start_at,
                        end_at=window.end_at,
                        content=window.content,
                        metadata_json={
                            "episode_id": episode.id,
                            "source_msg_ids": list(
                                window.source_platform_msg_ids
                            ),
                            "message_ids": list(window.source_message_ids),
                            "compaction_generation": (
                                expected_compaction_generation
                            ),
                        },
                        content_hash=content_hash,
                        source_message_ids=list(window.source_message_ids),
                        embedding_provider=self.embedding_provider,
                        embedding_model=self.embedding_model,
                        embedding_version=self.embedding_version,
                        embedding_dimensions=self.embedding_dimensions,
                        embedding_generation=self.embedding_generation,
                        embedding_eligible=True,
                        embedding_status=(
                            "pending" if has_embedding else "disabled"
                        ),
                    )
                    session.flush()
                    if has_embedding:
                        vector_rows.append(
                            (
                                int(document.id),
                                int(episode.group_id),
                                tuple(window.embedding or ()),
                            )
                        )

                if messages and derivation.summary.strip():
                    summary = SummaryRepository(session).upsert_summary(
                        scope_type="group",
                        scope_id=str(episode.group_id),
                        summary_level="episode",
                        summary_key=(
                            f"episode:{episode.id}:"
                            f"{expected_compaction_generation}"
                        ),
                        start_at=messages[0].timestamp,
                        end_at=messages[-1].timestamp,
                        content=derivation.summary,
                        source_count=len(messages),
                        source_start_msg_id=messages[0].platform_msg_id,
                        source_end_msg_id=messages[-1].platform_msg_id,
                    )
                    session.flush()
                    documents.upsert_document(
                        scope_type="group",
                        scope_id=str(episode.group_id),
                        group_id=episode.group_id,
                        episode_id=episode.id,
                        document_kind="episode_summary",
                        source_table="summaries",
                        source_id=str(summary.id),
                        start_at=messages[0].timestamp,
                        end_at=messages[-1].timestamp,
                        content=derivation.summary,
                        metadata_json={
                            "episode_id": episode.id,
                            "compaction_generation": (
                                expected_compaction_generation
                            ),
                        },
                        content_hash=hashlib.sha256(
                            derivation.summary.encode("utf-8")
                        ).hexdigest(),
                        source_message_ids=[message.id for message in messages],
                        embedding_eligible=False,
                        embedding_status="disabled",
                    )

                memories = MemoryRepository(session)
                for fact in derivation.facts:
                    source_ids = list(dict.fromkeys(fact.source_msg_ids))
                    source_message_ids = [
                        platform_to_id[source_id] for source_id in source_ids
                    ]
                    if fact.kind == "expired":
                        memories.supersede_current_memories(
                            scope_id=str(episode.group_id),
                            subject_id=fact.subject_id,
                            predicate=fact.predicate,
                            object_text=fact.object_text,
                            valid_until=_parse_timestamp(fact.valid_until) or now,
                        )
                        continue
                    memory = memories.upsert_canonical_memory(
                        scope_type="group",
                        scope_id=str(episode.group_id),
                        subject_type=(
                            "group"
                            if fact.subject_id == "group"
                            else "user"
                        ),
                        subject_id=fact.subject_id,
                        memory_kind=fact.kind,
                        canonical_key=canonical_key(
                            fact.kind,
                            fact.subject_id,
                            fact.predicate,
                            fact.object_text,
                        ),
                        predicate=fact.predicate,
                        object_text=fact.object_text,
                        content=fact.content,
                        importance=fact.importance,
                        confidence=fact.confidence,
                        source_msg_ids=source_ids,
                        valid_from=episode.ended_at or now,
                        valid_until=_parse_timestamp(fact.valid_until),
                    )
                    session.flush()
                    documents.upsert_document(
                        scope_type="group",
                        scope_id=str(episode.group_id),
                        group_id=episode.group_id,
                        episode_id=episode.id,
                        document_kind="memory",
                        source_table="memory_items",
                        source_id=str(memory.id),
                        start_at=episode.started_at,
                        end_at=episode.ended_at or now,
                        content=fact.content,
                        metadata_json={
                            "episode_id": episode.id,
                            "subject_id": fact.subject_id,
                            "kind": fact.kind,
                            "compaction_generation": (
                                expected_compaction_generation
                            ),
                        },
                        content_hash=hashlib.sha256(
                            fact.content.encode("utf-8")
                        ).hexdigest(),
                        source_message_ids=source_message_ids,
                        embedding_eligible=False,
                        embedding_status="disabled",
                    )

            if vector_rows:
                try:
                    written = write_retrieval_vector_embeddings(
                        self.engine,
                        generation=int(self.embedding_generation),
                        rows=vector_rows,
                    )
                    if written != len(vector_rows):
                        raise RuntimeError("retrieval vector batch was incomplete")
                except Exception as exc:
                    logger.warning(
                        "memory_episode_embedding_failed episode_id=%s "
                        "generation=%s error=%s",
                        episode_id,
                        self.embedding_generation,
                        type(exc).__name__,
                    )
                    try:
                        mark_retrieval_vector_embeddings_failed(
                            self.engine,
                            generation=int(self.embedding_generation),
                            group_id=int(resolved_group_id),
                            document_ids=[row[0] for row in vector_rows],
                            error_code=type(exc).__name__,
                        )
                    except Exception:
                        logger.exception(
                            "memory_episode_embedding_status_failed "
                            "episode_id=%s generation=%s",
                            episode_id,
                            self.embedding_generation,
                        )
            with session_scope(self.engine) as session:
                if resolved_group_id is None or not EpisodeRepository(
                    session
                ).compare_and_set_status(
                    episode_id=episode_id,
                    group_id=resolved_group_id,
                    expected_statuses=("processing",),
                    new_status="processed",
                    compaction_version=expected_compaction_generation,
                ):
                    raise _StaleDerivation
            return True
        except _StaleDerivation:
            return False

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
        del now
        with session_scope(self.engine) as session:
            episodes = EpisodeRepository(session)
            affected = episodes.find_episode_for_late_arrival(
                group_id=group_id,
                timestamp=message_timestamp,
                segmentation_version=segmentation_generation,
            )
            replay_ids = episodes.prepare_late_arrival_resegment(
                group_id=group_id,
                message_id=message_id,
                timestamp=message_timestamp,
                segmentation_version=segmentation_generation,
                compaction_version=compaction_generation,
            )
            if not replay_ids:
                return None
            return LateArrivalPlan(
                group_id=group_id,
                through_message_id=max(int(message_id), *replay_ids),
                watermark_message_id=None,
                backfill_run_id=None,
                superseded_episode_ids=(affected.id,) if affected is not None else (),
            )


class MemoryBackgroundService:
    """Idempotent episode allocation and derivation worker.

    The store owns transactions and compare-and-set operations. This service
    owns deterministic segmentation, the single blocked-content derivation
    boundary, finite retry policy, and worker lifecycle.
    """

    allocator_job_type = "episode_allocate"
    episode_job_type = "memory_episode_process"
    shadow_job_type = "memory_shadow_evaluate"

    def __init__(
        self,
        *,
        store: MemoryBackgroundStore,
        deriver: EpisodeDeriver,
        worker_id: str,
        segmentation_generation: str,
        compaction_generation: str,
        index_generation: str | None = None,
        idle_minutes: int = 30,
        max_messages: int = 50,
        max_tokens: int = 8000,
        chunk_max_tokens: int = 1800,
        chunk_overlap_messages: int = 5,
        bot_user_id: int | None = None,
        embedder: DocumentEmbedder | None = None,
        shadow_evaluator: ShadowEvaluator | None = None,
        lease_seconds: int = 60,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.store = store
        self.deriver = deriver
        self.worker_id = str(worker_id)
        self.segmentation_generation = str(segmentation_generation)
        self.compaction_generation = str(compaction_generation)
        self.index_generation = (
            str(index_generation) if index_generation is not None else None
        )
        self.idle_minutes = max(1, int(idle_minutes))
        self.max_messages = max(1, int(max_messages))
        self.max_tokens = max(1, int(max_tokens))
        self.chunk_max_tokens = max(1, int(chunk_max_tokens))
        self.chunk_overlap_messages = max(0, int(chunk_overlap_messages))
        self.bot_user_id = int(bot_user_id) if bot_user_id is not None else None
        self.embedder = embedder
        self.shadow_evaluator = shadow_evaluator
        self.lease_seconds = max(1, int(lease_seconds))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        configure_generations = getattr(store, "configure_generations", None)
        if callable(configure_generations):
            configure_generations(
                segmentation_generation=self.segmentation_generation,
                compaction_generation=self.compaction_generation,
            )
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def worker_task(self) -> asyncio.Task[None] | None:
        return self._worker_task

    def validate_backfill_contract(
        self,
        *,
        segmentation_generation: str,
        compaction_generation: str,
        index_generation: str,
    ) -> None:
        requested = (
            str(segmentation_generation),
            str(compaction_generation),
            str(index_generation),
        )
        expected_index = self.index_generation
        store_vector_generation = getattr(self.store, "embedding_generation", None)
        if expected_index is None and store_vector_generation is not None:
            expected_index = f"vector:{int(store_vector_generation)}"
        if (
            requested[0] != self.segmentation_generation
            or requested[1] != self.compaction_generation
            or (expected_index is not None and requested[2] != expected_index)
            or (requested[2].startswith("vector:") and store_vector_generation is None)
        ):
            raise ValueError(
                "backfill worker generation identity does not match requested contract"
            )
        if self.index_generation is None:
            self.index_generation = requested[2]

    def enqueue_message(
        self,
        *,
        group_id: int,
        message_id: int,
        now: datetime | None = None,
        backfill_run_id: int | None = None,
        watermark_message_id: int | None = None,
    ) -> BackgroundJob:
        """Persist/rearm one stable allocator job without reading history."""
        queued = self.store.enqueue_allocator(
            group_id=int(group_id),
            latest_message_id=int(message_id),
            segmentation_generation=self.segmentation_generation,
            backfill_run_id=backfill_run_id,
            watermark_message_id=watermark_message_id,
            now=_utc(now or datetime.now(UTC)),
        )
        self._wake_event.set()
        return queued

    def enqueue_late_arrival(
        self,
        *,
        group_id: int,
        message_id: int,
        message_timestamp: datetime,
        now: datetime | None = None,
    ) -> BackgroundJob | None:
        """Version and rearm the bounded region affected by an old timestamp.

        The store performs the membership/document invalidation transaction.
        The replay itself remains a normal generation-aware allocator job.
        """
        resolved_now = _utc(now or datetime.now(UTC))
        plan = self.store.prepare_late_arrival_resegment(
            group_id=int(group_id),
            message_id=int(message_id),
            message_timestamp=_utc(message_timestamp),
            segmentation_generation=self.segmentation_generation,
            compaction_generation=self.compaction_generation,
            now=resolved_now,
        )
        if plan is None:
            return None
        late_generation = (
            f"{self.segmentation_generation}:late:{int(message_id)}"
        )
        queued = self.store.enqueue_allocator(
            group_id=plan.group_id,
            latest_message_id=plan.through_message_id,
            segmentation_generation=late_generation,
            backfill_run_id=plan.backfill_run_id,
            watermark_message_id=plan.watermark_message_id,
            now=resolved_now,
        )
        self._wake_event.set()
        logger.info(
            "memory_late_arrival_resegment_queued group_id=%s message_id=%s "
            "superseded_episodes=%s generation=%s",
            group_id,
            message_id,
            len(plan.superseded_episode_ids),
            late_generation,
        )
        return queued

    def enqueue_shadow(
        self,
        request: ShadowJobRequest,
        *,
        now: datetime | None = None,
    ) -> BackgroundJob | None:
        if self.shadow_evaluator is None:
            return None
        queued = self.store.enqueue_shadow(
            request=request,
            now=_utc(now or datetime.now(UTC)),
        )
        self._wake_event.set()
        return queued

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        await asyncio.to_thread(
            self.store.recover_stale_jobs,
            now=datetime.now(UTC),
        )
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name=f"memory-background-{self.worker_id}",
        )

    async def wake(self) -> None:
        self._wake_event.set()

    async def stop(self) -> None:
        task = self._worker_task
        if task is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._worker_task = None

    async def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    did_work = await asyncio.to_thread(self.run_once)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "memory_background_loop_failed worker_id=%s",
                        self.worker_id,
                    )
                    did_work = False
                if did_work:
                    continue
                self._wake_event.clear()
                if self._stop_event.is_set():
                    return
                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            self._wake_event.clear()

    def run_once(self, *, now: datetime | None = None) -> bool:
        resolved_now = _utc(now or datetime.now(UTC))
        job = self.store.claim_next_job(
            worker_id=self.worker_id,
            now=resolved_now,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return self._close_idle_episode(resolved_now)

        try:
            if job.job_type == self.allocator_job_type:
                self._allocate(job=job, now=resolved_now)
            elif job.job_type == self.episode_job_type:
                self._derive_episode(job=job, now=resolved_now)
            elif job.job_type == self.shadow_job_type:
                self._evaluate_shadow(job=job, now=resolved_now)
            else:
                raise ValueError("unsupported memory background job type")
        except Exception as exc:
            retry_delay = min(300, 5 * (2 ** max(0, job.attempt_count)))
            will_retry = self.store.record_failure(
                job=job,
                error_code=type(exc).__name__[:96],
                now=resolved_now,
                retry_delay_seconds=retry_delay,
            )
            logger.warning(
                "memory_background_job_failed job_id=%s job_type=%s group_id=%s "
                "attempt=%s retry=%s error=%s",
                job.id,
                job.job_type,
                job.group_id,
                job.attempt_count + 1,
                int(will_retry),
                type(exc).__name__,
            )
            return True

        completed = self.store.complete_job(
            job=job,
            worker_id=self.worker_id,
            now=resolved_now,
        )
        if not completed:
            logger.warning(
                "memory_background_job_completion_rejected job_id=%s generation=%s",
                job.id,
                job.claimed_generation,
            )
        return True

    def recover_stale_jobs(self, *, now: datetime | None = None) -> int:
        """Release expired leases before a synchronous worker resumes processing."""
        return self.store.recover_stale_jobs(now=_utc(now or datetime.now(UTC)))

    def _evaluate_shadow(self, *, job: BackgroundJob, now: datetime) -> None:
        if self.shadow_evaluator is None:
            raise RuntimeError("shadow evaluator is disabled")
        request = ShadowJobRequest(
            group_id=job.group_id,
            message_id=int(job.payload["message_id"]),
            config_generation=str(job.payload["config_generation"]),
            index_generation=str(job.payload["index_generation"]),
        )
        result = self.shadow_evaluator.evaluate(request)
        if any(not str(source_id).strip() for source_id in result.source_message_ids):
            raise ValueError("shadow evaluation contains an invalid source id")
        _validated_candidate_scores(result.candidate_scores)
        if not self.store.persist_shadow_result(
            job=job,
            result=result,
            now=now,
        ):
            raise RuntimeError("shadow result persistence CAS failed")
        logger.info(
            "memory_shadow_completed group_id=%s message_id=%s sources=%s "
            "routes=%s tokens=%s latency_ms=%s rewrite=%s fallback=%s error=%s",
            job.group_id,
            request.message_id,
            len(set(result.source_message_ids)),
            sum(max(0, int(count)) for count in result.route_counts.values()),
            max(0, int(result.token_count)),
            max(0, int(result.latency_ms)),
            int(result.rewrite_used),
            int(result.fallback_used),
            str(result.error_category or "")[:96],
        )

    def _allocate(self, *, job: BackgroundJob, now: datetime) -> None:
        if not _compatible_segmentation_generation(
            job.target_generation,
            self.segmentation_generation,
        ):
            return
        through_message_id = int(job.payload["latest_message_id"])
        raw_watermark = job.payload.get("watermark_message_id")
        watermark_message_id = (
            int(raw_watermark) if raw_watermark is not None else None
        )
        open_episode = self.store.get_open_episode(
            group_id=job.group_id,
            segmentation_generation=job.target_generation,
        )
        conflict_restarts = 0
        while True:
            messages = self.store.list_unassigned_messages(
                group_id=job.group_id,
                segmentation_generation=job.target_generation,
                through_message_id=through_message_id,
                watermark_message_id=watermark_message_id,
            )
            if not messages:
                break
            appended = 0
            restart_batch = False
            for message in sorted(
                messages,
                key=lambda item: (_utc(item.timestamp), item.id),
            ):
                if message.group_id != job.group_id or message.is_reserved:
                    continue
                if open_episode is None:
                    open_episode = self.store.create_episode(
                        group_id=job.group_id,
                        first_message=message,
                        segmentation_generation=job.target_generation,
                        backfill_run_id=job.backfill_run_id,
                    )
                episode_messages = self.store.list_episode_messages(
                    episode_id=open_episode.id,
                    group_id=job.group_id,
                )
                if episode_messages:
                    previous = episode_messages[-1]
                    decision = decide_episode_boundary(
                        previous=previous,
                        current=message,
                        open_message_count=open_episode.message_count,
                        open_token_count=open_episode.token_count,
                        open_platform_msg_ids={
                            row.platform_msg_id for row in episode_messages
                        },
                        idle_minutes=self.idle_minutes,
                        max_messages=self.max_messages,
                        max_tokens=self.max_tokens,
                        bot_user_id=self.bot_user_id,
                    )
                    if decision.should_close:
                        self._close_and_enqueue(
                            episode=open_episode,
                            reason=decision.reason,
                            now=now,
                            backfill_run_id=job.backfill_run_id,
                        )
                        open_episode = self.store.create_episode(
                            group_id=job.group_id,
                            first_message=message,
                            segmentation_generation=job.target_generation,
                            backfill_run_id=job.backfill_run_id,
                        )
                try:
                    open_episode = self.store.append_message(
                        episode_id=open_episode.id,
                        message=message,
                        estimated_tokens=estimate_message_tokens(
                            message.plain_text
                        ),
                    )
                except EpisodeAppendConflict:
                    conflict_restarts += 1
                    if conflict_restarts >= 3:
                        raise EpisodeAppendConflict(
                            "open episode changed repeatedly during append"
                        )
                    open_episode = self.store.get_open_episode(
                        group_id=job.group_id,
                        segmentation_generation=job.target_generation,
                    )
                    restart_batch = True
                    break
                appended += 1
            if restart_batch:
                continue
            if appended == 0:
                raise RuntimeError("allocator made no progress")
        if (
            watermark_message_id is not None
            and through_message_id >= watermark_message_id
            and open_episode is not None
            and open_episode.status == "open"
        ):
            self._close_and_enqueue(
                episode=open_episode,
                reason="backfill_watermark",
                now=now,
                backfill_run_id=job.backfill_run_id,
            )

    def _close_idle_episode(self, now: datetime) -> bool:
        idle_before = now - timedelta(minutes=self.idle_minutes)
        episodes = self.store.list_idle_open_episodes(
            idle_before=idle_before,
            segmentation_generation=self.segmentation_generation,
        )
        if not episodes:
            return False
        self._close_and_enqueue(
            episode=min(episodes, key=lambda item: (item.ended_at or item.started_at, item.id)),
            reason="idle",
            now=now,
            backfill_run_id=None,
        )
        return True

    def _close_and_enqueue(
        self,
        *,
        episode: BackgroundEpisode,
        reason: str,
        now: datetime,
        backfill_run_id: int | None,
    ) -> None:
        rows = self.store.list_episode_messages(
            episode_id=episode.id,
            group_id=episode.group_id,
        )
        if not rows:
            raise RuntimeError("cannot close an empty episode")
        content_hash = _episode_content_hash(rows)
        closed = self.store.close_episode(
            episode_id=episode.id,
            reason=reason,
            content_hash=content_hash,
            compaction_generation=self.compaction_generation,
            now=now,
        )
        self.store.enqueue_episode_processing(
            episode_id=closed.id,
            group_id=closed.group_id,
            compaction_generation=self.compaction_generation,
            backfill_run_id=(
                backfill_run_id
                if backfill_run_id is not None
                else closed.backfill_run_id
            ),
            now=now,
        )

    def _derive_episode(self, *, job: BackgroundJob, now: datetime) -> None:
        episode_id = int(job.payload["episode_id"])
        loaded = self.store.load_episode(
            episode_id=episode_id,
            group_id=job.group_id,
        )
        if loaded is None:
            return
        episode, raw_messages = loaded
        # A late-arrival rebuild or new compaction version may supersede an
        # already-claimed job. Never publish results for the stale generation.
        if (
            episode.status not in {"closed", "processing", "failed"}
            or episode.compaction_generation != job.target_generation
        ):
            return

        safe_messages = tuple(
            message
            for message in raw_messages
            if not message.is_reserved and not message.is_blocked
        )
        if not safe_messages:
            derivation = EpisodeDerivation(
                summary="",
                facts=(),
                events=(),
                windows=(),
            )
        else:
            windows = self._build_windows(safe_messages)
            derivation = self.deriver.derive(
                episode=episode,
                messages=safe_messages,
                windows=windows,
            )
            _validate_derivation_sources(
                derivation,
                allowed_source_ids={
                    message.platform_msg_id for message in safe_messages
                },
            )
        published = self.store.persist_episode_derivation(
            episode_id=episode.id,
            expected_compaction_generation=job.target_generation,
            derivation=derivation,
            now=now,
        )
        if not published:
            logger.info(
                "memory_episode_derivation_stale episode_id=%s generation=%s",
                episode.id,
                job.target_generation,
            )

    def _build_windows(
        self,
        messages: tuple[BackgroundMessage, ...],
    ) -> tuple[RetrievalWindow, ...]:
        segmented = build_overlap_windows(
            messages,
            min_messages=12,
            max_messages=24,
            max_tokens=self.chunk_max_tokens,
            overlap_messages=self.chunk_overlap_messages,
        )
        contents = [
            "\n".join(
                f"{message.user_id}: {str(message.plain_text or '').strip()}"
                for message in window.messages
                if str(message.plain_text or "").strip()
            )
            for window in segmented
        ]
        embeddings: list[list[float]] | None = None
        if self.embedder is not None and self.embedder.available and contents:
            embeddings = self.embedder.embed_documents(contents)
            if embeddings is not None and len(embeddings) != len(contents):
                raise ValueError("embedding count does not match retrieval windows")
        windows: list[RetrievalWindow] = []
        for index, (window, content) in enumerate(zip(segmented, contents, strict=True)):
            embedding = None
            if embeddings is not None:
                embedding = tuple(float(value) for value in embeddings[index])
            windows.append(
                RetrievalWindow(
                    source_message_ids=window.source_message_ids,
                    source_platform_msg_ids=window.source_platform_msg_ids,
                    content=content,
                    start_at=_utc(window.messages[0].timestamp),
                    end_at=_utc(window.messages[-1].timestamp),
                    token_count=window.token_count,
                    embedding=embedding,
                )
            )
        return tuple(windows)


def _episode_content_hash(messages: Sequence[BackgroundMessage]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(str(message.id).encode("ascii"))
        digest.update(b"\0")
        digest.update(message.platform_msg_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(message.plain_text or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _background_message(row) -> BackgroundMessage:
    group_id = getattr(row, "group_id", None)
    if group_id is None:
        raise ValueError("private messages cannot enter group episodes")
    return BackgroundMessage(
        id=int(row.id),
        platform_msg_id=str(row.platform_msg_id),
        group_id=int(group_id),
        user_id=int(row.user_id),
        timestamp=_utc(row.timestamp),
        plain_text=str(row.plain_text or ""),
        reply_to_msg_id=(
            str(row.reply_to_msg_id)
            if row.reply_to_msg_id is not None
            else None
        ),
        mentioned_bot=bool(row.mentioned_bot),
        is_blocked=MessageRepository.is_qq_blocked_outbound(row),
        is_reserved=MessageRepository.is_reserved_outbound(row),
    )


def _background_episode(row) -> BackgroundEpisode:
    return BackgroundEpisode(
        id=int(row.id),
        group_id=int(row.group_id),
        status=str(row.status),
        segmentation_generation=str(row.segmentation_version or ""),
        compaction_generation=str(row.compaction_version or ""),
        message_count=int(row.message_count or 0),
        token_count=int(row.token_count or 0),
        started_at=_utc(row.started_at),
        ended_at=_utc(row.ended_at) if row.ended_at is not None else None,
        content_hash=str(row.content_hash or ""),
        backfill_run_id=None,
    )


def _background_job(row) -> BackgroundJob:
    payload = dict(row.payload_json or {})
    locked_by = str(getattr(row, "locked_by", "") or "")
    if locked_by:
        payload["_worker_id"] = locked_by
    group_id = payload.get("group_id")
    if group_id is None:
        raise ValueError("memory background job is missing group_id")
    return BackgroundJob(
        id=int(row.id),
        job_type=str(row.job_type),
        group_id=int(group_id),
        payload=payload,
        status=str(row.status),
        requested_generation=int(row.requested_generation or 0),
        claimed_generation=int(row.claimed_generation or 0),
        attempt_count=int(row.attempt_count or 0),
        max_attempts=max(1, int(row.max_attempts or 1)),
        target_generation=str(
            row.target_generation
            or payload.get("segmentation_generation")
            or payload.get("compaction_generation")
            or ""
        ),
        backfill_run_id=(
            int(row.backfill_run_id)
            if row.backfill_run_id is not None
            else None
        ),
    )


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(parsed)


def _safe_metric_key(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{1,32}", normalized):
        return "other"
    return normalized


def _validated_candidate_scores(
    values: Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    if len(values) > 200:
        raise ValueError("shadow candidate score list is too large")
    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for document_id, fused_score in values:
        if isinstance(document_id, bool) or int(document_id) <= 0:
            raise ValueError("shadow candidate score has an invalid document id")
        resolved_id = int(document_id)
        resolved_score = float(fused_score)
        if not math.isfinite(resolved_score):
            raise ValueError("shadow candidate score must be finite")
        if resolved_id in seen:
            raise ValueError("shadow candidate score contains a duplicate document id")
        seen.add(resolved_id)
        normalized.append((resolved_id, resolved_score))
    return tuple(normalized)


def _compatible_segmentation_generation(left: str, right: str) -> bool:
    left_base = left.split(":late:", 1)[0]
    right_base = right.split(":late:", 1)[0]
    return left == right or (
        left_base == right_base
        and (":late:" in left or ":late:" in right)
    )


def _validate_derivation_sources(
    derivation: EpisodeDerivation,
    *,
    allowed_source_ids: set[str],
) -> None:
    for item in (*derivation.facts, *derivation.events):
        source_ids = tuple(dict.fromkeys(item.source_msg_ids))
        if not source_ids or any(source_id not in allowed_source_ids for source_id in source_ids):
            raise ValueError("derived fact/event has invalid source message provenance")
    for window in derivation.windows:
        if not window.source_platform_msg_ids or any(
            source_id not in allowed_source_ids
            for source_id in window.source_platform_msg_ids
        ):
            raise ValueError("retrieval window has invalid source message provenance")
