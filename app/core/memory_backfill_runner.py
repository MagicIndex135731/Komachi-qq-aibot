from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import text

from app.core.memory_backfill import message_ledger_manifest_sha256
from app.storage.db import (
    activate_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
)
from app.storage.repositories import MemoryBackfillRunRepository


MANDATORY_JOB_TYPES = ("episode_allocate", "memory_episode_process")


class BackfillBackgroundService(Protocol):
    def enqueue_message(
        self,
        *,
        group_id: int,
        message_id: int,
        now: datetime | None = None,
        backfill_run_id: int | None = None,
        watermark_message_id: int | None = None,
    ) -> object: ...

    def run_once(self, *, now: datetime | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class GroupBackfillCoverage:
    group_id: int
    watermark_message_id: int
    messages: int
    eligible_messages: int
    assigned_messages: int
    retrieval_documents: int
    embedding_ready: int
    vector_eligible_documents: int
    embedding_pending: int
    embedding_disabled: int
    embedding_failed: int
    embedding_generation_mismatch: int
    blocked_derived_documents: int


@dataclass(frozen=True, slots=True)
class BackfillCoverageReport:
    run_id: int
    run_key: str
    status: str
    pending_jobs: int
    running_jobs: int
    failed_jobs: int
    generation_mismatch_jobs: int
    index_generation: str
    index_generation_status: str
    index_generation_active: bool
    orphan_episode_messages: int
    orphan_document_sources: int
    groups: tuple[GroupBackfillCoverage, ...]

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_key": self.run_key,
            "status": self.status,
            "pending_jobs": self.pending_jobs,
            "running_jobs": self.running_jobs,
            "failed_jobs": self.failed_jobs,
            "generation_mismatch_jobs": self.generation_mismatch_jobs,
            "index_generation": self.index_generation,
            "index_generation_status": self.index_generation_status,
            "index_generation_active": self.index_generation_active,
            "orphan_episode_messages": self.orphan_episode_messages,
            "orphan_document_sources": self.orphan_document_sources,
            "groups": [asdict(group) for group in self.groups],
        }


def group_watermarks_from_manifest(manifest: dict[str, Any]) -> dict[int, int]:
    buckets = manifest.get("buckets")
    if not isinstance(buckets, dict):
        raise ValueError("backup manifest buckets must be an object")
    watermarks: dict[int, int] = {}
    for key, value in buckets.items():
        if not str(key).startswith("group:"):
            continue
        if not isinstance(value, dict):
            raise ValueError("backup manifest group bucket must be an object")
        group_id = value.get("group_id")
        watermark = value.get("watermark")
        if (
            isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id <= 0
            or isinstance(watermark, bool)
            or not isinstance(watermark, int)
            or watermark < 0
        ):
            raise ValueError("backup manifest contains an invalid group watermark")
        watermarks[group_id] = watermark
    if not watermarks:
        raise ValueError("backup manifest contains no group watermarks")
    return dict(sorted(watermarks.items()))


def run_memory_backfill(
    *,
    engine: object,
    background_service: BackfillBackgroundService,
    manifest: dict[str, Any],
    run_key: str,
    segmentation_generation: str,
    compaction_generation: str,
    index_generation: str,
    max_steps: int = 1_000_000,
    finalize: bool = True,
) -> BackfillCoverageReport:
    """Run or resume one manifest-bounded, generation-pinned backfill.

    The persisted job payload contains only IDs, watermarks, and generation
    metadata. Raw message content remains in the source-of-truth message table.
    """

    normalized_key = str(run_key).strip()
    if not normalized_key:
        raise ValueError("run_key is required")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    watermarks = group_watermarks_from_manifest(manifest)
    now = datetime.now(UTC)
    with session_scope(engine) as session:
        runs = MemoryBackfillRunRepository(session)
        run = runs.create_run(
            run_key=normalized_key,
            snapshot_watermarks={
                str(group_id): watermark
                for group_id, watermark in watermarks.items()
            },
            manifest=manifest,
            segmentation_generation=str(segmentation_generation),
            compaction_generation=str(compaction_generation),
            index_generation=str(index_generation),
            created_at=now,
        )
        if (
            dict(run.snapshot_watermarks_json or {})
            != {str(group_id): watermark for group_id, watermark in watermarks.items()}
            or message_ledger_manifest_sha256(dict(run.manifest_json or {}))
            != message_ledger_manifest_sha256(manifest)
            or run.segmentation_generation != str(segmentation_generation)
            or run.compaction_generation != str(compaction_generation)
            or run.index_generation != str(index_generation)
        ):
            raise ValueError("run_key already belongs to a different backfill contract")
        validator = getattr(background_service, "validate_backfill_contract", None)
        if callable(validator):
            validator(
                segmentation_generation=str(segmentation_generation),
                compaction_generation=str(compaction_generation),
                index_generation=str(index_generation),
            )
        run_id = int(run.id)
        if run.status == "completed":
            completed_report = collect_backfill_coverage(
                engine=engine,
                run_id=run_id,
                run_key=normalized_key,
                watermarks=watermarks,
            )
            if _coverage_is_complete(completed_report):
                return completed_report
        runs.update_status(
            run_id=run_id,
            status="running",
            completed_at=None,
            last_error_code="",
        )

    recover_stale = getattr(background_service, "recover_stale_jobs", None)
    if callable(recover_stale):
        recover_stale(now=now)

    for group_id, watermark in watermarks.items():
        if watermark <= 0:
            continue
        background_service.enqueue_message(
            group_id=group_id,
            message_id=watermark,
            now=now,
            backfill_run_id=run_id,
            watermark_message_id=watermark,
        )

    try:
        for _ in range(max_steps):
            if not background_service.run_once():
                break
        else:
            raise RuntimeError("backfill exceeded its bounded processing steps")

        _activate_target_vector_generation(
            engine=engine,
            index_generation=str(index_generation),
        )
        report = collect_backfill_coverage(
            engine=engine,
            run_id=run_id,
            run_key=normalized_key,
            watermarks=watermarks,
        )
        if not _coverage_is_complete(report):
            raise RuntimeError("backfill coverage is incomplete")
    except Exception as exc:
        with session_scope(engine) as session:
            MemoryBackfillRunRepository(session).update_status(
                run_id=run_id,
                status="failed",
                completed_at=None,
                last_error_code=type(exc).__name__,
            )
        raise

    if finalize:
        finalize_backfill_run(engine=engine, run_id=run_id)
    return collect_backfill_coverage(
        engine=engine,
        run_id=run_id,
        run_key=normalized_key,
        watermarks=watermarks,
    )


def finalize_backfill_run(*, engine: object, run_id: int) -> None:
    """Commit the completed state only after all external final gates pass."""
    with session_scope(engine) as session:
        MemoryBackfillRunRepository(session).update_status(
            run_id=int(run_id),
            status="completed",
            completed_at=datetime.now(UTC),
            last_error_code="",
        )


def collect_backfill_coverage(
    *,
    engine: object,
    run_id: int,
    run_key: str,
    watermarks: dict[int, int],
) -> BackfillCoverageReport:
    with session_scope(engine) as session:
        run = session.execute(
            text(
                "SELECT status, run_key, segmentation_generation, "
                "compaction_generation, index_generation "
                "FROM memory_backfill_runs WHERE id = :run_id"
            ),
            {"run_id": int(run_id)},
        ).one()
        if str(run.run_key) != str(run_key):
            raise ValueError("backfill coverage run identity does not match")
        job_counts = {
            str(row.status): int(row.count)
            for row in session.execute(
                text(
                    "SELECT status, COUNT(*) AS count FROM jobs "
                    "WHERE backfill_run_id = :run_id "
                    "AND job_type IN ('episode_allocate','memory_episode_process') "
                    "GROUP BY status"
                ),
                {"run_id": int(run_id)},
            )
        }
        generation_mismatch_jobs = int(
            session.execute(
                text(
                    "SELECT COUNT(*) FROM jobs WHERE backfill_run_id = :run_id AND ("
                    "(job_type = 'episode_allocate' "
                    "AND target_generation != :segmentation_generation "
                    "AND target_generation NOT LIKE :late_generation_pattern) OR "
                    "(job_type = 'memory_episode_process' "
                    "AND target_generation != :compaction_generation))"
                ),
                {
                    "run_id": int(run_id),
                    "segmentation_generation": str(run.segmentation_generation),
                    "late_generation_pattern": (
                        f"{str(run.segmentation_generation)}:late:%"
                    ),
                    "compaction_generation": str(run.compaction_generation),
                },
            ).scalar_one()
        )
        index_generation_status, index_generation_active = _index_generation_state(
            session=session,
            index_generation=str(run.index_generation),
        )
        orphan_episode_messages = int(
            session.execute(
                text(
                    "SELECT COUNT(*) FROM episode_messages em "
                    "LEFT JOIN conversation_episodes e ON e.id = em.episode_id "
                    "LEFT JOIN messages m ON m.id = em.message_id "
                    "WHERE e.id IS NULL OR m.id IS NULL OR e.group_id != em.group_id "
                    "OR m.group_id != em.group_id"
                )
            ).scalar_one()
        )
        orphan_document_sources = int(
            session.execute(
                text(
                    "SELECT COUNT(*) FROM retrieval_document_messages rdm "
                    "LEFT JOIN retrieval_documents rd ON rd.id = rdm.document_id "
                    "LEFT JOIN messages m ON m.id = rdm.message_id "
                    "WHERE rd.id IS NULL OR m.id IS NULL OR rd.group_id != rdm.group_id "
                    "OR m.group_id != rdm.group_id"
                )
            ).scalar_one()
        )
        groups = tuple(
            _collect_group_coverage(
                session=session,
                group_id=group_id,
                watermark=watermark,
                segmentation_generation=str(run.segmentation_generation),
                compaction_generation=str(run.compaction_generation),
                vector_generation=_vector_generation(str(run.index_generation)),
            )
            for group_id, watermark in sorted(watermarks.items())
        )
    return BackfillCoverageReport(
        run_id=int(run_id),
        run_key=str(run_key),
        status=str(run.status),
        pending_jobs=job_counts.get("queued", 0) + job_counts.get("pending", 0),
        running_jobs=job_counts.get("running", 0),
        failed_jobs=job_counts.get("failed", 0),
        generation_mismatch_jobs=generation_mismatch_jobs,
        index_generation=str(run.index_generation),
        index_generation_status=index_generation_status,
        index_generation_active=index_generation_active,
        orphan_episode_messages=orphan_episode_messages,
        orphan_document_sources=orphan_document_sources,
        groups=groups,
    )


def _collect_group_coverage(
    *,
    session: object,
    group_id: int,
    watermark: int,
    segmentation_generation: str,
    compaction_generation: str,
    vector_generation: int | None,
) -> GroupBackfillCoverage:
    parameters = {
        "group_id": int(group_id),
        "watermark": int(watermark),
        "segmentation_generation": str(segmentation_generation),
        "late_generation_pattern": f"{str(segmentation_generation)}:late:%",
        "compaction_generation": str(compaction_generation),
        "vector_generation": vector_generation,
    }
    messages = int(
        session.execute(
            text(
                "SELECT COUNT(*) FROM messages "
                "WHERE group_id = :group_id AND id <= :watermark"
            ),
            parameters,
        ).scalar_one()
    )
    assigned = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT em.message_id) FROM episode_messages em "
                "JOIN messages m ON m.id = em.message_id "
                "JOIN conversation_episodes e ON e.id = em.episode_id "
                "WHERE em.group_id = :group_id AND m.id <= :watermark "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern)"
            ),
            parameters,
        ).scalar_one()
    )
    eligible_messages = int(
        session.execute(
            text(
                "SELECT COUNT(*) FROM messages "
                "WHERE group_id = :group_id AND id <= :watermark "
                "AND (json_extract(raw_json, '$.delivery_state') IS NULL "
                "OR json_extract(raw_json, '$.delivery_state') <> 'reserved')"
            ),
            parameters,
        ).scalar_one()
    )
    documents = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT rd.id) FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' AND e.is_current = 1 "
                "AND (e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation"
            ),
            parameters,
        ).scalar_one()
    )
    vector_eligible = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT rd.id) FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' AND rd.embedding_eligible = 1 "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation"
            ),
            parameters,
        ).scalar_one()
    )
    embedding_ready = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT rd.id) FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' AND rd.embedding_eligible = 1 "
                "AND rd.embedding_status = 'ready' "
                "AND (:vector_generation IS NULL OR "
                "rd.embedding_generation = :vector_generation) "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation"
            ),
            parameters,
        ).scalar_one()
    )
    embedding_status_counts = {
        str(row.embedding_status): int(row.count)
        for row in session.execute(
            text(
                "SELECT rd.embedding_status, COUNT(DISTINCT rd.id) AS count "
                "FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' AND rd.embedding_eligible = 1 "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation GROUP BY rd.embedding_status"
            ),
            parameters,
        )
    }
    embedding_failed = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT rd.id) FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' AND rd.embedding_eligible = 1 "
                "AND rd.embedding_status = 'failed' "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation"
            ),
            parameters,
        ).scalar_one()
    )
    embedding_generation_mismatch = 0
    if vector_generation is not None:
        embedding_generation_mismatch = max(0, vector_eligible - embedding_ready)
    blocked_derived = int(
        session.execute(
            text(
                "SELECT COUNT(DISTINCT rd.id) FROM retrieval_documents rd "
                "JOIN retrieval_document_messages rdm ON rdm.document_id = rd.id "
                "JOIN messages m ON m.id = rdm.message_id "
                "JOIN conversation_episodes e ON e.id = rd.episode_id "
                "AND e.group_id = rd.group_id "
                "WHERE rd.group_id = :group_id AND m.id <= :watermark "
                "AND rd.status = 'active' "
                "AND e.is_current = 1 AND ("
                "e.segmentation_version = :segmentation_generation OR "
                "e.segmentation_version LIKE :late_generation_pattern) "
                "AND e.compaction_version = :compaction_generation "
                "AND json_extract(rd.metadata_json, '$.compaction_generation') "
                "= :compaction_generation "
                "AND json_extract(m.raw_json, '$.delivery_state') = 'blocked' "
                "AND json_extract(m.raw_json, '$.failure_kind') = "
                "'qq_sensitive_content'"
            ),
            parameters,
        ).scalar_one()
    )
    return GroupBackfillCoverage(
        group_id=int(group_id),
        watermark_message_id=int(watermark),
        messages=messages,
        eligible_messages=eligible_messages,
        assigned_messages=assigned,
        retrieval_documents=documents,
        embedding_ready=embedding_ready,
        vector_eligible_documents=vector_eligible,
        embedding_pending=embedding_status_counts.get("pending", 0),
        embedding_disabled=embedding_status_counts.get("disabled", 0),
        embedding_failed=embedding_failed,
        embedding_generation_mismatch=embedding_generation_mismatch,
        blocked_derived_documents=blocked_derived,
    )


def _vector_generation(index_generation: str) -> int | None:
    prefix = "vector:"
    if not str(index_generation).startswith(prefix):
        return None
    raw_generation = str(index_generation)[len(prefix) :]
    try:
        generation = int(raw_generation)
    except ValueError as exc:
        raise ValueError("backfill vector generation is invalid") from exc
    if generation <= 0:
        raise ValueError("backfill vector generation is invalid")
    return generation


def _index_generation_state(
    *,
    session: object,
    index_generation: str,
) -> tuple[str, bool]:
    vector_generation = _vector_generation(index_generation)
    if vector_generation is not None:
        row = session.execute(
            text(
                "SELECT status, is_active FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": vector_generation},
        ).one_or_none()
    elif str(index_generation).startswith("fts"):
        row = session.execute(
            text(
                "SELECT status, is_active FROM retrieval_index_state "
                "WHERE channel = 'fts' AND is_active = 1 "
                "ORDER BY generation DESC LIMIT 1"
            )
        ).one_or_none()
    else:
        raise ValueError("backfill index generation is invalid")
    if row is None:
        return "missing", False
    return str(row.status), bool(row.is_active)


def _activate_target_vector_generation(*, engine: object, index_generation: str) -> None:
    generation = _vector_generation(index_generation)
    if generation is None:
        return
    coverage = refresh_retrieval_vector_generation(
        engine,
        generation=generation,
        mark_ready=True,
    )
    if coverage.status != "ready":
        return
    with engine.connect() as connection:
        active = connection.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).scalar_one_or_none()
    expected_active = int(active) if active is not None else None
    activate_retrieval_vector_generation(
        engine,
        generation=generation,
        expected_active_generation=expected_active,
    )


def _coverage_is_complete(report: BackfillCoverageReport) -> bool:
    vector_required = report.index_generation.startswith("vector:")
    index_ready = (
        report.index_generation_status == "ready"
        and report.index_generation_active
    )
    return (
        report.pending_jobs == 0
        and report.running_jobs == 0
        and report.failed_jobs == 0
        and report.generation_mismatch_jobs == 0
        and report.orphan_episode_messages == 0
        and report.orphan_document_sources == 0
        and index_ready
        and all(
            group.assigned_messages == group.eligible_messages
            and group.blocked_derived_documents == 0
            and (
                not vector_required
                or (
                    group.embedding_ready == group.vector_eligible_documents
                    and group.embedding_pending == 0
                    and group.embedding_disabled == 0
                    and group.embedding_failed == 0
                    and group.embedding_generation_mismatch == 0
                )
            )
            for group in report.groups
        )
    )
