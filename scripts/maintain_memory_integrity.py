"""Audit and deterministically repair Memory V3 persistence invariants.

The default ``audit`` command is read-only.  ``repair --apply`` creates an
SQLite online backup before changing lifecycle or derived-index state.  The
report contains counts only and never emits chat or memory content.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3
from typing import Sequence

from sqlalchemy import Integer, bindparam, cast, func, or_, select, text

from app.core.memory_background_service import _episode_summary_bounds
from app.storage.db import build_engine, session_scope
from app.storage.models import Job, MemoryItem, RetrievalDocument, Summary
from app.storage.repositories import EpisodeRepository, JobRepository, MemoryRepository


_EPISODE_SUMMARY_KEY = re.compile(r"^episode:(\d+):")


def _scalar(connection, statement: str) -> int:
    return int(connection.execute(text(statement)).scalar_one() or 0)


def audit_memory_integrity(engine, *, now: datetime) -> dict[str, int]:
    """Return content-free invariant counts for one database snapshot."""
    normalized_now = now.astimezone(UTC).replace(tzinfo=None)
    with engine.connect() as connection:
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            ).scalars()
        )

        def optional_scalar(statement: str, *required_tables: str) -> int:
            if any(table not in table_names for table in required_tables):
                return -1
            return _scalar(connection, statement)

        return {
            "missing_optional_schema_tables": sum(
                table not in table_names
                for table in (
                    "memory_item_semantic_vectors",
                    "retrieval_documents",
                    "memory_items_fts",
                    "retrieval_index_state",
                )
            ),
            "active_memories": _scalar(
                connection,
                "SELECT COUNT(*) FROM memory_items WHERE status='active'",
            ),
            "expired_active_memories": int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM memory_items WHERE status='active' "
                        "AND valid_until IS NOT NULL AND valid_until < :now"
                    ),
                    {"now": normalized_now},
                ).scalar_one()
                or 0
            ),
            "inverted_active_summaries": _scalar(
                connection,
                "SELECT COUNT(*) FROM summaries WHERE status='active' "
                "AND end_at < start_at",
            ),
            "provenance_less_legacy_summaries": _scalar(
                connection,
                "SELECT COUNT(*) FROM summaries WHERE status='active' "
                "AND summary_level='window' "
                "AND COALESCE(source_start_msg_id,'')='' "
                "AND COALESCE(source_end_msg_id,'')=''",
            ),
            "legacy_compaction_jobs_needing_cleanup": _scalar(
                connection,
                "SELECT COUNT(*) FROM jobs WHERE job_type='memory_compaction' "
                "AND status IN ('queued','running','failed')",
            ),
            "semantic_vectors_for_nonactive_memories": optional_scalar(
                "SELECT COUNT(*) FROM memory_item_semantic_vectors v "
                "LEFT JOIN memory_items m ON m.id=v.memory_id "
                "WHERE m.id IS NULL OR m.status<>'active'",
                "memory_item_semantic_vectors",
            ),
            "active_memories_missing_semantic_vectors": optional_scalar(
                "SELECT COUNT(*) FROM memory_items m WHERE m.status='active' "
                "AND NOT EXISTS (SELECT 1 FROM memory_item_semantic_vectors v "
                "WHERE v.memory_id=m.id)",
                "memory_item_semantic_vectors",
            ),
            "active_docs_for_nonactive_memories": optional_scalar(
                "SELECT COUNT(*) FROM retrieval_documents d "
                "LEFT JOIN memory_items m ON m.id=CAST(d.source_id AS INTEGER) "
                "WHERE d.document_kind='memory' AND d.source_table='memory_items' "
                "AND d.status='active' AND (m.id IS NULL OR m.status<>'active')",
                "retrieval_documents",
            ),
            "active_memories_without_active_docs": optional_scalar(
                "SELECT COUNT(*) FROM memory_items m WHERE m.status='active' "
                "AND NOT EXISTS (SELECT 1 FROM retrieval_documents d "
                "WHERE d.document_kind='memory' AND d.source_table='memory_items' "
                "AND d.source_id=CAST(m.id AS TEXT) AND d.status='active')",
                "retrieval_documents",
            ),
            "active_memories_only_with_inactive_docs": optional_scalar(
                "SELECT COUNT(*) FROM memory_items m WHERE m.status='active' "
                "AND EXISTS (SELECT 1 FROM retrieval_documents d "
                "WHERE d.document_kind='memory' AND d.source_table='memory_items' "
                "AND d.source_id=CAST(m.id AS TEXT) AND d.status<>'active') "
                "AND NOT EXISTS (SELECT 1 FROM retrieval_documents d "
                "WHERE d.document_kind='memory' AND d.source_table='memory_items' "
                "AND d.source_id=CAST(m.id AS TEXT) AND d.status='active')",
                "retrieval_documents",
            ),
            "active_memories_missing_fts": optional_scalar(
                "SELECT COUNT(*) FROM memory_items m WHERE m.status='active' "
                "AND NOT EXISTS (SELECT 1 FROM memory_items_fts f "
                "WHERE f.memory_id=CAST(m.id AS TEXT))",
                "memory_items_fts",
            ),
            "fts_rows_for_nonactive_memories": optional_scalar(
                "SELECT COUNT(*) FROM memory_items_fts f "
                "LEFT JOIN memory_items m ON CAST(m.id AS TEXT)=f.memory_id "
                "WHERE m.id IS NULL OR m.status<>'active'",
                "memory_items_fts",
            ),
            "duplicate_active_memory_excess": _scalar(
                connection,
                "SELECT COALESCE(SUM(n-1),0) FROM ("
                "SELECT COUNT(*) AS n FROM memory_items WHERE status='active' "
                "GROUP BY scope_type,scope_id,subject_id,memory_kind,content "
                "HAVING COUNT(*)>1)",
            ),
            "long_member_plan_or_decision": _scalar(
                connection,
                "SELECT COUNT(*) FROM memory_items WHERE status='active' "
                "AND subject_type='user' AND memory_kind IN ('plan','decision') "
                "AND length(content)>320",
            ),
            "unstructured_plan_or_decision": _scalar(
                connection,
                "SELECT COUNT(*) FROM memory_items WHERE status='active' "
                "AND memory_kind IN ('plan','decision') "
                "AND (COALESCE(canonical_key,'')='' OR COALESCE(predicate,'')='' "
                "OR COALESCE(object_text,'')='')",
            ),
            "active_index_states_with_impossible_counts": optional_scalar(
                "SELECT COUNT(*) FROM retrieval_index_state WHERE is_active=1 "
                "AND (indexed_documents<0 OR total_documents<0 "
                "OR indexed_documents>total_documents)",
                "retrieval_index_state",
            ),
        }


def backup_database(database: Path, backup_dir: Path) -> Path:
    """Create a consistent SQLite online backup without copying WAL by hand."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"{database.stem}.memory-integrity-{stamp}.db"
    if target.exists():
        raise FileExistsError(target)
    source_uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def repair_memory_integrity(engine, *, now: datetime) -> dict[str, int]:
    """Repair safe lifecycle/index invariants in one transaction."""
    repaired = {
        "expired_memories": 0,
        "normalized_summaries": 0,
        "deactivated_legacy_summaries": 0,
        "cancelled_legacy_jobs": 0,
        "deleted_orphan_semantic_vectors": 0,
        "deactivated_orphan_memory_docs": 0,
        "deleted_stale_fts_rows": 0,
        "inserted_missing_fts_rows": 0,
    }
    with session_scope(engine) as session:
        memories = MemoryRepository(session)
        repaired["expired_memories"] = memories.expire_stale_memories(now=now)

        inverted = list(
            session.scalars(
                select(Summary).where(
                    Summary.status == "active",
                    Summary.end_at < Summary.start_at,
                )
            )
        )
        for summary in inverted:
            start_at, end_at = summary.end_at, summary.start_at
            start_source = summary.source_end_msg_id
            end_source = summary.source_start_msg_id
            match = _EPISODE_SUMMARY_KEY.match(str(summary.summary_key or ""))
            if match is not None:
                episode_id = int(match.group(1))
                episode = EpisodeRepository(session).get_episode(episode_id)
                if episode is not None and str(episode.group_id) == str(summary.scope_id):
                    messages = EpisodeRepository(session).list_episode_messages(
                        episode_id=episode_id,
                        group_id=int(episode.group_id),
                    )
                    if messages:
                        start_message, end_message = _episode_summary_bounds(messages)
                        start_at, end_at = start_message.timestamp, end_message.timestamp
                        start_source = start_message.platform_msg_id
                        end_source = end_message.platform_msg_id
            summary.start_at = start_at
            summary.end_at = end_at
            summary.source_start_msg_id = start_source
            summary.source_end_msg_id = end_source
            session.execute(
                text(
                    "UPDATE retrieval_documents SET start_at=:start_at,end_at=:end_at,"
                    "updated_at=:updated_at WHERE source_table='summaries' AND source_id=:source_id"
                ),
                {
                    "start_at": start_at,
                    "end_at": end_at,
                    "updated_at": now.astimezone(UTC).replace(tzinfo=None),
                    "source_id": str(summary.id),
                },
            )
        repaired["normalized_summaries"] = len(inverted)

        legacy_summaries = list(
            session.scalars(
                select(Summary).where(
                    Summary.status == "active",
                    Summary.summary_level == "window",
                    or_(
                        Summary.source_start_msg_id.is_(None),
                        Summary.source_start_msg_id == "",
                    ),
                    or_(
                        Summary.source_end_msg_id.is_(None),
                        Summary.source_end_msg_id == "",
                    ),
                )
            )
        )
        legacy_ids = [str(summary.id) for summary in legacy_summaries]
        for summary in legacy_summaries:
            summary.status = "inactive"
        if legacy_ids:
            session.execute(
                text(
                    "UPDATE retrieval_documents SET status='inactive',embedding_status='stale' "
                    "WHERE source_table='summaries' AND source_id IN :source_ids"
                ).bindparams(bindparam("source_ids", expanding=True)),
                {"source_ids": tuple(legacy_ids)},
            )
        repaired["deactivated_legacy_summaries"] = len(legacy_ids)

        repaired["cancelled_legacy_jobs"] = JobRepository(session).cancel_jobs(
            job_type="memory_compaction",
            statuses=("queued", "running", "failed"),
            now=now,
        )

        semantic = session.execute(
            text(
                "DELETE FROM memory_item_semantic_vectors WHERE memory_id IN ("
                "SELECT v.memory_id FROM memory_item_semantic_vectors v "
                "LEFT JOIN memory_items m ON m.id=v.memory_id "
                "WHERE m.id IS NULL OR m.status<>'active')"
            )
        )
        repaired["deleted_orphan_semantic_vectors"] = int(semantic.rowcount or 0)

        documents = list(
            session.scalars(
                select(RetrievalDocument).outerjoin(
                    MemoryItem,
                    MemoryItem.id == cast(RetrievalDocument.source_id, Integer),
                ).where(
                    RetrievalDocument.document_kind == "memory",
                    RetrievalDocument.source_table == "memory_items",
                    RetrievalDocument.status == "active",
                    or_(MemoryItem.id.is_(None), MemoryItem.status != "active"),
                )
            )
        )
        for document in documents:
            document.status = "inactive"
            document.embedding_status = "stale"
        repaired["deactivated_orphan_memory_docs"] = len(documents)

        stale_fts = session.execute(
            text(
                "DELETE FROM memory_items_fts WHERE memory_id IN ("
                "SELECT f.memory_id FROM memory_items_fts f "
                "LEFT JOIN memory_items m ON CAST(m.id AS TEXT)=f.memory_id "
                "WHERE m.id IS NULL OR m.status<>'active')"
            )
        )
        repaired["deleted_stale_fts_rows"] = int(stale_fts.rowcount or 0)
        before_missing = int(
            session.execute(
                text(
                    "SELECT COUNT(*) FROM memory_items m WHERE m.status='active' "
                    "AND NOT EXISTS (SELECT 1 FROM memory_items_fts f "
                    "WHERE f.memory_id=CAST(m.id AS TEXT))"
                )
            ).scalar_one()
            or 0
        )
        session.execute(
            text(
                "INSERT INTO memory_items_fts(content,scope_type,scope_id,memory_id) "
                "SELECT m.content,m.scope_type,m.scope_id,CAST(m.id AS TEXT) "
                "FROM memory_items m WHERE m.status='active' AND NOT EXISTS ("
                "SELECT 1 FROM memory_items_fts f WHERE f.memory_id=CAST(m.id AS TEXT))"
            )
        )
        repaired["inserted_missing_fts_rows"] = before_missing
    return repaired


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "repair"), nargs="?", default="audit")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if args.command == "repair" and not args.apply:
        raise SystemExit("repair requires explicit --apply")
    if args.command == "audit" and args.apply:
        raise SystemExit("--apply is only valid with repair")

    backup = None
    if args.command == "repair":
        backup = backup_database(
            database,
            (args.backup_dir or database.parent / "backups").resolve(),
        )
    engine = build_engine(database, read_only=args.command == "audit")
    try:
        now = datetime.now(UTC)
        before = audit_memory_integrity(engine, now=now)
        repaired = repair_memory_integrity(engine, now=now) if args.command == "repair" else {}
        after = audit_memory_integrity(engine, now=now) if args.command == "repair" else before
    finally:
        engine.dispose()
    print(
        json.dumps(
            {
                "mode": args.command,
                "backup": str(backup) if backup is not None else None,
                "before": before,
                "repaired": repaired,
                "after": after,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
