from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

from sqlalchemy import select, text

from app.storage.db import session_scope
from app.storage.models import Job, MemoryItem, RetrievalDocument, Summary
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    UserRepository,
)
from scripts.maintain_memory_integrity import (
    audit_memory_integrity,
    backup_database,
    repair_memory_integrity,
)


NOW = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)


def _seed_drift(engine) -> None:
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=900000001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=900000101,
            nickname="member",
            group_card="",
        )
        message = MessageRepository(session).add_group_message(
            platform_msg_id="integrity-source",
            group_id=900000001,
            user_id=900000101,
            timestamp=NOW - timedelta(hours=2),
            plain_text="source",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        memories = MemoryRepository(session)
        expired = memories.add_memory(
            scope_type="group",
            scope_id="900000001",
            subject_type="user",
            subject_id="900000101",
            memory_kind="plan",
            content="short expired plan",
            importance=2,
            confidence=0.8,
            source_msg_id="integrity-source",
            valid_until=NOW - timedelta(seconds=1),
        )
        stale = memories.add_memory(
            scope_type="group",
            scope_id="900000001",
            subject_type="user",
            subject_id="900000101",
            memory_kind="fact",
            content="manually stale fact",
            importance=2,
            confidence=0.8,
            source_msg_id="integrity-source",
        )
        active = memories.add_memory(
            scope_type="group",
            scope_id="900000001",
            subject_type="user",
            subject_id="900000101",
            memory_kind="fact",
            content="active missing fts",
            importance=2,
            confidence=0.8,
            source_msg_id="integrity-source",
        )
        session.flush()
        stale.status = "inactive"
        memories.upsert_memory_item_semantic_vectors(
            [
                {
                    "memory_id": stale.id,
                    "group_id": 900000001,
                    "provider": "fake",
                    "model": "fake",
                    "dimensions": 2,
                    "version": "v1",
                    "vector_json": "[0.1,0.2]",
                }
            ]
        )
        session.add(
            RetrievalDocument(
                scope_type="group",
                scope_id="900000001",
                group_id=900000001,
                episode_id=None,
                document_kind="memory",
                source_table="memory_items",
                source_id=str(stale.id),
                start_at=NOW - timedelta(hours=2),
                end_at=NOW - timedelta(hours=1),
                content="stale",
                metadata_json={},
                content_hash="stale-doc",
                status="active",
                embedding_status="disabled",
            )
        )
        session.add(
            Summary(
                scope_type="group",
                scope_id="900000001",
                summary_level="episode",
                summary_key="",
                start_at=NOW,
                end_at=NOW - timedelta(hours=1),
                content="inverted",
                source_count=1,
                source_start_msg_id="later",
                source_end_msg_id="earlier",
                source_summary_ids=[],
                status="active",
            )
        )
        session.add(
            Summary(
                scope_type="group",
                scope_id="900000001",
                summary_level="window",
                summary_key="",
                start_at=NOW - timedelta(days=1),
                end_at=NOW,
                content="legacy no provenance",
                source_count=1,
                source_start_msg_id=None,
                source_end_msg_id=None,
                source_summary_ids=[],
                status="active",
            )
        )
        session.add(
            Job(
                job_type="memory_compaction",
                job_key="legacy-test",
                payload_json={},
                status="queued",
                run_at=NOW - timedelta(days=1),
            )
        )
        session.execute(
            text("DELETE FROM memory_items_fts WHERE memory_id=:memory_id"),
            {"memory_id": str(active.id)},
        )
        assert expired.id is not None
        assert message.id is not None


def test_audit_and_repair_memory_integrity_are_idempotent(sqlite_engine) -> None:
    _seed_drift(sqlite_engine)

    before = audit_memory_integrity(sqlite_engine, now=NOW)
    assert before["expired_active_memories"] == 1
    assert before["inverted_active_summaries"] == 1
    assert before["provenance_less_legacy_summaries"] == 1
    assert before["legacy_compaction_jobs_needing_cleanup"] == 1
    assert before["semantic_vectors_for_nonactive_memories"] == 1
    assert before["active_docs_for_nonactive_memories"] == 1
    assert before["active_memories_missing_fts"] == 1
    assert before["fts_rows_for_nonactive_memories"] == 1

    repaired = repair_memory_integrity(sqlite_engine, now=NOW)
    assert repaired["expired_memories"] == 1
    assert repaired["normalized_summaries"] == 1
    assert repaired["deactivated_legacy_summaries"] == 1
    assert repaired["cancelled_legacy_jobs"] == 1

    after = audit_memory_integrity(sqlite_engine, now=NOW)
    for key in (
        "expired_active_memories",
        "inverted_active_summaries",
        "provenance_less_legacy_summaries",
        "legacy_compaction_jobs_needing_cleanup",
        "semantic_vectors_for_nonactive_memories",
        "active_docs_for_nonactive_memories",
        "active_memories_missing_fts",
        "fts_rows_for_nonactive_memories",
    ):
        assert after[key] == 0

    second = repair_memory_integrity(sqlite_engine, now=NOW)
    assert all(value == 0 for value in second.values())
    with session_scope(sqlite_engine) as session:
        assert session.scalar(
            select(Job.status).where(Job.job_key == "legacy-test")
        ) == "cancelled"
        assert session.scalar(
            select(MemoryItem.status).where(MemoryItem.content == "short expired plan")
        ) == "inactive"


def test_backup_database_uses_online_backup(sqlite_engine, tmp_path: Path) -> None:
    source = Path(str(sqlite_engine.url.database))

    backup = backup_database(source, tmp_path / "backups")

    assert backup.is_file()
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
