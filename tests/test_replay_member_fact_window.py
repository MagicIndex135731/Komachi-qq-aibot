from __future__ import annotations

from datetime import UTC, datetime
import sqlite3
from types import SimpleNamespace

import pytest

from app.storage.db import build_engine, create_all, session_scope
from app.storage.models import (
    Group,
    MemoryItem,
    MemoryItemSemanticVector,
    Message,
    User,
)
from scripts.replay_member_fact_window import (
    _backfill_replayed_semantic_vectors,
    _verify_replay_backup,
)


def test_replay_vector_backfill_is_scoped_to_window_sources(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        session.add(Group(group_id=900000001, group_name="test"))
        session.add(User(user_id=900000101, nickname="member", group_card="member"))
        session.flush()
        message = Message(
            platform_msg_id="replay-vector-source",
            group_id=900000001,
            user_id=900000101,
            timestamp=datetime(2026, 9, 27, tzinfo=UTC),
            raw_json={},
            plain_text="我最近在学日语",
            msg_type="text",
            mentioned_bot=False,
        )
        session.add(message)
        session.flush()
        source_message_id = int(message.id)
        session.add_all(
            (
                MemoryItem(
                    scope_type="group",
                    scope_id="900000001",
                    subject_type="user",
                    subject_id="900000101",
                    memory_kind="current",
                    canonical_key="in-window",
                    predicate="学习",
                    object_text="日语",
                    content="该成员最近在学日语",
                    source_msg_id="replay-vector-source",
                    source_msg_ids=["replay-vector-source"],
                    status="active",
                ),
                MemoryItem(
                    scope_type="group",
                    scope_id="900000001",
                    subject_type="user",
                    subject_id="900000101",
                    memory_kind="current",
                    canonical_key="outside-window",
                    predicate="学习",
                    object_text="英语",
                    content="该成员以前学过英语",
                    source_msg_id="outside-source",
                    source_msg_ids=["outside-source"],
                    status="active",
                ),
            )
        )

    class FakeEmbedder:
        identity = SimpleNamespace(
            provider="test",
            model="test-model",
            dimensions=2,
            version="v1",
        )

        @staticmethod
        def embed_documents(contents):
            assert contents == ["该成员最近在学日语"]
            return [[0.25, 0.75]]

    indexed = _backfill_replayed_semantic_vectors(
        sqlite_engine,
        embedder=FakeEmbedder(),
        group_id=900000001,
        user_id=900000101,
        start_message_id=source_message_id,
        end_message_id=source_message_id,
        max_messages=10,
    )

    assert indexed == 1
    with session_scope(sqlite_engine) as session:
        vectors = session.query(MemoryItemSemanticVector).all()
        assert len(vectors) == 1
        assert vectors[0].provider == "test"
        assert vectors[0].dimensions == 2


def test_replay_vector_backfill_fails_on_incomplete_embedding_coverage(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        session.add(Group(group_id=900000001, group_name="test"))
        session.add(User(user_id=900000101, nickname="member", group_card="member"))
        session.flush()
        message = Message(
            platform_msg_id="replay-vector-source",
            group_id=900000001,
            user_id=900000101,
            timestamp=datetime(2026, 9, 27, tzinfo=UTC),
            raw_json={},
            plain_text="我最近在学日语",
            msg_type="text",
            mentioned_bot=False,
        )
        session.add(message)
        session.flush()
        source_message_id = int(message.id)
        session.add(
            MemoryItem(
                scope_type="group",
                scope_id="900000001",
                subject_type="user",
                subject_id="900000101",
                memory_kind="current",
                canonical_key="in-window",
                predicate="学习",
                object_text="日语",
                content="该成员最近在学日语",
                source_msg_id="replay-vector-source",
                source_msg_ids=["replay-vector-source"],
                status="active",
            )
        )

    class IncompleteEmbedder:
        identity = SimpleNamespace(
            provider="test",
            model="test-model",
            dimensions=2,
            version="v1",
        )

        @staticmethod
        def embed_documents(_contents):
            return []

    with pytest.raises(RuntimeError, match="incomplete replay coverage"):
        _backfill_replayed_semantic_vectors(
            sqlite_engine,
            embedder=IncompleteEmbedder(),
            group_id=900000001,
            user_id=900000101,
            start_message_id=source_message_id,
            end_message_id=source_message_id,
            max_messages=10,
        )


def test_replay_backup_must_match_window_ledger(tmp_path) -> None:
    live_path = tmp_path / "live.db"
    backup_path = tmp_path / "backup.db"
    engine = build_engine(live_path)
    create_all(engine)
    with session_scope(engine) as session:
        session.add(Group(group_id=900000001, group_name="test"))
        session.add(User(user_id=900000101, nickname="member", group_card="member"))
        session.flush()
        session.add(
            Message(
                platform_msg_id="source-1",
                group_id=900000001,
                user_id=900000101,
                timestamp=datetime(2026, 9, 27, tzinfo=UTC),
                raw_json={},
                plain_text="我最近在学日语",
                msg_type="text",
                mentioned_bot=False,
            )
        )
    with sqlite3.connect(live_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)

    digest = _verify_replay_backup(
        live_path,
        backup_path,
        group_id=900000001,
        user_id=900000101,
        start_message_id=1,
        end_message_id=None,
        max_messages=10,
    )
    assert len(digest) == 64

    with session_scope(engine) as session:
        session.add(
            Message(
                platform_msg_id="source-2",
                group_id=900000001,
                user_id=900000101,
                timestamp=datetime(2026, 9, 27, 1, tzinfo=UTC),
                raw_json={},
                plain_text="后来又学了韩语",
                msg_type="text",
                mentioned_bot=False,
            )
        )

    with pytest.raises(RuntimeError, match="does not match"):
        _verify_replay_backup(
            live_path,
            backup_path,
            group_id=900000001,
            user_id=900000101,
            start_message_id=1,
            end_message_id=None,
            max_messages=10,
        )
