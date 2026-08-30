from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.storage.db import session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    UserRepository,
)


def _seed(engine) -> list[int]:
    observed_at = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=100,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        messages = __import__("app.storage.repositories", fromlist=["MessageRepository"]).MessageRepository(session)
        row = messages.add_group_message(
            platform_msg_id="vec-source",
            group_id=100,
            user_id=20001,
            timestamp=observed_at,
            plain_text="阿渣喜欢看动画",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        first = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣喜欢看动画",
            importance=2,
            confidence=0.8,
            source_msg_id="vec-source",
            valid_from=observed_at,
        )
        second = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="20001",
            memory_kind="current",
            content="阿渣在做前后端",
            importance=5,
            confidence=0.9,
            source_msg_id="vec-source",
            valid_from=observed_at,
        )
        session.flush()
        return [int(first.id), int(second.id)]


def test_semantic_vector_upsert_load_delete_and_deactivate(sqlite_engine) -> None:
    memory_ids = _seed(sqlite_engine)
    rows = [
        {
            "memory_id": memory_ids[0],
            "group_id": 100,
            "provider": "fake",
            "model": "semantic-test",
            "dimensions": 2,
            "version": "v1",
            "vector_json": "[0.9,0.1]",
        },
        {
            "memory_id": memory_ids[1],
            "group_id": 100,
            "provider": "fake",
            "model": "semantic-test",
            "dimensions": 2,
            "version": "v1",
            "vector_json": "[0.1,0.9]",
        },
    ]
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        assert memories.upsert_memory_item_semantic_vectors(rows) == 2
        loaded = memories.load_memory_item_semantic_vectors(
            memory_ids,
            provider="fake",
            model="semantic-test",
            dimensions=2,
            version="v1",
        )
        assert loaded == {
            memory_ids[0]: [0.9, 0.1],
            memory_ids[1]: [0.1, 0.9],
        }
        assert memories.load_memory_item_semantic_vectors(
            memory_ids,
            provider="other",
        ) == {}

        deactivated = memories.deactivate_memory_items(
            [memory_ids[0]],
            valid_until=datetime.now(UTC),
        )
        assert deactivated == 1
        assert memories.delete_memory_item_semantic_vectors([memory_ids[0]]) == 0
        remaining = memories.load_memory_item_semantic_vectors(
            memory_ids,
            provider="fake",
            model="semantic-test",
            dimensions=2,
            version="v1",
        )
        assert memory_ids[0] not in remaining
        assert memory_ids[1] in remaining

    with session_scope(sqlite_engine) as session:
        rows = MemoryRepository(session).list_active_memory_items_for_indexing(
            after_id=0,
            limit=10,
        )
        assert [int(row.id) for row in rows] == [memory_ids[1]]


def test_expiry_deactivates_every_memory_kind(sqlite_engine) -> None:
    memory_ids = _seed(sqlite_engine)
    now = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        rows = [session.get(MemoryItem, memory_id) for memory_id in memory_ids]
        assert all(row is not None for row in rows)
        for row in rows:
            row.valid_until = now - timedelta(seconds=1)
            row.expires_at = row.valid_until

    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        assert memories.expire_stale_memories(now=now) == 2
        remaining = memories.list_active_memory_items_for_indexing(limit=10)
        assert remaining == []
