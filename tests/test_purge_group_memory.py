from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

import scripts.purge_group_memory as purge
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    UserRepository,
)


def _seed(sqlite_engine) -> dict[int, int]:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    memory_ids: dict[int, int] = {}
    with session_scope(sqlite_engine) as session:
        for group_id in (100, 200):
            GroupRepository(session).upsert_group(
                group_id=group_id,
                group_name=f"g{group_id}",
                enabled=True,
                speak_enabled=True,
            )
        UserRepository(session).upsert_user(
            user_id=1,
            nickname="a",
            group_card="a",
        )
        messages = MessageRepository(session)
        for group_id in (100, 200):
            messages.add_group_message(
                platform_msg_id=f"m-{group_id}",
                group_id=group_id,
                user_id=1,
                timestamp=now,
                plain_text="hi",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        memories = MemoryRepository(session)
        for group_id in (100, 200):
            row = memories.add_memory(
                scope_type="group",
                scope_id=str(group_id),
                subject_type="user",
                subject_id="1",
                memory_kind="profile",
                content=f"画像{group_id}",
                importance=1,
                confidence=0.9,
                source_msg_id=f"m-{group_id}",
                valid_from=now,
            )
            session.flush()
            memory_ids[group_id] = int(row.id)
        memories.upsert_memory_item_semantic_vectors(
            [
                {
                    "memory_id": memory_ids[100],
                    "group_id": 100,
                    "provider": "x",
                    "model": "y",
                    "dimensions": 2,
                    "version": "v",
                    "vector_json": "[0.1,0.2]",
                },
                {
                    "memory_id": memory_ids[200],
                    "group_id": 200,
                    "provider": "x",
                    "model": "y",
                    "dimensions": 2,
                    "version": "v",
                    "vector_json": "[0.3,0.4]",
                },
            ]
        )
    return memory_ids


def _scalar(engine, statement: str, parameters: dict | None = None) -> int:
    with engine.connect() as connection:
        return int(
            connection.execute(
                text(statement),
                parameters or {},
            ).scalar_one()
        )


def test_purge_group_memory_dry_run_and_run(sqlite_engine) -> None:
    _seed(sqlite_engine)
    dry = purge.purge_groups(sqlite_engine, [100], dry_run=True)
    assert dry[100]["memory_items"] == 1
    assert dry[100]["memory_item_semantic_vectors"] == 1
    assert _scalar(sqlite_engine, "SELECT COUNT(*) FROM memory_items WHERE scope_id='100'") == 1

    purge.purge_groups(sqlite_engine, [100])

    assert _scalar(sqlite_engine, "SELECT COUNT(*) FROM memory_items WHERE scope_id='100'") == 0
    assert _scalar(sqlite_engine, "SELECT COUNT(*) FROM memory_items WHERE scope_id='200'") == 1
    assert _scalar(
        sqlite_engine,
        "SELECT COUNT(*) FROM memory_item_semantic_vectors WHERE group_id=100",
    ) == 0
    assert _scalar(
        sqlite_engine,
        "SELECT COUNT(*) FROM memory_item_semantic_vectors WHERE group_id=200",
    ) == 1
    assert _scalar(sqlite_engine, "SELECT COUNT(*) FROM messages WHERE group_id=100") == 1
