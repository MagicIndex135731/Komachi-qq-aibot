from __future__ import annotations

from datetime import UTC, datetime

import scripts.cleanup_memory_noise as cleanup_module
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    UserRepository,
)


def _seed(sqlite_engine) -> dict[str, int]:
    observed_at = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
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
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="noise-source",
            group_id=100,
            user_id=20001,
            timestamp=observed_at,
            plain_text="source",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        memories = MemoryRepository(session)

        def add_memory(content: str, object_text: str) -> int:
            row = memories.add_memory(
                scope_type="group",
                scope_id="100",
                subject_type="user",
                subject_id="20001",
                memory_kind="preference",
                content=content,
                importance=1,
                confidence=0.8,
                source_msg_id="noise-source",
                valid_from=observed_at,
            )
            row.object_text = object_text
            session.flush()
            return int(row.id)

        ids = {
            "template": add_memory("阿渣 likes 坐床上看动画.", "坐床上看动画"),
            "fragment": add_memory("足泽满灰交（QQ昵称 likes 16的.", ""),
            "clean_short": add_memory("不吃香菜", "香菜"),
            "clean_normal": add_memory("用户表示自己一直看《海贼王》。", "《海贼王》"),
        }
        return ids


def test_cleanup_candidates_match_only_template_marker(sqlite_engine) -> None:
    ids = _seed(sqlite_engine)
    candidates = cleanup_module._candidate_ids(sqlite_engine, limit=None)
    assert sorted(candidates) == sorted([ids["template"], ids["fragment"]])


def test_cleanup_run_keeps_clean_short_facts(sqlite_engine) -> None:
    ids = _seed(sqlite_engine)
    candidates = cleanup_module._candidate_ids(sqlite_engine, limit=None)
    now = datetime.now(UTC)
    with session_scope(sqlite_engine) as session:
        deactivated = MemoryRepository(session).deactivate_memory_items(
            candidates,
            valid_until=now,
        )
        MemoryRepository(session).delete_memory_item_semantic_vectors(candidates)
    assert deactivated == 2
    with session_scope(sqlite_engine) as session:
        rows = MemoryRepository(session).list_active_memory_items_for_indexing(
            after_id=0,
            limit=100,
        )
        active = {int(row.id) for row in rows}
    assert ids["clean_short"] in active
    assert ids["clean_normal"] in active
    assert ids["template"] not in active
    assert ids["fragment"] not in active
