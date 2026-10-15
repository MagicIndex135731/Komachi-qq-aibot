from __future__ import annotations

from datetime import UTC, datetime

from app.storage.db import session_scope
from app.storage.repositories import MemoryRepository


def test_group_fact_search_applies_explicit_subject_set_without_changing_unbound_search(
    sqlite_engine,
) -> None:
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        for subject_id, name in (("42", "阿渣"), ("43", "加菲猫"), ("44", "小明")):
            memories.add_memory(
                scope_type="group",
                scope_id="10001",
                subject_type="user",
                subject_id=subject_id,
                memory_kind="preference",
                content=f"{name}最喜欢动画。",
                importance=4,
                confidence=0.9,
                source_msg_id=f"source-{subject_id}",
                valid_from=observed_at,
            )

        bound = memories.search_group_memories_fts(
            scope_id="10001",
            query="最喜欢什么动画？",
            limit=10,
            as_of=observed_at,
            subject_ids=("42", "43"),
        )
        empty_binding = memories.search_group_memories_fts(
            scope_id="10001",
            query="最喜欢什么动画？",
            limit=10,
            as_of=observed_at,
            subject_ids=(),
        )
        unbound = memories.search_group_memories_fts(
            scope_id="10001",
            query="最喜欢什么动画？",
            limit=10,
            as_of=observed_at,
            subject_ids=None,
        )

    assert {memory.subject_id for memory in bound} == {"42", "43"}
    assert empty_binding == []
    assert {memory.subject_id for memory in unbound} == {"42", "43", "44"}
