"""Profile correction regression: an old nationality/portrait value must be
retired when the same member supplies a corrected value.

Covers both the explicit-denial exact invalidation path (日本人 -> 广东人 with
an explicit denial) and the same-slot single-value replacement path.
"""

from datetime import UTC, datetime

from app.core.memory_compaction import (
    derive_explicit_memory_invalidations,
    single_value_profile_attribute_predicates,
)
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    UserRepository,
)


def test_nationality_correction_with_explicit_denial_retires_old_value(
    sqlite_engine,
) -> None:
    observed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(user_id=42, nickname="Maple", group_card="")
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="old-nationality",
            group_id=10001,
            user_id=42,
            timestamp=observed_at,
            plain_text="Maple 是日本人",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="corrects",
            group_id=10001,
            user_id=42,
            timestamp=observed_at,
            plain_text="其实我不是日本人，我是广东人",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )
        memories = MemoryRepository(session)
        old = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="profile",
            canonical_key="profile|42|nationality|日本人",
            predicate="nationality",
            object_text="日本人",
            content="Maple 是日本人",
            importance=4,
            confidence=0.9,
            source_msg_ids=["old-nationality"],
        )
        correction_targets = (
            {
                "target_canonical_key": "profile|42|nationality|日本人",
                "memory_kind": "profile",
                "subject_id": "42",
                "predicate": "nationality",
                "object_text": "日本人",
            },
        )
        invalidations = derive_explicit_memory_invalidations(
            messages=(
                {
                    "source_msg_id": "corrects",
                    "user_id": "42",
                    "plain_text": "其实我不是日本人，我是广东人",
                },
            ),
            active_correction_targets=correction_targets,
        )
        assert invalidations
        assert invalidations[0].target_canonical_key == "profile|42|nationality|日本人"
        assert invalidations[0].source_msg_ids == ("corrects",)
        memories.invalidate_canonical_memory(
            scope_id="10001",
            target_canonical_key="profile|42|nationality|日本人",
            source_msg_ids=["corrects"],
            valid_until=observed_at,
        )
        replacement = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="profile",
            canonical_key="profile|42|origin|广东人",
            predicate="origin",
            object_text="广东人",
            content="Maple 是广东人",
            importance=4,
            confidence=0.9,
            source_msg_ids=["corrects"],
            replace_previous=True,
            replacement_predicates=single_value_profile_attribute_predicates("origin"),
        )
        japanese = memories.search_group_memories_fts(
            scope_id="10001",
            query="日本人",
            limit=5,
            subject_ids=("42",),
        )
        cantonese = memories.search_group_memories_fts(
            scope_id="10001",
            query="广东人",
            limit=5,
            subject_ids=("42",),
        )

    assert old.status == "superseded"
    assert replacement.id is not None
    assert [memory.id for memory in japanese] == []
    assert [memory.id for memory in cantonese] == [replacement.id]


def test_same_slot_nationality_replacement_supersedes_old_value(sqlite_engine) -> None:
    observed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(user_id=42, nickname="Maple", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id="old-nationality",
            group_id=10001,
            user_id=42,
            timestamp=observed_at,
            plain_text="Maple 是日本人",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        memories = MemoryRepository(session)
        old = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="profile",
            canonical_key="profile|42|nationality|日本人",
            predicate="nationality",
            object_text="日本人",
            content="Maple 是日本人",
            importance=4,
            confidence=0.9,
            source_msg_ids=["old-nationality"],
        )
        replacement = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="profile",
            canonical_key="profile|42|nationality|广东人",
            predicate="nationality",
            object_text="广东人",
            content="Maple 是广东人",
            importance=4,
            confidence=0.9,
            source_msg_ids=["old-nationality"],
            replace_previous=True,
            replacement_predicates=("nationality", "国籍"),
        )

    assert old.status == "superseded"
    assert replacement.id is not None
