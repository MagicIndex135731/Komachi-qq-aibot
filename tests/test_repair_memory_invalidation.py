from datetime import UTC, datetime
import hashlib
from pathlib import Path

from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    UserRepository,
)
from scripts.repair_memory_invalidation import repair_memory_invalidation


def test_manual_review_invalidation_is_dry_run_by_default_and_exact(sqlite_engine) -> None:
    reviewed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    target_key = "profile|42|nationality|wrong"
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001, group_name="test", enabled=True, speak_enabled=True
        )
        UserRepository(session).upsert_user(user_id=42, nickname="Alice", group_card="")
        UserRepository(session).upsert_user(user_id=43, nickname="Reviewer", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id="review-source",
            group_id=10001,
            user_id=43,
            timestamp=reviewed_at,
            plain_text="这条旧画像不对。",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )
        target = MemoryRepository(session).upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="profile",
            canonical_key=target_key,
            predicate="nationality",
            object_text="wrong",
            content="Wrong legacy profile.",
            importance=4,
            confidence=0.9,
            source_msg_ids=["old-source"],
        )
        target_id = target.id

    arguments = {
        "database": Path(sqlite_engine.url.database),
        "group_id": 10001,
        "target_memory_id": target_id,
        "expected_target_sha256": hashlib.sha256(target_key.encode()).hexdigest(),
        "source_msg_ids": ["review-source"],
        "reason": "manual_review_rejected",
    }
    preview = repair_memory_invalidation(**arguments, apply=False)
    with session_scope(sqlite_engine) as session:
        assert session.get(type(target), target_id).status == "active"

    applied = repair_memory_invalidation(**arguments, apply=True)
    with session_scope(sqlite_engine) as session:
        repaired = session.get(type(target), target_id)

    assert preview.applied is False
    assert preview.receipt_memory_id is None
    assert applied.applied is True
    assert applied.receipt_memory_id is not None
    assert repaired.status == "superseded"
