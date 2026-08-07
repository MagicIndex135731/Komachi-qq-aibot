from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.storage.db import session_scope
from app.storage.models import MemoryItem, Message, RetrievalDocument
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)
from scripts.repair_memory_correction import repair_memory_correction


def test_repair_memory_correction_is_idempotent_and_preserves_evidence(sqlite_engine) -> None:
    corrected_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=100000001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        users = UserRepository(session)
        users.upsert_user(user_id=200000001, nickname="Reporter", group_card="")
        users.upsert_user(user_id=200000002, nickname="A-Zha", group_card="小明")
        messages = MessageRepository(session)
        target_seed = messages.add_group_message(
            platform_msg_id="target-seed",
            group_id=100000001,
            user_id=200000002,
            timestamp=corrected_at - timedelta(days=2),
            plain_text="普通群聊。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "小明"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        wrong_source = messages.add_group_message(
            platform_msg_id="900000001",
            group_id=100000001,
            user_id=200000001,
            timestamp=corrected_at - timedelta(days=1),
            plain_text="小明最喜欢坐床上看动画片。",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="900000002",
            group_id=100000001,
            user_id=200000001,
            timestamp=corrected_at,
            plain_text="你记错了，是小明喜欢坐床上看动画。",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        old = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100000001",
            subject_type="user",
            subject_id="200000001",
            memory_kind="preference",
            content="Reporter likes 坐床上看动画片.",
            importance=4,
            confidence=0.8,
            source_msg_id="900000001",
            valid_from=corrected_at - timedelta(days=1),
        )
        unrelated_plan = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100000001",
            subject_type="user",
            subject_id="200000001",
            memory_kind="plan",
            content="Reporter plans to watch animation.",
            importance=3,
            confidence=0.8,
            source_msg_id="900000001",
            valid_from=corrected_at - timedelta(days=1),
        )
        unrelated_preference = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100000001",
            subject_type="user",
            subject_id="200000001",
            memory_kind="preference",
            content="Reporter likes hotpot.",
            importance=4,
            confidence=0.8,
            source_msg_id="900000001",
            valid_from=corrected_at - timedelta(days=1),
        )
        document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="100000001",
            group_id=100000001,
            episode_id=None,
            document_kind="memory",
            source_table="memory_items",
            source_id=str(old.id),
            start_at=corrected_at - timedelta(days=1),
            end_at=corrected_at - timedelta(days=1),
            content=old.content,
            metadata_json={},
            content_hash="old-memory-document",
            source_message_ids=[wrong_source.id],
            embedding_status="ready",
        )
        episode_document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="100000001",
            group_id=100000001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="historical-episode",
            start_at=corrected_at - timedelta(days=1),
            end_at=corrected_at - timedelta(days=1),
            content="小明最喜欢坐床上看动画片。",
            metadata_json={},
            content_hash="historical-episode-document",
            source_message_ids=[wrong_source.id],
            embedding_status="ready",
        )
        old_id = old.id
        unrelated_plan_id = unrelated_plan.id
        unrelated_preference_id = unrelated_preference.id
        document_id = document.id
        episode_document_id = episode_document.id
        assert target_seed.id is not None

    kwargs = {
        "group_id": 100000001,
        "subject_id": 200000002,
        "subject_alias": "小明",
        "predicate": "likes",
        "object_text": "坐床上看动画",
        "supporting_source_ids": ("900000001", "900000002"),
        "erroneous_source_ids": ("900000001",),
    }
    first = repair_memory_correction(sqlite_engine, **kwargs)
    second = repair_memory_correction(sqlite_engine, **kwargs)

    with session_scope(sqlite_engine) as session:
        facts = list(session.scalars(select(MemoryItem).order_by(MemoryItem.id)))
        old = session.get(MemoryItem, old_id)
        replacement = session.get(MemoryItem, first.replacement_memory_id)
        unrelated_plan = session.get(MemoryItem, unrelated_plan_id)
        unrelated_preference = session.get(MemoryItem, unrelated_preference_id)
        document = session.get(RetrievalDocument, document_id)
        episode_document = session.get(RetrievalDocument, episode_document_id)
        message_count = session.scalar(select(func.count(Message.id)))
        source_rows = {
            message.platform_msg_id: (message.plain_text, message.raw_json)
            for message in session.scalars(select(Message).order_by(Message.id))
        }

    assert first.superseded_count == 1
    assert first.already_superseded_count == 0
    assert second.replacement_memory_id == first.replacement_memory_id
    assert second.superseded_count == 0
    assert second.already_superseded_count == 1
    assert len(facts) == 4
    assert old is not None and old.status == "superseded"
    assert old.superseded_by_id == replacement.id
    assert replacement is not None and replacement.status == "active"
    assert replacement.subject_id == "200000002"
    assert replacement.source_msg_ids == ["900000001", "900000002"]
    assert unrelated_plan is not None and unrelated_plan.status == "active"
    assert unrelated_preference is not None and unrelated_preference.status == "active"
    assert document is not None and document.status == "inactive"
    assert document.embedding_status == "stale"
    assert episode_document is not None and episode_document.status == "active"
    assert episode_document.embedding_status == "ready"
    assert message_count == 3
    assert source_rows == {
        "target-seed": ("普通群聊。", {"sender": {"nickname": "A-Zha", "card": "小明"}}),
        "900000001": ("小明最喜欢坐床上看动画片。", {}),
        "900000002": ("你记错了，是小明喜欢坐床上看动画。", {}),
    }

    with pytest.raises(ValueError, match="does not support the requested fact"):
        repair_memory_correction(sqlite_engine, **{**kwargs, "object_text": "火锅"})

    with session_scope(sqlite_engine) as session:
        assert session.scalar(select(func.count(MemoryItem.id))) == 4
