from datetime import UTC, datetime

import pytest

from app.core.memory_contracts import MemoryObservationRequest
from app.core.memory_observation import MemoryObservationReader
from app.storage.db import session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)


def _seed(engine):
    with session_scope(engine) as session:
        for group in (100, 200):
            GroupRepository(session).upsert_group(
                group_id=group, group_name="test", enabled=True, speak_enabled=True
            )
        for user in (11, 22):
            UserRepository(session).upsert_user(
                user_id=user, nickname="test", group_card=""
            )
        for source, group, user, content in (
            ("history", 100, 11, "我一直在等待游戏延期"),
            ("first", 100, 11, "游戏又延期了"),
            ("second", 100, 22, "我也在等游戏"),
            ("cross", 200, 11, "游戏跨群内容"),
        ):
            row = MessageRepository(session).add_group_message(
                platform_msg_id=source,
                group_id=group,
                user_id=user,
                timestamp=datetime(2026, 10, 9, tzinfo=UTC),
                plain_text=content,
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            session.flush()
            RetrievalDocumentRepository(session).project_raw_message_v3(
                group_id=group, message_id=row.id
            )
        MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="11",
            memory_kind="preference",
            content="游戏是他的长期关注话题",
            importance=3,
            confidence=1.0,
            source_msg_id="history",
        )


def test_observation_keeps_batch_authors_and_does_not_write(sqlite_engine):
    _seed(sqlite_engine)
    request = MemoryObservationRequest(
        group_id=100, topic_query="游戏", source_message_ids=("first", "second")
    )
    result = MemoryObservationReader(engine=sqlite_engine).read(request)
    assert result.group_id == 100
    assert result.facts[0].subject_id == "11"
    assert any(
        item.source_msg_id == "history" and item.user_id == 11
        for item in result.evidence
    )
    assert not any(
        item.source_msg_id in {"cross", "first", "second"} for item in result.evidence
    )
    assert "author=11" in "\n".join(result.prompt_lines())
    with session_scope(sqlite_engine) as session:
        assert session.query(MemoryItem).count() == 1


def test_observation_rejects_cross_group_batch(sqlite_engine):
    _seed(sqlite_engine)
    with pytest.raises(ValueError, match="observation_source_out_of_scope"):
        MemoryObservationReader(engine=sqlite_engine).read(
            MemoryObservationRequest(
                group_id=100, topic_query="游戏", source_message_ids=("first", "cross")
            )
        )


def test_observation_revalidates_recalled_fact_sources(sqlite_engine):
    _seed(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(
            group_id=100, platform_msg_id="history"
        )
    result = MemoryObservationReader(engine=sqlite_engine).read(
        MemoryObservationRequest(
            group_id=100, topic_query="游戏", source_message_ids=("first", "second")
        )
    )
    assert result.evidence == ()
    assert result.facts == ()


def test_observation_is_bounded_and_subject_focus_is_explicit(sqlite_engine):
    _seed(sqlite_engine)
    result = MemoryObservationReader(engine=sqlite_engine).read(
        MemoryObservationRequest(
            group_id=100,
            topic_query="游戏",
            source_message_ids=("first", "second"),
            focus_subject_ids=(22,),
            context_char_budget=100,
        )
    )
    assert result.facts == ()
    assert all(item.user_id == 22 for item in result.evidence)
    assert len("\n".join(result.prompt_lines())) <= 100


def test_observation_recalls_topic_at_the_end_of_a_long_chinese_batch(sqlite_engine):
    _seed(sqlite_engine)
    query = "今天下午我们大家凑在一起聊了很久也说了很多其它事情现在又聊到那个游戏延期了这个游戏还要继续等多久"
    result = MemoryObservationReader(engine=sqlite_engine).read(
        MemoryObservationRequest(
            group_id=100, topic_query=query, source_message_ids=("first", "second")
        )
    )
    assert any(item.source_msg_id == "history" for item in result.evidence)
    assert any(item.subject_id == "11" for item in result.facts)
