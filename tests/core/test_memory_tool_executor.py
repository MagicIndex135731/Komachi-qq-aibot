from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.memory_tool_executor import MemoryToolExecutor, _relative_day_range
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    SummaryRepository,
    UserRepository,
)


def _seed(engine) -> None:
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(engine) as session:
        for group_id in (100, 200):
            GroupRepository(session).upsert_group(
                group_id=group_id,
                group_name=f"group-{group_id}",
                enabled=True,
                speak_enabled=True,
            )
        users = UserRepository(session)
        users.upsert_user(user_id=20001, nickname="A-Zha", group_card="阿渣")
        users.upsert_user(user_id=99, nickname="Questioner", group_card="提问者")
        users.upsert_user(user_id=20002, nickname="Other", group_card="别人")
        messages = MessageRepository(session)
        fact_source = messages.add_group_message(
            platform_msg_id="tool-source",
            group_id=100,
            user_id=20001,
            timestamp=observed_at,
            plain_text="阿渣喜欢喝冰美式",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        query = messages.add_group_message(
            platform_msg_id="tool-query",
            group_id=100,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=1),
            plain_text="阿渣以前说过喜欢喝什么？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        cross = messages.add_group_message(
            platform_msg_id="tool-cross-200",
            group_id=200,
            user_id=20002,
            timestamp=observed_at,
            plain_text="别的群的内容",
            raw_json={"sender": {"nickname": "Other", "card": "别人"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        documents.project_raw_message_v3(
            group_id=100,
            message_id=int(fact_source.id),
        )
        documents.project_raw_message_v3(
            group_id=100,
            message_id=int(query.id),
        )
        documents.project_raw_message_v3(
            group_id=200,
            message_id=int(cross.id),
        )
        memory = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣喜欢喝冰美式",
            importance=4,
            confidence=0.9,
            source_msg_id="tool-source",
            valid_from=observed_at,
        )
        session.flush()
        documents.upsert_document(
            scope_type="group",
            scope_id="100",
            group_id=100,
            episode_id=None,
            document_kind="memory",
            source_table="memory_items",
            source_id=str(memory.id),
            start_at=observed_at,
            end_at=observed_at,
            content="阿渣喜欢喝冰美式",
            metadata_json={"subject_id": "20001", "kind": "preference"},
            content_hash="hash-tool-memory",
            source_message_ids=[int(fact_source.id)],
        )
        SummaryRepository(session).upsert_summary(
            scope_type="group",
            scope_id="100",
            summary_level="episode",
            summary_key="episode:tool:test",
            start_at=observed_at,
            end_at=observed_at + timedelta(hours=1),
            content="阿渣动画偏好总结",
            source_count=1,
            source_start_msg_id="tool-source",
            source_end_msg_id="tool-source",
        )


def _executor(engine, *, recent_source_ids=("tool-source", "tool-query")):
    return MemoryToolExecutor(
        engine=engine,
        group_id=100,
        current_user_id=99,
        now=datetime(2026, 7, 23, 12, 0, tzinfo=UTC),
        recent_source_msg_ids=recent_source_ids,
        member_names={"阿渣": 20001, "提问者": 99, "20001": 20001, "99": 99},
        timeout_seconds=2.0,
        max_results=5,
    )


def test_memory_search_facts_returns_scoped_fact(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_search",
        {"query": "冰美式", "layer": "facts"},
    )
    assert "阿渣喜欢喝冰美式" in output
    assert "tool-source" in output


def test_memory_search_raw_returns_raw_message(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_search",
        {"query": "冰美式", "layer": "raw"},
    )
    assert "阿渣喜欢喝冰美式" in output
    assert "tool-source" in output


def test_memory_search_summaries_returns_summary(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_search",
        {"query": "偏好", "layer": "summaries"},
    )
    assert "阿渣动画偏好总结" in output


def test_memory_search_member_filter_and_unresolved(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_search",
        {"query": "冰美式", "layer": "facts", "member": "阿渣"},
    )
    assert "阿渣喜欢喝冰美式" in output
    unresolved = _executor(sqlite_engine).execute(
        "memory_search",
        {"query": "冰美式", "member": "不存在的人"},
    )
    assert unresolved == '{"error":"member_unresolved"}'


def test_memory_search_empty_member_string_is_unrestricted(sqlite_engine) -> None:
    _seed(sqlite_engine)
    for member in ("", "   "):
        output = _executor(sqlite_engine).execute(
            "memory_search",
            {"query": "冰美式", "layer": "facts", "member": member},
        )
        assert "阿渣喜欢喝冰美式" in output
        assert "member_unresolved" not in output


def test_memory_read_returns_profile_and_recent_count(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute("memory_read", {"member": "阿渣"})
    assert "阿渣喜欢喝冰美式" in output
    assert "recent messages: 1" in output


def test_memory_read_returns_a_diverse_stable_portrait(sqlite_engine) -> None:
    _seed(sqlite_engine)
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        for index, (kind, content) in enumerate(
            (
                ("profile", "阿渣长期从事软件开发"),
                ("fact", "阿渣养了一只猫"),
                ("taboo", "阿渣不接受剧透"),
                ("relationship", "阿渣是提问者的朋友"),
                ("current", "阿渣最近在看某部动画"),
            ),
            start=1,
        ):
            memories.add_memory(
                scope_type="group",
                scope_id="100",
                subject_type="user",
                subject_id="20001",
                memory_kind=kind,
                content=content,
                importance=5 if kind == "current" else 2,
                confidence=0.9,
                source_msg_id="tool-source",
                valid_from=observed_at + timedelta(seconds=index),
            )

    output = _executor(sqlite_engine).execute("memory_read", {"member": "阿渣"})

    for kind in ("profile", "fact", "preference", "taboo", "relationship"):
        assert f"{kind} (source:" in output
    assert "current (source:" not in output


def test_memory_write_persists_source_backed_fact(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_write",
        {
            "kind": "preference",
            "subject": "99",
            "predicate": "likes",
            "object_text": "冰美式",
            "content": "提问者喜欢喝冰美式",
            "source_msg_ids": ["tool-query"],
        },
    )
    assert output.startswith('{"memory_id":')
    with session_scope(sqlite_engine) as session:
        rows = MemoryRepository(session).list_group_memories_for_subject(
            scope_id="100",
            subject_id="99",
            limit=10,
        )
    assert any("提问者喜欢喝冰美式" in row.content for row in rows)


def test_memory_write_accepts_source_backed_self_relationship(sqlite_engine) -> None:
    _seed(sqlite_engine)
    output = _executor(sqlite_engine).execute(
        "memory_write",
        {
            "kind": "relationship",
            "subject": "99",
            "predicate": "identity relation",
            "object_text": "bot owner",
            "content": "提问者是机器人的主人",
            "source_msg_ids": ["tool-query"],
        },
    )

    assert output.startswith('{"memory_id":')
    with session_scope(sqlite_engine) as session:
        rows = MemoryRepository(session).list_group_memories_for_subject(
            scope_id="100",
            subject_id="99",
            limit=10,
        )
    assert any(row.memory_kind == "relationship" for row in rows)


def test_memory_write_rejects_scope_and_source_violations(sqlite_engine) -> None:
    _seed(sqlite_engine)
    executor = _executor(
        sqlite_engine,
        recent_source_ids=("tool-query", "tool-cross-200"),
    )
    base = {
        "kind": "preference",
        "subject": "99",
        "predicate": "likes",
        "object_text": "冰美式",
        "content": "提问者喜欢喝冰美式",
        "source_msg_ids": ["tool-query"],
    }
    assert executor.execute("memory_write", {**base, "subject": "20001"}) == (
        '{"error":"subject_out_of_scope"}'
    )
    assert executor.execute(
        "memory_write",
        {**base, "source_msg_ids": ["tool-missing"]},
    ) == '{"error":"source_not_in_conversation"}'
    assert executor.execute(
        "memory_write",
        {**base, "source_msg_ids": ["tool-cross-200"]},
    ) == '{"error":"source_not_in_group"}'
    assert executor.execute("memory_write", {**base, "kind": "expired"}) == (
        '{"error":"kind is not in the allowed set"}'
    )
    assert executor.execute("unknown_tool", {}) == '{"error":"unknown_tool"}'


def test_relative_day_range_resolves_shanghai_days() -> None:
    now = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)  # 2026-08-10 20:00 +08

    yesterday = _relative_day_range("昨天我说了什么", now)
    assert yesterday is not None
    assert yesterday[0] == datetime(2026, 8, 8, 16, 0, tzinfo=UTC)
    assert yesterday[1] == datetime(2026, 8, 9, 16, 0, tzinfo=UTC)

    today = _relative_day_range("今天群聊内容", now)
    assert today is not None
    assert today[0] == datetime(2026, 8, 9, 16, 0, tzinfo=UTC)
    assert today[1] == datetime(2026, 8, 10, 16, 0, tzinfo=UTC)

    assert _relative_day_range("没有时间词", now) is None
