from __future__ import annotations

from datetime import UTC, datetime

from app.config import AppSettings
from app.core.legacy_memory_context import LegacyMemoryContext, LegacyMemoryRequest
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    SummaryRepository,
    UserRepository,
)


def _settings() -> AppSettings:
    return AppSettings.model_construct(
        context_recent_limit=2,
        context_summary_limit=3,
        context_history_limit=8,
        bot_qq=123456789,
    )


def _request(*, group_id: int = 10001, use_full_history: bool = False) -> LegacyMemoryRequest:
    return LegacyMemoryRequest(
        group_id=group_id,
        query="送外卖去了 加班",
        recent_messages=(),
        quoted_message=None,
        target_message_id="target-message",
        available_input=10_000,
        now=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        current_user_id=20001,
        use_full_history=use_full_history,
    )


def test_legacy_context_preserves_v1_recent_history_summary_memory_and_member_focus(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        summaries = SummaryRepository(session)
        memories = MemoryRepository(session)
        groups.upsert_group(group_id=10001, group_name="group", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=10002, nickname="熟人A", group_card="送外卖去了")
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="older-work",
            group_id=10001,
            user_id=10002,
            timestamp=datetime(2026, 5, 9, 10, 0, tzinfo=UTC),
            plain_text="今天又加班，刚送完最后一单。",
            raw_json={"sender": {"nickname": "熟人A", "card": "送外卖去了"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        for index in range(2):
            messages.add_group_message(
                platform_msg_id=f"recent-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 11, index, tzinfo=UTC),
                plain_text=f"最近闲聊 {index}",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        summaries.upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="daily",
            summary_key="daily:2026-05-09",
            start_at=datetime(2026, 5, 9, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 5, 9, 11, 0, tzinfo=UTC),
            content="送外卖去了最近工作很忙。",
            source_count=1,
            source_start_msg_id="older-work",
            source_end_msg_id="older-work",
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="10002",
            memory_kind="fact",
            content="10002 最近经常加班。",
            importance=8,
            confidence=0.9,
            source_msg_id="older-work",
        )

    result = LegacyMemoryContext(
        engine=sqlite_engine,
        settings=_settings(),
        bot_user_id=123456789,
        bot_display_name="Mira",
    ).build_context(_request())
    context = result.packed_context

    assert result.mode == "v1"
    assert context.recent_messages == ["Alice: 最近闲聊 0", "Alice: 最近闲聊 1"]
    assert any("今天又加班，刚送完最后一单。" in line for line in context.relevant_history_messages)
    assert any("送外卖去了最近工作很忙。" in line for line in context.summaries)
    assert any("10002 最近经常加班。" in line for line in context.memories)
    assert context.member_focus_lines[0] == "Referenced member: 送外卖去了（QQ昵称：熟人A）"
    assert "older-work" in result.selected_source_msg_ids


def test_legacy_context_full_history_is_group_scoped_and_reports_blocked_output(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        for group_id in (10001, 20002):
            groups.upsert_group(group_id=group_id, group_name=str(group_id), enabled=True, speak_enabled=True)
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="blocked-local",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 10, 0, tzinfo=UTC),
            plain_text="blocked source text",
            raw_json={
                "direction": "outbound",
                "delivery_state": "blocked",
                "failure_kind": "qq_sensitive_content",
            },
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="other-group-secret",
            group_id=20002,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 10, 0, tzinfo=UTC),
            plain_text="must never cross groups",
            raw_json={"direction": "outbound", "delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    result = LegacyMemoryContext(
        engine=sqlite_engine,
        settings=_settings(),
        bot_user_id=123456789,
        bot_display_name="Mira",
    ).build_context(_request(use_full_history=True))
    context = result.packed_context

    assert context.blocked_output_present is True
    assert any("blocked source text" in line for line in context.full_history_messages)
    assert all("must never cross groups" not in line for line in context.full_history_messages)
    assert "other-group-secret" not in result.selected_source_msg_ids
