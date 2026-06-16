from __future__ import annotations

from datetime import UTC, datetime

from app.core.group_weekly_report import build_group_weekly_report, mask_profane_text


class FakeWeeklyReportLlm:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[list[str]] = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines, *, conversation_key=None):
        self.calls.append(prompt_lines)
        self.conversation_keys.append(conversation_key)
        return self.response_text


class DummyMessage:
    def __init__(self, *, platform_msg_id: str, user_id: int, plain_text: str, timestamp: datetime) -> None:
        self.platform_msg_id = platform_msg_id
        self.user_id = user_id
        self.plain_text = plain_text
        self.timestamp = timestamp


class DummyUser:
    def __init__(self, *, user_id: int, nickname: str, group_card: str = "") -> None:
        self.user_id = user_id
        self.nickname = nickname
        self.group_card = group_card


def make_message(platform_msg_id: str, user_id: int, plain_text: str) -> DummyMessage:
    return DummyMessage(
        platform_msg_id=platform_msg_id,
        user_id=user_id,
        plain_text=plain_text,
        timestamp=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
    )


def make_user(user_id: int, nickname: str, group_card: str = "") -> DummyUser:
    return DummyUser(user_id=user_id, nickname=nickname, group_card=group_card)


def test_mask_profane_text_keeps_shape() -> None:
    assert mask_profane_text("你他妈真离谱") == "你他*真离谱"


def test_build_weekly_report_returns_insufficient_data_for_empty_candidates() -> None:
    result = build_group_weekly_report(
        group_id=10001,
        now=datetime(2026, 5, 15, tzinfo=UTC),
        messages=[],
        users_by_id={},
        llm_client=object(),
    )

    assert result.ok is False
    assert result.error_code == "insufficient_data"
    assert result.reply_text == ""


def test_build_weekly_report_formats_llm_output_into_sendable_text() -> None:
    llm = FakeWeeklyReportLlm(
        "1|Alice|你他*真离谱|火药味拉满\n2|Bob|这也太炸了吧|节目效果很强"
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=datetime(2026, 5, 15, tzinfo=UTC),
        messages=[make_message("m-1", 20001, "你他妈真离谱"), make_message("m-2", 20002, "这也太炸了吧")],
        users_by_id={20001: make_user(20001, "Alice"), 20002: make_user(20002, "Bob")},
        llm_client=llm,
    )

    assert result.ok is True
    assert "本群近一周高能雷霆发言周报" in result.reply_text
    assert "第1名" in result.reply_text
    assert "你他*真离谱" in result.reply_text
    assert "节目效果很强" in result.reply_text
    assert llm.conversation_keys == ["group-weekly-report:10001"]
    assert any("Candidate messages:" in line for line in llm.calls[0])
