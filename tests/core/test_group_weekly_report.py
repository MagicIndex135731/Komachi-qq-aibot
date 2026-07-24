from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.group_weekly_report import (
    MAX_UNCOVERED_MESSAGES,
    MAX_SUMMARY_DOCUMENTS,
    MAX_SUMMARY_DOCUMENTS_PER_PAGE,
    build_group_weekly_report,
    build_group_weekly_report_from_evidence,
    build_group_weekly_outline,
    mask_profane_text,
)


class FakeWeeklyReportLlm:
    def __init__(self, responses: str | list[str | Exception]) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[list[str]] = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines, *, conversation_key=None):
        self.calls.append(prompt_lines)
        self.conversation_keys.append(conversation_key)
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response


class DummyMessage:
    def __init__(
        self,
        *,
        platform_msg_id: str,
        user_id: int,
        plain_text: str,
        timestamp: datetime,
        raw_json: dict | None = None,
    ) -> None:
        self.platform_msg_id = platform_msg_id
        self.user_id = user_id
        self.plain_text = plain_text
        self.timestamp = timestamp
        self.raw_json = raw_json or {}


class DummyUser:
    def __init__(self, *, user_id: int, nickname: str, group_card: str = "") -> None:
        self.user_id = user_id
        self.nickname = nickname
        self.group_card = group_card


class DummyEpisodeSummary:
    def __init__(
        self,
        *,
        document_id: int,
        content: str,
        source_msg_ids: tuple[str, ...],
        start_at: datetime,
        end_at: datetime,
    ) -> None:
        self.document_id = document_id
        self.content = content
        self.source_msg_ids = source_msg_ids
        self.start_at = start_at
        self.end_at = end_at


NOW = datetime(2026, 5, 15, tzinfo=UTC)


def make_message(
    platform_msg_id: str,
    user_id: int,
    plain_text: str,
    *,
    timestamp: datetime | None = None,
    card: str = "",
    nickname: str = "",
) -> DummyMessage:
    return DummyMessage(
        platform_msg_id=platform_msg_id,
        user_id=user_id,
        plain_text=plain_text,
        timestamp=timestamp or datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
        raw_json={"sender": {"card": card, "nickname": nickname}},
    )


def make_user(user_id: int, nickname: str, group_card: str = "") -> DummyUser:
    return DummyUser(user_id=user_id, nickname=nickname, group_card=group_card)


def make_summary(
    document_id: int,
    source_msg_ids: tuple[str, ...],
    *,
    content: str | None = None,
    timestamp: datetime | None = None,
) -> DummyEpisodeSummary:
    at = timestamp or datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
    return DummyEpisodeSummary(
        document_id=document_id,
        content=content or f"第 {document_id} 个 episode 摘要",
        source_msg_ids=source_msg_ids,
        start_at=at - timedelta(minutes=5),
        end_at=at,
    )


def joined_prompt(call: list[str]) -> str:
    return "\n".join(call)


def test_mask_profane_text_keeps_shape() -> None:
    assert mask_profane_text("你他妈真离谱") == "你他*真离谱"


def test_build_weekly_report_returns_insufficient_data_for_empty_candidates() -> None:
    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=[],
        users_by_id={},
        llm_client=object(),
        episode_summaries=[make_summary(1, ("m-1",))],
    )

    assert result.ok is False
    assert result.error_code == "insufficient_data"
    assert result.reply_text == ""


def test_early_v2_summary_and_uncovered_raw_tail_reach_selection_prompt() -> None:
    messages = [
        make_message("m-early", 1, "周初的高能发言", timestamp=NOW - timedelta(days=6)),
        make_message("m-covered", 2, "周中的普通发言", timestamp=NOW - timedelta(days=3)),
        make_message("m-tail", 3, "刚刚出现的尾部高能发言", timestamp=NOW - timedelta(minutes=1)),
    ]
    summaries = [
        make_summary(11, ("m-early",), content="周初爆发了一场重要讨论", timestamp=NOW - timedelta(days=6)),
        make_summary(12, ("m-covered",), content="周中转为平静", timestamp=NOW - timedelta(days=3)),
    ]
    llm = FakeWeeklyReportLlm(
        [
            '{"overview":"本周先激烈后平静","document_ids":[11]}',
            "m-tail",
            "1|m-early|开启了整周话题\n2|m-tail|周末尾部突发",
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={
            1: make_user(1, "早期成员"),
            2: make_user(2, "中期成员"),
            3: make_user(3, "尾部成员"),
        },
        llm_client=llm,
        episode_summaries=summaries,
    )

    assert result.ok is True
    assert "本周概况：本周先激烈后平静" in result.reply_text
    selection_prompt = joined_prompt(llm.calls[-1])
    assert "m-early" in selection_prompt
    assert "m-tail" in selection_prompt
    assert "m-covered" not in selection_prompt
    assert "周初的高能发言" in result.reply_text
    assert "刚刚出现的尾部高能发言" in result.reply_text


def test_v2_summaries_are_paginated_and_page_results_are_merged() -> None:
    document_count = MAX_SUMMARY_DOCUMENTS_PER_PAGE + 1
    messages = [
        make_message(f"m-{index}", index, f"消息 {index}")
        for index in range(document_count)
    ]
    summaries = [
        make_summary(index + 1, (f"m-{index}",), content=f"摘要 {index}")
        for index in range(document_count)
    ]
    llm = FakeWeeklyReportLlm(
        [
            '{"overview":"第一页","document_ids":[1]}',
            f'{{"overview":"第二页","document_ids":[{document_count}]}}',
            f'{{"overview":"全周合并","document_ids":[1,{document_count}]}}',
            f"1|m-{document_count - 1}|来自最后一页",
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=summaries,
    )

    assert result.ok is True
    assert len(llm.calls) == 4
    assert "document_id=1" in joined_prompt(llm.calls[0])
    assert f"document_id={document_count}" in joined_prompt(llm.calls[1])
    assert "第一页" in joined_prompt(llm.calls[2])
    assert "第二页" in joined_prompt(llm.calls[2])
    assert f"消息 {document_count - 1}" in result.reply_text
    assert "本周概况：全周合并" in result.reply_text


def test_public_outline_returns_only_whitelisted_document_ids() -> None:
    llm = FakeWeeklyReportLlm(
        '{"overview":"全周概况","document_ids":[1,999,1]}'
    )

    result = build_group_weekly_outline(
        group_id=10001,
        now=NOW,
        episode_summaries=[
            make_summary(1, ()),
            make_summary(2, ()),
        ],
        llm_client=llm,
    )

    assert result.ok is True
    assert result.overview == "全周概况"
    assert result.selected_document_ids == ("1",)


def test_v2_outline_limits_selected_episodes_to_the_raw_evidence_budget() -> None:
    messages = [
        make_message(f"m-{index}", index, f"消息 {index}")
        for index in range(8)
    ]
    summaries = [
        make_summary(index + 1, (f"m-{index}",), content=f"摘要 {index}")
        for index in range(8)
    ]
    llm = FakeWeeklyReportLlm(
        [
            '{"overview":"完整周概况","document_ids":[1,2,3,4,5,6,7,8]}',
            "1|m-6|来自可验证 episode",
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=summaries,
    )

    assert result.ok is True
    selection_prompt = joined_prompt(llm.calls[-1])
    assert "m-6" in selection_prompt
    assert "m-7" not in selection_prompt


def test_too_many_summary_pages_fail_explicitly_without_dropping_early_pages() -> None:
    messages = [make_message("m-1", 1, "真实一"), make_message("m-2", 2, "真实二")]
    summaries = [
        make_summary(index, ("m-1",), content=f"摘要 {index}")
        for index in range(MAX_SUMMARY_DOCUMENTS + 1)
    ]
    llm = FakeWeeklyReportLlm([])

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=summaries,
    )

    assert result.ok is False
    assert result.error_code == "summary_limit_exceeded"
    assert llm.calls == []


def test_selection_rejects_forged_ids_and_duplicate_rank_or_message_id() -> None:
    messages = [
        make_message("m-1", 1, "真实原话一"),
        make_message("m-2", 2, "真实原话二"),
        make_message("m-3", 3, "真实原话三"),
    ]
    llm = FakeWeeklyReportLlm(
        [
            "m-1\nm-2\nm-3",
            "\n".join(
                [
                    "1|m-1|合法理由一",
                    "1|m-2|重复 rank 应丢弃",
                    "9|forged|伪造 ID",
                    "2|m-2|合法理由",
                    "3|m-2|重复 message id 应丢弃",
                    "4|m-3|合法理由三",
                    "bad|m-1|非法 rank",
                    "5|m-1|",
                ]
            ),
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
    )

    assert result.ok is True
    assert "第1名" in result.reply_text
    assert "第2名" in result.reply_text
    assert "第4名" in result.reply_text
    assert result.reply_text.count("真实原话二") == 1
    assert "forged" not in result.reply_text


def test_model_cannot_forge_quote_or_name_and_server_applies_masking_and_limit() -> None:
    original = "你他妈" + ("真离谱" * 30)
    llm = FakeWeeklyReportLlm(
        [
            "m-1",
            "1|m-1|合法理由|假名=Mallory|假原话=模型编造\n"
            "1|m-1|火药味拉满",
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=[
            make_message("m-1", 1, original, card="群名片", nickname="消息昵称"),
            make_message("m-2", 2, "陪衬消息"),
        ],
        users_by_id={1: make_user(1, "昵称", "群名片")},
        llm_client=llm,
    )

    assert result.ok is True
    assert "第1名 群名片" in result.reply_text
    assert "Mallory" not in result.reply_text
    assert "模型编造" not in result.reply_text
    assert "原话：你他*" in result.reply_text
    assert original not in result.reply_text
    quote_line = next(line for line in result.reply_text.splitlines() if line.startswith("原话："))
    assert len(quote_line.removeprefix("原话：")) <= 80


def test_display_name_uses_message_sender_snapshot_not_global_user_card() -> None:
    llm = FakeWeeklyReportLlm(["m-1", "1|m-1|同群证据"])

    result = build_group_weekly_report_from_evidence(
        group_id=10001,
        now=NOW,
        document_message_groups=[],
        uncovered_messages=[
            make_message("m-1", 1, "同群原话", card="A群名片", nickname="A群昵称"),
            make_message("m-2", 2, "陪衬", nickname="陪衬昵称"),
        ],
        users_by_id={1: make_user(1, "全局昵称", "B群名片")},
        llm_client=llm,
    )

    assert result.ok is True
    assert "第1名 A群名片" in result.reply_text
    assert "B群名片" not in result.reply_text
    assert "全局昵称" not in result.reply_text


def test_all_52_uncovered_messages_reach_bounded_page_prompts() -> None:
    messages = [
        make_message(f"u-{index}", index, f"未覆盖消息 {index}")
        for index in range(52)
    ]
    llm = FakeWeeklyReportLlm(
        [
            "u-0\nu-49",
            "u-50\nu-51",
            "1|u-0|第一页候选",
        ]
    )

    result = build_group_weekly_report_from_evidence(
        group_id=10001,
        now=NOW,
        document_message_groups=[],
        uncovered_messages=messages,
        llm_client=llm,
    )

    assert result.ok is True
    assert len(llm.calls) == 3
    page_prompts = [joined_prompt(call) for call in llm.calls[:2]]
    prompted_ids = {
        f"u-{index}"
        for index in range(52)
        if any(f"source_msg_id=u-{index} " in prompt for prompt in page_prompts)
    }
    assert prompted_ids == {f"u-{index}" for index in range(52)}


def test_more_than_800_uncovered_messages_fails_without_model_call() -> None:
    messages = [
        make_message(f"u-{index}", index, f"未覆盖消息 {index}")
        for index in range(MAX_UNCOVERED_MESSAGES + 1)
    ]
    llm = FakeWeeklyReportLlm([])

    result = build_group_weekly_report_from_evidence(
        group_id=10001,
        now=NOW,
        document_message_groups=[],
        uncovered_messages=messages,
        llm_client=llm,
    )

    assert result.ok is False
    assert result.error_code == "uncovered_limit_exceeded"
    assert llm.calls == []


def test_document_evidence_budget_round_robins_across_all_groups() -> None:
    groups = [
        [
            make_message(
                f"d-{group_index}-{message_index}",
                group_index,
                f"文档 {group_index} 消息 {message_index}",
            )
            for message_index in range(100)
        ]
        for group_index in range(8)
    ]
    llm = FakeWeeklyReportLlm("1|d-7-0|后段文档也有预算")

    result = build_group_weekly_report_from_evidence(
        group_id=10001,
        now=NOW,
        document_message_groups=groups,
        uncovered_messages=[],
        llm_client=llm,
        overview="完整周概况",
    )

    assert result.ok is True
    final_prompt = joined_prompt(llm.calls[-1])
    for group_index in range(8):
        assert f"source_msg_id=d-{group_index}-0 " in final_prompt
    assert "文档 7 消息 0" in result.reply_text


def test_missing_v2_documents_falls_back_to_bounded_raw_id_selection() -> None:
    messages = [
        make_message(f"m-{index}", index, f"原始消息 {index}")
        for index in range(250)
    ]
    llm = FakeWeeklyReportLlm(
        ["", "", "", "", "m-249", "1|m-249|最近消息"]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=[],
    )

    assert result.ok is True
    assert len(llm.calls) == 6
    page_prompts = "\n".join(
        joined_prompt(call)
        for call in llm.calls[:-1]
    )
    assert "m-249" in page_prompts
    assert "m-0 " in page_prompts
    assert "原始消息 249" in result.reply_text
    assert "rank|source_msg_id|reason" in joined_prompt(llm.calls[-1])


def test_outline_failure_falls_back_to_raw_id_selection() -> None:
    messages = [make_message("m-1", 1, "真实一"), make_message("m-2", 2, "真实二")]
    llm = FakeWeeklyReportLlm(["not-json", "m-2", "1|m-2|降级后选中"])

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=[make_summary(1, ("m-1",), content="摘要")],
    )

    assert result.ok is True
    assert len(llm.calls) == 3
    assert "Candidate messages:" in joined_prompt(llm.calls[1])
    assert "真实二" in result.reply_text


def test_outline_provider_error_falls_back_to_raw_id_selection() -> None:
    messages = [make_message("m-1", 1, "真实一"), make_message("m-2", 2, "真实二")]
    llm = FakeWeeklyReportLlm(
        [RuntimeError("provider down"), "m-1", "1|m-1|降级后选中"]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=messages,
        users_by_id={},
        llm_client=llm,
        episode_summaries=[make_summary(1, ("m-1",), content="摘要")],
    )

    assert result.ok is True
    assert len(llm.calls) == 3
    assert "Candidate messages:" in joined_prompt(llm.calls[1])
    assert "真实一" in result.reply_text


def test_malformed_v2_provenance_falls_back_without_partial_outline() -> None:
    malformed_summary = make_summary(1, ("m-1",), content="不应局部使用")
    malformed_summary.source_msg_ids = "m-1"
    llm = FakeWeeklyReportLlm(["m-2", "1|m-2|原始消息降级"])

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=[make_message("m-1", 1, "真实一"), make_message("m-2", 2, "真实二")],
        users_by_id={},
        llm_client=llm,
        episode_summaries=[malformed_summary],
    )

    assert result.ok is True
    assert len(llm.calls) == 2
    assert "Episode summaries:" not in joined_prompt(llm.calls[0])
    assert "真实二" in result.reply_text


def test_final_reply_format_remains_compatible() -> None:
    llm = FakeWeeklyReportLlm(
        [
            "m-1\nm-2",
            "1|m-1|火药味拉满\n2|m-2|节目效果很强",
        ]
    )

    result = build_group_weekly_report(
        group_id=10001,
        now=NOW,
        messages=[
            make_message("m-1", 20001, "你他妈真离谱", nickname="Alice"),
            make_message("m-2", 20002, "这也太炸了吧", nickname="Bob"),
        ],
        users_by_id={
            20001: make_user(20001, "Alice"),
            20002: make_user(20002, "Bob"),
        },
        llm_client=llm,
    )

    assert result.ok is True
    assert "本群近一周高能雷霆发言周报" in result.reply_text
    assert "第1名 Alice" in result.reply_text
    assert "原话：你他*真离谱" in result.reply_text
    assert "上榜理由：节目效果很强" in result.reply_text
    assert llm.conversation_keys == [
        "group-weekly-report:10001:uncovered:1-of-1",
        "group-weekly-report:10001",
    ]
