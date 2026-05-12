from datetime import datetime

from app.core.search_policy import (
    build_forced_search_query,
    build_current_datetime_facts,
    build_search_decision_prompt,
    detect_address_intent,
    is_general_search_decision_candidate,
    is_explicit_search_request,
    is_search_verification_query,
    needs_external_lookup_search,
    needs_reference_search,
    is_time_sensitive_request,
    needs_current_datetime_context,
    normalize_relative_time_query,
    parse_search_decision,
)


def test_detect_address_intent_matches_explicit_name() -> None:
    decision = detect_address_intent(
        text="\u6bd4\u4f01\u8c37\u5c0f\u753a\u4f60\u600e\u4e48\u770b\u8fd9\u4ef6\u4e8b",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
        reply_to_bot=False,
        quoted_bot=False,
        bot_recently_participated=False,
        recent_bot_message_count=0,
    )

    assert decision.is_addressed is True
    assert decision.reason == "named_bot"
    assert decision.score == 10


def test_detect_address_intent_matches_recent_followup() -> None:
    decision = detect_address_intent(
        text="\u90a3\u4f60\u600e\u4e48\u770b",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
        reply_to_bot=False,
        quoted_bot=False,
        bot_recently_participated=True,
        recent_bot_message_count=1,
    )

    assert decision.is_addressed is True
    assert decision.reason == "recent_followup"
    assert decision.score == 7


def test_detect_address_intent_rejects_stray_second_person_pronoun() -> None:
    decision = detect_address_intent(
        text="\u4f60\u89c9\u5f97\u8fd9\u4e2a\u89d2\u8272\u600e\u4e48\u6837",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
        reply_to_bot=False,
        quoted_bot=False,
        bot_recently_participated=False,
        recent_bot_message_count=0,
    )

    assert decision.is_addressed is False
    assert decision.reason == "not_addressed"
    assert decision.score == 0


def test_detect_address_intent_matches_short_alias_name() -> None:
    decision = detect_address_intent(
        text="\u5c0f\u753a\uff0c\u4f60\u597d\u53ef\u7231",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
        reply_to_bot=False,
        quoted_bot=False,
        bot_recently_participated=False,
        recent_bot_message_count=0,
    )

    assert decision.is_addressed is True
    assert decision.reason == "named_bot"
    assert decision.score == 10


def test_detect_address_intent_ignores_meta_discussion_about_bot() -> None:
    decision = detect_address_intent(
        text="\u5c0f\u753a\u600e\u4e48\u8fd9\u4e48\u5361",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
        reply_to_bot=False,
        quoted_bot=False,
        bot_recently_participated=False,
        recent_bot_message_count=0,
    )

    assert decision.is_addressed is False
    assert decision.reason == "bot_meta_discussion"
    assert decision.score == 0


def test_time_sensitive_request_matches_recent_anime_evaluation() -> None:
    assert is_time_sensitive_request("\u6700\u8fd1\u65b0\u756a\u53e3\u7891\u600e\u4e48\u6837") is True
    assert is_time_sensitive_request("\u4f60\u559c\u6b22\u4ec0\u4e48\u7c7b\u578b\u52a8\u753b") is False
    assert is_time_sensitive_request("\u4f60\u4eca\u5929\u600e\u4e48\u770b\u8fd9\u4e2a\u89d2\u8272") is False
    assert is_time_sensitive_request("\u4eca\u5929\u6700\u65b0\u65b0\u95fb\u600e\u4e48\u6837") is True


def test_explicit_search_request_matches_search_language_only() -> None:
    assert is_explicit_search_request("\u5c0f\u753a\uff0c\u5e2e\u6211\u8054\u7f51\u641c\u4e00\u4e0baqua") is True
    assert is_explicit_search_request("\u5c0f\u753a\uff0c\u6211\u7231\u4f60") is False
    assert is_explicit_search_request("\u4f60\u771f\u7684\u4e0a\u7f51\u641c\u4e86\u5417") is False


def test_search_verification_query_matches_meta_followup_without_triggering_new_search() -> None:
    assert is_search_verification_query("\u4f60\u771f\u7684\u4e0a\u7f51\u641c\u4e86\u5417") is True
    assert is_search_verification_query("\u4f60\u521a\u521a\u67e5\u4e86\u5417") is True
    assert is_search_verification_query("\u5e2e\u6211\u4e0a\u7f51\u641c\u4e00\u4e0b\u4eca\u5929\u897f\u5b89\u5929\u6c14") is False


def test_reference_search_matches_media_and_internet_topics() -> None:
    assert (
        needs_reference_search(
            "\u5ba2\u89c2\u8bc4\u4ef7\u4e00\u4e0b\u4e0a\u4f0a\u90a3\u7261\u4e39\uff0c\u9189\u59ff\u5982\u767e\u5408\u5230\u5e95\u662f\u4e0d\u662f\u4e00\u90e8\u597d\u4f5c\u54c1\uff0c\u5206\u6790\u4e3a\u4ec0\u4e48\u5f88\u591a\u4eba\u9a82\u5b83"
        )
        is True
    )
    assert needs_reference_search("\u8fd9\u90e8\u8001\u52a8\u753b\u5230\u5e95\u503c\u4e0d\u503c\u5f97\u770b") is True
    assert needs_reference_search("\u8fd9\u4e2a\u4e8b\u4ef6\u7f51\u4e0a\u600e\u4e48\u8bf4") is True
    assert needs_reference_search("\u5c0f\u753a\u4f60\u559c\u6b22\u4ec0\u4e48\u7c7b\u578b\u7684\u52a8\u753b") is False
    assert needs_reference_search("\u8bc4\u4ef7\u4e00\u4e0b\u7fa4\u91cc\u7684\u963f\u798f") is False


def test_external_lookup_search_matches_real_world_recommendation_requests() -> None:
    assert needs_external_lookup_search("\u897f\u5b89\u7535\u5b50\u79d1\u6280\u5927\u5b66\u5357\u6821\u533a\u90a3\u5bb6\u5e97\u6700\u597d\u5403") is True
    assert needs_external_lookup_search("\u897f\u7535\u5357\u6821\u533a\u9644\u8fd1\u6709\u4ec0\u4e48\u63a8\u8350\u7684\u70e7\u70e4") is True
    assert needs_external_lookup_search("\u8bc4\u4ef7\u4e00\u4e0b\u7fa4\u91cc\u7684\u963f\u798f") is False
    assert needs_external_lookup_search("\u6211\u7231\u4f60") is False


def test_general_search_decision_candidate_covers_information_requests_but_not_smalltalk() -> None:
    assert is_general_search_decision_candidate("\u6c34\u7684\u6cb8\u70b9\u662f\u591a\u5c11") is True
    assert is_general_search_decision_candidate("\u897f\u7535\u5357\u6821\u533a\u54ea\u5bb6\u5e97\u597d\u5403") is True
    assert is_general_search_decision_candidate("\u4f60\u5728\u5417") is False
    assert is_general_search_decision_candidate("\u6211\u7231\u4f60") is False


def test_build_forced_search_query_strips_bot_addressing_noise() -> None:
    query = build_forced_search_query(
        "@\u6bd4\u4f01\u8c37\u5c0f\u753a \u91cd\u65b0\u5ba2\u89c2\u8bc4\u4ef7\u4e00\u4e0b\u4e0a\u4f0a\u90a3\u7261\u4e39\uff0c\u9189\u59ff\u5982\u767e\u5408\u5230\u5e95\u662f\u4e0d\u662f\u4e00\u90e8\u597d\u4f5c\u54c1\uff0c\u4ece\u591a\u65b9\u9762\u8bc4\u4ef7\uff0c\u5206\u6790\u4e3a\u4ec0\u4e48\u5f88\u591a\u4eba\u9a82\u5b83",
        bot_names={"\u5c0f\u753a", "\u6bd4\u4f01\u8c37\u5c0f\u753a"},
    )

    assert "\u4e0a\u4f0a\u90a3\u7261\u4e39" in query
    assert "\u9189\u59ff\u5982\u767e\u5408" in query
    assert "@" not in query
    assert "\u6bd4\u4f01\u8c37\u5c0f\u753a" not in query
    assert "\u91cd\u65b0" not in query


def test_current_datetime_context_matches_date_and_time_questions() -> None:
    assert needs_current_datetime_context("\u4eca\u5929\u51e0\u53f7") is True
    assert needs_current_datetime_context("\u4eca\u5929\u661f\u671f\u51e0") is True
    assert needs_current_datetime_context("\u73b0\u5728\u51e0\u70b9") is True
    assert needs_current_datetime_context("\u5f53\u524d\u65e5\u671f\u662f\u4ec0\u4e48") is True
    assert needs_current_datetime_context("\u73b0\u5728\u662f\u54ea\u4e00\u5e74") is True
    assert needs_current_datetime_context("\u4eca\u5e74\u662f\u51e0\u5e74") is True
    assert needs_current_datetime_context("\u4eca\u5e74\u662f\u51e0\u51e0\u5e74") is True
    assert needs_current_datetime_context("\u660e\u5e74\u662f\u54ea\u4e00\u5e74") is True
    assert needs_current_datetime_context("\u53bb\u5e74\u662f\u51e0\u5e74") is True
    assert needs_current_datetime_context("\u4f60\u4eca\u5929\u600e\u4e48\u770b\u8fd9\u4e2a\u89d2\u8272") is False


def test_build_current_datetime_facts_formats_local_clock() -> None:
    facts = build_current_datetime_facts(datetime.fromisoformat("2026-05-09T13:26:00+08:00"))

    assert facts == [
        "Current local datetime: 2026-05-09 13:26:00 +08:00",
        "Current local date: 2026-05-09",
        "Current local weekday: Saturday",
    ]


def test_normalize_relative_time_query_expands_relative_year_terms() -> None:
    now = datetime.fromisoformat("2026-05-09T13:26:00+08:00")

    assert normalize_relative_time_query("今年欧冠冠军 当前结果", now=now) == "2026年欧冠冠军 当前结果"
    assert normalize_relative_time_query("去年欧冠冠军", now=now) == "2025年欧冠冠军"
    assert normalize_relative_time_query("明年欧冠冠军", now=now) == "2027年欧冠冠军"
    assert normalize_relative_time_query("欧冠冠军 当前结果", now=now) == "欧冠冠军 当前结果"


def test_parse_search_decision_reads_strict_three_line_payload() -> None:
    decision = parse_search_decision(
        "SEARCH: yes\nQUERY: \u6700\u8fd1 \u65b0\u756a \u67d0\u67d0\u52a8\u753b \u53e3\u7891\nREASON: current-facts-needed"
    )

    assert decision.should_search is True
    assert decision.query == "\u6700\u8fd1 \u65b0\u756a \u67d0\u67d0\u52a8\u753b \u53e3\u7891"
    assert decision.reason == "current-facts-needed"


def test_parse_search_decision_rejects_malformed_payload() -> None:
    decision = parse_search_decision(
        "SEARCH: yes\nQUERY: \u6700\u8fd1 \u65b0\u756a\nREASON: current-facts-needed\nEXTRA: no"
    )

    assert decision.should_search is False
    assert decision.query == ""
    assert decision.reason == "malformed"


def test_parse_search_decision_rejects_blank_physical_line() -> None:
    decision = parse_search_decision("SEARCH: yes\n\nQUERY: \u6700\u8fd1 \u65b0\u756a\nREASON: current-facts-needed")

    assert decision.should_search is False
    assert decision.query == ""
    assert decision.reason == "malformed"


def test_parse_search_decision_rejects_invalid_search_token() -> None:
    decision = parse_search_decision("SEARCH: maybe\nQUERY: \u6700\u8fd1 \u65b0\u756a\nREASON: current-facts-needed")

    assert decision.should_search is False
    assert decision.query == ""
    assert decision.reason == "malformed"


def test_parse_search_decision_allows_blank_query_line_to_fail_cleanly() -> None:
    decision = parse_search_decision("SEARCH: yes\nQUERY: \nREASON: current-facts-needed")

    assert decision.should_search is False
    assert decision.query == ""
    assert decision.reason == "empty_query"


def test_parse_search_decision_falls_back_for_blank_reason_line() -> None:
    decision = parse_search_decision("SEARCH: yes\nQUERY: \u6700\u8fd1 \u65b0\u756a\nREASON: ")

    assert decision.should_search is True
    assert decision.query == "\u6700\u8fd1 \u65b0\u756a"
    assert decision.reason == "current-facts-needed"


def test_build_search_decision_prompt_matches_plan_structure() -> None:
    prompt = build_search_decision_prompt(
        bot_name="\u5c0f\u753a",
        target_message="\u6700\u8fd1\u65b0\u756a\u53e3\u7891\u600e\u4e48\u6837",
        recent_messages=["1: old msg", "2: newer msg"],
        proactive_turn=True,
        now=datetime.fromisoformat("2026-05-09T13:26:00+08:00"),
    )

    assert prompt == [
        "System persona: Decide whether \u5c0f\u753a needs current web facts before replying.",
        "Safety rules: Reply with exactly three lines in this grammar: SEARCH: yes|no / QUERY: <text> / REASON: <text>.",
        "Group policy: Search mode=proactive. Prefer search for real-world facts, local places, stores, recommendations, public events, reviews, and recent topics. Decline only for stable commonsense, personal chat, or group-internal talk.",
        "Current local date: 2026-05-09. Resolve relative time words like 今天、今年、明年、去年 against this date.",
        "Recent messages:\n1: old msg\n2: newer msg",
        "Target message: \u6700\u8fd1\u65b0\u756a\u53e3\u7891\u600e\u4e48\u6837",
    ]
