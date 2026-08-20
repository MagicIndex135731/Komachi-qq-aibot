from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import time

import pytest

from app.core.memory_query_resolver import MemoryQueryResolver, ResolvedMemoryQuery, TimeRange
from app.core.member_identity import GroupMemberIdentity


@dataclass(frozen=True)
class Recent:
    message_id: str
    speaker: str
    content: str
    sent_at: datetime
    reply_to_msg_id: str | None = None
    blocked: bool = False
    user_id: int | str | None = None
    is_bot: bool = False


NOW = datetime(2026, 7, 23, 0, 10)


def test_deterministic_follow_up_uses_quoted_message_without_rewrite() -> None:
    resolver = MemoryQueryResolver()
    quoted = Recent("42", "小王", "服务器迁移已经完成，但还要观察。", datetime(2026, 7, 22, 23, 55))

    result = resolver.resolve("后来呢？", recent_messages=(quoted,), quoted_message=quoted, now=NOW)

    assert result.retrieval_query == quoted.content
    assert result.reference_msg_ids == ("42",)
    assert result.subject_ids is None
    assert result.rewrite_used is False


def test_explicit_quote_does_not_require_a_textual_follow_up_marker() -> None:
    resolver = MemoryQueryResolver()
    quoted = Recent(
        "explicit-quote",
        "member",
        "source evidence",
        datetime(2026, 7, 22, 23, 55),
        user_id=10001,
    )

    result = resolver.resolve(
        "please verify the source evidence",
        recent_messages=(quoted,),
        quoted_message=quoted,
        now=NOW,
    )

    assert result.retrieval_query == quoted.content
    assert result.reference_msg_ids == ("explicit-quote",)
    assert result.subject_ids is None
    assert result.rewrite_used is False


def test_named_follow_up_binds_unique_recent_entity() -> None:
    resolver = MemoryQueryResolver()
    recent = (
        Recent("1", "Alice", "张三说发布已经延期。", datetime(2026, 7, 22, 23, 50)),
        Recent("2", "Bob", "我在等结果。", datetime(2026, 7, 22, 23, 55)),
    )

    result = resolver.resolve("那个人最后怎么样？", recent_messages=recent, now=NOW)

    assert result.entities == ("张三",)
    assert result.retrieval_query == "张三 张三说发布已经延期。"
    assert result.rewrite_used is False


def test_nickname_in_question_binds_the_matching_recent_speaker() -> None:
    resolver = MemoryQueryResolver()
    recent = (
        Recent(
            "7",
            "小王",
            "发布已经完成。",
            datetime(2026, 7, 22, 23, 58),
            user_id=10001,
        ),
    )

    result = resolver.resolve("小王他说了什么？", recent_messages=recent, now=NOW)

    assert result.entities == ("小王",)
    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)
    assert result.reference_msg_ids == ("7",)
    assert result.rewrite_used is False


def test_direct_nickname_question_binds_unique_group_member_without_rewriting_query() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="Garfield", group_card="加菲猫"),
        GroupMemberIdentity(user_id=10002, nickname="Bob", group_card=""),
    )

    result = resolver.resolve(
        "加菲猫最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.retrieval_query == "动画"
    assert result.topic_query == "动画"
    assert result.entities == ("加菲猫",)
    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)


def test_bound_assessment_extracts_subject_independent_topic_query() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )

    result = resolver.resolve(
        "阿渣如何评价八仙动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.original_query == "阿渣如何评价八仙动画？"
    assert result.retrieval_query == "八仙动画"
    assert result.topic_query == "八仙动画"
    assert result.topic_terms == ("八仙动画", "八仙")
    assert result.topic_extraction == "deterministic"
    assert result.subject_aliases_removed == ("阿渣",)
    assert result.subject_ids == ("10001",)
    assert result.answer_mode == "assessment"
    assert result.coverage_mode == "relevance"


def test_possessive_assessment_binds_speaker_and_extracts_media_core_topic() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )

    result = resolver.resolve(
        "阿渣对八仙电影的评价",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)
    assert result.speaker_ids == ("10001",)
    assert result.answer_mode == "assessment"
    assert result.coverage_mode == "relevance"
    assert result.topic_query == "八仙电影"
    assert result.topic_terms == ("八仙电影", "八仙")


@pytest.mark.parametrize(
    ("query", "topic"),
    (
        ("阿渣如何评价八仙动画？", "八仙动画"),
        ("阿渣对八仙电影的评价", "八仙电影"),
    ),
)
def test_media_topic_wins_when_title_core_collides_with_member_alias(
    query: str,
    topic: str,
) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="阿渣"),
            GroupMemberIdentity(user_id=10002, nickname="八仙"),
        ),
    )

    assert result.subject_ids == ("10001",)
    assert result.speaker_ids == ("10001",)
    assert result.topic_query == topic
    assert result.coverage_mode == "relevance"


def test_subject_only_assessment_has_no_topic_instead_of_alias_query_fallback() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )

    result = resolver.resolve(
        "如何评价阿渣？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.original_query == "如何评价阿渣？"
    assert result.retrieval_query == "如何评价阿渣？"
    assert result.topic_query is None
    assert result.topic_terms == ()
    assert result.topic_extraction == "none"
    assert result.subject_ids == ("10001",)


def test_dated_history_binds_unique_member_despite_common_word_alias_collision() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="昨天", group_card=""),
    )

    result = resolver.resolve(
        "阿渣昨天的劲爆发言",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.entities == ("阿渣",)
    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)
    assert result.time_range == TimeRange(
        start=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
    )
    assert result.answer_mode == "dated_history"


def test_relative_day_alias_with_member_joiner_still_fails_closed() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="昨天"),
    )

    result = resolver.resolve(
        "昨天和阿渣都说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣昨天说了哪些话？",
        "阿渣昨天有哪些发言？",
        "阿渣昨天有什么内容？",
    ),
)
def test_plural_dated_speech_query_uses_summary_coverage_instead_of_top_one(
    query: str,
) -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10001, nickname="阿渣"),)

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)
    assert result.answer_mode == "summary"
    assert result.coverage_mode == "time_buckets"


@pytest.mark.parametrize(
    ("query", "expected_answer_mode"),
    (
        ("昨天群里聊了什么", "summary"),
        ("昨天群里说了什么", "summary"),
        ("群里以前提到“动画”时说了什么", "general_history"),
        ('群里以前提到"动画"时说了什么', "general_history"),
        ("之前关于“动画”说过什么", "general_history"),
        ("之前关于动画说过什么", "general_history"),
        ("群里过去讨论「动画」时聊过什么", "general_history"),
    ),
)
def test_explicit_group_history_prefixes_remain_subjectless(
    query: str,
    expected_answer_mode: str,
) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        # A quoted topic may collide with a real member alias. The explicit
        # group-history grammar still asks about the topic across the group.
        GroupMemberIdentity(user_id=10002, nickname="动画"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids is None
    assert result.answer_mode == expected_answer_mode
    assert result.needs_history is True
    if "动画" in query:
        assert result.topic_query == "动画"
        assert result.retrieval_query == "动画"
        assert result.topic_terms == ("动画",)


def test_group_running_joke_query_marks_group_subject_role() -> None:
    result = MemoryQueryResolver().resolve(
        "群里有什么梗",
        recent_messages=(),
        now=NOW,
        group_id=10001,
    )

    assert result.subject_ids is None
    assert result.subject_role == "group"


@pytest.mark.parametrize(
    "query",
    (
        "阿渣 和 昨天说了什么？",
        "阿渣和@昨天说了什么？",
    ),
)
def test_spaced_or_at_relative_day_member_joiner_still_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="昨天"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_excluded_alias_cannot_redirect_dated_query_to_relative_day_member() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="昨天"),
    )

    result = resolver.resolve(
        "阿渣昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        excluded_member_ids={10001},
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_bare_relative_day_alias_does_not_hijack_temporal_or_unknown_person_query() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10003, nickname="昨天"),)

    temporal = resolver.resolve(
        "昨天发生了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    unknown_person = resolver.resolve(
        "陌生猫昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert temporal.speaker_ids == ()
    assert temporal.subject_ids is None
    assert temporal.answer_mode == "summary"
    assert temporal.time_range is not None
    assert unknown_person.speaker_ids == ()
    assert unknown_person.subject_ids == ()


def test_explicit_member_qq_binds_and_conflicting_alias_fails_closed() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    explicit = resolver.resolve(
        "@10002 昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    conflicting = resolver.resolve(
        "@10002 阿渣昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert explicit.speaker_ids == ("10002",)
    assert explicit.subject_ids == ("10002",)
    assert conflicting.speaker_ids == ()
    assert conflicting.subject_ids == ()


def test_chinese_user_prefix_qq_binds_to_requester_id() -> None:
    resolver = MemoryQueryResolver()
    result = resolver.resolve(
        "用户10002最近决定了什么",
        recent_messages=(),
        now=NOW,
        group_members=(),
        requester_id=10002,
    )

    assert result.speaker_ids == ("10002",)
    assert result.subject_ids == ("10002",)


def test_chinese_user_prefix_qq_binds_to_group_member() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10002, nickname="加菲猫"),)
    result = resolver.resolve(
        "用户10002最近决定了什么",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ("10002",)
    assert result.subject_ids == ("10002",)


@pytest.mark.parametrize(
    ("query", "excluded"),
    (
        ("@99999 昨天说了什么？", frozenset()),
        ("@10001 昨天说了什么？", frozenset({10001})),
        ("@10002 @99999 昨天说了什么？", frozenset()),
        ("@10002 @10001 昨天说了什么？", frozenset({10001})),
    ),
)
def test_unknown_or_excluded_explicit_qq_fails_closed(
    query: str,
    excluded: frozenset[int],
) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
        GroupMemberIdentity(user_id=10003, nickname="昨天"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
        excluded_member_ids=excluded,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_unknown_text_mention_fails_closed_and_bare_group_qq_binds() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
        GroupMemberIdentity(user_id=10003, nickname="昨天"),
    )

    unknown = resolver.resolve(
        "@陌生猫昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    bare_qq = resolver.resolve(
        "10002昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert unknown.speaker_ids == ()
    assert unknown.subject_ids == ()
    assert bare_qq.speaker_ids == ("10002",)
    assert bare_qq.subject_ids == ("10002",)


@pytest.mark.parametrize(
    ("query", "excluded"),
    (
        ("10002和99999昨天说了什么？", frozenset()),
        ("10002和10001昨天说了什么？", frozenset({10001})),
    ),
)
def test_known_bare_qq_joined_with_unknown_or_excluded_qq_fails_closed(
    query: str,
    excluded: frozenset[int],
) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
        excluded_member_ids=excluded,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣和陌生猫昨天说了什么？",
        "@加菲猫和陌生猫昨天说了什么？",
    ),
)
def test_known_alias_joined_with_unknown_alias_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣还有陌生猫昨天说了什么？",
        "阿渣以及陌生猫昨天说了什么？",
        "10002还有99999昨天说了什么？",
        "10002以及99999昨天说了什么？",
    ),
)
def test_multichar_joiner_with_unknown_member_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_joined_relative_days_remain_a_subjectless_temporal_query() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="昨天"),
        GroupMemberIdentity(user_id=10002, nickname="今天"),
    )

    result = resolver.resolve(
        "昨天和今天发生了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids is None
    assert result.time_range is not None


@pytest.mark.parametrize(
    "query",
    (
        "阿渣昨天的消息",
        "@10001 昨天的消息",
    ),
)
def test_strong_member_reference_wins_over_common_word_alias(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="消息"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣和加菲猫昨天说了什么？",
        "@阿渣和@加菲猫昨天说了什么？",
    ),
)
def test_joined_excluded_alias_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
        excluded_member_ids={10002},
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣再加上陌生猫昨天都说了什么？",
        "阿渣并且陌生猫昨天都说了什么？",
        "10001并且99999昨天都说了什么？",
    ),
)
def test_additional_joiner_with_unknown_member_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣外加陌生猫昨天都说了什么？",
        "10001外加99999昨天都说了什么？",
    ),
)
def test_unenumerated_join_phrase_with_unknown_member_fails_closed(query: str) -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == ()
    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣外加陌生猫发生了什么？",
        "阿渣外加陌生猫怎么了？",
        "说说阿渣外加陌生猫昨天的事",
        "@10001 阿渣和加菲猫昨天都说了什么？",
        "QQ号10001 阿渣加菲猫昨天都说了什么？",
        "@10001 阿渣外加陌生猫昨天都说了什么？",
    ),
)
def test_unknown_conjunction_shape_fails_closed_without_predicate_enumeration(
    query: str,
) -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣2026-07-21说了什么？",
        "阿渣7月21日说了什么？",
        "阿渣上周说了什么？",
    ),
)
def test_strong_member_alias_allows_trailing_explicit_date(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)
    assert result.time_range is not None


def test_same_member_nickname_and_card_in_query_are_one_identity() -> None:
    result = MemoryQueryResolver().resolve(
        "阿渣（A-Zha）昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(
                user_id=10001,
                nickname="阿渣",
                group_card="A-Zha",
            ),
        ),
    )

    assert result.speaker_ids == ("10001",)
    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣昨天外加陌生猫都说了什么？",
        "阿渣昨天外加加菲猫都说了什么？",
        "阿渣2026-07-21外加陌生猫都说了什么？",
        "@10001 阿渣昨天外加陌生猫都说了什么？",
        "阿渣也和陌生猫昨天都说了什么？",
        "阿渣为什么和陌生猫昨天都说了什么？",
    ),
)
def test_safe_prefix_is_consumed_before_checking_for_second_member(query: str) -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize("query", ("阿渣7/21说了什么？", "阿渣7-21说了什么？"))
def test_strong_member_alias_allows_short_trailing_date(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ("10001",)
    assert result.time_range is not None


@pytest.mark.parametrize(
    "query",
    (
        "请问阿渣昨天的劲爆发言",
        "关于阿渣昨天的劲爆发言",
        "帮我看看阿渣昨天的劲爆发言",
    ),
)
def test_polite_intro_keeps_prefix_alias_strong(query: str) -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="发言"),
    )

    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣的朋友加菲猫昨天说了什么？",
        "阿渣说加菲猫昨天做了什么？",
        "我想问阿渣外加陌生猫昨天都说了什么？",
        "能不能说说阿渣外加陌生猫昨天的事",
    ),
)
def test_complete_second_alias_or_unknown_suffix_fails_closed(query: str) -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
    )

    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣刚刚说了什么？",
        "阿渣之前说了什么？",
        "阿渣曾说过什么？",
    ),
)
def test_common_single_member_time_adverbs_remain_bound(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣的朋友陌生猫昨天说了什么？",
        "阿渣说陌生猫昨天做了什么？",
        "阿渣提到陌生猫昨天的事",
        "我想问阿渣的朋友陌生猫昨天说了什么？",
    ),
)
def test_unknown_relation_or_quote_target_fails_closed(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="阿渣"),
            GroupMemberIdentity(user_id=10002, nickname="加菲猫"),
        ),
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣在群里说了什么？",
        "阿渣最近在群里说了什么？",
        "阿渣刚才说了什么？",
        "阿渣上次说了什么？",
    ),
)
def test_common_group_preposition_and_time_adverb_remain_bound(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣的室友陌生猫昨天说了什么？",
        "阿渣的好友陌生猫昨天说了什么？",
    ),
)
def test_unenumerated_relation_target_fails_closed(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "阿渣说动画昨天上映了吗？",
        "阿渣提到项目昨天上线了吗？",
        "阿渣聊到游戏昨天更新了吗？",
    ),
)
def test_quote_verb_with_known_topic_remains_single_member(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ("10001",)


def test_cross_group_alias_is_detected_but_never_bound() -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=20001, nickname="动画", in_scope=False),
    )

    relation = MemoryQueryResolver().resolve(
        "阿渣的室友动画昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    direct = MemoryQueryResolver().resolve(
        "动画昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert relation.subject_ids == ()
    assert direct.subject_ids == ()


@pytest.mark.parametrize(
    ("query", "members"),
    (
        (
            "王小明昨天说了什么？",
            (
                GroupMemberIdentity(user_id=10001, nickname="小明"),
                GroupMemberIdentity(user_id=20001, nickname="王小明", in_scope=False),
            ),
        ),
        (
            "小明王小明昨天说了什么？",
            (
                GroupMemberIdentity(user_id=10001, nickname="小明"),
                GroupMemberIdentity(user_id=10001, nickname="王小明", in_scope=False),
            ),
        ),
    ),
)
def test_cross_group_superstring_alias_never_binds_shorter_target_alias(
    query: str,
    members: tuple[GroupMemberIdentity, ...],
) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ()


def test_exact_target_alias_wins_over_same_named_cross_group_alias() -> None:
    result = MemoryQueryResolver().resolve(
        "小明昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="小明"),
            GroupMemberIdentity(user_id=20001, nickname="小明", in_scope=False),
        ),
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "我想问王小明昨天说了什么？",
        "能不能说说王小明昨天的事",
    ),
)
def test_polite_prefix_cannot_bypass_cross_group_superstring_shadow(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="小明"),
            GroupMemberIdentity(user_id=20001, nickname="王小明", in_scope=False),
        ),
    )

    assert result.subject_ids == ()


def test_shadow_check_covers_every_matching_alias_of_resolved_member() -> None:
    result = MemoryQueryResolver().resolve(
        "我想问王小明A-Zha昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="小明", group_card="A-Zha"),
            GroupMemberIdentity(user_id=20001, nickname="王小明", in_scope=False),
        ),
    )

    assert result.subject_ids == ()


@pytest.mark.parametrize(
    "query",
    (
        "我想问加菲猫，阿渣昨天说了什么？",
        "能不能说说加菲猫阿渣昨天的事",
    ),
)
def test_non_overlapping_restricted_alias_before_target_fails_closed(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="阿渣"),
            GroupMemberIdentity(user_id=20001, nickname="加菲猫", in_scope=False),
        ),
    )

    assert result.subject_ids == ()


def test_excluded_superstring_alias_shadows_shorter_allowed_alias() -> None:
    result = MemoryQueryResolver().resolve(
        "王小明昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="小明"),
            GroupMemberIdentity(user_id=10002, nickname="王小明"),
        ),
        excluded_member_ids={10002},
    )

    assert result.subject_ids == ()


def test_text_mention_uses_longest_alias_before_scope_conflict_check() -> None:
    result = MemoryQueryResolver().resolve(
        "@王小明 昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="王小明"),
            GroupMemberIdentity(user_id=20001, nickname="王小", in_scope=False),
        ),
    )

    assert result.subject_ids == ("10001",)


def test_text_mention_exact_target_alias_wins_over_same_named_external_alias() -> None:
    result = MemoryQueryResolver().resolve(
        "@小明昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="小明"),
            GroupMemberIdentity(user_id=20001, nickname="小明", in_scope=False),
        ),
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    "query",
    (
        "阿渣的室友消息昨天说了什么？",
        "阿渣的好友服务昨天说了什么？",
        "阿渣的老师大家昨天说了什么？",
    ),
)
def test_possessive_relation_rejects_common_or_topic_words(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ()


def test_long_possessive_relation_target_fails_closed() -> None:
    result = MemoryQueryResolver().resolve(
        "阿渣的室友这是一个超过旧长度限制的陌生人物名字昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ()


def test_cross_group_common_word_alias_overrides_topic_word_exception() -> None:
    result = MemoryQueryResolver().resolve(
        "阿渣说消息昨天发了什么？",
        recent_messages=(),
        now=NOW,
        group_members=(
            GroupMemberIdentity(user_id=10001, nickname="阿渣"),
            GroupMemberIdentity(user_id=20001, nickname="消息", in_scope=False),
        ),
    )

    assert result.subject_ids == ()


def test_text_mention_binds_target_when_same_qq_also_has_external_snapshots() -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣", in_scope=True),
        GroupMemberIdentity(user_id=10001, nickname="阿渣", in_scope=False),
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", in_scope=False),
    )

    result = MemoryQueryResolver().resolve(
        "@阿渣 昨天说了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)


@pytest.mark.parametrize(
    ("query", "members", "expected_subject"),
    (
        (
            "加菲猫最喜欢什么动画？",
            (
                GroupMemberIdentity(user_id=10001, nickname="加菲猫"),
                GroupMemberIdentity(user_id=10002, nickname="动画"),
            ),
            ("10001",),
        ),
        (
            "@10001 加菲猫最喜欢什么动画？",
            (
                GroupMemberIdentity(user_id=10001, nickname="加菲猫"),
                GroupMemberIdentity(user_id=10002, nickname="动画"),
            ),
            ("10001",),
        ),
        (
            "阿渣昨天的劲爆发言",
            (
                GroupMemberIdentity(user_id=10001, nickname="阿渣"),
                GroupMemberIdentity(user_id=10002, nickname="发言"),
            ),
            ("10001",),
        ),
        (
            "结果昨天说了什么？",
            (GroupMemberIdentity(user_id=10003, nickname="结果"),),
            ("10003",),
        ),
    ),
)
def test_prefix_alias_is_strong_without_common_word_enumeration(
    query: str,
    members: tuple[GroupMemberIdentity, ...],
    expected_subject: tuple[str, ...],
) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.speaker_ids == expected_subject
    assert result.subject_ids == expected_subject


def test_joined_relative_days_ignore_unrelated_predicate_alias() -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="昨天"),
        GroupMemberIdentity(user_id=10002, nickname="今天"),
        GroupMemberIdentity(user_id=10003, nickname="发生"),
    )

    result = MemoryQueryResolver().resolve(
        "昨天和今天发生了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids is None


def test_excluded_bot_text_mention_does_not_bind_a_person() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="Mira"),
        GroupMemberIdentity(user_id=10003, nickname="昨天"),
    )

    result = resolver.resolve(
        "@Mira 昨天发生了什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        excluded_member_ids={10001},
    )

    assert result.speaker_ids == ()
    assert result.subject_ids is None
    assert result.time_range is not None


def test_direct_nickname_question_fails_closed_for_duplicate_aliases() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, group_card="阿渣"),
    )

    result = resolver.resolve(
        "阿渣最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.retrieval_query == "阿渣最喜欢什么动画？"
    assert result.entities == ()
    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_direct_multi_member_question_fails_closed_for_fact_subjects() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣"),
        GroupMemberIdentity(user_id=10002, group_card="加菲猫"),
    )

    result = resolver.resolve(
        "阿渣和加菲猫最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.entities == ()
    assert result.speaker_ids == ()
    assert result.subject_ids == ()


def test_unknown_person_memory_query_fails_closed_but_topic_query_remains_unbound() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10001, nickname="阿渣"),)

    ordinary = resolver.resolve(
        "动画有什么推荐？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    unknown = resolver.resolve(
        "陌生猫最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    first_person = resolver.resolve(
        "我最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    second_person = resolver.resolve(
        "你最喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    first_person_likes = resolver.resolve(
        "我喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    subjectless = tuple(
        resolver.resolve(
            query,
            recent_messages=(),
            now=NOW,
            group_members=members,
        )
        for query in (
            "最喜欢什么动画？",
            "喜欢什么动画？",
            "讨厌什么动画？",
            "不喜欢什么动画？",
        )
    )

    assert ordinary.subject_ids is None
    assert unknown.subject_ids == ()
    assert first_person.subject_ids is None
    assert second_person.subject_ids is None
    assert first_person_likes.subject_ids is None
    assert all(result.subject_ids is None for result in subjectless)


def test_ambiguous_history_question_can_use_one_safe_rewrite() -> None:
    calls: list[tuple[str, tuple[Recent, ...], float]] = []

    def rewrite(query: str, recent: tuple[Recent, ...], timeout_seconds: float) -> str:
        calls.append((query, recent, timeout_seconds))
        return '{"retrieval_query":"发布延期的结果","entities":["张三"]}'

    resolver = MemoryQueryResolver(rewrite_provider=rewrite, rewrite_timeout_seconds=0.25)
    recent = (Recent("1", "Alice", "张三和李四都提到发布。", datetime(2026, 7, 22, 23, 50)),)

    result = resolver.resolve("之前那个怎么样？", recent_messages=recent, now=NOW)

    assert result.retrieval_query == "发布延期的结果"
    assert result.entities == ("张三",)
    assert result.rewrite_used is True
    assert calls[0][2] == 0.25


def test_rewrite_cannot_drop_deterministic_time_boundary() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=lambda *_: '{"retrieval_query":"那个话题怎么样"}'
    )

    result = resolver.resolve(
        "昨天之前那个怎么样",
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.rewrite_used is True
    assert result.time_range == TimeRange(
        start=datetime(2026, 7, 21, 16, tzinfo=UTC),
        end=datetime(2026, 7, 22, 16, tzinfo=UTC),
    )
    assert result.retrieval_mode == "temporal"


@pytest.mark.parametrize("query", ("以前说过什么", "曾经聊过什么", "过去的发言"))
def test_general_history_markers_never_degrade_to_recent(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.needs_history is True


def test_first_person_history_binds_requester_and_unknown_speaker_fails_closed() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10001, nickname="阿渣"),)

    requester = resolver.resolve(
        "我以前说过什么劲爆话题",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
        requester_id=10001,
    )
    unknown = resolver.resolve(
        "陌生猫昨天说了什么",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
        requester_id=10001,
    )
    unknown_assessment = resolver.resolve(
        "如何评价陌生猫这个人",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
        requester_id=10001,
    )

    assert requester.subject_ids == ("10001",)
    assert requester.subject_binding == "requester"
    assert unknown.subject_ids == ()
    assert unknown_assessment.subject_ids == ()


def test_temporal_prefix_before_first_person_still_binds_requester() -> None:
    result = MemoryQueryResolver().resolve(
        "昨天我说了什么？",
        recent_messages=(),
        now=NOW,
        group_id=10001,
        requester_id=20001,
    )

    assert result.subject_ids == ("20001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "dated_history"


@pytest.mark.parametrize(
    "query",
    (
        "我喜欢看什么动画？",
        "我爱看什么动画？",
        "我想看什么动画？",
        "我平时喜欢看什么动画？",
        "我最喜欢看什么动画？",
        "我喜欢听什么歌？",
        "我喜欢玩什么游戏？",
        "我喜欢吃什么？",
        "我喜欢喝什么？",
    ),
)
def test_first_person_verb_infix_variants_bind_requester(query: str) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_id=10001,
        requester_id=20001,
    )

    assert result.subject_ids == ("20001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "current_fact"


def test_first_person_verb_infix_without_requester_stays_unbound() -> None:
    result = MemoryQueryResolver().resolve(
        "我喜欢看什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids is None


def test_unknown_person_verb_infix_memory_query_fails_closed() -> None:
    result = MemoryQueryResolver().resolve(
        "陌生猫喜欢看什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ()


def test_member_verb_infix_memory_query_binds_member() -> None:
    result = MemoryQueryResolver().resolve(
        "阿渣喜欢看什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=(GroupMemberIdentity(user_id=10001, nickname="阿渣"),),
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "explicit"
    assert result.answer_mode == "current_fact"


def test_english_media_title_with_short_latin_member_aliases_stays_single_subject() -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="", group_card="加菲猫"),
        GroupMemberIdentity(user_id=10002, nickname="", group_card="To"),
        GroupMemberIdentity(user_id=10003, nickname="", group_card="V"),
    )
    result = MemoryQueryResolver().resolve(
        "加菲猫喜欢ToLove吗？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "explicit"


def test_standalone_latin_member_alias_still_marks_second_person() -> None:
    members = (
        GroupMemberIdentity(user_id=10001, nickname="", group_card="加菲猫"),
        GroupMemberIdentity(user_id=10002, nickname="", group_card="To"),
    )
    result = MemoryQueryResolver().resolve(
        "加菲猫和To喜欢什么动画？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
    )

    assert result.subject_ids == ()


def test_hyphenated_group_card_binds_member() -> None:
    members = (
        GroupMemberIdentity(
            user_id=10001,
            nickname="Ray Fluorite",
            group_card="21-集成-Ray",
        ),
    )
    result = MemoryQueryResolver().resolve(
        "21-集成-Ray以前说过风吹着吗？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "explicit"


def test_mixed_alias_containing_another_latin_alias_stays_single_subject() -> None:
    members = (
        GroupMemberIdentity(
            user_id=10001,
            nickname="Ray Fluorite",
            group_card="21-集成-Ray",
        ),
        GroupMemberIdentity(user_id=10002, nickname="Ray", group_card=""),
    )
    result = MemoryQueryResolver().resolve(
        "21-集成-Ray以前说过风吹着吗？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "explicit"


def test_single_cjk_char_alias_inside_chinese_word_is_not_a_person() -> None:
    members = (
        GroupMemberIdentity(
            user_id=10001,
            nickname="Ray Fluorite",
            group_card="21-集成-Ray",
        ),
        GroupMemberIdentity(user_id=10002, nickname="Ray", group_card=""),
        GroupMemberIdentity(user_id=10003, nickname="风", group_card=""),
    )
    result = MemoryQueryResolver().resolve(
        "21-集成-Ray以前说过风吹着吗？",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "explicit"


@pytest.mark.parametrize(
    ("query", "expect_detail"),
    (
        ("我自称巴萨球迷是哪条，发一下原话", True),
        ("我具体什么时候说的自己是巴萨球迷？", True),
        ("我是巴萨球迷是哪句话说的？", True),
        ("我哪句话说过我是巴萨球迷？", True),
        ("我什么时候发过我是巴萨球迷？", False),
        ("我发的哪条消息说了我是巴萨球迷？", True),
    ),
)
def test_first_person_claim_quote_variants_bind_requester(
    query: str,
    expect_detail: bool,
) -> None:
    result = MemoryQueryResolver().resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_id=10001,
        requester_id=20001,
    )

    assert result.subject_ids == ("20001",)
    assert result.subject_binding == "requester"
    assert result.needs_history is True
    assert result.needs_detail is expect_detail


def test_first_person_claim_quote_with_other_person_pronoun_stays_unbound() -> None:
    result = MemoryQueryResolver().resolve(
        "我想知道你什么时候说过我是巴萨球迷？",
        recent_messages=(),
        now=NOW,
        group_id=10001,
        requester_id=20001,
    )

    # Pronoun points at another person: never bind the requester. The query is
    # treated as an ambiguous person-memory question and fails closed.
    assert result.subject_ids == ()


def test_first_person_claim_quote_without_requester_stays_unbound() -> None:
    result = MemoryQueryResolver().resolve(
        "我自称巴萨球迷是哪条，发一下原话",
        recent_messages=(),
        now=NOW,
        group_id=10001,
    )

    assert result.subject_ids is None


def test_bad_or_unsafe_rewrite_json_falls_back_to_original_question() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=lambda *_: '{"retrieval_query":"x","group_id":"forged"}'
    )

    result = resolver.resolve("之前那个怎么样？", recent_messages=(), now=NOW)

    assert result.retrieval_query == "之前那个怎么样？"
    assert result.rewrite_used is False


def test_blocked_quote_is_excluded_from_rewrite_context() -> None:
    seen: list[tuple[Recent, ...]] = []

    def rewrite(_query: str, recent: tuple[Recent, ...], _timeout: float) -> str:
        seen.append(recent)
        return "not json"

    resolver = MemoryQueryResolver(rewrite_provider=rewrite)
    blocked = Recent("blocked", "bot", "敏感原文", datetime(2026, 7, 22, 23, 50), blocked=True)
    result = resolver.resolve("之前那个怎么样？", recent_messages=(blocked,), quoted_message=blocked, now=NOW)

    assert result.retrieval_query == "之前那个怎么样？"
    assert seen == [()]


def test_relative_and_absolute_dates_are_a_local_calendar_range() -> None:
    resolver = MemoryQueryResolver()

    yesterday = resolver.resolve("昨天发生了什么", recent_messages=(), now=NOW)
    explicit = resolver.resolve("2026-07-21 的消息", recent_messages=(), now=NOW)

    assert yesterday.time_range == TimeRange(
        datetime(2026, 7, 21, 16, tzinfo=UTC),
        datetime(2026, 7, 22, 16, tzinfo=UTC),
    )
    assert explicit.time_range == TimeRange(
        datetime(2026, 7, 20, 16, tzinfo=UTC),
        datetime(2026, 7, 21, 16, tzinfo=UTC),
    )


def test_resolved_query_contract_exposes_speaker_ids_and_confidence() -> None:
    resolver = MemoryQueryResolver()
    recent = (
        Recent(
            "7",
            "小王",
            "发布已经完成。",
            datetime(2026, 7, 22, 23, 58),
            user_id=10001,
        ),
    )

    result = resolver.resolve("小王他说了什么？", recent_messages=recent, now=NOW)

    assert result.resolved_query == result.retrieval_query
    assert result.speaker_ids == ("10001",)
    assert result.confidence == 1.0


def test_quoted_bot_reply_recovers_its_upstream_user_message_from_recent() -> None:
    resolver = MemoryQueryResolver()
    upstream = Recent(
        "user-question",
        "小王",
        "服务器迁移最后怎么处理的？",
        datetime(2026, 7, 22, 23, 50),
        user_id=10001,
    )
    quoted_bot = Recent(
        "bot-answer",
        "小町",
        "当时已经处理好了。",
        datetime(2026, 7, 22, 23, 51),
        reply_to_msg_id="user-question",
        user_id=99999,
        is_bot=True,
    )

    result = resolver.resolve(
        "详细讲讲",
        recent_messages=(upstream, quoted_bot),
        quoted_message=quoted_bot,
        now=NOW,
    )

    assert result.retrieval_query == upstream.content
    assert result.speaker_ids == ("10001",)
    assert result.reference_msg_ids == ("user-question", "bot-answer")


def test_rewrite_identity_outside_group_fails_closed_when_validator_is_injected() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=lambda *_: (
            '{"resolved_query":"外群用户的计划","entity_ids":["foreign-user"],'
            '"speaker_ids":["foreign-user"]}'
        ),
        identity_validator=lambda identity: identity == "10001",
    )

    result = resolver.resolve("之前那个怎么样？", recent_messages=(), now=NOW)

    assert result.resolved_query == "之前那个怎么样？"
    assert result.entity_ids == ()
    assert result.speaker_ids == ()
    assert result.rewrite_used is False


def test_last_week_is_a_temporal_half_open_range() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve("上周发生了什么", recent_messages=(), now=NOW)

    assert result.time_range == TimeRange(
        datetime(2026, 7, 12, 16, tzinfo=UTC),
        datetime(2026, 7, 19, 16, tzinfo=UTC),
    )
    assert result.retrieval_mode == "temporal"
    assert result.needs_history is True


def test_rewrite_timeout_is_enforced_by_resolver_boundary() -> None:
    def slow_rewrite(*_args) -> str:
        time.sleep(0.2)
        return '{"resolved_query":"too late"}'

    resolver = MemoryQueryResolver(
        rewrite_provider=slow_rewrite,
        rewrite_timeout_seconds=0.01,
    )
    started = time.perf_counter()

    result = resolver.resolve("之前那个怎么样？", recent_messages=(), now=NOW)

    assert time.perf_counter() - started < 0.1
    assert result.resolved_query == "之前那个怎么样？"
    assert result.rewrite_used is False


def test_query_plan_binds_requester_and_keeps_group_and_typed_modes() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "评价一下我过去的表现",
        recent_messages=(),
        now=datetime(2026, 7, 23, 8, 10, tzinfo=UTC),
        group_id=12345,
        requester_id=10001,
    )

    assert result.group_id == 12345
    assert result.requester_id == "10001"
    assert result.requester_uin == "10001"
    assert result.subject_binding == "requester"
    assert result.subject_ids == ("10001",)
    assert result.subject_uins == ("10001",)
    assert result.answer_mode == "assessment"
    assert result.coverage_mode == "relevance"
    assert result.coverage_strategy == "relevance"
    assert result.needs_history is True


def test_named_assessment_without_time_range_uses_relevance() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10002, nickname="Garfield", group_card="加菲猫"),)

    result = resolver.resolve(
        "评价加菲猫的性格",
        recent_messages=(),
        now=NOW,
        group_members=members,
        group_id=12345,
        requester_id="10001",
    )

    assert result.speaker_ids == ("10002",)
    assert result.subject_ids == ("10002",)
    assert result.subject_binding == "explicit"
    assert result.answer_mode == "assessment"
    assert result.coverage_mode == "relevance"
    assert result.topic_query == "性格"


def test_named_assessment_with_time_range_keeps_time_bucket_coverage() -> None:
    resolver = MemoryQueryResolver()
    members = (GroupMemberIdentity(user_id=10002, nickname="Garfield", group_card="加菲猫"),)

    result = resolver.resolve(
        "评价加菲猫昨天的表现",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.answer_mode == "assessment"
    assert result.time_range is not None
    assert result.coverage_mode == "time_buckets"
    assert result.topic_query == "昨天的表现"


def test_shanghai_yesterday_is_converted_once_to_strict_utc_boundaries() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "昨天谁发了消息",
        recent_messages=(),
        now=datetime(2026, 7, 23, 18, 30, tzinfo=UTC),
        group_id=12345,
        requester_id=10001,
    )

    assert result.start_at_utc == datetime(2026, 7, 22, 16, tzinfo=UTC)
    assert result.end_at_utc == datetime(2026, 7, 23, 16, tzinfo=UTC)
    assert result.answer_mode == "dated_history"
    assert result.coverage_mode == "chronological"


def test_summary_mode_with_date_uses_time_bucket_coverage() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "总结昨天发生了什么",
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.answer_mode == "summary"
    assert result.coverage_mode == "time_buckets"


def test_explicit_date_interval_is_inclusive_by_shanghai_day_and_half_open_in_utc() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "总结 2026-07-20 到 7月22日 的聊天",
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.start_at_utc == datetime(2026, 7, 19, 16, tzinfo=UTC)
    assert result.end_at_utc == datetime(2026, 7, 22, 16, tzinfo=UTC)
    assert result.answer_mode == "summary"


@pytest.mark.parametrize(
    "query",
    (
        "他们最近有没有叫我",
        "最近谁@我",
        "谁提到我",
    ),
)
def test_requester_mention_queries_bind_requester_and_require_history(query: str) -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.answer_mode == "mention"
    assert result.subject_binding == "requester"
    assert result.subject_ids == ("10001",)
    assert result.speaker_ids == ("10001",)
    assert result.needs_history is True


def test_requester_mention_query_without_requester_fails_closed() -> None:
    result = MemoryQueryResolver().resolve(
        "他们最近有没有叫我",
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=None,
    )

    assert result.answer_mode == "mention"
    assert result.subject_ids == ()
    assert result.needs_history is True


def test_generic_mention_query_keeps_author_subject_empty_and_binds_target() -> None:
    result = MemoryQueryResolver(mention_target_ids=(90001,)).resolve(
        "谁提到小町",
        recent_messages=(),
        now=NOW,
        group_id=12345,
        requester_id=10001,
    )

    assert result.answer_mode == "mention"
    assert result.subject_ids == ()
    assert result.mentioned_user_ids == ("90001",)
    assert result.subject_binding == "unbound"


def test_invalid_or_conflicting_scope_identity_is_rejected() -> None:
    resolver = MemoryQueryResolver()

    with pytest.raises(ValueError, match="group_id"):
        resolver.resolve("历史", recent_messages=(), group_id=0)
    with pytest.raises(ValueError, match="same user"):
        resolver.resolve(
            "历史",
            recent_messages=(),
        requester_id=10001,
        requester_uin=10002,
        )


def test_opinion_phrasing_binds_member_deterministically() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=200000002, nickname="A-Zha", group_card="阿渣"),
    )
    for query_text in (
        "阿渣觉得八仙怎么样？",
        "阿渣感觉八仙如何？",
        "阿渣认为八仙咋样？",
        "阿渣怎么看八仙？",
        "阿渣对八仙什么看法？",
        "阿渣对八仙的印象如何？",
    ):
        result = resolver.resolve(
            query_text,
            recent_messages=(),
            now=NOW,
            group_members=members,
        )
        assert result.subject_ids == ("200000002",), query_text
        assert result.answer_mode == "assessment", query_text
        assert result.needs_history is True, query_text


def test_current_plan_decision_relationship_phrasings_bind_member() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=200000002, nickname="A-Zha", group_card="阿渣"),
    )
    for query_text in (
        "阿渣最近在做什么？",
        "阿渣在干嘛？",
        "阿渣决定了什么？",
        "阿渣打算做什么？",
        "阿渣的计划是什么？",
        "阿渣和谁是什么关系？",
        "阿渣是什么样的人？",
        "阿渣是哪里人？",
        "阿渣是做什么的？",
        "阿渣支持哪个足球队？",
        "阿渣支持哪支球队？",
        "阿渣是哪个足球队粉丝？",
        "阿渣看好哪支球队？",
    ):
        result = resolver.resolve(
            query_text,
            recent_messages=(),
            now=NOW,
            group_members=members,
        )
        assert result.subject_ids == ("200000002",), query_text


def test_default_allow_binding_and_evidence_based_multi_subject_rejection() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
        GroupMemberIdentity(user_id=10002, nickname="张三", group_card=""),
    )

    multi = resolver.resolve(
        "阿渣和张三聊过什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert multi.subject_ids == ()

    with_other = resolver.resolve(
        "阿渣和别人谁更厉害？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert with_other.subject_ids == ()

    relation_placeholder = resolver.resolve(
        "阿渣和谁是什么关系？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert relation_placeholder.subject_ids == ("10001",)

    classmate = resolver.resolve(
        "阿渣的同学说过什么？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert classmate.subject_ids == ()

    same_name_topic = resolver.resolve(
        "阿渣这部动画怎么样？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert same_name_topic.subject_ids == ("10001",)


def test_two_member_assessment_can_resolve_via_rewrite() -> None:
    def rewrite(_query, _recent, _timeout) -> str:
        return '{"resolved_query":"加菲猫 评价","speaker_ids":["10001"]}'

    resolver = MemoryQueryResolver(rewrite_provider=rewrite)
    members = (
        GroupMemberIdentity(user_id=10001, nickname="阿渣", group_card=""),
        GroupMemberIdentity(user_id=10002, nickname="加菲猫", group_card=""),
    )
    result = resolver.resolve(
        "阿渣评价加菲猫怎么样？",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )
    assert result.rewrite_used is True
    assert result.subject_ids == ("10001",)


def test_rewrite_fallback_normalizes_unbound_opinion_query() -> None:
    def rewrite(_query, _recent, _timeout) -> str:
        return '{"resolved_query":"八仙 评价 阿渣"}'

    resolver = MemoryQueryResolver(rewrite_provider=rewrite)
    result = resolver.resolve(
        "阿渣觉得八仙怎么样？",
        recent_messages=(),
        now=NOW,
    )
    assert result.rewrite_used is True
    assert result.retrieval_query == "八仙 评价 阿渣"
    assert result.subject_ids is None


def test_rewrite_subject_must_be_group_member_and_mentioned() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )
    base = ResolvedMemoryQuery(
        original_query="阿渣觉得八仙怎么样？",
        retrieval_query="八仙 评价 阿渣",
        speaker_ids=("10001",),
        subject_ids=("10001",),
    )
    kept = resolver._constrain_rewritten_subject(
        base,
        group_members=members,
        original="阿渣觉得八仙怎么样？",
        recent=(),
    )
    assert kept is not None
    assert kept.subject_ids == ("10001",)

    unknown = replace(base, speaker_ids=("99999",), subject_ids=("99999",))
    assert (
        resolver._constrain_rewritten_subject(
            unknown,
            group_members=members,
            original="阿渣觉得八仙怎么样？",
            recent=(),
        )
        is None
    )

    unmentioned = replace(base, speaker_ids=(), subject_ids=("10001",))
    assert (
        resolver._constrain_rewritten_subject(
            unmentioned,
            group_members=members,
            original="八仙怎么样？",
            recent=(),
        )
        is None
    )

    recent = (
        Recent(
            "1",
            "阿渣",
            "我喜欢八仙",
            datetime(2026, 7, 22, 23, 50),
            user_id=10001,
        ),
    )
    pronoun = replace(base, speaker_ids=("10001",), subject_ids=("10001",))
    kept_pronoun = resolver._constrain_rewritten_subject(
        pronoun,
        group_members=members,
        original="他对八仙什么看法？",
        recent=recent,
    )
    assert kept_pronoun is not None
    assert kept_pronoun.subject_ids == ("10001",)


def _rewrite_provider(payload: dict) -> object:
    return lambda query, recent, timeout_seconds: json.dumps(payload)


def test_bare_first_person_viewing_question_binds_requester_without_whitelist() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "我最近在看什么动画",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "requester"
    assert result.topic_query == "最近在看什么动画"


def test_first_person_profile_portrait_binds_requester() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "给出我的完整个人画像",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "current_fact"
    assert result.topic_query == "画像"


def test_ownership_question_binds_requester_and_uses_fact_intent() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "我和逆蝶蝶两个人到底谁是你的主人",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "current_fact"


def test_ownership_question_with_member_alias_still_binds_requester() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=20001, nickname="N-Zha", group_card="逆蝶蝶"),
    )

    result = resolver.resolve(
        "我和逆蝶蝶两个人到底谁是你的主人",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "current_fact"


def test_member_profile_portrait_is_current_fact_intent() -> None:
    resolver = MemoryQueryResolver()
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )

    result = resolver.resolve(
        "给出阿渣的完整个人画像",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)
    assert result.answer_mode == "current_fact"
    assert result.topic_query == "画像"


@pytest.mark.parametrize(
    "query",
    (
        "我最近在看什么动画",
        "我在追什么番",
        "最近我在看什么动画",
        "我最近在补什么番剧",
        "我现在在看什么",
    ),
)
def test_current_viewing_question_variants_bind_requester_without_whitelist(
    query: str,
) -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_ids == ("10001",)


def test_semantic_rewrite_sets_current_fact_intent_for_viewing_question() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "最近在看 动画",
                "answer_mode": "current_fact",
                "subject_role": "requester",
                "fact_kinds": ["current"],
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "我最近在看什么动画",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_ids == ("10001",)
    assert result.subject_binding == "requester"
    assert result.answer_mode == "current_fact"
    assert result.retrieval_query == "最近在看 动画"
    assert result.preferred_fact_kinds == ("current",)
    assert result.needs_history is False
    assert result.rewrite_used is True


def test_semantic_rewrite_keeps_member_subject_for_member_question() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "阿渣 最近在看 动画",
                "answer_mode": "current_fact",
                "subject_role": "member",
                "speaker_ids": ["10001"],
                "fact_kinds": ["current"],
                "confidence": 0.9,
            }
        )
    )
    members = (
        GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),
    )

    result = resolver.resolve(
        "阿渣最近在看什么动画",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert result.subject_ids == ("10001",)
    assert result.answer_mode == "current_fact"
    assert result.preferred_fact_kinds == ("current",)


def test_semantic_rewrite_accepts_string_time_range_marker() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "最近在看 动画",
                "answer_mode": "current_fact",
                "subject_role": "requester",
                "fact_kinds": ["current"],
                "time_range": "recent",
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "我最近在看什么动画",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.rewrite_used is True
    assert result.answer_mode == "current_fact"
    assert result.retrieval_query == "最近在看 动画"
    assert result.time_range is None


def test_semantic_rewrite_model_denial_clears_deterministic_time_range() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "今天天气",
                "answer_mode": "general",
                "subject_role": "none",
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "今天天气怎么样",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.rewrite_used is True
    assert result.subject_ids in (None, ())
    assert result.time_range is None
    assert result.needs_history is False


def test_mention_pattern_does_not_capture_at_plus_arbitrary_who() -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        "@小町 锐评下群里阿玙和阿渣谁对群更有贡献",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.answer_mode != "mention"


@pytest.mark.parametrize("query", ("谁提到我", "最近谁@我", "他们@我了吗"))
def test_mention_pattern_still_captures_real_mention_queries(query: str) -> None:
    resolver = MemoryQueryResolver()

    result = resolver.resolve(
        query,
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.answer_mode == "mention"


def test_semantic_rewrite_general_mode_clears_memory_intent() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "宜宾地震 页岩气",
                "answer_mode": "general",
                "subject_role": "none",
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "宜宾地震真的是在采集页岩吗",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.rewrite_used is True
    assert result.subject_ids in (None, ())
    assert result.time_range is None
    assert result.needs_history is False
    assert result.preferred_fact_kinds == ()


def test_semantic_rewrite_general_keeps_rule_bound_member_subject() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "laptop display output ports",
                "answer_mode": "general",
                "subject_role": "none",
                "confidence": 0.98,
            }
        )
    )

    result = resolver.resolve(
        "what display outputs does Noir et Noir's laptop have",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
        group_members=(
            GroupMemberIdentity(
                user_id=900000102,
                nickname="Noir et Noir",
                group_card="唉，gpt5.6 sol",
                in_scope=True,
            ),
        ),
    )

    assert result.rewrite_used is True
    assert result.subject_ids == ("900000102",)
    assert result.subject_binding == "explicit"
    assert result.retrieval_query == "laptop display output ports"


def test_semantic_rewrite_member_role_without_id_is_ignored() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "贵州习酒 来历",
                "answer_mode": "general",
                "subject_role": "member",
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "贵州习酒什么来历",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.rewrite_used is True
    assert result.subject_role != "member"
    assert result.subject_ids in (None, ())


def test_semantic_rewrite_group_role_with_personal_id_is_cleared_but_subject_kept() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "我都在群里讨论过哪些话题",
                "answer_mode": "general_history",
                "subject_role": "group",
                "speaker_ids": ["10001"],
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "我都在群里讨论过哪些话题",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.subject_role == ""
    assert result.subject_ids == ("10001",)


def test_explicit_search_questions_skip_semantic_rewrite() -> None:
    resolver = MemoryQueryResolver(
        rewrite_provider=_rewrite_provider(
            {
                "resolved_query": "台风 登陆",
                "answer_mode": "general_history",
                "subject_role": "none",
                "confidence": 0.9,
            }
        )
    )

    result = resolver.resolve(
        "联网搜索最新台风登陆地点",
        recent_messages=(),
        now=NOW,
        requester_id=10001,
    )

    assert result.rewrite_used is False


def test_dated_member_question_skips_semantic_rewrite() -> None:
    calls = {"count": 0}

    def rewrite(query, recent, timeout_seconds):
        calls["count"] += 1
        return '{"resolved_query":"昨天 阿渣 说了什么"}'

    resolver = MemoryQueryResolver(rewrite_provider=rewrite, rewrite_timeout_seconds=0.25)
    members = (GroupMemberIdentity(user_id=10001, nickname="A-Zha", group_card="阿渣"),)

    result = resolver.resolve(
        "阿渣昨天说了什么",
        recent_messages=(),
        now=NOW,
        group_members=members,
    )

    assert calls["count"] == 0
    assert result.rewrite_used is False
    assert result.subject_ids == ("10001",)
    assert result.time_range is not None


def test_rewrite_time_range_cannot_override_deterministic_window() -> None:
    conflicting = TimeRange(
        start=datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
        end=datetime(2026, 7, 23, 16, 0, tzinfo=UTC),
    )

    def rewrite(query, recent, timeout_seconds):
        del query, recent, timeout_seconds
        return json.dumps(
            {
                "resolved_query": "昨天群里发生了什么",
                "time_range": {
                    "start": conflicting.start.isoformat(),
                    "end": conflicting.end.isoformat(),
                },
            }
        )

    resolver = MemoryQueryResolver(rewrite_provider=rewrite, rewrite_timeout_seconds=0.25)

    result = resolver.resolve(
        "昨天群里发生了什么",
        recent_messages=(),
        now=NOW,
        group_id=10,
        requester_id=42,
    )

    assert result.rewrite_used is True
    assert result.time_range == TimeRange(
        start=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
    )
