from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

from app.core.memory_query_resolver import MemoryQueryResolver, TimeRange


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
    assert result.reference_msg_ids == ("7",)
    assert result.rewrite_used is False


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

    assert yesterday.time_range == TimeRange(datetime(2026, 7, 22), datetime(2026, 7, 23))
    assert explicit.time_range == TimeRange(datetime(2026, 7, 21), datetime(2026, 7, 22))


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

    assert result.time_range == TimeRange(datetime(2026, 7, 13), datetime(2026, 7, 20))
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
