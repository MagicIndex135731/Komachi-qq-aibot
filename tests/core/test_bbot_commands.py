from __future__ import annotations

import pytest

from app.core.bbot_commands import (
    BBOT_ADMIN_DENIED_TEXT,
    BbotCommandContext,
    BbotCommandSpec,
    BbotCommandResolver,
)


def test_bbot_command_resolver_parses_today_anime_query() -> None:
    resolver = BbotCommandResolver()

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text="@Mira 帮我看看今天有什么新番")

    assert parsed is not None
    assert parsed.command_id == "today_anime"
    assert parsed.command_text == "今日新番"
    assert parsed.denied_reason is None


@pytest.mark.parametrize(
    ("plain_text", "expected_command_text"),
    [
        ("@Mira 帮我看看今天有什么新番", "今日新番"),
        ("@Mira 告诉我今天有什么动画", "今日新番"),
        ("@Mira 今天更新了哪些新番", "今日新番"),
    ],
)
def test_bbot_command_resolver_supports_multiple_today_anime_phrasings(
    plain_text: str,
    expected_command_text: str,
) -> None:
    resolver = BbotCommandResolver()

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text=plain_text)

    assert parsed is not None
    assert parsed.command_id == "today_anime"
    assert parsed.command_text == expected_command_text


def test_bbot_command_resolver_marks_listener_admin_command() -> None:
    resolver = BbotCommandResolver()

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text="@Mira 帮我监听一下 elonmusk 的推特")

    assert parsed is not None
    assert parsed.command_id == "add_twitter_listener"
    assert parsed.command_text is None
    assert parsed.denied_reason == BBOT_ADMIN_DENIED_TEXT


def test_bbot_command_resolver_marks_bilibili_dynamic_query_as_cacheable() -> None:
    resolver = BbotCommandResolver()

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text="@Mira 看一下猫雷最新的b站动态")

    assert parsed is not None
    assert parsed.command_id == "latest_bilibili_dynamic"
    assert parsed.command_text == "最新动态 猫雷"
    assert parsed.cache_platform == "bilibili"


@pytest.mark.parametrize(
    ("plain_text", "expected_command_id", "expected_command_text"),
    [
        ("@Mira 帮我看下老番茄的b站动态", "latest_bilibili_dynamic", "最新动态 老番茄"),
        ("@Mira 看看 elonmusk 最近发了什么推文", "latest_tweet", "最新推文 elonmusk"),
        ("@Mira 帮我看下 elonmusk 的推特", "latest_tweet", "最新推文 elonmusk"),
        ("@Mira 来个6657的烂梗", "random_joke", "随机烂梗 6657"),
        ("@Mira 随机来一条烂梗", "random_joke", "随机烂梗"),
    ],
)
def test_bbot_command_resolver_supports_multiple_phrasings_for_other_commands(
    plain_text: str,
    expected_command_id: str,
    expected_command_text: str,
) -> None:
    resolver = BbotCommandResolver()

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text=plain_text)

    assert parsed is not None
    assert parsed.command_id == expected_command_id
    assert parsed.command_text == expected_command_text


def test_bbot_command_resolver_builds_outbound_message() -> None:
    resolver = BbotCommandResolver()

    outbound = resolver.build_outbound_message("最新动态 697091119")

    assert outbound == "[CQ:at,qq=20002] 最新动态 697091119"


def test_bbot_command_resolver_allows_registering_custom_command_specs() -> None:
    def match_custom(context: BbotCommandContext) -> str | None:
        if "刷新BBot状态" not in context.text:
            return None
        return "刷新状态"

    resolver = BbotCommandResolver(
        extra_command_specs=[
            BbotCommandSpec(
                command_id="refresh_status",
                matcher=match_custom,
            )
        ]
    )

    parsed = resolver.resolve(group_id=10001, mentioned_bot=True, plain_text="@Mira 刷新BBot状态")

    assert parsed is not None
    assert parsed.command_id == "refresh_status"
    assert parsed.command_text == "刷新状态"
