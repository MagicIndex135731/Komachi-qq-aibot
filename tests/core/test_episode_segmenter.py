from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.episode_segmenter import (
    build_overlap_windows,
    decide_episode_boundary,
    estimate_message_tokens,
)


@dataclass(frozen=True)
class StubMessage:
    id: int
    platform_msg_id: str
    timestamp: datetime
    plain_text: str
    reply_to_msg_id: str | None = None
    mentioned_bot: bool = False
    user_id: int = 42


def _message(
    index: int,
    *,
    at: datetime | None = None,
    text: str | None = None,
    reply_to: str | None = None,
    mentioned_bot: bool = False,
    user_id: int = 42,
) -> StubMessage:
    return StubMessage(
        id=index,
        platform_msg_id=f"m-{index}",
        timestamp=at or datetime(2026, 7, 23, 8, index, tzinfo=UTC),
        plain_text=text or f"message {index}",
        reply_to_msg_id=reply_to,
        mentioned_bot=mentioned_bot,
        user_id=user_id,
    )


def test_episode_boundaries_cover_idle_day_count_and_token_limits() -> None:
    previous = _message(1, at=datetime(2026, 7, 23, 8, 0, tzinfo=UTC))

    idle = decide_episode_boundary(
        previous=previous,
        current=_message(2, at=previous.timestamp + timedelta(minutes=30)),
        open_message_count=2,
        open_token_count=10,
        open_platform_msg_ids={"m-1"},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
    )
    day = decide_episode_boundary(
        previous=previous,
        current=_message(2, at=datetime(2026, 7, 24, 0, 1, tzinfo=UTC)),
        open_message_count=2,
        open_token_count=10,
        open_platform_msg_ids={"m-1"},
        idle_minutes=60 * 24,
        max_messages=50,
        max_tokens=8000,
    )
    count = decide_episode_boundary(
        previous=previous,
        current=_message(2),
        open_message_count=50,
        open_token_count=10,
        open_platform_msg_ids={"m-1"},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
    )
    token = decide_episode_boundary(
        previous=previous,
        current=_message(2, text="甲乙"),
        open_message_count=2,
        open_token_count=7999,
        open_platform_msg_ids={"m-1"},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
    )

    assert (idle.should_close, idle.reason) == (True, "idle")
    assert (day.should_close, day.reason) == (True, "day")
    assert (count.should_close, count.reason) == (True, "message_limit")
    assert (token.should_close, token.reason) == (True, "token_limit")


def test_reply_and_continuous_bot_turn_extend_soft_boundary_but_not_hard_limit() -> (
    None
):
    previous = _message(50, user_id=999)
    referenced = decide_episode_boundary(
        previous=previous,
        current=_message(51, reply_to="m-7"),
        open_message_count=50,
        open_token_count=100,
        open_platform_msg_ids={f"m-{index}" for index in range(1, 51)},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
        bot_user_id=999,
    )
    addressed = decide_episode_boundary(
        previous=previous,
        current=_message(51, mentioned_bot=True),
        open_message_count=50,
        open_token_count=100,
        open_platform_msg_ids={f"m-{index}" for index in range(1, 51)},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
        bot_user_id=999,
    )
    hard = decide_episode_boundary(
        previous=previous,
        current=_message(
            64,
            at=datetime(2026, 7, 23, 9, 4, tzinfo=UTC),
            reply_to="m-7",
        ),
        open_message_count=63,
        open_token_count=100,
        open_platform_msg_ids={f"m-{index}" for index in range(1, 64)},
        idle_minutes=30,
        max_messages=50,
        max_tokens=8000,
        bot_user_id=999,
    )

    assert referenced.should_close is False
    assert referenced.extended_for_continuity is True
    assert addressed.should_close is False
    assert addressed.extended_for_continuity is True
    assert (hard.should_close, hard.reason) == (True, "hard_limit")


def test_overlap_windows_stay_ordered_bounded_and_keep_about_five_message_overlap() -> (
    None
):
    messages = [_message(index) for index in range(1, 31)]
    windows = build_overlap_windows(
        messages,
        min_messages=12,
        max_messages=24,
        max_tokens=1800,
        overlap_messages=5,
    )

    assert [len(window.messages) for window in windows] == [23, 12]
    assert [item.id for item in windows[0].messages[-5:]] == [
        item.id for item in windows[1].messages[:5]
    ]
    assert all(
        [item.timestamp for item in window.messages]
        == sorted(item.timestamp for item in window.messages)
        for window in windows
    )
    assert all(window.token_count <= 1800 for window in windows)


def test_overlap_windows_respect_token_limit_and_keep_reply_with_quoted_message() -> (
    None
):
    messages = [
        _message(
            index,
            text="甲乙丙丁戊己庚辛壬癸",
            reply_to="m-8" if index == 9 else None,
        )
        for index in range(1, 21)
    ]
    windows = build_overlap_windows(
        messages,
        min_messages=4,
        max_messages=8,
        max_tokens=80,
        overlap_messages=3,
    )

    assert all(len(window.messages) <= 8 for window in windows)
    assert all(window.token_count <= 80 for window in windows)
    assert any(
        {"m-8", "m-9"} <= {message.platform_msg_id for message in window.messages}
        for window in windows
    )
    assert estimate_message_tokens("甲乙 hello") == 3
