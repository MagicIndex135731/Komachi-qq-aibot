from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from typing import Protocol, Sequence

from app.core.context_builder import _estimate_tokens


class EpisodeMessageLike(Protocol):
    id: int
    platform_msg_id: str
    timestamp: datetime
    plain_text: str
    reply_to_msg_id: str | None
    mentioned_bot: bool
    user_id: int


@dataclass(frozen=True, slots=True)
class EpisodeBoundaryDecision:
    should_close: bool
    reason: str = ""
    extended_for_continuity: bool = False


@dataclass(frozen=True, slots=True)
class EpisodeWindow:
    messages: tuple[EpisodeMessageLike, ...]
    token_count: int

    @property
    def source_message_ids(self) -> tuple[int, ...]:
        return tuple(message.id for message in self.messages)

    @property
    def source_platform_msg_ids(self) -> tuple[str, ...]:
        return tuple(message.platform_msg_id for message in self.messages)


def estimate_message_tokens(text: str) -> int:
    """Use the same deterministic token-ish estimate as prompt budgeting."""
    return _estimate_tokens(str(text or ""))


def _normalized_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _has_continuity(
    *,
    previous: EpisodeMessageLike,
    current: EpisodeMessageLike,
    open_platform_msg_ids: set[str],
    bot_user_id: int | None,
) -> bool:
    if current.reply_to_msg_id and current.reply_to_msg_id in open_platform_msg_ids:
        return True
    if bot_user_id is None:
        return False
    # An addressed user turn immediately following a bot reply is the common
    # continuous @bot Q&A shape. The explicit mention is intentionally
    # required; ordinary chatter after a bot message does not extend episodes.
    return current.mentioned_bot and previous.user_id == bot_user_id


def decide_episode_boundary(
    *,
    previous: EpisodeMessageLike,
    current: EpisodeMessageLike,
    open_message_count: int,
    open_token_count: int,
    open_platform_msg_ids: set[str],
    idle_minutes: int,
    max_messages: int,
    max_tokens: int,
    bot_user_id: int | None = None,
    hard_limit_factor: float = 1.25,
) -> EpisodeBoundaryDecision:
    """Decide whether ``current`` starts a new episode.

    Soft boundaries may be delayed for an in-episode reply or a continuous
    addressed bot turn. A bounded hard limit prevents a reply chain from
    growing an episode forever.
    """
    previous_at = _normalized_timestamp(previous.timestamp)
    current_at = _normalized_timestamp(current.timestamp)
    message_limit = max(1, int(max_messages))
    token_limit = max(1, int(max_tokens))
    factor = max(1.0, float(hard_limit_factor))
    projected_tokens = open_token_count + estimate_message_tokens(current.plain_text)

    if open_message_count >= ceil(message_limit * factor) or projected_tokens > ceil(
        token_limit * factor
    ):
        return EpisodeBoundaryDecision(True, "hard_limit")

    reason = ""
    if previous_at.date() != current_at.date():
        reason = "day"
    elif (
        current_at > previous_at
        and (current_at - previous_at).total_seconds() >= max(0, int(idle_minutes)) * 60
    ):
        reason = "idle"
    elif open_message_count >= message_limit:
        reason = "message_limit"
    elif projected_tokens > token_limit:
        reason = "token_limit"

    if not reason:
        return EpisodeBoundaryDecision(False)

    if _has_continuity(
        previous=previous,
        current=current,
        open_platform_msg_ids=open_platform_msg_ids,
        bot_user_id=bot_user_id,
    ):
        return EpisodeBoundaryDecision(
            False,
            reason=reason,
            extended_for_continuity=True,
        )
    return EpisodeBoundaryDecision(True, reason)


def _window_end(
    messages: Sequence[EpisodeMessageLike],
    *,
    start: int,
    min_messages: int,
    max_messages: int,
    max_tokens: int,
    overlap_messages: int,
) -> tuple[int, int]:
    end = start
    tokens = 0
    while end < len(messages) and end - start < max_messages:
        next_tokens = estimate_message_tokens(messages[end].plain_text)
        if end > start and tokens + next_tokens > max_tokens:
            break
        tokens += next_tokens
        end += 1

    if end == start:
        # A single oversized message remains intact. This is preferable to an
        # infinite loop or silently dropping source provenance.
        end += 1
        tokens = estimate_message_tokens(messages[start].plain_text)

    remaining_with_overlap = len(messages) - max(start + 1, end - overlap_messages)
    if (
        end < len(messages)
        and remaining_with_overlap < min_messages
        and end - start > min_messages
    ):
        shrink_by = min(
            min_messages - remaining_with_overlap, end - start - min_messages
        )
        end -= shrink_by
        tokens = sum(
            estimate_message_tokens(message.plain_text)
            for message in messages[start:end]
        )
    return end, tokens


def build_overlap_windows(
    messages: Sequence[EpisodeMessageLike],
    *,
    min_messages: int = 12,
    max_messages: int = 24,
    max_tokens: int = 1800,
    overlap_messages: int = 5,
) -> list[EpisodeWindow]:
    """Split ordered episode messages into bounded overlapping evidence windows."""
    if not messages:
        return []
    resolved_min = max(1, int(min_messages))
    resolved_max = max(resolved_min, int(max_messages))
    resolved_tokens = max(1, int(max_tokens))
    overlap = min(max(0, int(overlap_messages)), resolved_max - 1)
    ordered = sorted(
        messages,
        key=lambda message: (_normalized_timestamp(message.timestamp), message.id),
    )

    windows: list[EpisodeWindow] = []
    start = 0
    while start < len(ordered):
        end, token_count = _window_end(
            ordered,
            start=start,
            min_messages=resolved_min,
            max_messages=resolved_max,
            max_tokens=resolved_tokens,
            overlap_messages=overlap,
        )
        windows.append(
            EpisodeWindow(
                messages=tuple(ordered[start:end]),
                token_count=token_count,
            )
        )
        if end >= len(ordered):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return windows
