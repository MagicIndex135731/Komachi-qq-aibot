"""Freshness gate for live group messages.

QQ bridges hand over everything that arrived while the session was offline as
soon as it comes back (offline messages), and SnowLuma starts delivering them
right after login.  Answering those reads as "the bot suddenly replies to a
message from hours ago", so the live path archives stale messages instead of
replying to them.
"""

from __future__ import annotations

from datetime import UTC, datetime


def message_age_seconds(event: object, *, now: datetime | None = None) -> float:
    """Seconds between the event timestamp and now (never negative)."""

    timestamp = getattr(event, "timestamp", None)
    if not isinstance(timestamp, datetime):
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return max(0.0, (reference - timestamp).total_seconds())


def is_stale_message(
    event: object,
    *,
    max_age_seconds: int,
    now: datetime | None = None,
) -> bool:
    """True when a message is older than the reply window.

    ``max_age_seconds`` of 0 (or less) disables the gate, which restores the
    previous "answer everything that arrives" behaviour.
    """

    if int(max_age_seconds) <= 0:
        return False
    return message_age_seconds(event, now=now) > float(max_age_seconds)
