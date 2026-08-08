from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")


def shanghai_naive(value: datetime) -> datetime:
    """Convert any datetime to a naive Shanghai clock face for SQLite storage."""
    if value.tzinfo is None:
        return value
    return value.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)


def shanghai_aware(value: datetime) -> datetime:
    """Interpret a stored naive datetime as Shanghai, or convert aware to Shanghai."""
    if value.tzinfo is None:
        return value.replace(tzinfo=ASIA_SHANGHAI)
    return value.astimezone(ASIA_SHANGHAI)


def stored_as_utc(value: datetime) -> datetime:
    """Interpret a stored naive datetime as Shanghai clock face and return aware UTC."""
    return shanghai_aware(value).astimezone(UTC)


def shanghai_now_naive() -> datetime:
    return datetime.now(ASIA_SHANGHAI).replace(tzinfo=None)


def shanghai_now_aware() -> datetime:
    return datetime.now(ASIA_SHANGHAI)
