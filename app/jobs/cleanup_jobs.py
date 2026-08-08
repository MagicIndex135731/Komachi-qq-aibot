from __future__ import annotations

from datetime import UTC, datetime

from app.core.time_utils import shanghai_now_naive


def is_memory_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        return expires_at <= shanghai_now_naive()
    return expires_at <= datetime.now(UTC)
