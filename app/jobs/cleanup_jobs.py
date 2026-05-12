from __future__ import annotations

from datetime import UTC, datetime


def is_memory_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    return expires_at <= datetime.now(UTC)
