from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.message_freshness import is_stale_message, message_age_seconds


@dataclass
class FakeEvent:
    timestamp: datetime


def test_fresh_message_is_not_stale() -> None:
    now = datetime(2026, 9, 11, 21, 0, tzinfo=UTC)
    event = FakeEvent(timestamp=now - timedelta(seconds=30))

    assert message_age_seconds(event, now=now) == 30
    assert is_stale_message(event, max_age_seconds=300, now=now) is False


def test_offline_backlog_message_is_stale() -> None:
    now = datetime(2026, 9, 11, 21, 0, tzinfo=UTC)
    # This is the incident that motivated the gate: a 18:25 message delivered
    # right after the 20:52 login.
    event = FakeEvent(timestamp=now - timedelta(hours=2, minutes=27))

    assert is_stale_message(event, max_age_seconds=300, now=now) is True


def test_gate_boundary_and_disable_switch() -> None:
    now = datetime(2026, 9, 11, 21, 0, tzinfo=UTC)
    at_limit = FakeEvent(timestamp=now - timedelta(seconds=300))
    over_limit = FakeEvent(timestamp=now - timedelta(seconds=301))

    assert is_stale_message(at_limit, max_age_seconds=300, now=now) is False
    assert is_stale_message(over_limit, max_age_seconds=300, now=now) is True
    assert is_stale_message(over_limit, max_age_seconds=0, now=now) is False


def test_naive_timestamps_are_treated_as_utc() -> None:
    now = datetime(2026, 9, 11, 21, 0, tzinfo=UTC)
    event = FakeEvent(timestamp=datetime(2026, 9, 11, 18, 25))

    assert message_age_seconds(event, now=now) == 2 * 3600 + 35 * 60
    assert is_stale_message(event, max_age_seconds=300, now=now) is True


def test_event_without_timestamp_is_never_stale() -> None:
    assert message_age_seconds(object()) == 0.0
    assert is_stale_message(object(), max_age_seconds=300) is False
