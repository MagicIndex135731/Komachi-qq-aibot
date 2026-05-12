from datetime import UTC, datetime, timedelta

from app.jobs.cleanup_jobs import is_memory_expired
from app.jobs.summary_jobs import should_schedule_window_summary


def test_should_schedule_window_summary_every_twenty_five_messages() -> None:
    assert should_schedule_window_summary(message_count=25) is True
    assert should_schedule_window_summary(message_count=24) is False


def test_is_memory_expired_handles_none_past_and_future_values() -> None:
    assert is_memory_expired(None) is False
    assert is_memory_expired(datetime.now(UTC) - timedelta(days=1)) is True
    assert is_memory_expired(datetime.now(UTC) + timedelta(days=1)) is False
