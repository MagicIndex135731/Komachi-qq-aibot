from datetime import UTC, datetime, timedelta, timezone

from app.core.usage_meter import build_local_day_utc_window


def test_build_local_day_utc_window_uses_local_calendar_day() -> None:
    china = timezone(timedelta(hours=8))

    start_at, end_at = build_local_day_utc_window(
        datetime(2026, 5, 9, 1, 0, tzinfo=UTC),
        local_timezone=china,
    )

    assert start_at == datetime(2026, 5, 8, 16, 0, tzinfo=UTC)
    assert end_at == datetime(2026, 5, 9, 1, 0, tzinfo=UTC)
