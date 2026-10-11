from datetime import UTC, datetime
import time

import pytest

from app.core.search_policy import build_current_datetime_facts, normalize_relative_time_query
from app.core.time_utils import shanghai_naive, stored_as_utc


def test_shanghai_anchor_when_utc_is_still_yesterday():
    from app.core.time_utils import conversation_time_context
    result = conversation_time_context(datetime(2026, 10, 10, 19, 0, tzinfo=UTC))
    assert result == {"timezone": "Asia/Shanghai", "now": "2026-10-11T03:00:00+08:00",
        "today": "2026-10-11", "yesterday": "2026-10-10", "tomorrow": "2026-10-12", "weekday": "星期日"}


@pytest.mark.parametrize("stamp,today,yesterday", [
    ("2026-10-10T15:59:59+00:00", "2026-10-10", "2026-10-09"),
    ("2026-10-10T16:00:00+00:00", "2026-10-11", "2026-10-10"),
    ("2026-12-31T16:00:00+00:00", "2027-01-01", "2026-12-31"),
    ("2028-02-28T16:00:00+00:00", "2028-02-29", "2028-02-28"),
])
def test_calendar_boundaries(stamp, today, yesterday):
    from app.core.time_utils import conversation_time_context
    context = conversation_time_context(datetime.fromisoformat(stamp))
    assert context["today"] == today and context["yesterday"] == yesterday


def test_naive_message_clock_remains_shanghai_without_double_conversion():
    from app.core.time_utils import conversation_time_context
    instant = datetime(2026, 10, 10, 19, tzinfo=UTC)
    stored = shanghai_naive(instant)
    assert stored == datetime(2026, 10, 11, 3)
    assert stored_as_utc(stored) == instant
    assert conversation_time_context(stored)["now"] == "2026-10-11T03:00:00+08:00"


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX timezone test")
def test_clock_and_relative_year_do_not_depend_on_host_timezone(monkeypatch):
    with monkeypatch.context() as scope:
        scope.setenv("TZ", "UTC")
        time.tzset()
        try:
            facts = build_current_datetime_facts(datetime(2026, 10, 10, 19, tzinfo=UTC))
            assert "Current local date: 2026-10-11" in facts
            assert facts[0].endswith("+08:00")
            assert normalize_relative_time_query("今年记录", now=datetime(2026, 12, 31, 19, tzinfo=UTC)) == "2027年记录"
        finally:
            scope.undo()
            time.tzset()
