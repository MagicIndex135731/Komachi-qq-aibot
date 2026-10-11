from app.jobs.summary_jobs import should_schedule_window_summary


def test_should_schedule_window_summary_every_twenty_five_messages() -> None:
    assert should_schedule_window_summary(message_count=25) is True
    assert should_schedule_window_summary(message_count=24) is False
