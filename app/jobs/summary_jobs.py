from __future__ import annotations


def should_schedule_window_summary(*, message_count: int) -> bool:
    return message_count > 0 and message_count % 25 == 0


def format_summary_source_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]
