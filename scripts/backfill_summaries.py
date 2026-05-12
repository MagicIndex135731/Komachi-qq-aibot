from __future__ import annotations

from app.core.summarizer import summarize_window


def backfill_lines(lines: list[str], *, window_size: int = 25) -> list[str]:
    summaries: list[str] = []
    for start in range(0, len(lines), window_size):
        window = lines[start : start + window_size]
        if window:
            summaries.append(summarize_window(window))
    return summaries
