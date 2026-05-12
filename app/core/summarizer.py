from __future__ import annotations


def summarize_window(lines: list[str]) -> str:
    topic_preview = " | ".join(line.strip() for line in lines[:3] if line.strip())
    if not topic_preview:
        return "Recent chat summary: (empty)"
    return f"Recent chat summary: {topic_preview}"
