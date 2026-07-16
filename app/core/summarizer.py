from __future__ import annotations


def summarize_window(lines: list[str]) -> str:
    normalized = [line.strip() for line in lines if line.strip()]
    if not normalized:
        return "Recent chat summary: (empty)"
    sample_count = min(6, len(normalized))
    indices = sorted({round(index * (len(normalized) - 1) / max(1, sample_count - 1)) for index in range(sample_count)})
    highlights = " | ".join(normalized[index] for index in indices)
    return f"Recent chat summary: {highlights}"
