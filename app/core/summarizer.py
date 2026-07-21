from __future__ import annotations

import re


def summarize_window(lines: list[str]) -> str:
    normalized = [line.strip() for line in lines if line.strip()]
    if not normalized:
        return "Recent chat summary: (empty)"
    sample_count = min(6, len(normalized))
    indices = sorted({round(index * (len(normalized) - 1) / max(1, sample_count - 1)) for index in range(sample_count)})
    highlights = " | ".join(normalized[index] for index in indices)
    return f"Recent chat summary: {highlights}"


_HIGH_SIGNAL_PATTERN = re.compile(
    r"(?:\u51b3\u5b9a|\u8ba1\u5212|\u51c6\u5907|\u53d6\u6d88|\u6539\u6210|\u7ea6|\u660e\u5929|\u4e0b\u5468|\u95ee\u9898|\u7ed3\u8bba|\?|\uff1f|decid|plan|cancel|change|tomorrow|next week)",
    re.IGNORECASE,
)


def summarize_recursive(*, previous_summary: str, new_window_summary: str, max_chars: int = 1800) -> str:
    """Merge a rolling group summary without rereading all historical messages.

    The original window summaries remain in storage as evidence. This compact
    layer deliberately retains decisions, plans, questions, and evenly spaced
    coverage from the prior rolling summary when it reaches its budget.
    """
    def _without_prefix(value: str) -> str:
        return re.sub(r"^(?:Rolling group memory|Recent chat summary):\s*", "", value.strip(), flags=re.IGNORECASE)

    candidates = [_without_prefix(line) for line in (previous_summary, new_window_summary) if line and line.strip()]
    if not candidates:
        return "Rolling group memory: (empty)"

    signals = [line for line in candidates if _HIGH_SIGNAL_PATTERN.search(line)]
    coverage = candidates if len(candidates) <= 3 else [candidates[0], candidates[-1]]
    selected = list(dict.fromkeys([*signals, *coverage]))
    content = " | ".join(selected)
    if len(content) > max_chars:
        # Preserve both the established narrative and the newly observed
        # window. A prefix-only trim would silently discard the latest change.
        prior_budget = max(1, (max_chars * 2) // 5)
        latest_budget = max(1, max_chars - prior_budget - 5)
        prior = _without_prefix(previous_summary)
        latest = _without_prefix(new_window_summary)
        if len(prior) > prior_budget:
            prior = "..." + prior[-max(1, prior_budget - 3) :]
        if len(latest) > latest_budget:
            latest = latest[: max(1, latest_budget - 3)].rstrip() + "..."
        content = f"{prior} | {latest}".strip(" |")
    return f"Rolling group memory: {content}"
