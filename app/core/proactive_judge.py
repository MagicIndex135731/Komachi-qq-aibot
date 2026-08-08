from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re


@dataclass(slots=True)
class ProactiveJudgeResult:
    should_interject: bool
    reason: str = ""


def build_proactive_judge_prompt(
    *,
    bot_name: str,
    target_message: str,
    recent_messages: list[str],
    now: datetime,
    context_messages: int = 5,
    max_chars_per_message: int = 120,
) -> list[str]:
    recent = [
        message[:max_chars_per_message]
        for message in recent_messages[-context_messages:]
        if message and message.strip()
    ]
    joined_recent = "\n".join(recent)
    local_now = now.astimezone()
    return [
        f"System persona: Decide whether {bot_name} should proactively chime in with one short teasing message in this group chat.",
        "Safety rules: Reply with exactly two lines in this grammar: DECISION: yes|no / REASON: <one short line>. Do not add anything else.",
        "Group policy: Say yes only when the conversation is lively and a short, smug, playful interjection would naturally fit (a funny claim, a dumb take, big news, a teasing opportunity, or a topic the bot clearly has a strong opinion on). Say no for boring logistics, private personal details, off-topic noise, long technical discussion without a hook, or anything where butting in would be annoying. Stay selective: silence is the default.",
        f"Current time: {local_now.strftime('%Y-%m-%d %H:%M')}.",
        "Recent messages:",
        joined_recent,
        f"Target message: {target_message}",
    ]


_DECISION_PATTERN = re.compile(r"^\s*DECISION\s*:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_REASON_PATTERN = re.compile(r"^\s*REASON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_proactive_judge(text: str | None) -> ProactiveJudgeResult:
    if not text:
        return ProactiveJudgeResult(False, "malformed")
    decision_match = _DECISION_PATTERN.search(text)
    if decision_match is None:
        return ProactiveJudgeResult(False, "malformed")
    decision = decision_match.group(1).lower()
    reason_match = _REASON_PATTERN.search(text)
    reason = reason_match.group(1).strip() if reason_match is not None else ""
    return ProactiveJudgeResult(decision == "yes", reason)


def judge_proactive_interjection(
    *,
    client,
    prompt_lines: list[str],
) -> ProactiveJudgeResult:
    """Ask the upstream model whether this turn deserves a proactive interjection.

    Any transport/parse failure degrades to ``no`` so the bot never butts in
    because of a flaky call.
    """
    try:
        raw = client.generate_text(prompt_lines)
    except Exception:  # noqa: BLE001 - degrade safely on any provider failure
        return ProactiveJudgeResult(False, "judge_error")
    return parse_proactive_judge(raw)
