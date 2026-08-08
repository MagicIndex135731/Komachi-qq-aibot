from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import re


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicJudgeResult:
    switched: bool
    reason: str = ""


def build_topic_judge_prompt(
    *,
    recent_messages: list[str],
    current_message: str,
    now: datetime,
    context_messages: int = 8,
    max_chars_per_message: int = 160,
) -> list[str]:
    recent = [
        message[:max_chars_per_message]
        for message in recent_messages[-context_messages:]
        if message and message.strip()
    ]
    joined_recent = "\n".join(recent)
    local_now = now.astimezone()
    return [
        "System persona: You segment a QQ group chat log into topics.",
        "Safety rules: Reply with exactly two lines in this grammar: TOPIC: same|different / REASON: <one short line>. Do not add anything else.",
        "Group policy: Answer 'different' when the current message clearly leaves the topic established by the recent messages, OR when the recent window already mixes multiple unrelated topics and the current message does not clearly continue the latest one. Answer 'same' when it continues, replies to, jokes about, or naturally follows the nearest topic. Brief tangents that get pulled back count as 'same'; when in doubt after seeing multiple topics in the window, prefer 'different'.",
        f"Current time: {local_now.strftime('%Y-%m-%d %H:%M')}.",
        "Recent messages:",
        joined_recent,
        f"Current message: {current_message}",
    ]


_TOPIC_PATTERN = re.compile(r"TOPIC\s*:\s*(same|different)", re.IGNORECASE)
_REASON_PATTERN = re.compile(r"REASON\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def parse_topic_judge(text: str | None) -> TopicJudgeResult:
    if not text:
        return TopicJudgeResult(True, "malformed")
    topic_match = _TOPIC_PATTERN.search(text)
    if topic_match is None:
        return TopicJudgeResult(True, "malformed")
    switched = topic_match.group(1).lower() == "different"
    reason_match = _REASON_PATTERN.search(text)
    reason = reason_match.group(1).strip() if reason_match is not None else ""
    return TopicJudgeResult(switched, reason)


def judge_topic_switch(
    *,
    client,
    prompt_lines: list[str],
) -> TopicJudgeResult:
    """Ask the upstream model whether the current message starts a new topic.

    Any failure degrades to ``switched=True`` so segmentation never grows an
    episode forever because of a flaky call.
    """
    try:
        raw = client.generate_text(prompt_lines)
    except Exception:  # noqa: BLE001
        return TopicJudgeResult(True, "judge_error")
    result = parse_topic_judge(raw)
    if result.reason == "malformed":
        logger.warning(
            "episode_topic_judge_parse_failed raw=%s",
            (raw or "")[:300].replace("\n", "\\n"),
        )
    return result
