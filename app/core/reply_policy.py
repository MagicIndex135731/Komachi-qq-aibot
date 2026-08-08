from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zlib import crc32


@dataclass(slots=True)
class PolicyInput:
    group_speak_enabled: bool
    mentioned_bot: bool
    named_bot: bool
    same_thread_followup: bool
    recent_bot_reply_at: datetime | None
    now: datetime
    quiet_hours: tuple[time, time] | None
    proactive_enabled: bool
    group_traffic_last_minute: int
    proactive_judge_enabled: bool = False
    addressed_without_at: bool = False
    proactive_interval_seconds: tuple[int, int] = (180, 480)
    event_id: str = ""


@dataclass(slots=True)
class ReplyDecision:
    should_reply: bool
    reason: str
    score: int


class ReplyPolicy:
    def __init__(self, *, cooldown_seconds: int = 60) -> None:
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.cooldown_seconds = cooldown_seconds

    def decide(self, policy_input: PolicyInput) -> ReplyDecision:
        if not policy_input.group_speak_enabled:
            return ReplyDecision(False, "group_not_allowlisted", 0)

        guaranteed_trigger = any(
            [
                policy_input.mentioned_bot,
                policy_input.named_bot,
                policy_input.same_thread_followup,
                policy_input.addressed_without_at,
            ]
        )
        if guaranteed_trigger:
            return ReplyDecision(True, "direct_trigger", 10)

        if self._in_quiet_hours(policy_input):
            return ReplyDecision(False, "quiet_hours", 0)
        if self._in_cooldown(policy_input):
            return ReplyDecision(False, "cooldown", 0)

        if not policy_input.proactive_enabled:
            return ReplyDecision(False, "proactive_disabled", 0)

        if policy_input.group_traffic_last_minute < 2:
            return ReplyDecision(False, "below_threshold", 0)

        if not policy_input.proactive_judge_enabled:
            return ReplyDecision(False, "proactive_judge_disabled", 0)

        return ReplyDecision(True, "proactive_candidate", 1)

    def _in_cooldown(self, policy_input: PolicyInput) -> bool:
        if policy_input.recent_bot_reply_at is None:
            return False
        cooldown = timedelta(seconds=self._proactive_cooldown_seconds(policy_input))
        return (policy_input.now - policy_input.recent_bot_reply_at) < cooldown

    def _proactive_cooldown_seconds(self, policy_input: PolicyInput) -> int:
        minimum, maximum = policy_input.proactive_interval_seconds
        if maximum < minimum:
            maximum = minimum
        if minimum == maximum:
            return max(self.cooldown_seconds, minimum)
        span = maximum - minimum
        deterministic = minimum + (crc32(policy_input.event_id.encode("utf-8")) % (span + 1))
        return max(self.cooldown_seconds, deterministic)

    def _in_quiet_hours(self, policy_input: PolicyInput) -> bool:
        if policy_input.quiet_hours is None:
            return False
        start, end = policy_input.quiet_hours
        current = policy_input.now.timetz().replace(tzinfo=None)
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end
