"""Typed observations and actions for scheduled account participation.

Group sources remain canonical message IDs. Space sources are actual post IDs;
neither surface fabricates an inbound message to obtain send permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SocialKind = Literal["group", "qzone_publish", "qzone_comment", "profile"]
SocialAction = Literal["silent", "send_group", "publish_qzone", "comment_qzone", "run_tools"]


@dataclass(frozen=True, slots=True)
class SocialPost:
    owner_id: int
    post_id: str
    text: str
    published_at: datetime
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialJob:
    key: str
    kind: SocialKind
    scope_id: int
    interval_seconds: float = 1800
    cooldown_seconds: float = 1800
    max_daily: int = 2
    min_idle_seconds: float = 1800
    max_post_age_seconds: float = 86400
    persona_key: str = "default"
    public_topics: tuple[str, ...] = ()
    friend_ids: tuple[int, ...] = ()
    visibility: int = 4
    quiet_hours: tuple[str, str] | None = ("23:00", "08:00")
    autonomous_tools: bool = False
    allow_images: bool = False
    identity_tools: tuple[str, ...] = ()
    max_steps: int = 8
    max_model_calls_daily: int = 60
    max_image_calls_daily: int = 2
    event_goals_only: bool = False
    chat_cadence_only: bool = False
    all_friends: bool = False
    max_daily_per_friend: int = 0
    patrol_interval_seconds: tuple[float, float] | None = None
    morning_delay_seconds: tuple[float, float] = (1800, 5400)
    shared_cooldown_seconds: float = 0

    def __post_init__(self) -> None:
        import math

        if self.kind not in {"group", "qzone_publish", "qzone_comment", "profile"}:
            raise ValueError("unsupported social job kind")
        if not self.key or len(self.key) > 160 or self.scope_id <= 0:
            raise ValueError("invalid social job identity")
        for value in (self.interval_seconds, self.cooldown_seconds, self.max_post_age_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("social timing must be finite and positive")
        if not math.isfinite(self.min_idle_seconds) or self.min_idle_seconds < 0:
            raise ValueError("invalid idle interval")
        if not 1 <= self.max_daily <= 50:
            raise ValueError("invalid daily action budget")
        if (type(self.max_daily_per_friend) is not int or not 0 <= self.max_daily_per_friend <= 50
                or self.max_daily_per_friend and self.kind != "qzone_comment"):
            raise ValueError("invalid per-friend daily action budget")
        if self.visibility not in {4, 64}:
            raise ValueError("autonomous space visibility must be friends or self")
        if self.patrol_interval_seconds is not None:
            if self.kind not in {"qzone_publish", "qzone_comment"} or not self.autonomous_tools:
                raise ValueError("patrol belongs to autonomous space jobs")
            for pair in (self.patrol_interval_seconds, self.morning_delay_seconds):
                if (not isinstance(pair, tuple) or len(pair) != 2 or
                        any(type(n) not in {int, float} or not math.isfinite(n) for n in pair) or
                        not 60 <= pair[0] <= pair[1] <= 86400):
                    raise ValueError("invalid patrol timing range")
        if (type(self.shared_cooldown_seconds) not in {int, float} or
                not math.isfinite(self.shared_cooldown_seconds) or not 0 <= self.shared_cooldown_seconds <= 86400):
            raise ValueError("invalid shared space cooldown")
        if any(value <= 0 for value in self.friend_ids) or len(self.friend_ids) > 100:
            raise ValueError("invalid friend allowlist")
        if len(self.public_topics) > 20 or any(not topic.strip() or len(topic) > 500 for topic in self.public_topics):
            raise ValueError("invalid public topics")
        if any(type(value) is not bool for value in (self.autonomous_tools, self.allow_images, self.event_goals_only, self.chat_cadence_only, self.all_friends)):
            raise ValueError("tool permissions must be booleans")
        if self.all_friends and (self.kind != "qzone_comment" or self.friend_ids):
            raise ValueError("all_friends belongs to comments and cannot overlap a static allowlist")
        if set(self.identity_tools) - {"nickname", "avatar", "card"}:
            raise ValueError("invalid own identity tools")
        for number, minimum, maximum in ((self.max_steps, 1, 32), (self.max_model_calls_daily, 1, 500), (self.max_image_calls_daily, 0, 10)):
            if type(number) is not int or not minimum <= number <= maximum:
                raise ValueError("invalid autonomous budget")
        if self.quiet_hours is not None:
            from datetime import time

            if len(self.quiet_hours) != 2:
                raise ValueError("invalid quiet hours")
            for value in self.quiet_hours:
                if time.fromisoformat(value).tzinfo is not None:
                    raise ValueError("quiet hours must be local Shanghai times without offsets")


@dataclass(frozen=True, slots=True)
class SocialObservation:
    job: SocialJob
    now: datetime
    persona: str
    context: tuple[str, ...] = ()
    source_message_ids: tuple[int, ...] = ()
    posts: tuple[SocialPost, ...] = ()
    suppressed_reason: str = ""
    own_output_action_ids: tuple[str, ...] = ()
    authorized_friend_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialDecision:
    action: SocialAction = "silent"
    text: str = ""
    post_id: str = ""
    owner_id: int = 0
    source_message_ids: tuple[int, ...] = ()
    reason: str = ""
    image_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialResult:
    status: Literal["sent", "uncertain", "failed", "blocked", "silent"]
    receipt_id: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SocialRuntimeOptions:
    poll_seconds: float = 15
    observation_timeout_seconds: float = 30
    decision_timeout_seconds: float = 60
    execution_timeout_seconds: float = 45
    lease_seconds: float = 300
    shutdown_timeout_seconds: float = 5

    def __post_init__(self) -> None:
        import math

        for value in (self.poll_seconds, self.observation_timeout_seconds, self.decision_timeout_seconds,
                      self.execution_timeout_seconds, self.lease_seconds, self.shutdown_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("social runtime deadlines must be finite and positive")
        if self.lease_seconds <= (self.observation_timeout_seconds + self.decision_timeout_seconds
                                  + self.execution_timeout_seconds):
            raise ValueError("lease must outlive the bounded social turn")
