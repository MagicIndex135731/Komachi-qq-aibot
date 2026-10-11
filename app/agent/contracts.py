from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Literal

from app.adapters.onebot_models import GroupMessageEvent


AgentAction = Literal["silent", "wait", "reply"]
AgentActionStatus = Literal["sent", "uncertain", "blocked", "failed", "silent"]


@dataclass(frozen=True, slots=True)
class AgentBatch:
    group_id: int
    events: tuple[GroupMessageEvent, ...]
    turn_id: str
    working_note: str = ""
    last_sent_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: AgentAction
    target_message_id: str | None = None
    reason: str = ""
    working_note: str = ""
    wait_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentActionResult:
    status: AgentActionStatus
    receipt_ids: tuple[str, ...] = ()
    note: str = ""
    charge_proactive_cadence: bool = True


@dataclass(frozen=True, slots=True)
class GroupAgentOptions:
    quiet_seconds: float = 1.5
    max_batch_wait_seconds: float = 5.0
    max_batch_messages: int = 20
    max_pending_per_group: int = 200
    max_message_age_seconds: float = 180.0
    max_active_groups: int = 32
    model_concurrency: int = 2
    decision_timeout_seconds: float = 45.0
    execution_timeout_seconds: float = 180.0
    shutdown_timeout_seconds: float = 5.0
    max_wait_rounds: int = 2
    max_wait_seconds: float = 10.0
    working_note_max_chars: int = 2000
    working_note_max_age_seconds: float = 1800.0
    working_note_max_sources: int = 200
    completed_retention_seconds: float = 7 * 24 * 60 * 60
    prune_interval_seconds: float = 60 * 60

    def __post_init__(self) -> None:
        positive = (
            "max_batch_wait_seconds", "max_batch_messages", "max_pending_per_group",
            "max_message_age_seconds", "max_active_groups", "model_concurrency",
            "decision_timeout_seconds", "execution_timeout_seconds",
            "shutdown_timeout_seconds", "working_note_max_chars", "working_note_max_age_seconds",
            "working_note_max_sources",
            "completed_retention_seconds", "prune_interval_seconds",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.quiet_seconds < 0 or self.max_wait_rounds < 0 or self.max_wait_seconds < 0:
            raise ValueError("quiet/wait bounds must not be negative")


class AgentStoppedError(RuntimeError):
    """The turn's runtime, connection or generation is no longer current."""
