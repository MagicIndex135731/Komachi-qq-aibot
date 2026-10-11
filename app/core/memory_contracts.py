"""Neutral read contracts for autonomous, multi-author group observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MemoryObservationRequest:
    group_id: int
    topic_query: str
    source_message_ids: tuple[str, ...]
    focus_subject_ids: tuple[int, ...] = ()
    now: datetime | None = None
    max_results: int = 5
    context_char_budget: int = 1800

    def __post_init__(self) -> None:
        if (
            isinstance(self.group_id, bool)
            or not isinstance(self.group_id, int)
            or self.group_id <= 0
        ):
            raise ValueError("observation_group_invalid")
        if (
            not isinstance(self.topic_query, str)
            or not self.topic_query.strip()
            or len(self.topic_query) > 1200
        ):
            raise ValueError("observation_query_invalid")
        if (
            not self.source_message_ids
            or len(self.source_message_ids) > 100
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.source_message_ids
            )
        ):
            raise ValueError("observation_sources_invalid")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in self.focus_subject_ids
        ):
            raise ValueError("observation_subjects_invalid")
        if (
            isinstance(self.max_results, bool)
            or not isinstance(self.max_results, int)
            or not 1 <= self.max_results <= 20
        ):
            raise ValueError("observation_limit_invalid")
        if (
            isinstance(self.context_char_budget, bool)
            or not isinstance(self.context_char_budget, int)
            or not 1 <= self.context_char_budget <= 6000
        ):
            raise ValueError("observation_budget_invalid")
        object.__setattr__(
            self, "source_message_ids", tuple(dict.fromkeys(self.source_message_ids))
        )
        object.__setattr__(
            self, "focus_subject_ids", tuple(dict.fromkeys(self.focus_subject_ids))
        )


@dataclass(frozen=True, slots=True)
class ObservationEvidence:
    source_msg_id: str
    user_id: int
    content: str


@dataclass(frozen=True, slots=True)
class ObservationFact:
    memory_id: int
    subject_id: str
    content: str
    source_msg_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryObservationResult:
    group_id: int
    evidence: tuple[ObservationEvidence, ...] = ()
    facts: tuple[ObservationFact, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    context_char_budget: int = 1800

    def prompt_lines(self) -> list[str]:
        lines = [
            f"Historical evidence source={item.source_msg_id} author={item.user_id}: {item.content}"
            for item in self.evidence
        ]
        lines.extend(
            f"Memory fact subject={item.subject_id} sources={','.join(item.source_msg_ids)}: {item.content}"
            for item in self.facts
        )
        packed: list[str] = []
        remaining = self.context_char_budget
        for line in lines:
            required = len(line) + (1 if packed else 0)
            if required <= remaining:
                packed.append(line)
                remaining -= required
        return packed
