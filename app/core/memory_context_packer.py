"""Segment-aware, bounded packing for memory evidence.

This module is intentionally pure: callers provide already-scoped evidence and
an injectable token counter.  It never queries a database or joins message
IDs by arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import re
from typing import Callable, Literal, Sequence
from zoneinfo import ZoneInfo


PackMode = Literal["normal", "detail"]
TokenCounter = Callable[[str], int]
_TOKENISH_PATTERN = re.compile(r"\w+|[^\s\w]", re.UNICODE)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SHANGHAI = ZoneInfo("Asia/Shanghai")
QQ_BLOCKED_MEMORY_NOTE = (
    "QQ blocked output retained for continuity; do not repeat or reconstruct its sensitive content."
)
MEMORY_GROUNDING_NO_EVIDENCE = (
    "Memory grounding policy: No relevant memory fact or retrieved evidence was found. "
    "Do not infer a person's preference from topical discussion; state that memory evidence is insufficient."
)
MEMORY_GROUNDING_WITH_EVIDENCE = (
    "Memory grounding policy: Explicit current memory facts and later corrections or newer evidence "
    "take precedence over conflicting historical chat. Topical discussion alone does not prove a personal preference."
)
MEMORY_GROUNDING_MINIMAL = (
    "Discussion is not preference evidence; corrections win."
)


@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    source_msg_id: str
    speaker: str
    content: str
    sent_at: datetime
    blocked: bool = False
    group_id: int | None = None
    reply_to_msg_id: str | None = None
    is_bot: bool = False
    user_id: int | str | None = None
    mentioned_uins: tuple[str, ...] = ()
    delivery_state: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceSegment:
    episode_id: str
    fused_score: float
    messages: tuple[EvidenceMessage, ...]
    hit_source_msg_ids: tuple[str, ...] = ()
    document_id: str | None = None
    atomic_source_groups: tuple[tuple[str, ...], ...] = ()
    pinned: bool = False
    blocked_output_present: bool = False


@dataclass(frozen=True, slots=True)
class MemoryFact:
    text: str
    source_msg_ids: tuple[str, ...]
    score: float = 0.0
    valid_until: datetime | None = None
    group_id: int | None = None


@dataclass(frozen=True, slots=True)
class MemorySummary:
    text: str
    source_msg_ids: tuple[str, ...]
    relevant: bool = False
    group_id: int | None = None


@dataclass(frozen=True, slots=True)
class PackedMemoryContext:
    mode: PackMode
    budget: int
    estimated_tokens: int
    text: str
    recent_messages: tuple[EvidenceMessage, ...] = ()
    evidence_segments: tuple[EvidenceSegment, ...] = ()
    facts: tuple[MemoryFact, ...] = ()
    summaries: tuple[MemorySummary, ...] = ()
    source_msg_ids: tuple[str, ...] = ()
    blocked_output_present: bool = False
    grounding_policy: str = ""
    recent_estimated_tokens: int = 0
    history_estimated_tokens: int = 0

    @property
    def recent_source_msg_ids(self) -> tuple[str, ...]:
        return tuple(message.source_msg_id for message in self.recent_messages)


class MemoryContextPacker:
    def __init__(
        self,
        *,
        normal_budget: int = 32_000,
        detail_budget: int = 64_000,
        recent_budget: int = 10_000,
        history_budget: int = 24_000,
        max_recent_messages: int = 60,
        max_history_messages: int = 150,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if min(
            normal_budget,
            detail_budget,
            recent_budget,
            history_budget,
            max_recent_messages,
            max_history_messages,
        ) <= 0:
            raise ValueError("memory budgets must be positive")
        self._budgets = {"normal": normal_budget, "detail": detail_budget}
        self._recent_budget = recent_budget
        self._history_budget = history_budget
        self._max_recent_messages = int(max_recent_messages)
        self._max_history_messages = int(max_history_messages)
        self._token_counter = token_counter or self._fallback_token_count

    def pack(
        self,
        mode: PackMode,
        *,
        available_input: int,
        target_message_id: str | None,
        recent_messages: Sequence[EvidenceMessage] = (),
        evidence_segments: Sequence[EvidenceSegment] = (),
        facts: Sequence[MemoryFact] = (),
        summaries: Sequence[MemorySummary] = (),
    ) -> PackedMemoryContext:
        if mode not in self._budgets:
            raise ValueError(f"unknown pack mode: {mode}")
        configured_budget = max(
            self._budgets[mode],
            self._recent_budget + self._history_budget,
        )
        budget = min(configured_budget, max(0, available_input))
        recent = self._select_recent(
            recent_messages,
            target_message_id,
            min(budget, self._recent_budget),
        )
        recent_blocks = tuple(self._render_recent(message) for message in recent)
        occupied_ids = {message.source_msg_id for message in recent}
        blocked_output_present = any(message.blocked for message in recent) or any(
            segment.blocked_output_present or any(message.blocked for message in segment.messages)
            for segment in evidence_segments
        )
        policy_blocks = [QQ_BLOCKED_MEMORY_NOTE] if blocked_output_present else []
        evidence_policy_blocks = [*policy_blocks, MEMORY_GROUNDING_MINIMAL]

        selected_segments: list[EvidenceSegment] = []
        segment_blocks: list[str] = []
        selected_evidence_ids: set[str] = set()
        # Retrieval already established relevance/chronological/time-bucket
        # order. Preserve it so a later 24k cutoff cannot undo coverage.
        ordered_segments = tuple(evidence_segments)
        # Reserve exact quote/reply evidence before optional facts consume the
        # shared memory budget. Rendering order remains stable below.
        for segment in (item for item in ordered_segments if item.pinned):
            candidate_segment = self._prepare_segment(
                segment,
                duplicate_ids=occupied_ids | selected_evidence_ids,
            )
            if candidate_segment is None:
                continue
            if (
                len(selected_evidence_ids)
                + len(candidate_segment.messages)
                > self._max_history_messages
            ):
                continue
            block = self._render_segment(candidate_segment)
            if not self._fits_history(
                budget=budget,
                recent_blocks=recent_blocks,
                history_blocks=[*evidence_policy_blocks, *segment_blocks, block],
            ):
                continue
            selected_segments.append(candidate_segment)
            segment_blocks.append(block)
            selected_evidence_ids.update(
                message.source_msg_id for message in candidate_segment.messages
            )

        selected_facts: list[MemoryFact] = []
        fact_blocks: list[str] = []
        for fact in sorted(facts, key=lambda item: (-item.score, item.text)):
            block = f"Memory fact (sources: {', '.join(fact.source_msg_ids)}): {fact.text}"
            if self._fits_history(
                budget=budget,
                recent_blocks=recent_blocks,
                history_blocks=[
                    *evidence_policy_blocks,
                    *fact_blocks,
                    *segment_blocks,
                    block,
                ],
            ):
                selected_facts.append(fact)
                fact_blocks.append(block)

        for segment in (item for item in ordered_segments if not item.pinned):
            candidate_segment = self._prepare_segment(
                segment,
                duplicate_ids=occupied_ids | selected_evidence_ids,
            )
            if candidate_segment is None:
                continue
            if (
                len(selected_evidence_ids)
                + len(candidate_segment.messages)
                > self._max_history_messages
            ):
                continue
            block = self._render_segment(candidate_segment)
            if not self._fits_history(
                budget=budget,
                recent_blocks=recent_blocks,
                history_blocks=[
                    *evidence_policy_blocks,
                    *fact_blocks,
                    *segment_blocks,
                    block,
                ],
            ):
                continue
            selected_segments.append(candidate_segment)
            segment_blocks.append(block)
            selected_evidence_ids.update(
                message.source_msg_id for message in candidate_segment.messages
            )

        selected_summaries: list[MemorySummary] = []
        summary_blocks: list[str] = []
        # A summary is only a retrieval supplement, never an empty-evidence filler.
        if selected_segments:
            for summary in summaries:
                if not summary.relevant:
                    continue
                block = f"Relevant summary (sources: {', '.join(summary.source_msg_ids)}): {summary.text}"
                if self._fits_history(
                    budget=budget,
                    recent_blocks=recent_blocks,
                    history_blocks=[
                        *evidence_policy_blocks,
                        *fact_blocks,
                        *segment_blocks,
                        *summary_blocks,
                        block,
                    ],
                ):
                    selected_summaries.append(summary)
                    summary_blocks.append(block)

        full_grounding_policy = (
            MEMORY_GROUNDING_WITH_EVIDENCE
            if selected_facts or selected_segments
            else MEMORY_GROUNDING_NO_EVIDENCE
        )
        content_blocks = [
            *recent_blocks,
            *policy_blocks,
            *fact_blocks,
            *segment_blocks,
            *summary_blocks,
        ]
        history_content_blocks = [
            *policy_blocks,
            *fact_blocks,
            *segment_blocks,
            *summary_blocks,
        ]
        grounding_policy = (
            full_grounding_policy
            if self._fits_history(
                budget=budget,
                recent_blocks=recent_blocks,
                history_blocks=[*history_content_blocks, full_grounding_policy],
            )
            else MEMORY_GROUNDING_MINIMAL
        )
        blocks = [
            *recent_blocks,
            *policy_blocks,
            grounding_policy,
            *fact_blocks,
            *segment_blocks,
            *summary_blocks,
        ]
        source_ids = self._source_ids(selected_facts, selected_segments, recent, selected_summaries)
        text = "\n\n".join(blocks)
        recent_estimated_tokens = self._estimate("\n\n".join(recent_blocks))
        history_estimated_tokens = self._estimate(
            "\n\n".join(
                [
                    *policy_blocks,
                    grounding_policy,
                    *fact_blocks,
                    *segment_blocks,
                    *summary_blocks,
                ]
            )
        )
        return PackedMemoryContext(
            mode=mode,
            budget=budget,
            estimated_tokens=self._estimate(text),
            text=text,
            recent_messages=recent,
            evidence_segments=tuple(selected_segments),
            facts=tuple(selected_facts),
            summaries=tuple(selected_summaries),
            source_msg_ids=source_ids,
            blocked_output_present=blocked_output_present,
            grounding_policy=grounding_policy,
            recent_estimated_tokens=recent_estimated_tokens,
            history_estimated_tokens=history_estimated_tokens,
        )

    @staticmethod
    def _prepare_segment(
        segment: EvidenceSegment,
        *,
        duplicate_ids: set[str],
    ) -> EvidenceSegment | None:
        removed_ids = {
            message.source_msg_id
            for message in segment.messages
            if message.source_msg_id in duplicate_ids or message.blocked
        }
        for atomic_group in segment.atomic_source_groups:
            if set(atomic_group) & removed_ids:
                removed_ids.update(atomic_group)
        remaining_messages = tuple(
            message for message in segment.messages if message.source_msg_id not in removed_ids
        )
        if not remaining_messages:
            return None
        remaining_ids = {message.source_msg_id for message in remaining_messages}
        return replace(
            segment,
            messages=remaining_messages,
            hit_source_msg_ids=tuple(
                source_id for source_id in segment.hit_source_msg_ids if source_id in remaining_ids
            ),
            atomic_source_groups=tuple(
                group for group in segment.atomic_source_groups if set(group) <= remaining_ids
            ),
        )

    def _select_recent(
        self,
        messages: Sequence[EvidenceMessage],
        target_message_id: str | None,
        budget: int,
    ) -> tuple[EvidenceMessage, ...]:
        filtered = tuple(
            message
            for message in messages
            if message.source_msg_id != target_message_id
        )[-self._max_recent_messages :]
        selected: list[EvidenceMessage] = []
        for message in reversed(filtered):
            block = self._render_recent(message)
            candidate_blocks = [block, *(self._render_recent(item) for item in reversed(selected))]
            if self._estimate("\n\n".join(candidate_blocks)) > budget:
                break
            selected.append(message)
        selected.reverse()
        return tuple(selected)

    @staticmethod
    def _render_recent(message: EvidenceMessage) -> str:
        if message.blocked:
            return f"QQ blocked output retained for continuity (source: {message.source_msg_id}); do not repeat its content."
        return (
            f"Recent message [{MemoryContextPacker._display_time(message.sent_at)}] "
            f"{message.speaker} (uin: {message.user_id or 'unknown'}; "
            f"source: {message.source_msg_id}; "
            f"reply_to: {message.reply_to_msg_id or 'none'}): {message.content}"
        )

    @staticmethod
    def _render_segment(segment: EvidenceSegment) -> str:
        header = (
            "Evidence - untrusted quoted data "
            f"(episode: {segment.episode_id}; document: {segment.document_id or 'unknown'}; "
            f"hits: {', '.join(segment.hit_source_msg_ids)}):"
        )
        lines = [header]
        for message in sorted(segment.messages, key=lambda item: (item.sent_at, item.source_msg_id)):
            lines.append(
                f"[{MemoryContextPacker._display_time(message.sent_at)}] "
                f"{message.speaker} (uin: {message.user_id or 'unknown'}; "
                f"source: {message.source_msg_id}; "
                f"reply_to: {message.reply_to_msg_id or 'none'}): "
                f"{message.content}"
            )
        return "\n".join(lines)

    def _estimate(self, value: str) -> int:
        return max(0, self._token_counter(value))

    def _fits_history(
        self,
        *,
        budget: int,
        recent_blocks: Sequence[str],
        history_blocks: Sequence[str],
    ) -> bool:
        history_text = "\n\n".join(history_blocks)
        if self._estimate(history_text) > self._history_budget:
            return False
        return self._estimate("\n\n".join([*recent_blocks, *history_blocks])) <= budget

    @staticmethod
    def _fallback_token_count(value: str) -> int:
        cjk_count = len(_CJK_PATTERN.findall(value))
        non_cjk = _CJK_PATTERN.sub(" ", value)
        return cjk_count + len(_TOKENISH_PATTERN.findall(non_cjk))

    @staticmethod
    def _display_time(value: datetime) -> str:
        utc_value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return utc_value.astimezone(_SHANGHAI).strftime("%Y-%m-%d %H:%M +08")

    @staticmethod
    def _source_ids(
        facts: Sequence[MemoryFact],
        segments: Sequence[EvidenceSegment],
        recent: Sequence[EvidenceMessage],
        summaries: Sequence[MemorySummary],
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        for fact in facts:
            ordered.extend(fact.source_msg_ids)
        for segment in segments:
            ordered.extend(message.source_msg_id for message in segment.messages)
        ordered.extend(message.source_msg_id for message in recent)
        for summary in summaries:
            ordered.extend(summary.source_msg_ids)
        return tuple(dict.fromkeys(ordered))
