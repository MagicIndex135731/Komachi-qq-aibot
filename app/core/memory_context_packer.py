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
    "Memory grounding policy: Answer historical-memory questions only with facts directly and "
    "unambiguously supported by the retrieved evidence. Do not infer, generalize, embellish, add "
    "reactions, or treat topical discussion as proof of a personal preference. Use the smallest "
    "sufficient set of evidence. Do not use memory excerpts to answer recommendation, opinion, "
    "general-knowledge, or action requests; if the packet does not directly contain what is "
    "requested, state that memory evidence is insufficient. For a single-event dated question, "
    "state only one fact from the "
    "top direct source and preserve its wording with minimal paraphrase; later corrections or newer "
    "evidence take precedence. If the "
    "retrieved evidence does not directly answer the question, state that memory evidence is insufficient."
)
MEMORY_GROUNDING_MINIMAL = (
    "Discussion is not preference evidence; corrections win."
)
MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL = (
    "No memory evidence; do not infer a person's preference."
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
    adaptive_enabled: bool = False
    spillover: Literal["none", "recent_to_history", "history_to_recent", "bidirectional"] = "none"
    degradation_reason: str = ""

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
        context_char_budget: int = 12_000,
        max_recent_messages: int = 60,
        max_history_messages: int = 150,
        adaptive_enabled: bool = False,
        recent_protected_min_tokens: int = 1_200,
        history_protected_min_tokens: int = 2_400,
        recent_protected_min_messages: int = 1,
        history_protected_min_messages: int = 1,
        adaptive_max_recent_messages: int = 120,
        adaptive_max_history_messages: int = 300,
        token_counter: TokenCounter | None = None,
    ) -> None:
        minimum_char_budget = len(MEMORY_GROUNDING_WITH_EVIDENCE) + len(
            QQ_BLOCKED_MEMORY_NOTE
        ) + 4
        if min(
            normal_budget,
            detail_budget,
            recent_budget,
            history_budget,
            context_char_budget,
            max_recent_messages,
            max_history_messages,
        ) <= 0:
            raise ValueError("memory budgets must be positive")
        if context_char_budget < minimum_char_budget:
            raise ValueError("memory context character budget is too small for safety policies")
        self._budgets = {"normal": normal_budget, "detail": detail_budget}
        self._recent_budget = recent_budget
        self._history_budget = history_budget
        self._context_char_budget = int(context_char_budget)
        self._max_recent_messages = int(max_recent_messages)
        self._max_history_messages = int(max_history_messages)
        protected_values = (
            recent_protected_min_tokens,
            history_protected_min_tokens,
            recent_protected_min_messages,
            history_protected_min_messages,
        )
        if min(protected_values) < 0:
            raise ValueError("adaptive protected minima must be non-negative")
        if min(adaptive_max_recent_messages, adaptive_max_history_messages) <= 0:
            raise ValueError("adaptive message safety caps must be positive")
        if recent_protected_min_messages > adaptive_max_recent_messages:
            raise ValueError("recent protected message minimum exceeds adaptive safety cap")
        if history_protected_min_messages > adaptive_max_history_messages:
            raise ValueError("history protected message minimum exceeds adaptive safety cap")
        if adaptive_enabled and (
            recent_protected_min_tokens + history_protected_min_tokens
            > normal_budget
        ):
            raise ValueError("adaptive protected token minima exceed normal memory budget")
        self._adaptive_enabled = bool(adaptive_enabled)
        self._recent_protected_min_tokens = int(recent_protected_min_tokens)
        self._history_protected_min_tokens = int(history_protected_min_tokens)
        self._recent_protected_min_messages = int(recent_protected_min_messages)
        self._history_protected_min_messages = int(history_protected_min_messages)
        self._adaptive_max_recent_messages = int(adaptive_max_recent_messages)
        self._adaptive_max_history_messages = int(adaptive_max_history_messages)
        self._token_counter = token_counter or self._fallback_token_count
        self._token_counter_is_additive = token_counter is None

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
        if self._adaptive_enabled:
            return self._pack_adaptive(
                mode,
                available_input=available_input,
                target_message_id=target_message_id,
                recent_messages=recent_messages,
                evidence_segments=evidence_segments,
                facts=facts,
                summaries=summaries,
            )
        return self._pack_legacy(
            mode,
            available_input=available_input,
            target_message_id=target_message_id,
            recent_messages=recent_messages,
            evidence_segments=evidence_segments,
            facts=facts,
            summaries=summaries,
        )

    def _pack_legacy(
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
        history_requested = bool(evidence_segments or facts or summaries)
        blocked_input_present = any(message.blocked for message in recent_messages) or any(
            segment.blocked_output_present or any(message.blocked for message in segment.messages)
            for segment in evidence_segments
        )
        reserved_policy = (
            MEMORY_GROUNDING_MINIMAL
            if history_requested
            else MEMORY_GROUNDING_NO_EVIDENCE
        )
        if blocked_input_present:
            reserved_policy = f"{QQ_BLOCKED_MEMORY_NOTE}\n\n{reserved_policy}"
        reserved_policy_tokens = self._estimate(reserved_policy)
        recent_char_budget = max(0, self._context_char_budget - len(reserved_policy) - 2)
        if history_requested:
            recent_char_budget //= 3
        recent = self._select_recent(
            recent_messages,
            target_message_id,
            max(0, min(budget, self._recent_budget) - reserved_policy_tokens),
            recent_char_budget,
        )
        recent_blocks = tuple(self._render_recent(message) for message in recent)
        block_token_cache: dict[str, int] = {}

        def fits_history(history_blocks: Sequence[str]) -> bool:
            combined_text = "\n\n".join([*recent_blocks, *history_blocks])
            if len(combined_text) > self._context_char_budget:
                return False
            if not self._token_counter_is_additive:
                return self._fits_history(
                    budget=budget,
                    recent_blocks=recent_blocks,
                    history_blocks=history_blocks,
                )

            def block_tokens(block: str) -> int:
                cached = block_token_cache.get(block)
                if cached is None:
                    cached = self._estimate(block)
                    block_token_cache[block] = cached
                return cached

            recent_tokens = sum(block_tokens(block) for block in recent_blocks)
            history_tokens = sum(block_tokens(block) for block in history_blocks)
            return (
                history_tokens <= self._history_budget
                and recent_tokens + history_tokens <= budget
            )

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
            if not fits_history(
                [*evidence_policy_blocks, *segment_blocks, block]
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
            if fits_history(
                [
                    *evidence_policy_blocks,
                    *fact_blocks,
                    *segment_blocks,
                    block,
                ]
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
            if not fits_history(
                [
                    *evidence_policy_blocks,
                    *fact_blocks,
                    *segment_blocks,
                    block,
                ]
            ):
                continue
            selected_segments.append(candidate_segment)
            segment_blocks.append(block)
            selected_evidence_ids.update(
                message.source_msg_id for message in candidate_segment.messages
            )

        selected_summaries: list[MemorySummary] = []
        summary_blocks: list[str] = []
        # A summary is normally a retrieval supplement; when the summary is
        # explicitly relevant (time-range overlap / summary intent) it may
        # stand alone so dated/summary questions still reach the summary layer.
        if selected_segments or any(summary.relevant for summary in summaries):
            for summary in summaries:
                if not summary.relevant:
                    continue
                block = f"Relevant summary (sources: {', '.join(summary.source_msg_ids)}): {summary.text}"
                if fits_history(
                    [
                        *evidence_policy_blocks,
                        *fact_blocks,
                        *segment_blocks,
                        *summary_blocks,
                        block,
                    ]
                ):
                    selected_summaries.append(summary)
                    summary_blocks.append(block)

        has_selected_evidence = bool(
            selected_facts or selected_segments or selected_summaries
        )
        full_grounding_policy = (
            MEMORY_GROUNDING_WITH_EVIDENCE
            if has_selected_evidence
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
            if fits_history([*history_content_blocks, full_grounding_policy])
            else MEMORY_GROUNDING_MINIMAL
        )
        blocks = [
            *policy_blocks,
            grounding_policy,
            *fact_blocks,
            *segment_blocks,
            *summary_blocks,
            *recent_blocks,
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
        total_estimated_tokens = self._estimate(text)
        if (
            len(text) > self._context_char_budget
            or history_estimated_tokens > self._history_budget
            or total_estimated_tokens > budget
        ):
            raise ValueError("packed memory context exceeds a hard budget")
        return PackedMemoryContext(
            mode=mode,
            budget=budget,
            estimated_tokens=total_estimated_tokens,
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

    def _pack_adaptive(
        self,
        mode: PackMode,
        *,
        available_input: int,
        target_message_id: str | None,
        recent_messages: Sequence[EvidenceMessage],
        evidence_segments: Sequence[EvidenceSegment],
        facts: Sequence[MemoryFact],
        summaries: Sequence[MemorySummary],
    ) -> PackedMemoryContext:
        if mode not in self._budgets:
            raise ValueError(f"unknown pack mode: {mode}")
        budget = min(self._budgets[mode], max(0, available_input))
        history_requested = bool(evidence_segments or facts or summaries)
        blocked_output_present = any(message.blocked for message in recent_messages) or any(
            segment.blocked_output_present or any(message.blocked for message in segment.messages)
            for segment in evidence_segments
        )
        policy_blocks = [QQ_BLOCKED_MEMORY_NOTE] if blocked_output_present else []
        reserve_policy = (
            MEMORY_GROUNDING_MINIMAL
            if history_requested
            else MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL
        )
        block_token_cache: dict[str, int] = {}

        def block_tokens(block: str) -> int:
            cached = block_token_cache.get(block)
            if cached is None:
                cached = self._estimate(block)
                block_token_cache[block] = cached
            return cached

        def joined(recent_blocks: Sequence[str], history_blocks: Sequence[str], policy: str) -> str:
            # New messages belong closest to the target instruction, so the
            # packed context renders policy/history first and recent last.
            return "\n\n".join([*policy_blocks, policy, *history_blocks, *recent_blocks])

        def fits(recent_blocks: Sequence[str], history_blocks: Sequence[str], policy: str = reserve_policy) -> bool:
            if self._token_counter_is_additive:
                blocks = [*recent_blocks, *policy_blocks, policy, *history_blocks]
                rendered_length = sum(len(block) for block in blocks) + max(
                    0, len(blocks) - 1
                ) * 2
                return (
                    rendered_length <= self._context_char_budget
                    and sum(block_tokens(block) for block in blocks) <= budget
                )
            value = joined(recent_blocks, history_blocks, policy)
            return len(value) <= self._context_char_budget and self._estimate(value) <= budget

        if not fits((), ()):
            policy = self._shortest_fitting_policy(
                budget=budget,
                policies=(
                    MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL,
                    MEMORY_GROUNDING_MINIMAL,
                ),
            )
            text = policy
            degradation_reason = "policy_only" if policy else "policy_unrepresentable"
            if blocked_output_present:
                blocked_fallbacks = tuple(
                    f"{QQ_BLOCKED_MEMORY_NOTE}\n\n{candidate}"
                    for candidate in (
                        MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL,
                        MEMORY_GROUNDING_MINIMAL,
                    )
                ) + (QQ_BLOCKED_MEMORY_NOTE,)
                text = self._shortest_fitting_policy(
                    budget=budget,
                    policies=blocked_fallbacks,
                )
                policy = next(
                    (
                        candidate
                        for candidate in (
                            MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL,
                            MEMORY_GROUNDING_MINIMAL,
                        )
                        if text.endswith(candidate)
                    ),
                    "",
                )
                degradation_reason = (
                    "blocked_policy_only"
                    if policy
                    else "blocked_note_only"
                    if text
                    else "blocked_policy_unrepresentable"
                )
            return PackedMemoryContext(
                mode=mode,
                budget=budget,
                estimated_tokens=self._estimate(text),
                text=text,
                blocked_output_present=blocked_output_present,
                grounding_policy=policy,
                history_estimated_tokens=self._estimate(text),
                adaptive_enabled=True,
                degradation_reason=degradation_reason,
            )

        selected_segments: list[EvidenceSegment] = []
        selected_facts: list[MemoryFact] = []
        selected_summaries: list[MemorySummary] = []
        selected_recent: list[EvidenceMessage] = []
        history_blocks: list[str] = []
        summary_blocks: list[str] = []
        selected_history_ids: set[str] = set()
        history_message_count = 0

        def add_segment(segment: EvidenceSegment) -> bool:
            nonlocal history_message_count
            candidate = self._prepare_segment(
                segment,
                duplicate_ids=selected_history_ids
                | {message.source_msg_id for message in selected_recent},
            )
            if candidate is None:
                return True
            if history_message_count + len(candidate.messages) > self._adaptive_max_history_messages:
                return False
            block = self._render_segment(candidate)
            recent_blocks = [self._render_recent(message) for message in selected_recent]
            if not fits(recent_blocks, [*history_blocks, block, *summary_blocks]):
                return False
            selected_segments.append(candidate)
            history_blocks.append(block)
            selected_history_ids.update(message.source_msg_id for message in candidate.messages)
            history_message_count += len(candidate.messages)
            return True

        # A pin is an immutable evidence unit in adaptive mode. If the next pin
        # cannot fit, optional lower-priority history must not bypass it.
        pins_fit = True
        for segment in (item for item in evidence_segments if item.pinned):
            if not add_segment(segment):
                pins_fit = False
                break

        recent_candidates = tuple(
            message
            for message in recent_messages
            if message.source_msg_id != target_message_id
            and message.source_msg_id not in selected_history_ids
        )[-self._adaptive_max_recent_messages :]

        # Protect the newest contiguous suffix. A non-fitting newest row stops
        # the suffix instead of allowing an older row to leapfrog it.
        for message in reversed(recent_candidates):
            current_tokens = self._estimate(
                "\n\n".join(self._render_recent(item) for item in selected_recent)
            )
            if (
                len(selected_recent) >= self._recent_protected_min_messages
                and current_tokens >= self._recent_protected_min_tokens
            ):
                break
            candidate_recent = [message, *selected_recent]
            candidate_blocks = [self._render_recent(item) for item in candidate_recent]
            if not fits(candidate_blocks, history_blocks):
                break
            selected_recent = candidate_recent

        def history_content_tokens() -> int:
            if self._token_counter_is_additive:
                return sum(
                    block_tokens(block)
                    for block in (*history_blocks, *summary_blocks)
                )
            return self._estimate(
                "\n\n".join([*history_blocks, *summary_blocks])
            )

        ordered_facts = sorted(facts, key=lambda item: (-item.score, item.text))
        remaining_segments = [item for item in evidence_segments if not item.pinned]

        def add_fact(fact: MemoryFact) -> bool:
            block = f"Memory fact (sources: {', '.join(fact.source_msg_ids)}): {fact.text}"
            recent_blocks = [self._render_recent(message) for message in selected_recent]
            if not fits(recent_blocks, [*history_blocks, block, *summary_blocks]):
                return False
            selected_facts.append(fact)
            history_blocks.append(block)
            return True

        # Establish the history floor before either section consumes optional
        # shared capacity. Facts retain their legacy priority over raw segments.
        if history_requested and pins_fit:
            for fact in ordered_facts:
                if (
                    history_message_count >= self._history_protected_min_messages
                    and history_content_tokens() >= self._history_protected_min_tokens
                ):
                    break
                if not add_fact(fact):
                    break
            for segment in tuple(remaining_segments):
                if (
                    history_message_count >= self._history_protected_min_messages
                    and history_content_tokens() >= self._history_protected_min_tokens
                ):
                    break
                if not add_segment(segment):
                    break
                remaining_segments.remove(segment)

        # Explicitly relevant summaries answer summary/time-window intents and
        # must not be crowded out by optional broad facts or raw segments.
        # Pinned evidence and the protected history floor still retain their
        # safety/coverage priority.
        if selected_segments or any(summary.relevant for summary in summaries):
            for summary in summaries:
                if not summary.relevant:
                    continue
                block = f"Relevant summary (sources: {', '.join(summary.source_msg_ids)}): {summary.text}"
                recent_blocks = [
                    self._render_recent(message) for message in selected_recent
                ]
                if fits(
                    recent_blocks,
                    [*history_blocks, *summary_blocks, block],
                ):
                    selected_summaries.append(summary)
                    summary_blocks.append(block)

        # Historical evidence gets first use of unclaimed shared capacity on a
        # history turn. Once it is exhausted, recent context borrows the rest.
        if pins_fit:
            for fact in ordered_facts:
                if fact in selected_facts:
                    continue
                if not add_fact(fact):
                    continue
            for segment in remaining_segments:
                add_segment(segment)

        selected_recent_ids = {message.source_msg_id for message in selected_recent}
        for message in reversed(recent_candidates):
            if (
                message.source_msg_id in selected_recent_ids
                or message.source_msg_id in selected_history_ids
            ):
                continue
            candidate_recent = [message, *selected_recent]
            candidate_blocks = [self._render_recent(item) for item in candidate_recent]
            if not fits(candidate_blocks, [*history_blocks, *summary_blocks]):
                break
            selected_recent = candidate_recent
            selected_recent_ids.add(message.source_msg_id)

        recent_blocks = [self._render_recent(message) for message in selected_recent]
        # Canonical render order so the context builder can reconstruct the
        # same block sequence for trimming: facts -> segments -> summaries.
        packed_history_blocks = [
            *(f"Memory fact (sources: {', '.join(fact.source_msg_ids)}): {fact.text}" for fact in selected_facts),
            *(self._render_segment(segment) for segment in selected_segments),
            *(
                f"Relevant summary (sources: {', '.join(summary.source_msg_ids)}): {summary.text}"
                for summary in selected_summaries
            ),
        ]
        has_selected_evidence = bool(
            selected_facts or selected_segments or selected_summaries
        )
        full_policy = (
            MEMORY_GROUNDING_WITH_EVIDENCE
            if has_selected_evidence
            else MEMORY_GROUNDING_NO_EVIDENCE
        )
        fallback_policy = (
            MEMORY_GROUNDING_MINIMAL
            if has_selected_evidence
            else MEMORY_GROUNDING_NO_EVIDENCE_MINIMAL
        )
        if fits(recent_blocks, packed_history_blocks, full_policy):
            grounding_policy = full_policy
        elif fits(recent_blocks, packed_history_blocks, fallback_policy):
            grounding_policy = fallback_policy
        else:
            grounding_policy = reserve_policy
        text = joined(recent_blocks, packed_history_blocks, grounding_policy)
        recent_tokens = self._estimate("\n\n".join(recent_blocks))
        history_tokens = self._estimate(
            "\n\n".join(
                [*policy_blocks, grounding_policy, *packed_history_blocks]
            )
        )
        total_tokens = self._estimate(text)
        if len(text) > self._context_char_budget or total_tokens > budget:
            raise ValueError("packed adaptive memory context exceeds a hard budget")

        recent_borrowed = recent_tokens > self._recent_protected_min_tokens
        history_borrowed = history_content_tokens() > self._history_protected_min_tokens
        spillover: Literal["none", "recent_to_history", "history_to_recent", "bidirectional"]
        if recent_borrowed and history_borrowed:
            spillover = "bidirectional"
        elif history_borrowed:
            spillover = "recent_to_history"
        elif recent_borrowed:
            spillover = "history_to_recent"
        else:
            spillover = "none"
        degradation_reason = ""
        if grounding_policy != full_policy:
            degradation_reason = "minimal_policy"
        elif history_requested and (
            history_message_count < self._history_protected_min_messages
            or history_content_tokens() < self._history_protected_min_tokens
        ):
            degradation_reason = "history_floor_limited"
        elif recent_candidates and (
            len(selected_recent) < self._recent_protected_min_messages
            or recent_tokens < self._recent_protected_min_tokens
        ):
            degradation_reason = "recent_floor_limited"

        source_ids = self._source_ids(
            selected_facts,
            selected_segments,
            selected_recent,
            selected_summaries,
        )
        return PackedMemoryContext(
            mode=mode,
            budget=budget,
            estimated_tokens=total_tokens,
            text=text,
            recent_messages=tuple(selected_recent),
            evidence_segments=tuple(selected_segments),
            facts=tuple(selected_facts),
            summaries=tuple(selected_summaries),
            source_msg_ids=source_ids,
            blocked_output_present=blocked_output_present,
            grounding_policy=grounding_policy,
            recent_estimated_tokens=recent_tokens,
            history_estimated_tokens=history_tokens,
            adaptive_enabled=True,
            spillover=spillover,
            degradation_reason=degradation_reason,
        )

    def _shortest_fitting_policy(
        self,
        *,
        budget: int,
        policies: Sequence[str],
    ) -> str:
        for policy in sorted(policies, key=len):
            if len(policy) <= self._context_char_budget and self._estimate(policy) <= budget:
                return policy
        return ""

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
        char_budget: int,
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
            candidate_text = "\n\n".join(candidate_blocks)
            if self._estimate(candidate_text) > budget or len(candidate_text) > char_budget:
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
            "Evidence - quoted chat data "
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
        if len("\n\n".join([*recent_blocks, *history_blocks])) > self._context_char_budget:
            return False
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
        utc_value = (
            value.replace(tzinfo=_SHANGHAI).astimezone(UTC)
            if value.tzinfo is None
            else value.astimezone(UTC)
        )
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
