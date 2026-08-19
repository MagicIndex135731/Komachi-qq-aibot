"""Synchronous V2 query-side pipeline composed from offline-testable stages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
from time import perf_counter
from typing import Protocol

from app.core.memory_context_packer import (
    EvidenceMessage,
    EvidenceSegment,
    MemoryContextPacker,
    MemoryFact,
    MemorySummary,
)
from app.core.memory_eligibility import eligible
from app.core.hybrid_memory_retriever import HybridRetrievalResult, MemoryScopeViolation
from app.core.memory_fact_ranking import temporal_recency_required
from app.core.memory_orchestrator import MemoryContextResult
from app.core.memory_query_resolver import RecentMemoryMessage, ResolvedMemoryQuery
from app.core.member_identity import GroupMemberIdentity

_PROFILE_MARKERS = ("画像", "介绍", "是什么样的人", "哪里人", "做什么的")


logger = logging.getLogger(__name__)


class QueryResolver(Protocol):
    def resolve(
        self,
        query: str,
        *,
        recent_messages: Sequence[RecentMemoryMessage],
        quoted_message: RecentMemoryMessage | None,
        now: datetime | None,
        group_members: Sequence[GroupMemberIdentity] = (),
        excluded_member_ids: set[int] | frozenset[int] = frozenset(),
        group_id: int | None = None,
        requester_id: int | None = None,
    ) -> ResolvedMemoryQuery: ...


class Retriever(Protocol):
    def retrieve(self, *, group_id: int, resolved_query: object) -> object: ...


class Expander(Protocol):
    def expand(self, *, group_id: int, candidates: Sequence[object], mode: str) -> Sequence[object]: ...


FactLoader = Callable[..., Sequence[MemoryFact]]
SummaryLoader = Callable[..., Sequence[MemorySummary]]
SourceScopeValidator = Callable[[int, tuple[str, ...]], bool]
MemberLoader = Callable[[int], Sequence[GroupMemberIdentity]]
CandidateFilter = Callable[..., Sequence[object]]


@dataclass(frozen=True, slots=True)
class MemoryV2Request:
    group_id: int
    query: str
    recent_messages: tuple[EvidenceMessage, ...]
    quoted_message: EvidenceMessage | None
    target_message_id: str | None
    available_input: int
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryV2EvaluationTrace:
    result: MemoryContextResult
    resolved_query: ResolvedMemoryQuery
    retrieved_source_msg_ids: tuple[str, ...]
    retrieved_source_units: tuple[tuple[str, ...], ...]
    candidate_scores: tuple[tuple[int, float], ...]
    attempted_channels: tuple[str, ...] = ()
    failed_channels: tuple[str, ...] = ()
    channel_candidate_counts: tuple[tuple[str, int], ...] = ()
    expansion_mode: str = "legacy"
    expansion_reasons: tuple[str, ...] = ()
    phase_timings_ms: tuple[tuple[str, float], ...] = ()


class MemoryV2ContextProvider:
    """Run resolver -> retrieval -> expansion -> packing as one fallback unit."""

    def __init__(
        self,
        *,
        resolver: QueryResolver,
        retriever: Retriever,
        expander: Expander,
        packer: MemoryContextPacker,
        source_scope_validator: SourceScopeValidator,
        fact_loader: FactLoader | None = None,
        summary_loader: SummaryLoader | None = None,
        member_loader: MemberLoader | None = None,
        candidate_filter: CandidateFilter | None = None,
        max_direct_replies_per_source: int = 2,
        excluded_member_ids: set[int] | frozenset[int] = frozenset(),
        historical_no_hit_omit_recent: bool = False,
        observability_route: str = "",
        adaptive_context_enabled: bool = False,
        compact_candidate_limit: int = 150,
        recent_intent_candidate_limit: int | None = None,
    ) -> None:
        self._resolver = resolver
        self._retriever = retriever
        self._expander = expander
        self._packer = packer
        self._source_scope_validator = source_scope_validator
        self._fact_loader = fact_loader or (lambda **_: ())
        self._summary_loader = summary_loader or (lambda **_: ())
        self._member_loader = member_loader
        self._candidate_filter = candidate_filter
        if max_direct_replies_per_source < 0:
            raise ValueError("direct reply limit cannot be negative")
        self._max_direct_replies_per_source = max_direct_replies_per_source
        self._excluded_member_ids = frozenset(int(item) for item in excluded_member_ids)
        self._historical_no_hit_omit_recent = bool(historical_no_hit_omit_recent)
        self._observability_route = str(observability_route).strip()
        if compact_candidate_limit <= 0:
            raise ValueError("compact candidate limit must be positive")
        self._adaptive_context_enabled = bool(adaptive_context_enabled)
        self._compact_candidate_limit = int(compact_candidate_limit)
        if recent_intent_candidate_limit is not None and recent_intent_candidate_limit <= 0:
            raise ValueError("recent intent candidate limit must be positive")
        self._recent_intent_candidate_limit = (
            int(recent_intent_candidate_limit)
            if recent_intent_candidate_limit is not None
            else None
        )

    def __call__(self, request: MemoryV2Request) -> MemoryContextResult:
        return self.evaluate(request).result

    def evaluate(self, request: MemoryV2Request) -> MemoryV2EvaluationTrace:
        """Run V2 and expose an in-memory, content-free evaluation trace."""
        evaluation_started = perf_counter()
        self._validate_recent_scope(request)
        resolve_kwargs = {
            "recent_messages": request.recent_messages,
            "quoted_message": request.quoted_message,
            "now": request.now,
            "group_id": request.group_id,
            "requester_id": getattr(request, "current_user_id", None),
        }
        if self._member_loader is not None:
            resolve_kwargs["group_members"] = tuple(self._member_loader(request.group_id))
            resolve_kwargs["excluded_member_ids"] = self._excluded_member_ids
        resolve_started = perf_counter()
        resolved = self._resolver.resolve(request.query, **resolve_kwargs)
        resolve_ms = (perf_counter() - resolve_started) * 1000
        retrieval_started = perf_counter()
        if self._should_skip_retrieval(resolved):
            # Plain chit-chat/general questions without history or fact intent
            # do not need memory retrieval; keep the reply grounded in recent
            # context only and avoid pulling irrelevant facts/raw noise.
            retrieval_result = HybridRetrievalResult(())
            candidates: tuple[object, ...] = ()
            retrieval_skipped = True
        else:
            retrieval_result = self._retriever.retrieve(
                group_id=request.group_id,
                resolved_query=resolved,
            )
            retrieval_skipped = False
            candidates = (
                ()
                if bool(getattr(retrieval_result, "all_channels_failed", False))
                else tuple(getattr(retrieval_result, "candidates"))
            )
        # Retrieval accelerators are optional. If every channel is unavailable,
        # keep the request inside the scoped V3 path and emit explicit
        # no-evidence grounding instead of falling into unscoped legacy memory.
        if self._candidate_filter is not None:
            candidates = tuple(
                self._candidate_filter(
                    request=request,
                    resolved_query=resolved,
                    candidates=candidates,
                )
            )
        expansion_candidates, expansion_mode, expansion_reasons = (
            self._select_expansion_candidates(
                candidates=candidates,
                retrieval_result=retrieval_result,
                needs_history=resolved.needs_history,
                profile_intent=(
                    resolved.answer_mode == "current_fact"
                    and any(
                        marker in (resolved.original_query or "")
                        for marker in _PROFILE_MARKERS
                    )
                ),
                recent_intent=temporal_recency_required(
                    query=str(resolved.original_query or "")
                ),
            )
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        mode = "detail" if resolved.needs_detail else "normal"
        expansion_started = perf_counter()
        expanded_segments = tuple(
            self._expander.expand(
                group_id=request.group_id,
                candidates=expansion_candidates,
                mode=mode,
            )
        )
        segments = self._pin_required_segments(
            self._eligible_segments(expanded_segments, resolved),
            resolved,
        )
        if (
            self._adaptive_context_enabled
            and resolved.needs_history
            and expansion_mode == "compact"
            and not segments
            and len(expansion_candidates) < len(candidates)
        ):
            expansion_mode = "expanded"
            expansion_reasons = ("post_eligibility_no_evidence",)
            expanded_segments = tuple(
                self._expander.expand(
                    group_id=request.group_id,
                    candidates=candidates,
                    mode=mode,
                )
            )
            segments = self._pin_required_segments(
                self._eligible_segments(expanded_segments, resolved),
                resolved,
            )
        expansion_ms = (perf_counter() - expansion_started) * 1000
        derived_started = perf_counter()
        facts = (
            ()
            if retrieval_skipped
            else tuple(
                self._fact_loader(
                    group_id=request.group_id,
                    resolved_query=resolved,
                )
            )
        )
        summaries = (
            ()
            if retrieval_skipped
            else tuple(
                self._summary_loader(
                    group_id=request.group_id,
                    resolved_query=resolved,
                )
            )
        )
        self._validate_derived_scope(
            group_id=request.group_id,
            facts=facts,
            summaries=summaries,
        )
        derived_ms = (perf_counter() - derived_started) * 1000
        packing_started = perf_counter()
        packed = self._packer.pack(
            mode,
            available_input=request.available_input,
            target_message_id=request.target_message_id,
            recent_messages=(
                ()
                if self._historical_no_hit_omit_recent
                and resolved.needs_history
                and resolved.time_range is not None
                and not segments
                else request.recent_messages
            ),
            evidence_segments=segments,
            facts=facts,
            summaries=summaries,
        )
        if (
            self._historical_no_hit_omit_recent
            and resolved.needs_history
            and resolved.time_range is not None
            and packed.recent_messages
            and not packed.evidence_segments
        ):
            packed = self._packer.pack(
                mode,
                available_input=request.available_input,
                target_message_id=request.target_message_id,
                recent_messages=(),
                evidence_segments=segments,
                facts=facts,
                summaries=summaries,
            )
        packing_ms = (perf_counter() - packing_started) * 1000
        if packed.source_msg_ids and not self._source_scope_validator(
            request.group_id,
            packed.source_msg_ids,
        ):
            raise MemoryScopeViolation("packed memory source scope mismatch")
        result = MemoryContextResult(
            group_id=request.group_id,
            packed_context=packed,
            selected_source_msg_ids=packed.source_msg_ids,
            estimated_tokens=packed.estimated_tokens,
            mode="v2",
        )
        total_ms = (perf_counter() - evaluation_started) * 1000
        if self._observability_route:
            expanded_source_count = sum(
                len(segment.messages) for segment in expanded_segments
            )
            eligible_source_count = sum(
                len(segment.messages) for segment in segments
            )
            logger.info(
                "memory_query_metrics route=%s group_id=%s answer_mode=%s "
                "coverage=%s has_subject=%s subject_ambiguous=%s has_time=%s "
                "topic_extraction=%s topic_terms=%s "
                "adaptive_enabled=%s expansion_mode=%s expansion_reasons=%s "
                "attempted_channels=%s failed_channels=%s channel_counts=%s "
                "pin_counts=%s pin_overflow=%s "
                "candidate_units=%s expanded_sources=%s rejected_sources=%s "
                "selected_source_count=%s recent_messages=%s history_messages=%s "
                "effective_budget=%s recent_tokens=%s history_tokens=%s total_tokens=%s "
                "spillover=%s degradation_reason=%s "
                "resolve_ms=%.3f retrieval_ms=%.3f expansion_ms=%.3f "
                "derived_ms=%.3f packing_ms=%.3f total_ms=%.3f rewrite=%s",
                self._observability_route,
                request.group_id,
                resolved.answer_mode,
                resolved.coverage_mode,
                resolved.subject_ids is not None,
                resolved.subject_ids == (),
                resolved.time_range is not None,
                resolved.topic_extraction,
                len(resolved.topic_terms),
                self._adaptive_context_enabled,
                expansion_mode,
                json.dumps(list(expansion_reasons), separators=(",", ":")),
                json.dumps(
                    list(getattr(retrieval_result, "attempted_channels", ())),
                    separators=(",", ":"),
                ),
                json.dumps(
                    list(getattr(retrieval_result, "failed_channels", ())),
                    separators=(",", ":"),
                ),
                json.dumps(
                    list(
                        getattr(
                            retrieval_result,
                            "channel_candidate_counts",
                            (),
                        )
                    ),
                    separators=(",", ":"),
                ),
                json.dumps(
                    list(getattr(retrieval_result, "pin_counts", ())),
                    separators=(",", ":"),
                ),
                int(getattr(retrieval_result, "pin_overflow_count", 0)),
                len(candidates),
                expanded_source_count,
                max(0, expanded_source_count - eligible_source_count),
                len(packed.source_msg_ids),
                len(packed.recent_messages),
                sum(len(segment.messages) for segment in packed.evidence_segments),
                packed.budget,
                packed.recent_estimated_tokens,
                packed.history_estimated_tokens,
                packed.estimated_tokens,
                getattr(packed, "spillover", "none"),
                getattr(packed, "degradation_reason", ""),
                resolve_ms,
                retrieval_ms,
                expansion_ms,
                derived_ms,
                packing_ms,
                total_ms,
                int(bool(resolved.rewrite_used)),
            )
        retrieved_source_msg_ids = tuple(
            dict.fromkeys(
                str(source_id)
                for candidate in candidates
                for source_id in getattr(candidate, "source_msg_ids", ())
                if str(source_id)
            )
        )
        return MemoryV2EvaluationTrace(
            result=result,
            resolved_query=resolved,
            retrieved_source_msg_ids=retrieved_source_msg_ids,
            retrieved_source_units=tuple(
                tuple(
                    dict.fromkeys(
                        str(source_id)
                        for source_id in getattr(candidate, "source_msg_ids", ())
                        if str(source_id)
                    )
                )
                for candidate in candidates
                if str(getattr(candidate, "document_kind", "")) == "episode"
            ),
            candidate_scores=tuple(
                (int(getattr(candidate, "document_id")), float(getattr(candidate, "fused_score")))
                for candidate in candidates
            ),
            attempted_channels=tuple(
                str(channel)
                for channel in getattr(retrieval_result, "attempted_channels", ())
            ),
            failed_channels=tuple(
                str(channel)
                for channel in getattr(retrieval_result, "failed_channels", ())
            ),
            channel_candidate_counts=tuple(
                (str(channel), int(count))
                for channel, count in getattr(
                    retrieval_result,
                    "channel_candidate_counts",
                    (),
                )
            ),
            expansion_mode=expansion_mode,
            expansion_reasons=expansion_reasons,
            phase_timings_ms=(
                ("resolve", resolve_ms),
                ("retrieval", retrieval_ms),
                ("expansion", expansion_ms),
                ("derived", derived_ms),
                ("packing", packing_ms),
                ("total", total_ms),
            ),
        )

    def _select_expansion_candidates(
        self,
        *,
        candidates: Sequence[object],
        retrieval_result: object,
        needs_history: bool,
        profile_intent: bool = False,
        recent_intent: bool = False,
    ) -> tuple[tuple[object, ...], str, tuple[str, ...]]:
        available = tuple(candidates)
        if profile_intent:
            # Profile/画像 questions should not flood the context with hundreds
            # of raw segments; keep a small evidence window so profile facts
            # and recent messages survive packing.
            return available[:12], "compact", ("profile_intent",)
        if (
            self._recent_intent_candidate_limit is not None
            and recent_intent
            and available
        ):
            # "最近/现在" questions only need a bounded recency window; hundreds
            # of raw segments drown out the freshest facts and citations.
            return (
                available[: self._recent_intent_candidate_limit],
                "compact",
                ("recent_intent",),
            )
        if not self._adaptive_context_enabled or not needs_history:
            return available, "legacy", ()
        if not available:
            return (), "no_evidence", ("no_candidates",)

        compact = available[: self._compact_candidate_limit]
        failed_channels = tuple(
            str(channel)
            for channel in getattr(retrieval_result, "failed_channels", ())
        )
        has_strong_local_signal = any(
            str(getattr(candidate, "pin_reason", ""))
            in {"direct", "lexical", "semantic"}
            or len(set(getattr(candidate, "routes", ()))) >= 2
            for candidate in compact
        )
        reasons: list[str] = []
        if failed_channels:
            reasons.append("channel_failure")
        if not has_strong_local_signal:
            reasons.append("weak_channel_agreement")
        if reasons:
            return available, "expanded", tuple(reasons)
        return compact, "compact", ()

    @staticmethod
    def _should_skip_retrieval(resolved: ResolvedMemoryQuery) -> bool:
        """True for plain general questions that need no memory retrieval."""
        if resolved.answer_mode != "general_history":
            return False
        if resolved.needs_history or resolved.time_range is not None:
            return False
        if getattr(resolved, "preferred_fact_kinds", ()) or ():
            return False
        if getattr(resolved, "subject_ids", None) is not None:
            return False
        return True

    def _eligible_segments(
        self,
        segments: Sequence[EvidenceSegment],
        resolved: ResolvedMemoryQuery,
    ) -> tuple[EvidenceSegment, ...]:
        """Revalidate every expanded raw source against the resolved plan."""

        validated: list[EvidenceSegment] = []
        direct_reply_counts: dict[str, int] = {}
        for segment in segments:
            allowed_messages = tuple(
                message
                for message in segment.messages
                if not message.is_bot and eligible(message, resolved)
            )
            allowed_ids = {message.source_msg_id for message in allowed_messages}
            # A hit source is the provenance that authorized this segment. If
            # any hit fails closed, discard the segment instead of retaining
            # surrounding context under a different identity/time boundary.
            if any(
                source_id not in allowed_ids
                for source_id in segment.hit_source_msg_ids
            ):
                continue
            if not allowed_messages:
                continue
            hit_ids = frozenset(segment.hit_source_msg_ids)
            if segment.episode_id.startswith("raw:"):
                allowed_messages = tuple(
                    message
                    for message in allowed_messages
                    if (
                        message.source_msg_id in hit_ids
                        or message.reply_to_msg_id not in hit_ids
                        or self._consume_direct_reply_quota(
                            message.reply_to_msg_id,
                            direct_reply_counts,
                        )
                    )
                )
            if not allowed_messages:
                continue
            validated.append(
                replace(
                    segment,
                    messages=allowed_messages,
                    hit_source_msg_ids=tuple(
                        source_id
                        for source_id in segment.hit_source_msg_ids
                        if source_id in allowed_ids
                    ),
                    atomic_source_groups=tuple(
                        group
                        for group in segment.atomic_source_groups
                        if set(group) <= allowed_ids
                    ),
                )
            )
        return tuple(validated)

    def _consume_direct_reply_quota(
        self,
        parent_id: str | None,
        counts: dict[str, int],
    ) -> bool:
        if not parent_id or self._max_direct_replies_per_source <= 0:
            return False
        count = counts.get(parent_id, 0)
        if count >= self._max_direct_replies_per_source:
            return False
        counts[parent_id] = count + 1
        return True

    @staticmethod
    def _pin_required_segments(
        segments: Sequence[EvidenceSegment],
        resolved: ResolvedMemoryQuery,
    ) -> tuple[EvidenceSegment, ...]:
        """Protect direct references and mention hits from later truncation."""

        required_ids = frozenset(resolved.reference_msg_ids)
        pin_all = resolved.answer_mode == "mention"
        return tuple(
            replace(
                segment,
                pinned=(
                    segment.pinned
                    or pin_all
                    or bool(
                        required_ids.intersection(
                            message.source_msg_id
                            for message in segment.messages
                        )
                    )
                ),
            )
            for segment in segments
        )

    @staticmethod
    def _validate_recent_scope(request: MemoryV2Request) -> None:
        scoped = (*request.recent_messages,) + (
            (request.quoted_message,) if request.quoted_message is not None else ()
        )
        if any(
            message.group_id is None or int(message.group_id) != int(request.group_id)
            for message in scoped
        ):
            raise ValueError("memory recent snapshot scope mismatch")

    @staticmethod
    def _validate_derived_scope(
        *,
        group_id: int,
        facts: Sequence[MemoryFact],
        summaries: Sequence[MemorySummary],
    ) -> None:
        for item in (*facts, *summaries):
            if item.group_id is None or int(item.group_id) != int(group_id):
                raise MemoryScopeViolation("derived memory scope mismatch")
            if not item.source_msg_ids or any(not str(source_id) for source_id in item.source_msg_ids):
                raise MemoryScopeViolation("derived memory provenance is missing")
