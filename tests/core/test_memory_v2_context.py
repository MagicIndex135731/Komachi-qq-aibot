from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import logging

import pytest

from app.core.hybrid_memory_retriever import (
    FusedRetrievalCandidate,
    HybridRetrievalResult,
    MemoryScopeViolation,
)
from app.core.memory_context_packer import (
    EvidenceMessage,
    EvidenceSegment,
    PackedMemoryContext,
)
from app.core.memory_orchestrator import MemoryContextResult
from app.core.memory_query_resolver import ResolvedMemoryQuery, TimeRange
from app.core.memory_evidence_expander import MemoryEvidenceExpander
from app.core.memory_v2_context import MemoryV2ContextProvider, MemoryV2Request


@dataclass
class Resolver:
    detail: bool = False
    time_range: TimeRange | None = None
    answer_mode: str = "general_history"
    needs_history: bool = True
    subject_ids: tuple[str, ...] | None = None
    subject_binding: str = "unbound"
    preferred_fact_kinds: tuple[str, ...] = ()
    original_query: str | None = None

    def resolve(self, query, **kwargs):
        return ResolvedMemoryQuery(
            original_query=self.original_query or query,
            retrieval_query=query,
            needs_history=self.needs_history,
            needs_detail=self.detail,
            group_id=kwargs.get("group_id"),
            time_range=self.time_range,
            answer_mode=self.answer_mode,
            subject_ids=self.subject_ids,
            subject_binding=self.subject_binding,
            preferred_fact_kinds=self.preferred_fact_kinds,
        )


class Retriever:
    def retrieve(self, **_):
        return HybridRetrievalResult(())


class FailedRetriever:
    def retrieve(self, **_):
        return HybridRetrievalResult(
            (),
            failed_channels=("bm25", "vector"),
            attempted_channels=("bm25", "vector"),
        )


class TracedRetriever:
    def retrieve(self, **_):
        return HybridRetrievalResult(
            (
                FusedRetrievalCandidate(
                    document_id=2,
                    group_id=100,
                    document_kind="episode_summary",
                    episode_id=1,
                    source_msg_ids=("summary-only-provenance",),
                    start_at=datetime(2026, 7, 22, tzinfo=UTC),
                    end_at=datetime(2026, 7, 22, tzinfo=UTC),
                    routes=("bm25",),
                    route_ranks=(("bm25", 1),),
                    fused_score=0.8,
                ),
                FusedRetrievalCandidate(
                    document_id=1,
                    group_id=100,
                    document_kind="episode",
                    episode_id=1,
                    source_msg_ids=("evidence-1", "evidence-2"),
                    start_at=datetime(2026, 7, 22, tzinfo=UTC),
                    end_at=datetime(2026, 7, 22, tzinfo=UTC),
                    routes=("bm25",),
                    route_ranks=(("bm25", 1),),
                    fused_score=0.75,
                ),
            ),
            attempted_channels=("bm25", "vector"),
            channel_candidate_counts=(("bm25", 2), ("vector", 1)),
        )


class Expander:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def expand(self, *, mode, **_):
        self.modes.append(mode)
        return ()


class CandidateCapturingExpander(Expander):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_ids: list[tuple[int, ...]] = []

    def expand(self, *, candidates, mode, **_):
        self.modes.append(mode)
        self.candidate_ids.append(
            tuple(int(candidate.document_id) for candidate in candidates)
        )
        if not candidates:
            return ()
        source_id = str(candidates[0].source_msg_ids[0])
        return (
            EvidenceSegment(
                episode_id=f"raw:{source_id}",
                fused_score=1.0,
                messages=(
                    EvidenceMessage(
                        source_id,
                        "member",
                        "eligible evidence",
                        datetime(2026, 7, 22, tzinfo=UTC),
                        group_id=100,
                    ),
                ),
                hit_source_msg_ids=(source_id,),
            ),
        )


class OneSegmentExpander(Expander):
    def expand(self, *, mode, **_):
        self.modes.append(mode)
        return (
            EvidenceSegment(
                episode_id="raw-1",
                fused_score=1.0,
                messages=(
                    EvidenceMessage(
                        "history",
                        "member",
                        "oversized history",
                        datetime(2026, 7, 22, tzinfo=UTC),
                        group_id=100,
                    ),
                ),
            ),
        )


class Packer:
    def pack(self, mode, **_):
        return PackedMemoryContext(
            mode=mode,
            budget=100,
            estimated_tokens=5,
            text="packed",
            source_msg_ids=("m-1",),
        )


class CapturingPacker(Packer):
    def __init__(self) -> None:
        self.recent_messages = None

    def pack(self, mode, **kwargs):
        self.recent_messages = kwargs["recent_messages"]
        return super().pack(mode, **kwargs)


class CapturingSegmentsPacker(Packer):
    def __init__(self) -> None:
        self.evidence_segments = None

    def pack(self, mode, **kwargs):
        self.evidence_segments = kwargs["evidence_segments"]
        return super().pack(mode, **kwargs)


class PostPackDropPacker(Packer):
    def __init__(self) -> None:
        self.recent_calls: list[tuple[EvidenceMessage, ...]] = []

    def pack(self, mode, **kwargs):
        recent = tuple(kwargs["recent_messages"])
        self.recent_calls.append(recent)
        return PackedMemoryContext(
            mode=mode,
            budget=100,
            estimated_tokens=1,
            text="recent only" if recent else "no evidence",
            recent_messages=recent,
            evidence_segments=(),
            source_msg_ids=tuple(item.source_msg_id for item in recent),
        )


def request(*, group_id: int = 100) -> MemoryV2Request:
    recent = EvidenceMessage(
        "recent",
        "member",
        "hello",
        datetime(2026, 7, 23, tzinfo=UTC),
        group_id=group_id,
    )
    return MemoryV2Request(
        group_id=group_id,
        query="后来呢？",
        recent_messages=(recent,),
        quoted_message=None,
        target_message_id="target",
        available_input=1000,
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )


def test_v2_provider_runs_resolve_retrieve_expand_pack_and_returns_common_contract() -> None:
    expander = Expander()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            detail=True,
            answer_mode="current_fact",
            subject_ids=("20001",),
            subject_binding="explicit",
        ),
        retriever=Retriever(),
        expander=expander,
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    result = provider(request())

    assert isinstance(result, MemoryContextResult)
    assert result.group_id == 100
    assert result.mode == "v2"
    assert result.packed_context.text == "packed"
    assert result.selected_source_msg_ids == ("m-1",)
    assert result.resolved_answer_mode == "current_fact"
    assert result.resolved_subject_ids == ("20001",)
    assert result.resolved_subject_binding == "explicit"
    assert expander.modes == ["detail"]


@pytest.mark.parametrize(
    ("routes", "failed_channels", "expected_ids", "expected_mode"),
    (
        (("bm25",), (), (1,), "compact"),
        (("vector",), (), (1, 2, 3), "expanded"),
        (("bm25",), ("vector",), (1, 2, 3), "expanded"),
    ),
)
def test_adaptive_provider_expands_only_when_local_retrieval_is_weak(
    routes,
    failed_channels,
    expected_ids,
    expected_mode,
) -> None:
    class AdaptiveRetriever:
        def retrieve(self, **_):
            candidates = tuple(
                FusedRetrievalCandidate(
                    document_id=document_id,
                    group_id=100,
                    document_kind="raw_message_v3",
                    episode_id=None,
                    source_msg_ids=(f"source-{document_id}",),
                    start_at=datetime(2026, 7, 22, tzinfo=UTC),
                    end_at=datetime(2026, 7, 22, tzinfo=UTC),
                    routes=routes,
                    route_ranks=tuple(
                        (route, document_id) for route in routes
                    ),
                    fused_score=1.0 / document_id,
                    lexical_match_kind=("exact" if routes == ("bm25",) else "none"),
                    pin_reason=("lexical" if routes == ("bm25",) else None),
                )
                for document_id in range(1, 4)
            )
            return HybridRetrievalResult(
                candidates,
                attempted_channels=("bm25", "vector"),
                failed_channels=failed_channels,
            )

    expander = CandidateCapturingExpander()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=AdaptiveRetriever(),
        expander=expander,
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
        adaptive_context_enabled=True,
        compact_candidate_limit=1,
    )

    trace = provider.evaluate(request())

    assert expander.candidate_ids == [expected_ids]
    assert trace.expansion_mode == expected_mode


def test_adaptive_provider_retries_full_candidates_after_compact_fails_eligibility() -> None:
    candidates = tuple(
        FusedRetrievalCandidate(
            document_id=document_id,
            group_id=100,
            document_kind="raw_message_v3",
            episode_id=None,
            source_msg_ids=(f"source-{document_id}",),
            start_at=datetime(2026, 7, 22, tzinfo=UTC),
            end_at=datetime(2026, 7, 22, tzinfo=UTC),
            routes=("bm25",),
            route_ranks=(("bm25", document_id),),
            fused_score=1.0 / document_id,
            lexical_match_kind="exact",
            pin_reason="lexical" if document_id == 1 else None,
        )
        for document_id in range(1, 3)
    )

    class AdaptiveRetriever:
        def retrieve(self, **_):
            return HybridRetrievalResult(candidates, attempted_channels=("bm25",))

    class EligibilityExpander(Expander):
        def __init__(self) -> None:
            super().__init__()
            self.candidate_ids: list[tuple[int, ...]] = []

        def expand(self, *, candidates, mode, **_):
            self.candidate_ids.append(tuple(item.document_id for item in candidates))
            return tuple(
                EvidenceSegment(
                    episode_id=f"raw:{item.document_id}",
                    fused_score=item.fused_score,
                    messages=(
                        EvidenceMessage(
                            f"source-{item.document_id}",
                            "bot" if item.document_id == 1 else "human",
                            "generated" if item.document_id == 1 else "human evidence",
                            datetime(2026, 7, 22, tzinfo=UTC),
                            group_id=100,
                            is_bot=item.document_id == 1,
                        ),
                    ),
                    hit_source_msg_ids=(f"source-{item.document_id}",),
                )
                for item in candidates
            )

    expander = EligibilityExpander()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=AdaptiveRetriever(),
        expander=expander,
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
        adaptive_context_enabled=True,
        compact_candidate_limit=1,
    )

    trace = provider.evaluate(request())

    assert expander.candidate_ids == [(1,), (1, 2)]
    assert trace.expansion_mode == "expanded"
    assert trace.expansion_reasons == ("post_eligibility_no_evidence",)


def test_v2_provider_fails_closed_when_recent_snapshot_contains_other_group() -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=Retriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    wrong_recent = request(group_id=200).recent_messages
    with pytest.raises(ValueError, match="scope"):
        provider(replace(request(group_id=100), recent_messages=wrong_recent))


def test_v2_provider_rejects_cross_group_fact_or_unverified_final_source() -> None:
    from app.core.memory_context_packer import MemoryFact

    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=Retriever(),
        expander=Expander(),
        packer=Packer(),
        fact_loader=lambda **_: (
            MemoryFact("foreign", ("g200-secret",), group_id=200),
        ),
        source_scope_validator=lambda _group_id, _source_ids: False,
    )

    with pytest.raises(MemoryScopeViolation, match="scope"):
        provider(request())


def test_v2_provider_keeps_all_channel_failure_inside_safe_empty_v3_path() -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=FailedRetriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    result = provider(request())

    assert result.mode == "v2"
    assert result.selected_source_msg_ids == ("m-1",)


def test_historical_v3_no_hit_does_not_inject_unrelated_recent_messages() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            time_range=TimeRange(
                datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
                datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
            )
        ),
        retriever=Retriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    provider(request())

    assert packer.recent_messages == ()


def test_historical_v3_no_hit_keeps_recent_without_time_range() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=Retriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    provider(request())

    assert packer.recent_messages == request().recent_messages


def test_historical_v3_post_validation_empty_does_not_inject_recent() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            time_range=TimeRange(
                datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
                datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
            )
        ),
        retriever=TracedRetriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    provider(request())

    assert packer.recent_messages == ()


def test_historical_v3_post_pack_empty_retries_without_recent() -> None:
    packer = PostPackDropPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            time_range=TimeRange(
                datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
                datetime(2026, 7, 22, 16, 0, tzinfo=UTC),
            )
        ),
        retriever=TracedRetriever(),
        expander=OneSegmentExpander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    result = provider(request())

    assert len(packer.recent_calls) == 2
    assert packer.recent_calls[0]
    assert packer.recent_calls[1] == ()
    assert result.packed_context.recent_messages == ()


def test_plain_general_query_skips_retrieval_and_keeps_recent() -> None:
    class ExplodingRetriever:
        def retrieve(self, **_):
            raise AssertionError("retrieval must be skipped for plain general chat")

    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            needs_history=False,
            answer_mode="general_history",
            subject_ids=None,
        ),
        retriever=ExplodingRetriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    result = provider(request())

    assert packer.recent_messages == request().recent_messages
    assert result.packed_context.evidence_segments == ()


def test_profile_intent_queries_cap_expansion_candidates() -> None:
    expander = CandidateCapturingExpander()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            needs_history=False,
            answer_mode="current_fact",
            subject_ids=("100",),
            original_query="给出阿渣的完整个人画像",
        ),
        retriever=TracedRetriever(),
        expander=expander,
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
        adaptive_context_enabled=True,
    )

    trace = provider.evaluate(request())

    assert trace.expansion_mode == "compact"
    assert "profile_intent" in trace.expansion_reasons
    assert len(expander.candidate_ids[0]) == 2


def test_temporal_current_fact_excludes_prior_bot_answers_from_recent_evidence() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            needs_history=True,
            answer_mode="current_fact",
            subject_ids=("20001",),
            original_query="阿渣最近在做什么？",
        ),
        retriever=Retriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
    )
    base_request = request()
    user_message = base_request.recent_messages[0]
    stale_bot_answer = replace(
        user_message,
        source_msg_id="bot-old",
        speaker="bot",
        content="阿渣仍在做旧项目",
        is_bot=True,
    )

    provider(replace(base_request, recent_messages=(user_message, stale_bot_answer)))

    assert packer.recent_messages == (user_message,)


def test_non_temporal_query_keeps_prior_bot_messages_for_conversation_continuity() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(
            needs_history=True,
            answer_mode="current_fact",
            subject_ids=("20001",),
            original_query="阿渣喜欢什么？",
        ),
        retriever=Retriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
    )
    base_request = request()
    stale_bot_answer = replace(
        base_request.recent_messages[0],
        source_msg_id="bot-old",
        is_bot=True,
    )

    provider(
        replace(
            base_request,
            recent_messages=(*base_request.recent_messages, stale_bot_answer),
        )
    )

    assert packer.recent_messages == (*base_request.recent_messages, stale_bot_answer)


def test_v2_evaluation_trace_exposes_only_ids_and_resolver_metrics() -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=TracedRetriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    trace = provider.evaluate(request())

    assert trace.result.mode == "v2"
    assert trace.retrieved_source_msg_ids == (
        "summary-only-provenance",
        "evidence-1",
        "evidence-2",
    )
    assert trace.retrieved_source_units == (("evidence-1", "evidence-2"),)
    assert trace.candidate_scores == ((2, 0.8), (1, 0.75))
    assert trace.attempted_channels == ("bm25", "vector")
    assert trace.failed_channels == ()
    assert trace.channel_candidate_counts == (("bm25", 2), ("vector", 1))
    assert trace.resolved_query.retrieval_query
    phase_timings = dict(trace.phase_timings_ms)
    assert set(phase_timings) == {
        "resolve",
        "retrieval",
        "expansion",
        "derived",
        "packing",
        "total",
    }
    assert all(value >= 0 for value in phase_timings.values())


def test_evaluation_candidate_filter_runs_before_expansion_and_trace() -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=TracedRetriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
        candidate_filter=lambda **values: tuple(values["candidates"])[1:],
    )

    trace = provider.evaluate(request())

    assert trace.retrieved_source_msg_ids == ("evidence-1", "evidence-2")
    assert trace.candidate_scores == ((1, 0.75),)


def test_v3_observability_logs_metrics_without_query_or_message_content(
    caplog,
) -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=TracedRetriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
        observability_route="raw_v3",
    )
    sensitive_query = "private-query-marker"
    sensitive_message = "private-message-marker"
    observed_request = replace(
        request(),
        query=sensitive_query,
        recent_messages=(
            replace(
                request().recent_messages[0],
                content=sensitive_message,
            ),
        ),
    )

    with caplog.at_level(logging.INFO, logger="app.core.memory_v2_context"):
        provider.evaluate(observed_request)

    metrics = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("memory_query_metrics ")
    )
    assert "route=raw_v3" in metrics
    assert "attempted_channels=[\"bm25\",\"vector\"]" in metrics
    assert "channel_counts=[[\"bm25\",2],[\"vector\",1]]" in metrics
    assert "selected_source_count=" in metrics
    assert "selected_source_ids=" not in metrics
    assert "rewrite=0" in metrics
    assert "candidate_units=2" in metrics
    assert "recent_messages=0" in metrics
    assert "history_messages=0" in metrics
    assert "recent_tokens=0" in metrics
    assert "history_tokens=0" in metrics
    assert sensitive_query not in metrics
    assert sensitive_message not in metrics


def test_expanded_sources_are_revalidated_against_subject_time_and_group() -> None:
    class ScopedResolver:
        def resolve(self, query, **_):
            return ResolvedMemoryQuery(
                original_query=query,
                retrieval_query=query,
                needs_history=True,
                group_id=100,
                subject_ids=("42",),
                answer_mode="dated_history",
                time_range=TimeRange(
                    start=datetime(2026, 7, 22, tzinfo=UTC),
                    end=datetime(2026, 7, 24, tzinfo=UTC),
                ),
            )

    class MixedExpander:
        def expand(self, **_):
            return (
                EvidenceSegment(
                    "raw:eligible",
                    1.0,
                    (
                        EvidenceMessage(
                            "eligible",
                            "Target",
                            "inside",
                            datetime(2026, 7, 23, tzinfo=UTC),
                            group_id=100,
                            user_id=42,
                        ),
                    ),
                    hit_source_msg_ids=("eligible",),
                ),
                EvidenceSegment(
                    "raw:wrong-subject",
                    100.0,
                    (
                        EvidenceMessage(
                            "wrong-subject",
                            "Other",
                            "must not pass",
                            datetime(2026, 7, 23, tzinfo=UTC),
                            group_id=100,
                            user_id=99,
                        ),
                    ),
                    hit_source_msg_ids=("wrong-subject",),
                ),
                EvidenceSegment(
                    "raw:wrong-time",
                    100.0,
                    (
                        EvidenceMessage(
                            "wrong-time",
                            "Target",
                            "must not pass",
                            datetime(2026, 7, 25, tzinfo=UTC),
                            group_id=100,
                            user_id=42,
                        ),
                    ),
                    hit_source_msg_ids=("wrong-time",),
                ),
                EvidenceSegment(
                    "raw:deleted",
                    100.0,
                    (
                        EvidenceMessage(
                            "deleted",
                            "Target",
                            "must not pass",
                            datetime(2026, 7, 23, tzinfo=UTC),
                            group_id=100,
                            user_id=42,
                            delivery_state="deleted",
                        ),
                    ),
                    hit_source_msg_ids=("deleted",),
                ),
            )

    packer = CapturingSegmentsPacker()
    provider = MemoryV2ContextProvider(
        resolver=ScopedResolver(),
        retriever=Retriever(),
        expander=MixedExpander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    provider(request())

    assert tuple(
        segment.episode_id for segment in packer.evidence_segments
    ) == ("raw:eligible",)


def test_direct_reply_quota_is_consumed_only_after_delivery_subject_and_time_checks() -> None:
    class ScopedResolver:
        def resolve(self, query, **_):
            return ResolvedMemoryQuery(
                original_query=query,
                retrieval_query=query,
                needs_history=True,
                group_id=100,
                subject_ids=("42",),
                answer_mode="dated_history",
                time_range=TimeRange(
                    start=datetime(2026, 7, 23, tzinfo=UTC),
                    end=datetime(2026, 7, 24, tzinfo=UTC),
                ),
            )

    class RawReplyRetriever:
        def retrieve(self, **_):
            return HybridRetrievalResult(
                (
                    FusedRetrievalCandidate(
                        document_id=10,
                        group_id=100,
                        document_kind="raw_message_v3",
                        episode_id=None,
                        source_msg_ids=("parent",),
                        start_at=datetime(2026, 7, 23, tzinfo=UTC),
                        end_at=datetime(2026, 7, 23, tzinfo=UTC),
                        routes=("bm25",),
                        route_ranks=(("bm25", 1),),
                        fused_score=1.0,
                    ),
                )
            )

    def message(
        source_id: str,
        *,
        sent_at: datetime,
        user_id: int = 42,
        delivery_state: str = "",
        reply_to: str | None = None,
    ) -> EvidenceMessage:
        return EvidenceMessage(
            source_id,
            "member",
            source_id,
            sent_at,
            group_id=100,
            user_id=user_id,
            delivery_state=delivery_state,
            reply_to_msg_id=reply_to,
        )

    rows = (
        message("parent", sent_at=datetime(2026, 7, 23, 10, tzinfo=UTC)),
        message(
            "outside-time",
            sent_at=datetime(2026, 7, 22, 23, 59, tzinfo=UTC),
            reply_to="parent",
        ),
        message(
            "deleted",
            sent_at=datetime(2026, 7, 23, 10, 1, tzinfo=UTC),
            delivery_state="deleted",
            reply_to="parent",
        ),
        message(
            "wrong-subject",
            sent_at=datetime(2026, 7, 23, 10, 2, tzinfo=UTC),
            user_id=99,
            reply_to="parent",
        ),
        message(
            "eligible-third",
            sent_at=datetime(2026, 7, 23, 10, 3, tzinfo=UTC),
            reply_to="parent",
        ),
        message(
            "eligible-fourth",
            sent_at=datetime(2026, 7, 23, 10, 4, tzinfo=UTC),
            reply_to="parent",
        ),
        message(
            "eligible-fifth",
            sent_at=datetime(2026, 7, 23, 10, 5, tzinfo=UTC),
            reply_to="parent",
        ),
    )
    packer = CapturingSegmentsPacker()
    provider = MemoryV2ContextProvider(
        resolver=ScopedResolver(),
        retriever=RawReplyRetriever(),
        expander=MemoryEvidenceExpander(
            episode_loader=lambda **_: (),
            source_loader=lambda **_: rows,
        ),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        max_direct_replies_per_source=2,
    )

    provider(request())

    assert tuple(
        message.source_msg_id
        for message in packer.evidence_segments[0].messages
    ) == ("parent", "eligible-third", "eligible-fourth")


def test_historical_evidence_excludes_bot_authored_hits_and_attached_replies() -> None:
    class BotEvidenceExpander:
        def expand(self, **_):
            human = EvidenceMessage(
                "human-source",
                "member",
                "supported statement",
                datetime(2026, 7, 23, tzinfo=UTC),
                group_id=100,
                user_id=42,
            )
            bot_hit = EvidenceMessage(
                "bot-hit",
                "bot",
                "unsupported generated answer",
                datetime(2026, 7, 23, 0, 1, tzinfo=UTC),
                group_id=100,
                user_id=999,
                is_bot=True,
            )
            bot_reply = replace(
                bot_hit,
                source_msg_id="bot-reply",
                reply_to_msg_id="human-source",
            )
            return (
                EvidenceSegment(
                    "raw:bot-hit",
                    10.0,
                    (bot_hit,),
                    hit_source_msg_ids=("bot-hit",),
                ),
                EvidenceSegment(
                    "raw:human-hit",
                    9.0,
                    (human, bot_reply),
                    hit_source_msg_ids=("human-source",),
                ),
            )

    packer = CapturingSegmentsPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=Retriever(),
        expander=BotEvidenceExpander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    provider(request())

    assert tuple(segment.episode_id for segment in packer.evidence_segments) == (
        "raw:human-hit",
    )
    assert tuple(
        message.source_msg_id
        for message in packer.evidence_segments[0].messages
    ) == ("human-source",)


def test_required_reference_and_mention_segments_are_pinned() -> None:
    segment = EvidenceSegment(
        "raw:required",
        1.0,
        (
            EvidenceMessage(
                "required-source",
                "Target",
                "inside",
                datetime(2026, 7, 23, tzinfo=UTC),
                group_id=100,
                user_id=42,
            ),
        ),
    )

    referenced = MemoryV2ContextProvider._pin_required_segments(
        (segment,),
        ResolvedMemoryQuery(
            original_query="that",
            retrieval_query="that",
            reference_msg_ids=("required-source",),
        ),
    )
    mentioned = MemoryV2ContextProvider._pin_required_segments(
        (segment,),
        ResolvedMemoryQuery(
            original_query="@ me",
            retrieval_query="@ me",
            answer_mode="mention",
        ),
    )

    assert referenced[0].pinned is True
    assert mentioned[0].pinned is True

def test_recent_intent_candidate_limit_compacts_evidence_window() -> None:
    provider = MemoryV2ContextProvider(
        resolver=object(),
        retriever=object(),
        expander=object(),
        packer=object(),
        source_scope_validator=object(),
        recent_intent_candidate_limit=3,
    )
    selected, mode, reasons = provider._select_expansion_candidates(
        candidates=tuple(range(10)),
        retrieval_result=object(),
        needs_history=True,
        recent_intent=True,
    )
    assert mode == "compact"
    assert reasons == ("recent_intent",)
    assert len(selected) == 3

    selected_plain, mode_plain, reasons_plain = provider._select_expansion_candidates(
        candidates=tuple(range(10)),
        retrieval_result=object(),
        needs_history=True,
    )
    assert mode_plain == "legacy"
    assert reasons_plain == ()
    assert len(selected_plain) == 10


def test_recent_intent_candidate_limit_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        MemoryV2ContextProvider(
            resolver=object(),
            retriever=object(),
            expander=object(),
            packer=object(),
            source_scope_validator=object(),
            recent_intent_candidate_limit=0,
        )
