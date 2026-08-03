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

    def resolve(self, query, **kwargs):
        return ResolvedMemoryQuery(
            original_query=query,
            retrieval_query=query,
            needs_history=True,
            needs_detail=self.detail,
            group_id=kwargs.get("group_id"),
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
        resolver=Resolver(detail=True),
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
    assert expander.modes == ["detail"]


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
        resolver=Resolver(),
        retriever=Retriever(),
        expander=Expander(),
        packer=packer,
        source_scope_validator=lambda _group_id, _source_ids: True,
        historical_no_hit_omit_recent=True,
    )

    provider(request())

    assert packer.recent_messages == ()


def test_historical_v3_post_validation_empty_does_not_inject_recent() -> None:
    packer = CapturingPacker()
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
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
        resolver=Resolver(),
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
