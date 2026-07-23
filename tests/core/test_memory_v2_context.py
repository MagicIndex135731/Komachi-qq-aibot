from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from app.core.hybrid_memory_retriever import (
    FusedRetrievalCandidate,
    HybridRetrievalResult,
    MemoryScopeViolation,
)
from app.core.memory_context_packer import EvidenceMessage, PackedMemoryContext
from app.core.memory_orchestrator import MemoryContextResult
from app.core.memory_query_resolver import ResolvedMemoryQuery
from app.core.memory_v2_context import MemoryV2ContextProvider, MemoryV2Request


@dataclass
class Resolver:
    detail: bool = False

    def resolve(self, query, **_):
        return ResolvedMemoryQuery(
            original_query=query,
            retrieval_query=query,
            needs_history=True,
            needs_detail=self.detail,
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


class Packer:
    def pack(self, mode, **_):
        return PackedMemoryContext(
            mode=mode,
            budget=100,
            estimated_tokens=5,
            text="packed",
            source_msg_ids=("m-1",),
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


def test_v2_provider_rejects_all_channel_failure_instead_of_claiming_abstention() -> None:
    provider = MemoryV2ContextProvider(
        resolver=Resolver(),
        retriever=FailedRetriever(),
        expander=Expander(),
        packer=Packer(),
        source_scope_validator=lambda _group_id, _source_ids: True,
    )

    with pytest.raises(RuntimeError, match="all memory retrieval channels"):
        provider(request())


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
