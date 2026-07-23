from __future__ import annotations

from datetime import UTC, datetime
from threading import Barrier
import time

import pytest

from app.core.hybrid_memory_retriever import (
    HybridMemoryRetriever,
    MemoryScopeViolation,
    RetrievalCandidate,
)


def candidate(
    document_id: int,
    *,
    group_id: int = 100,
    source_msg_ids: tuple[str, ...] = (),
    score: float = 0.0,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id=document_id,
        group_id=group_id,
        document_kind="episode",
        episode_id=document_id,
        source_msg_ids=source_msg_ids or (f"msg-{document_id}",),
        start_at=datetime(2026, 7, document_id, tzinfo=UTC),
        end_at=datetime(2026, 7, document_id, tzinfo=UTC),
        channel_score=score,
    )


def test_vector_only_candidate_survives_without_any_lexical_candidate() -> None:
    retriever = HybridMemoryRetriever(
        channels={
            "bm25": lambda **_: [],
            "vector": lambda **_: [candidate(7, score=0.92)],
        }
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == [7]
    assert result.candidates[0].routes == ("vector",)


def test_weighted_rrf_pins_exact_reference_above_multi_route_semantic_hit() -> None:
    retriever = HybridMemoryRetriever(
        channels={
            "exact_quote": lambda **_: [candidate(3)],
            "bm25": lambda **_: [candidate(9), candidate(3)],
            "vector": lambda **_: [candidate(9), candidate(3)],
        }
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates[:2]] == [3, 9]
    assert result.candidates[0].routes == ("exact_quote", "bm25", "vector")


def test_exact_only_reference_is_pinned_above_stronger_multi_route_candidate() -> None:
    semantic = candidate(9)
    retriever = HybridMemoryRetriever(
        channels={
            "exact_quote": lambda **_: [candidate(3)],
            "reply_graph": lambda **_: [semantic],
            "entity": lambda **_: [semantic],
            "fact": lambda **_: [semantic],
            "bm25": lambda **_: [semantic],
            "vector": lambda **_: [semantic],
            "temporal": lambda **_: [semantic],
        }
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates[:2]] == [3, 9]


def test_channels_run_in_parallel_with_independent_callables() -> None:
    barrier = Barrier(2)

    def channel(**_):
        barrier.wait(timeout=1)
        return [candidate(1)]

    retriever = HybridMemoryRetriever(channels={"bm25": channel, "vector": channel})

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == [1]


def test_one_failed_channel_does_not_discard_other_local_results() -> None:
    def broken(**_):
        raise RuntimeError("provider payload must not leak")

    retriever = HybridMemoryRetriever(
        channels={
            "bm25": broken,
            "entity": lambda **_: [candidate(4)],
        }
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == [4]
    assert result.failed_channels == ("bm25",)


def test_channel_timeout_returns_without_waiting_for_executor_shutdown() -> None:
    def slow(**_):
        time.sleep(0.2)
        return [candidate(8)]

    retriever = HybridMemoryRetriever(
        channels={
            "bm25": slow,
            "entity": lambda **_: [candidate(4)],
        },
        channel_timeout_seconds=0.01,
    )
    started = time.perf_counter()

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert time.perf_counter() - started < 0.1
    assert [item.document_id for item in result.candidates] == [4]
    assert result.failed_channels == ("bm25",)


def test_all_failed_channels_are_reported_to_the_caller() -> None:
    def broken(**_):
        raise RuntimeError("channel failed")

    retriever = HybridMemoryRetriever(
        channels={"bm25": broken, "vector": broken},
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert result.candidates == ()
    assert result.failed_channels == ("bm25", "vector")


def test_any_cross_group_candidate_fails_closed_for_the_whole_v2_batch() -> None:
    retriever = HybridMemoryRetriever(
        channels={
            "bm25": lambda **_: [candidate(1)],
            "vector": lambda **_: [candidate(2, group_id=200)],
        }
    )

    with pytest.raises(MemoryScopeViolation):
        retriever.retrieve(group_id=100, resolved_query=object())


def test_rrf_tie_break_is_stable_by_recency_then_document_id() -> None:
    first = candidate(1)
    second = candidate(2)
    retriever = HybridMemoryRetriever(
        channels={"bm25": lambda **_: [first, second]},
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == [1, 2]
