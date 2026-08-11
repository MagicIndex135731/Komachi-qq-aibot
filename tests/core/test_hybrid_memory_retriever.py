from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Barrier
import time
from types import SimpleNamespace

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
    at: datetime | None = None,
    lexical_match_kind: str = "none",
) -> RetrievalCandidate:
    timestamp = at or datetime(2026, 7, document_id, tzinfo=UTC)
    return RetrievalCandidate(
        document_id=document_id,
        group_id=group_id,
        document_kind="episode",
        episode_id=document_id,
        source_msg_ids=source_msg_ids or (f"msg-{document_id}",),
        start_at=timestamp,
        end_at=timestamp,
        channel_score=score,
        lexical_match_kind=lexical_match_kind,
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


def test_vector_cutoff_interleaves_only_the_near_boundary_slice() -> None:
    rows = tuple(candidate(index) for index in range(1, 21))
    retriever = HybridMemoryRetriever(
        channels={"vector": lambda **_: rows},
        candidate_limit=20,
        final_limit=10,
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
    ]
    assert len({item.document_id for item in result.candidates}) == 10


def test_vector_cutoff_stabilization_is_not_used_for_time_bucket_coverage() -> None:
    rows = tuple(candidate(index) for index in range(1, 21))
    retriever = HybridMemoryRetriever(
        channels={"vector": lambda **_: rows},
        candidate_limit=20,
        final_limit=10,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(coverage_mode="time_buckets"),
    )

    assert len(result.candidates) == 10
    assert [item.document_id for item in result.candidates] != [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
    ]


def test_vector_cutoff_keeps_true_top_k_when_boundary_scores_are_not_tied() -> None:
    rows = tuple(candidate(index, score=1.0 - index / 100.0) for index in range(1, 21))
    retriever = HybridMemoryRetriever(
        channels={"vector": lambda **_: rows},
        candidate_limit=20,
        final_limit=10,
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert [item.document_id for item in result.candidates] == list(range(1, 11))


def test_vector_cutoff_never_promotes_scores_outside_the_boundary_tie_group() -> None:
    scores = [2.3 - index / 10.0 for index in range(14)] + [0.9, 0.9, 0.2, 0.1, 0.0, -0.1]
    rows = tuple(
        candidate(index, score=score)
        for index, score in enumerate(scores, start=1)
    )
    retriever = HybridMemoryRetriever(
        channels={"vector": lambda **_: rows},
        candidate_limit=20,
        final_limit=16,
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert {item.document_id for item in result.candidates} == set(range(1, 17))


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


def test_explicit_reference_runs_only_deterministic_provenance_channels() -> None:
    called: list[str] = []

    def channel(name: str, result: list[RetrievalCandidate]):
        def run(**_):
            called.append(name)
            return result

        return run

    retriever = HybridMemoryRetriever(
        channels={
            "bm25": channel("bm25", [candidate(9)]),
            "vector": channel("vector", [candidate(9)]),
            "temporal": channel("temporal", [candidate(9)]),
            "entity": channel("entity", [candidate(9)]),
            "exact_quote": channel("exact_quote", [candidate(3)]),
            "reply_graph": channel("reply_graph", [candidate(4)]),
        }
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(reference_msg_ids=("quoted",)),
    )

    assert set(called) == {"exact_quote", "reply_graph"}
    assert result.attempted_channels == ("exact_quote", "reply_graph")
    assert {item.document_id for item in result.candidates} == {3, 4}


def test_mention_plan_runs_only_temporal_channel() -> None:
    called: list[str] = []

    def channel(name: str):
        def run(**_):
            called.append(name)
            return [candidate(4)]

        return run

    retriever = HybridMemoryRetriever(
        channels={
            "bm25": channel("bm25"),
            "vector": channel("vector"),
            "entity": channel("entity"),
            "temporal": channel("temporal"),
        }
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            reference_msg_ids=(),
            answer_mode="mention",
            retrieval_mode="hybrid",
        ),
    )

    assert called == ["temporal"]
    assert result.attempted_channels == ("temporal",)
    assert [item.document_id for item in result.candidates] == [4]


def test_temporal_history_combines_scoped_relevance_and_window_coverage() -> None:
    called: list[str] = []

    def channel(name: str):
        def run(**_):
            called.append(name)
            return [candidate(4)]

        return run

    retriever = HybridMemoryRetriever(
        channels={
            "bm25": channel("bm25"),
            "vector": channel("vector"),
            "entity": channel("entity"),
            "temporal": channel("temporal"),
            "reply_graph": channel("reply_graph"),
        }
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            reference_msg_ids=(),
            answer_mode="exact",
            retrieval_mode="temporal",
        ),
    )

    assert set(called) == {"bm25", "vector", "entity", "temporal"}
    assert result.attempted_channels == ("bm25", "vector", "entity", "temporal")
    assert [item.document_id for item in result.candidates] == [4]


def test_temporal_relevance_window_is_bounded_to_sixty_candidates() -> None:
    rows = tuple(
        candidate(
            index,
            at=datetime(2026, 7, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
        for index in range(1, 101)
    )
    retriever = HybridMemoryRetriever(
        channels={"temporal": lambda **_: rows},
        candidate_limit=100,
        final_limit=150,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            reference_msg_ids=(),
            answer_mode="exact",
            retrieval_mode="temporal",
            coverage_mode="relevance",
        ),
    )

    assert len(result.candidates) == 60


def test_temporal_mention_keeps_configured_limit_but_buckets_are_bounded() -> None:
    rows = tuple(
        candidate(
            index,
            at=datetime(2026, 7, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
        for index in range(1, 101)
    )
    retriever = HybridMemoryRetriever(
        channels={"temporal": lambda **_: rows},
        candidate_limit=100,
        final_limit=80,
    )

    mention = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            reference_msg_ids=(),
            answer_mode="mention",
            retrieval_mode="temporal",
            coverage_mode="relevance",
        ),
    )
    buckets = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            reference_msg_ids=(),
            answer_mode="exact",
            retrieval_mode="temporal",
            coverage_mode="time_buckets",
        ),
    )

    assert len(mention.candidates) == 80
    assert len(buckets.candidates) == 60


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


def test_time_bucket_coverage_spans_available_history_deterministically() -> None:
    retriever = HybridMemoryRetriever(
        channels={"entity": lambda **_: [candidate(index) for index in range(1, 11)]},
        candidate_limit=10,
        final_limit=3,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(coverage_mode="time_buckets"),
    )

    assert [item.document_id for item in result.candidates] == [1, 4, 10]


def test_time_bucket_coverage_cannot_evict_bounded_lexical_relevance_pins() -> None:
    timeline = [
        candidate(
            index,
            at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
        )
        for index in range(1, 21)
    ]
    lexical_hit = candidate(
        10,
        at=timeline[9].start_at,
        lexical_match_kind="exact",
    )
    retriever = HybridMemoryRetriever(
        channels={
            "bm25": lambda **_: [lexical_hit],
            "temporal": lambda **_: timeline,
        },
        candidate_limit=20,
        final_limit=3,
        relevance_pin_limit=1,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(coverage_mode="time_buckets"),
    )

    assert result.candidates[0].document_id == lexical_hit.document_id
    assert result.candidates[0].pin_reason == "lexical"
    assert result.candidates[0].pinned is True
    assert result.candidates[0].relevance_pinned is True
    assert len(result.candidates) == 3


def test_relevance_pin_limit_is_finite_and_reports_overflow() -> None:
    lexical = [
        candidate(index, lexical_match_kind="exact")
        for index in range(1, 6)
    ]
    retriever = HybridMemoryRetriever(
        channels={"bm25": lambda **_: lexical},
        final_limit=5,
        relevance_pin_limit=2,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(coverage_mode="relevance"),
    )

    assert [item.document_id for item in result.candidates if item.pinned] == [1, 2]
    assert result.pin_overflow_count == 3
    assert result.pin_counts == (("lexical", 2),)


def test_direct_pin_has_priority_over_lexical_pin_when_cap_is_exhausted() -> None:
    retriever = HybridMemoryRetriever(
        channels={
            "exact_quote": lambda **_: [candidate(9)],
            "bm25": lambda **_: [candidate(1, lexical_match_kind="exact")],
        },
        final_limit=2,
        relevance_pin_limit=1,
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert result.candidates[0].document_id == 9
    assert result.candidates[0].pin_reason == "direct"
    assert result.candidates[1].pinned is False
    assert result.pin_overflow_count == 1


def test_broad_bm25_fallback_is_not_pinned_as_strong_lexical_evidence() -> None:
    retriever = HybridMemoryRetriever(
        channels={
            "bm25": lambda **_: [candidate(1, lexical_match_kind="broad")],
        },
        final_limit=1,
    )

    result = retriever.retrieve(group_id=100, resolved_query=object())

    assert result.candidates[0].lexical_match_kind == "broad"
    assert result.candidates[0].pin_reason is None
    assert result.pin_counts == ()


def test_time_bucket_coverage_resists_busy_recent_interval() -> None:
    old = [
        candidate(
            index + 1,
            at=datetime(2025, 1, 1, tzinfo=UTC)
            + timedelta(days=index * 36),
        )
        for index in range(10)
    ]
    recent = [
        candidate(
            100 + index,
            at=datetime(2025, 12, 31, tzinfo=UTC)
            + timedelta(minutes=index),
        )
        for index in range(90)
    ]
    retriever = HybridMemoryRetriever(
        channels={"entity": lambda **_: [*recent, *old]},
        candidate_limit=100,
        final_limit=12,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(coverage_mode="time_buckets"),
    )

    represented_months = {
        (item.start_at.year, item.start_at.month)
        for item in result.candidates
    }
    assert len(result.candidates) == 12
    assert len(represented_months) >= 9


def test_temporal_route_pins_relevance_before_chronological_coverage() -> None:
    ordered = [
        candidate(
            index,
            at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
        for index in range(1, 101)
    ]
    relevant = ordered[-1]
    retriever = HybridMemoryRetriever(
        channels={
            "bm25": lambda **_: [relevant],
            "temporal": lambda **_: ordered,
        },
        candidate_limit=100,
        final_limit=60,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            retrieval_mode="temporal",
            answer_mode="summary",
            coverage_mode="chronological",
        ),
    )

    assert len(result.candidates) == 60
    assert result.candidates[0].document_id == relevant.document_id
    assert [item.document_id for item in result.candidates[-12:]] == list(range(48, 60))


def test_dated_history_uses_a_small_relevance_first_shortlist() -> None:
    ordered = [
        candidate(
            index,
            at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
        for index in range(1, 101)
    ]
    relevant = ordered[-1]
    retriever = HybridMemoryRetriever(
        channels={"bm25": lambda **_: [relevant], "temporal": lambda **_: ordered},
        candidate_limit=100,
        final_limit=150,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            retrieval_mode="temporal",
            answer_mode="dated_history",
            coverage_mode="chronological",
        ),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].document_id == relevant.document_id


def test_temporal_relevance_pins_leave_room_for_every_time_bucket() -> None:
    ordered = [
        candidate(
            index,
            at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index * 4),
        )
        for index in range(1, 101)
    ]
    relevance_order = list(reversed(ordered))
    retriever = HybridMemoryRetriever(
        channels={"bm25": lambda **_: relevance_order},
        candidate_limit=100,
        final_limit=60,
    )

    result = retriever.retrieve(
        group_id=100,
        resolved_query=SimpleNamespace(
            retrieval_mode="temporal",
            answer_mode="answer",
            coverage_mode="time_buckets",
        ),
    )

    assert len(result.candidates) == 60
    assert result.candidates[0].document_id == 100
    assert len({(item.start_at.year, item.start_at.month) for item in result.candidates}) >= 9


def test_weights_for_boosts_fact_channel_for_profile_preference_intent() -> None:
    retriever = HybridMemoryRetriever(channels={})
    base = retriever.channel_weights["fact"]

    preference_query = SimpleNamespace(
        preferred_fact_kinds=("preference",),
        answer_mode="assessment",
    )
    generic_query = SimpleNamespace(
        preferred_fact_kinds=(),
        answer_mode="general_history",
    )

    assert retriever._weights_for(preference_query)["fact"] == base * 2.5
    assert retriever._weights_for(generic_query)["fact"] == base
