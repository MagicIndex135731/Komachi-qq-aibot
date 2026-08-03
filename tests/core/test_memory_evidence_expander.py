from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.core.hybrid_memory_retriever import FusedRetrievalCandidate, MemoryScopeViolation
from app.core.memory_context_packer import EvidenceMessage
from app.core.memory_evidence_expander import MemoryEvidenceExpander


def item(
    identifier: str,
    offset: int,
    *,
    group_id: int = 100,
    reply_to: str | None = None,
    is_bot: bool = False,
    blocked: bool = False,
) -> EvidenceMessage:
    return EvidenceMessage(
        source_msg_id=identifier,
        speaker="bot" if is_bot else "member",
        content=f"text-{identifier}",
        sent_at=datetime(2026, 7, 23, tzinfo=UTC) + timedelta(minutes=offset),
        blocked=blocked,
        group_id=group_id,
        reply_to_msg_id=reply_to,
        is_bot=is_bot,
    )


def candidate(
    source_ids: tuple[str, ...],
    *,
    group_id: int = 100,
    episode_id: int = 7,
) -> FusedRetrievalCandidate:
    now = datetime(2026, 7, 23, tzinfo=UTC)
    return FusedRetrievalCandidate(
        document_id=3,
        group_id=group_id,
        document_kind="episode",
        episode_id=episode_id,
        source_msg_ids=source_ids,
        start_at=now,
        end_at=now,
        routes=("bm25",),
        route_ranks=(("bm25", 1),),
        fused_score=1.0,
    )


def test_expands_only_inside_requested_group_and_episode_radius() -> None:
    rows = tuple(item(str(index), index) for index in range(12))
    expander = MemoryEvidenceExpander(
        episode_loader=lambda *, group_id, episode_id: rows,
        normal_radius=2,
        detail_radius=4,
    )

    segment = expander.expand(
        group_id=100,
        candidates=(candidate(("6",)),),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == ("4", "5", "6", "7", "8")
    assert segment.hit_source_msg_ids == ("6",)


def test_reply_ancestor_and_direct_bot_reply_are_atomic_and_cycle_safe() -> None:
    rows = (
        item("root", 0),
        item("question", 1, reply_to="root"),
        item("answer", 20, reply_to="question", is_bot=True),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: rows,
        normal_radius=0,
        max_reply_depth=2,
    )

    segment = expander.expand(group_id=100, candidates=(candidate(("question",)),), mode="normal")[0]

    assert tuple(message.source_msg_id for message in segment.messages) == ("root", "question", "answer")
    assert ("question", "answer") in segment.atomic_source_groups


def test_reply_graph_hit_is_pinned_before_history_truncation() -> None:
    rows = (item("question", 0),)
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: rows,
        normal_radius=0,
    )
    reply_candidate = replace(
        candidate(("question",)),
        routes=("reply_graph",),
        route_ranks=(("reply_graph", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(reply_candidate,),
        mode="normal",
    )[0]

    assert segment.pinned is True


def test_fused_relevance_pin_survives_evidence_expansion() -> None:
    rows = (item("topic-hit", 0),)
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: rows,
        normal_radius=0,
    )
    lexical_candidate = replace(
        candidate(("topic-hit",)),
        pin_reason="lexical",
    )

    segment = expander.expand(
        group_id=100,
        candidates=(lexical_candidate,),
        mode="normal",
    )[0]

    assert segment.pinned is True


def test_missing_or_cross_group_provenance_fails_closed() -> None:
    cross_group = (item("hit", 0, group_id=200),)
    expander = MemoryEvidenceExpander(episode_loader=lambda **_: cross_group)

    with pytest.raises(MemoryScopeViolation):
        expander.expand(group_id=100, candidates=(candidate(("hit",)),), mode="normal")

    missing = MemoryEvidenceExpander(episode_loader=lambda **_: ())
    with pytest.raises(MemoryScopeViolation):
        missing.expand(group_id=100, candidates=(candidate(("missing",)),), mode="normal")


def test_blocked_neighbor_sets_policy_signal_without_exposing_derived_text() -> None:
    rows = (item("hit", 0), item("blocked", 1, blocked=True))
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: rows,
        normal_radius=1,
    )

    segment = expander.expand(group_id=100, candidates=(candidate(("hit",)),), mode="normal")[0]

    assert segment.blocked_output_present is True
    assert tuple(message.source_msg_id for message in segment.messages) == ("hit",)
    assert "text-blocked" not in " ".join(message.content for message in segment.messages)


def test_raw_message_document_loads_exact_provenance_without_episode() -> None:
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda *, group_id, source_msg_ids: (
            item(source_msg_ids[0], 3, group_id=group_id),
        ),
    )
    raw_candidate = replace(
        candidate(("raw-hit",), episode_id=None),
        document_kind="raw_message_v3",
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert segment.episode_id == "raw:3"
    assert tuple(message.source_msg_id for message in segment.messages) == ("raw-hit",)


def test_raw_message_documents_batch_source_loading_and_preserve_candidate_order() -> None:
    calls: list[tuple[int, tuple[str, ...]]] = []

    def load_sources(*, group_id: int, source_msg_ids: tuple[str, ...]):
        calls.append((group_id, source_msg_ids))
        return tuple(
            item(source_id, index, group_id=group_id)
            for index, source_id in enumerate(reversed(source_msg_ids))
        )

    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=load_sources,
        normal_segment_limit=4,
    )
    first = replace(
        candidate(("first-b", "first-a"), episode_id=None),
        document_id=30,
        document_kind="raw_message_v3",
    )
    second = replace(
        candidate(("second",), episode_id=None),
        document_id=31,
        document_kind="raw_message_v3",
    )

    segments = expander.expand(
        group_id=100,
        candidates=(first, second),
        mode="normal",
    )

    assert calls == [(100, ("first-b", "first-a", "second"))]
    assert tuple(segment.document_id for segment in segments) == ("30", "31")
    assert tuple(message.source_msg_id for message in segments[0].messages) == (
        "first-a",
        "first-b",
    )
    assert tuple(message.source_msg_id for message in segments[1].messages) == (
        "second",
    )


def test_raw_message_document_loads_direct_replies_for_later_eligibility_filtering() -> None:
    rows = (
        item("hit", 0),
        item("reply-1", 1, reply_to="hit"),
        item("reply-2", 2, reply_to="hit"),
        item("reply-3", 3, reply_to="hit"),
        item("unrelated", 4, reply_to="other"),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("hit",), episode_id=None),
        document_kind="raw_message_v3",
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == (
        "hit",
        "reply-1",
        "reply-2",
        "reply-3",
    )
    assert segment.hit_source_msg_ids == ("hit",)
    assert segment.atomic_source_groups == ()
