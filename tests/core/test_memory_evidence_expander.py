from __future__ import annotations

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
