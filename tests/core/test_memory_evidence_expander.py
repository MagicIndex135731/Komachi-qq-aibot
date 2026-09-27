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
    user_id: int | None = 10,
    content: str | None = None,
) -> EvidenceMessage:
    return EvidenceMessage(
        source_msg_id=identifier,
        speaker="bot" if is_bot else "member",
        content=content if content is not None else f"text-{identifier}",
        sent_at=datetime(2026, 7, 23, tzinfo=UTC) + timedelta(minutes=offset),
        blocked=blocked,
        group_id=group_id,
        reply_to_msg_id=reply_to,
        is_bot=is_bot,
        user_id=user_id,
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


def test_member_reference_raw_hit_keeps_bounded_preceding_context_as_atomic_evidence() -> None:
    context_calls: list[tuple[int, tuple[str, ...], int, int]] = []

    def load_context(*, group_id, source_msg_ids, limit_per_source, max_gap_seconds):
        context_calls.append(
            (group_id, source_msg_ids, limit_per_source, max_gap_seconds)
        )
        return (
            item("too-old", -1, group_id=group_id),
            item("named-item", 0, group_id=group_id),
            item("follow-up", 1, group_id=group_id),
        )

    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (item("member-reply", 2),),
        context_loader=load_context,
        context_radius=2,
        context_max_gap_seconds=120,
    )
    raw_candidate = replace(
        candidate(("member-reply",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert context_calls == [(100, ("member-reply",), 2, 120)]
    assert tuple(message.source_msg_id for message in segment.messages) == (
        "named-item",
        "follow-up",
        "member-reply",
    )
    assert segment.hit_source_msg_ids == ("member-reply",)
    assert segment.atomic_source_groups == (
        ("named-item", "follow-up", "member-reply"),
    )


def test_non_member_raw_hit_does_not_request_conversation_context() -> None:
    calls: list[object] = []
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (item("hit", 2),),
        context_loader=lambda **kwargs: calls.append(kwargs) or (),
    )
    raw_candidate = replace(
        candidate(("hit",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("bm25",),
        route_ranks=(("bm25", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert calls == []
    assert tuple(message.source_msg_id for message in segment.messages) == ("hit",)


def test_member_reference_links_bounded_clarification_thread_atomically() -> None:
    rows = (
        item("anchor", 0, user_id=30, content="我这边已经有一个大保底了"),
        item("noise", 3, user_id=30, content="面"),
        item("question", 9, user_id=20, content="大保底了哪家？"),
        item("guess", 10, user_id=40, content="示例公司"),
        item("answer", 13, user_id=30, content="示例公司"),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (rows[0],),
        context_loader=lambda **_: (),
        thread_context_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("anchor",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == (
        "anchor",
        "question",
        "answer",
    )
    assert ("anchor", "question", "answer") in segment.atomic_source_groups


def test_clarification_thread_rejects_conflicting_third_party_answer() -> None:
    rows = (
        item("anchor", 0, user_id=30, content="我这边已经有一个大保底了"),
        item("question", 5, user_id=20, content="大保底了哪家？"),
        item("conflict", 6, user_id=40, content="另一家公司"),
        item("answer", 7, user_id=30, content="示例公司"),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (rows[0],),
        context_loader=lambda **_: (),
        thread_context_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("anchor",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == ("anchor",)
    assert ("anchor", "question", "answer") not in segment.atomic_source_groups


def test_clarification_question_cue_selects_unique_matching_anchor() -> None:
    rows = (
        item("anchor", 0, user_id=30, content="我这边已经有一个大保底了"),
        item("unrelated-a", 3, user_id=30, content="刚吃完饭准备回去"),
        item("unrelated-b", 5, user_id=30, content="路上还挺凉快"),
        item("question", 8, user_id=20, content="大保底了哪家？"),
        item("guess", 9, user_id=40, content="示例公司"),
        item("answer", 12, user_id=30, content="示例公司"),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (rows[0],),
        context_loader=lambda **_: (),
        thread_context_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("anchor",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == (
        "anchor",
        "question",
        "answer",
    )
    assert ("anchor", "question", "answer") in segment.atomic_source_groups


def test_clarification_thread_rejects_multiple_possible_anchors() -> None:
    rows = (
        item("older-anchor", 0, user_id=30, content="我已经拿到一个明确结果了"),
        item("newer-anchor", 1, user_id=30, content="我还在同时推进另一个选择"),
        item("question", 2, user_id=20, content="哪家？"),
        item("answer", 3, user_id=30, content="示例公司"),
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (rows[0],),
        context_loader=lambda **_: (),
        thread_context_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("older-anchor",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == (
        "older-anchor",
    )
    assert segment.atomic_source_groups == ()


@pytest.mark.parametrize("unsafe_kind", ("blocked_answer", "bot_question", "timeout"))
def test_clarification_thread_rejects_unsafe_or_unbounded_rows(
    unsafe_kind: str,
) -> None:
    question = item("question", 1, user_id=20, content="哪家？")
    answer = item("answer", 2, user_id=30, content="示例公司")
    if unsafe_kind == "blocked_answer":
        answer = replace(answer, blocked=True)
    elif unsafe_kind == "bot_question":
        question = replace(question, is_bot=True)
    else:
        answer = item("answer", 17, user_id=30, content="示例公司")
    rows = (
        item("anchor", 0, user_id=30, content="我已经有一个明确结果了"),
        question,
        answer,
    )
    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **_: (rows[0],),
        context_loader=lambda **_: (),
        thread_context_loader=lambda **_: rows,
    )
    raw_candidate = replace(
        candidate(("anchor",), episode_id=None),
        document_kind="raw_message_v3",
        routes=("member_reference",),
        route_ranks=(("member_reference", 1),),
    )

    segment = expander.expand(
        group_id=100,
        candidates=(raw_candidate,),
        mode="normal",
    )[0]

    assert tuple(message.source_msg_id for message in segment.messages) == ("anchor",)
    assert segment.atomic_source_groups == ()


def test_clarification_loader_preserves_each_anchor_episode_boundary() -> None:
    calls: list[tuple[str, ...]] = []
    sources = {
        "anchor-a": item("anchor-a", 0, user_id=30, content="我有一个结果"),
        "anchor-b": item("anchor-b", 30, user_id=40, content="我也有一个结果"),
    }

    def load_threads(*, source_msg_ids, **_kwargs):
        calls.append(tuple(source_msg_ids))
        anchor = source_msg_ids[0]
        row = sources[anchor]
        return (
            row,
            item(f"question-{anchor}", 1 if anchor == "anchor-a" else 31, user_id=20, content="哪家？"),
            item(f"answer-{anchor}", 2 if anchor == "anchor-a" else 32, user_id=row.user_id, content=f"答案-{anchor}"),
        )

    expander = MemoryEvidenceExpander(
        episode_loader=lambda **_: (),
        source_loader=lambda **kwargs: tuple(
            sources[source_id] for source_id in kwargs["source_msg_ids"]
        ),
        context_loader=lambda **_: (),
        thread_context_loader=load_threads,
        normal_segment_limit=2,
    )
    candidates = tuple(
        replace(
            candidate((source_id,), episode_id=None),
            document_id=f"doc-{source_id}",
            document_kind="raw_message_v3",
            routes=("member_reference",),
            route_ranks=(("member_reference", 1),),
        )
        for source_id in ("anchor-a", "anchor-b")
    )

    segments = expander.expand(group_id=100, candidates=candidates, mode="normal")

    assert calls == [("anchor-a",), ("anchor-b",)]
    assert segments[0].atomic_source_groups == (
        ("anchor-a", "question-anchor-a", "answer-anchor-a"),
    )
    assert segments[1].atomic_source_groups == (
        ("anchor-b", "question-anchor-b", "answer-anchor-b"),
    )
