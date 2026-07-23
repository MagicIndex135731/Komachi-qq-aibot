from __future__ import annotations

from datetime import datetime

from app.core.memory_context_packer import (
    EvidenceMessage,
    EvidenceSegment,
    MemoryContextPacker,
    MemoryFact,
    MemorySummary,
)


def message(identifier: str, text: str, *, blocked: bool = False) -> EvidenceMessage:
    return EvidenceMessage(identifier, "Alice", text, datetime(2026, 7, 23, 10), blocked=blocked)


def test_recent_is_a_contiguous_suffix_and_target_is_not_repeated() -> None:
    packer = MemoryContextPacker(normal_budget=40, detail_budget=80, token_counter=lambda value: len(value.split()))
    recent = (message("1", "old"), message("2", "new"), message("target", "ask"))

    packed = packer.pack("normal", available_input=40, target_message_id="target", recent_messages=recent)

    assert packed.recent_source_msg_ids == ("1", "2")
    assert "ask" not in packed.text


def test_evidence_is_deduplicated_against_recent_and_quote_pair_is_atomic() -> None:
    packer = MemoryContextPacker(normal_budget=40, detail_budget=80, token_counter=lambda value: len(value.split()))
    pair = EvidenceSegment(
        "ep-1",
        2.0,
        (message("q", "question"), message("a", "answer")),
        hit_source_msg_ids=("a",),
        atomic_source_groups=(("q", "a"),),
    )
    packed = packer.pack(
        "normal", available_input=40, target_message_id=None, recent_messages=(message("a", "answer"),), evidence_segments=(pair,)
    )

    assert packed.evidence_segments == ()
    assert "question" not in packed.text


def test_non_atomic_overlap_deduplicates_sources_without_dropping_segment() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)
    segment = EvidenceSegment(
        "ep-1",
        2.0,
        (message("recent", "duplicate"), message("history", "useful history")),
        hit_source_msg_ids=("history",),
    )

    packed = packer.pack(
        "normal",
        available_input=100,
        target_message_id=None,
        recent_messages=(message("recent", "duplicate"),),
        evidence_segments=(segment,),
    )

    assert len(packed.evidence_segments) == 1
    assert tuple(item.source_msg_id for item in packed.evidence_segments[0].messages) == ("history",)
    assert packed.text.count("duplicate") == 1


def test_default_budgets_follow_v2_contract_and_recent_has_own_cap() -> None:
    packer = MemoryContextPacker(token_counter=lambda value: len(value))

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        recent_messages=(message("1", "x" * 12_000),),
    )

    assert packed.budget == 32_000
    assert packed.estimated_tokens <= 10_000


def test_detail_allows_more_segments_and_summary_requires_evidence() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)
    segments = tuple(EvidenceSegment(f"ep-{index}", 10 - index, (message(str(index), f"proof {index}"),)) for index in range(7))
    summary = MemorySummary("latest summary", ("x",), relevant=True)

    normal = packer.pack("normal", available_input=100, target_message_id=None, evidence_segments=segments, summaries=(summary,))
    detail = packer.pack("detail", available_input=100, target_message_id=None, evidence_segments=segments, summaries=(summary,))
    empty = packer.pack("normal", available_input=100, target_message_id=None, summaries=(summary,))

    assert len(normal.evidence_segments) == 4
    assert len(detail.evidence_segments) == 6
    assert "latest summary" in normal.text
    assert empty.summaries == ()


def test_blocked_message_becomes_safe_note_without_sensitive_text() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)

    packed = packer.pack("normal", available_input=100, target_message_id=None, recent_messages=(message("b", "不能复述的敏感原文", blocked=True),))

    assert "QQ blocked" in packed.text
    assert "不能复述" not in packed.text


def test_facts_keep_provenance_and_untrusted_evidence_label() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)
    fact = MemoryFact("发布已延期", ("f1",), score=1.0)
    segment = EvidenceSegment("ep-1", 2.0, (message("e1", "发布延期"),))

    packed = packer.pack("normal", available_input=100, target_message_id=None, evidence_segments=(segment,), facts=(fact,))

    assert "untrusted quoted data" in packed.text
    assert packed.source_msg_ids == ("f1", "e1")


def test_pinned_exact_evidence_is_selected_before_large_facts() -> None:
    packer = MemoryContextPacker(
        normal_budget=75,
        detail_budget=75,
        recent_budget=20,
        token_counter=lambda value: len(value.split()),
    )
    pinned = EvidenceSegment(
        "exact",
        0.01,
        (message("exact-source", "EXACT REPLY EVIDENCE"),),
        pinned=True,
    )
    fact = MemoryFact(" ".join(["long-fact"] * 60), ("fact-source",), score=100.0)

    packed = packer.pack(
        "normal",
        available_input=75,
        target_message_id=None,
        evidence_segments=(pinned,),
        facts=(fact,),
    )

    assert packed.evidence_segments == (pinned,)
    assert "EXACT REPLY EVIDENCE" in packed.text


def test_blocked_neighbor_adds_safe_policy_note_without_raw_text() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)
    segment = EvidenceSegment(
        "ep-1",
        2.0,
        (message("safe", "safe evidence"),),
        blocked_output_present=True,
    )

    packed = packer.pack(
        "normal",
        available_input=100,
        target_message_id=None,
        evidence_segments=(segment,),
    )

    assert packed.blocked_output_present is True
    assert "QQ blocked" in packed.text
    assert "raw-secret-marker" not in packed.text
