from __future__ import annotations

from datetime import UTC, datetime

from app.core.memory_context_packer import (
    EvidenceMessage,
    EvidenceSegment,
    MEMORY_GROUNDING_NO_EVIDENCE,
    QQ_BLOCKED_MEMORY_NOTE,
    MemoryContextPacker,
    MemoryFact,
    MemorySummary,
)


def test_default_additive_token_counter_counts_each_history_block_once() -> None:
    packer = MemoryContextPacker(
        normal_budget=100_000,
        detail_budget=100_000,
        recent_budget=10_000,
        history_budget=100_000,
        context_char_budget=100_000,
        max_history_messages=150,
    )
    calls = 0

    def counting_counter(value: str) -> int:
        nonlocal calls
        calls += 1
        return MemoryContextPacker._fallback_token_count(value)

    # Preserve the constructor-selected additive contract while observing how
    # often the default counter is invoked.
    packer._token_counter = counting_counter
    now = datetime(2026, 8, 1, tzinfo=UTC)
    segments = tuple(
        EvidenceSegment(
            episode_id=f"raw:{index}",
            document_id=str(index),
            fused_score=1.0,
            messages=(
                EvidenceMessage(
                    source_msg_id=f"source-{index}",
                    speaker="member",
                    content=f"evidence-{index}",
                    sent_at=now,
                    group_id=100,
                ),
            ),
            hit_source_msg_ids=(f"source-{index}",),
        )
        for index in range(150)
    )

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        evidence_segments=segments,
    )

    assert len(packed.evidence_segments) == 150
    assert calls < 500


def test_adaptive_additive_token_counter_does_not_rescan_selected_history() -> None:
    packer = MemoryContextPacker(
        normal_budget=100_000,
        detail_budget=100_000,
        context_char_budget=100_000,
        adaptive_enabled=True,
        adaptive_max_history_messages=300,
    )
    calls = 0

    def counting_counter(value: str) -> int:
        nonlocal calls
        calls += 1
        return MemoryContextPacker._fallback_token_count(value)

    packer._token_counter = counting_counter
    now = datetime(2026, 8, 1, tzinfo=UTC)
    segments = tuple(
        EvidenceSegment(
            episode_id=f"raw:{index}",
            document_id=str(index),
            fused_score=1.0,
            messages=(
                EvidenceMessage(
                    source_msg_id=f"source-{index}",
                    speaker="member",
                    content=f"evidence-{index}",
                    sent_at=now,
                    group_id=100,
                ),
            ),
            hit_source_msg_ids=(f"source-{index}",),
        )
        for index in range(300)
    )

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        evidence_segments=segments,
    )

    assert len(packed.evidence_segments) == 300
    assert calls < 1_000


def message(identifier: str, text: str, *, blocked: bool = False) -> EvidenceMessage:
    return EvidenceMessage(identifier, "Alice", text, datetime(2026, 7, 23, 10), blocked=blocked)


def test_recent_is_a_contiguous_suffix_and_target_is_not_repeated() -> None:
    packer = MemoryContextPacker(normal_budget=60, detail_budget=80, token_counter=lambda value: len(value.split()))
    recent = (message("1", "old"), message("2", "new"), message("target", "ask"))

    packed = packer.pack("normal", available_input=60, target_message_id="target", recent_messages=recent)

    assert packed.recent_source_msg_ids == ("1", "2")
    assert "ask" not in packed.text
    assert packed.grounding_policy
    assert packed.estimated_tokens <= packed.budget


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

    assert packed.budget == 34_000
    assert packed.estimated_tokens <= 10_000


def test_both_modes_allow_bounded_history_and_summary_requires_evidence() -> None:
    packer = MemoryContextPacker(normal_budget=100, detail_budget=100, token_counter=lambda value: 1)
    segments = tuple(EvidenceSegment(f"ep-{index}", 10 - index, (message(str(index), f"proof {index}"),)) for index in range(7))
    summary = MemorySummary("latest summary", ("x",), relevant=True)

    normal = packer.pack("normal", available_input=100, target_message_id=None, evidence_segments=segments, summaries=(summary,))
    detail = packer.pack("detail", available_input=100, target_message_id=None, evidence_segments=segments, summaries=(summary,))
    empty = packer.pack("normal", available_input=100, target_message_id=None, summaries=(summary,))

    assert len(normal.evidence_segments) == 7
    assert len(detail.evidence_segments) == 7
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


def test_adaptive_tiny_budget_preserves_blocked_safety_signal_and_flag() -> None:
    packer = MemoryContextPacker(
        normal_budget=200,
        detail_budget=200,
        adaptive_enabled=True,
        recent_protected_min_tokens=0,
        history_protected_min_tokens=0,
    )
    blocked = message("blocked", "raw-secret-marker", blocked=True)
    note_budget = packer._fallback_token_count(QQ_BLOCKED_MEMORY_NOTE)

    packed = packer.pack(
        "normal",
        available_input=note_budget,
        target_message_id=None,
        recent_messages=(blocked,),
    )

    assert packed.blocked_output_present is True
    assert packed.text == QQ_BLOCKED_MEMORY_NOTE
    assert "raw-secret-marker" not in packed.text
    assert packed.degradation_reason == "blocked_note_only"


def test_empty_v2_context_explicitly_marks_memory_evidence_as_insufficient() -> None:
    packer = MemoryContextPacker(normal_budget=200, detail_budget=200)

    packed = packer.pack(
        "normal",
        available_input=200,
        target_message_id=None,
    )

    assert "No relevant memory fact or retrieved evidence was found" in packed.text
    assert "Do not infer a person's preference from topical discussion" in packed.text


def test_v2_context_with_evidence_prioritizes_corrections_over_historical_chat() -> None:
    packer = MemoryContextPacker(normal_budget=300, detail_budget=300)
    segment = EvidenceSegment(
        "ep-correction",
        2.0,
        (message("correction", "explicit corrected fact"),),
    )

    packed = packer.pack(
        "normal",
        available_input=300,
        target_message_id=None,
        evidence_segments=(segment,),
    )

    assert "later corrections or newer evidence" in packed.text
    assert "directly and unambiguously supported" in packed.text
    assert "Do not infer, generalize, embellish" in packed.text
    assert "memory evidence is insufficient" in packed.text


def test_packed_memory_context_respects_provider_character_budget() -> None:
    packer = MemoryContextPacker(
        normal_budget=32_000,
        detail_budget=32_000,
        history_budget=24_000,
        context_char_budget=1_200,
    )
    segments = tuple(
        EvidenceSegment(
            f"ep-{index}",
            float(100 - index),
            (message(f"source-{index}", "evidence " + ("x" * 180)),),
        )
        for index in range(20)
    )

    packed = packer.pack(
        "normal",
        available_input=32_000,
        target_message_id=None,
        evidence_segments=segments,
    )

    assert len(packed.text) <= 1_200
    assert packed.evidence_segments
    assert packed.grounding_policy


def test_recent_near_character_cap_cannot_push_final_policy_over_budget() -> None:
    packer = MemoryContextPacker(
        normal_budget=32_000,
        detail_budget=32_000,
        context_char_budget=12_000,
    )

    packed = packer.pack(
        "normal",
        available_input=32_000,
        target_message_id=None,
        recent_messages=(message("recent", "x" * 11_900),),
    )

    assert len(packed.text) <= 12_000
    assert packed.grounding_policy == MEMORY_GROUNDING_NO_EVIDENCE


def test_historical_request_reserves_space_for_direct_evidence_before_recent() -> None:
    packer = MemoryContextPacker(
        normal_budget=32_000,
        detail_budget=32_000,
        context_char_budget=12_000,
    )
    segment = EvidenceSegment(
        "direct",
        1.0,
        (message("direct-source", "direct historical evidence"),),
        pinned=True,
    )

    packed = packer.pack(
        "normal",
        available_input=32_000,
        target_message_id=None,
        recent_messages=tuple(
            message(f"recent-{index}", "r" * 500) for index in range(30)
        ),
        evidence_segments=(segment,),
    )

    assert len(packed.text) <= 12_000
    assert [item.episode_id for item in packed.evidence_segments] == ["direct"]
    assert "direct-source" in packed.source_msg_ids


def test_recent_60_and_history_150_use_independent_message_quotas() -> None:
    packer = MemoryContextPacker(
        normal_budget=100_000,
        detail_budget=100_000,
        recent_budget=100_000,
        history_budget=100_000,
        context_char_budget=100_000,
        max_recent_messages=60,
        max_history_messages=150,
        token_counter=lambda _value: 1,
    )
    recent = tuple(message(f"r-{index}", "recent") for index in range(70))
    history = tuple(
        EvidenceSegment(
            f"raw:{index}",
            float(200 - index),
            (message(f"h-{index}", "history"),),
        )
        for index in range(170)
    )

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        recent_messages=recent,
        evidence_segments=history,
    )

    assert len(packed.recent_messages) == 60
    assert sum(len(segment.messages) for segment in packed.evidence_segments) == 150
    assert len(packed.source_msg_ids) == 210


def test_default_counter_enforces_24k_history_cap_for_long_chinese_text() -> None:
    packer = MemoryContextPacker(
        normal_budget=100_000,
        detail_budget=100_000,
        recent_budget=100_000,
        history_budget=24_000,
        max_history_messages=150,
    )
    history = tuple(
        EvidenceSegment(
            f"raw:{index}",
            float(200 - index),
            (message(f"zh-{index}", "中" * 1000),),
        )
        for index in range(40)
    )

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        evidence_segments=history,
    )

    assert packed.estimated_tokens <= 24_000
    assert len(packed.evidence_segments) < 40


def test_rendered_evidence_time_is_explicit_shanghai_time() -> None:
    source = EvidenceMessage(
        "utc-source",
        "Alice",
        "hello",
        datetime(2026, 7, 23, 16, 30),
    )
    packed = MemoryContextPacker(
        normal_budget=1000,
        detail_budget=1000,
    ).pack(
        "normal",
        available_input=1000,
        target_message_id=None,
        recent_messages=(source,),
    )

    assert "2026-07-24 00:30 +08" in packed.text


def test_rendered_history_includes_uin_source_and_reply_relationship() -> None:
    source = EvidenceMessage(
        "reply-source",
        "SameCard",
        "quoted reply",
        datetime(2026, 7, 23, 16, 30),
        group_id=100,
        reply_to_msg_id="parent-source",
        user_id=123456,
    )
    packed = MemoryContextPacker(
        normal_budget=1000,
        detail_budget=1000,
    ).pack(
        "normal",
        available_input=1000,
        target_message_id=None,
        evidence_segments=(EvidenceSegment("raw:1", 1.0, (source,)),),
    )

    assert "SameCard (uin: 123456;" in packed.text
    assert "source: reply-source;" in packed.text
    assert "reply_to: parent-source" in packed.text


def test_history_budget_preserves_retriever_time_bucket_order() -> None:
    packer = MemoryContextPacker(
        normal_budget=1000,
        detail_budget=1000,
        history_budget=20,
        token_counter=lambda value: value.count("TOKEN"),
    )
    segments = (
        EvidenceSegment("old-bucket", 0.1, (message("old", "TOKEN " * 10),)),
        EvidenceSegment("middle-bucket", 0.2, (message("middle", "TOKEN " * 10),)),
        EvidenceSegment("recent-high-score", 999.0, (message("recent", "TOKEN " * 10),)),
    )

    packed = packer.pack(
        "normal",
        available_input=1000,
        target_message_id=None,
        evidence_segments=segments,
    )

    assert tuple(segment.episode_id for segment in packed.evidence_segments) == (
        "old-bucket",
        "middle-bucket",
    )


def test_default_normal_budget_allows_independent_recent_and_history_caps() -> None:
    packer = MemoryContextPacker(
        normal_budget=32,
        detail_budget=64,
        recent_budget=10,
        history_budget=24,
        token_counter=lambda value: value.count("TOKEN"),
    )
    recent = (
        message("recent-independent", "TOKEN " * 10),
    )
    history = (
        EvidenceSegment(
            "history-a",
            1.0,
            (message("history-a", "TOKEN " * 12),),
        ),
        EvidenceSegment(
            "history-b",
            0.9,
            (message("history-b", "TOKEN " * 12),),
        ),
    )

    packed = packer.pack(
        "normal",
        available_input=34,
        target_message_id=None,
        recent_messages=recent,
        evidence_segments=history,
    )

    assert packed.recent_estimated_tokens == 10
    assert packed.history_estimated_tokens == 24
    assert len(packed.recent_messages) == 1
    assert len(packed.evidence_segments) == 2


def test_adaptive_pack_never_widens_or_exceeds_one_effective_total() -> None:
    packer = MemoryContextPacker(
        adaptive_enabled=True,
        token_counter=lambda value: value.count("TOKEN"),
    )
    recent = tuple(message(f"r-{index}", "TOKEN " * 1000) for index in range(12))
    history = tuple(
        EvidenceSegment(
            f"ep-{index}",
            float(20 - index),
            (message(f"h-{index}", "TOKEN " * 1000),),
        )
        for index in range(30)
    )

    packed = packer.pack(
        "normal",
        available_input=31_999,
        target_message_id=None,
        recent_messages=recent,
        evidence_segments=history,
    )

    assert packed.adaptive_enabled is True
    assert packed.budget == 31_999
    assert packed.estimated_tokens <= packed.budget
    assert packed.recent_estimated_tokens + packed.history_estimated_tokens <= packed.budget
    assert len(packed.text) <= 12_000


def test_adaptive_pack_borrows_unused_recent_capacity_for_history() -> None:
    packer = MemoryContextPacker(
        normal_budget=40,
        detail_budget=40,
        context_char_budget=20_000,
        adaptive_enabled=True,
        recent_protected_min_tokens=5,
        history_protected_min_tokens=5,
        token_counter=lambda value: value.count("TOKEN"),
    )
    history = tuple(
        EvidenceSegment(
            f"ep-{index}",
            float(10 - index),
            (message(f"h-{index}", "TOKEN " * 8),),
        )
        for index in range(5)
    )

    packed = packer.pack(
        "normal",
        available_input=40,
        target_message_id=None,
        recent_messages=(message("recent", "TOKEN"),),
        evidence_segments=history,
    )

    assert packed.recent_source_msg_ids == ("recent",)
    assert packed.history_estimated_tokens > 5
    assert packed.estimated_tokens <= 40
    assert len(packed.source_msg_ids) == len(set(packed.source_msg_ids))
    assert packed.spillover == "recent_to_history"


def test_adaptive_pack_borrows_unused_history_capacity_for_recent() -> None:
    packer = MemoryContextPacker(
        normal_budget=25,
        detail_budget=25,
        context_char_budget=20_000,
        adaptive_enabled=True,
        recent_protected_min_tokens=3,
        history_protected_min_tokens=5,
        token_counter=lambda value: value.count("TOKEN"),
    )
    recent = tuple(message(f"r-{index}", "TOKEN " * 3) for index in range(20))

    packed = packer.pack(
        "normal",
        available_input=25,
        target_message_id=None,
        recent_messages=recent,
    )

    assert packed.evidence_segments == ()
    assert packed.recent_estimated_tokens > 3
    assert packed.recent_source_msg_ids == tuple(f"r-{index}" for index in range(12, 20))
    assert packed.estimated_tokens <= 25
    assert packed.spillover == "history_to_recent"


def test_adaptive_pack_preserves_pinned_and_recent_before_optional_history() -> None:
    packer = MemoryContextPacker(
        normal_budget=15,
        detail_budget=15,
        context_char_budget=20_000,
        adaptive_enabled=True,
        recent_protected_min_tokens=5,
        history_protected_min_tokens=5,
        token_counter=lambda value: value.count("TOKEN"),
    )
    pinned = EvidenceSegment(
        "pinned",
        0.01,
        (message("pin-a", "TOKEN " * 5), message("pin-b", "TOKEN " * 5)),
        atomic_source_groups=(("pin-a", "pin-b"),),
        pinned=True,
    )
    optional = EvidenceSegment(
        "optional",
        100.0,
        (message("optional", "TOKEN " * 10),),
    )

    packed = packer.pack(
        "normal",
        available_input=15,
        target_message_id=None,
        recent_messages=(message("recent", "TOKEN " * 5),),
        evidence_segments=(pinned, optional),
    )

    assert tuple(segment.episode_id for segment in packed.evidence_segments) == ("pinned",)
    assert tuple(item.source_msg_id for item in packed.evidence_segments[0].messages) == (
        "pin-a",
        "pin-b",
    )
    assert packed.recent_source_msg_ids == ("recent",)
    assert packed.estimated_tokens == 15


def test_adaptive_message_limits_are_emergency_row_caps() -> None:
    packer = MemoryContextPacker(
        normal_budget=100_000,
        detail_budget=100_000,
        context_char_budget=100_000,
        adaptive_enabled=True,
        adaptive_max_recent_messages=120,
        adaptive_max_history_messages=300,
        token_counter=lambda _value: 1,
    )
    recent = tuple(message(f"r-{index}", "short") for index in range(140))
    history = tuple(
        EvidenceSegment(
            f"ep-{index}",
            float(400 - index),
            (message(f"h-{index}", "short"),),
        )
        for index in range(320)
    )

    packed = packer.pack(
        "normal",
        available_input=100_000,
        target_message_id=None,
        recent_messages=recent,
        evidence_segments=history,
    )

    assert len(packed.recent_messages) == 120
    assert sum(len(segment.messages) for segment in packed.evidence_segments) == 300


def test_adaptive_pack_degrades_to_no_evidence_policy_under_tiny_budget() -> None:
    packer = MemoryContextPacker(
        normal_budget=13,
        detail_budget=13,
        adaptive_enabled=True,
        recent_protected_min_tokens=0,
        history_protected_min_tokens=0,
    )
    oversized = EvidenceSegment(
        "oversized",
        1.0,
        (message("source", "中" * 100),),
        pinned=True,
    )

    packed = packer.pack(
        "normal",
        available_input=13,
        target_message_id=None,
        evidence_segments=(oversized,),
    )

    assert packed.evidence_segments == ()
    assert packed.grounding_policy == "No memory evidence; do not infer a person's preference."
    assert packed.estimated_tokens <= packed.budget
    assert len(packed.text) <= 12_000
    assert packed.degradation_reason == "minimal_policy"
