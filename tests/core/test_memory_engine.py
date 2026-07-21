from app.core.memory_engine import (
    extract_memory_candidates,
    extract_structured_memory_candidates,
    is_history_detail_query,
    retrieve_relevant_history,
    retrieve_relevant_memories,
)
from app.providers.embeddings import hashed_text_embedding, tokenize_text


def test_extract_memory_candidates_keeps_stable_preferences() -> None:
    candidates = extract_memory_candidates(
        scope_id="10001",
        source_msg_id="m-1",
        lines=[
            "Alice: I like sci-fi books.",
            "Bob: okay",
        ],
    )

    assert candidates == [
        {
            "scope_type": "group",
            "scope_id": "10001",
            "subject_type": "user",
            "subject_id": "Alice",
            "memory_kind": "preference",
            "content": "Alice likes sci-fi books.",
            "importance": 4,
            "confidence": 0.8,
            "source_msg_id": "m-1",
        }
    ]


def test_extract_memory_candidates_supports_chinese_preference_lines() -> None:
    candidates = extract_memory_candidates(
        scope_id="10001",
        source_msg_id="m-2",
        lines=["Alice\uff1a\u6211\u559c\u6b22\u706b\u9505\u3002"],
    )

    assert candidates == [
        {
            "scope_type": "group",
            "scope_id": "10001",
            "subject_type": "user",
            "subject_id": "Alice",
            "memory_kind": "preference",
            "content": "Alice likes \u706b\u9505.",
            "importance": 4,
            "confidence": 0.8,
            "source_msg_id": "m-2",
        }
    ]


def test_retrieve_relevant_memories_prefers_keyword_overlap() -> None:
    memories = [
        {"content": "Alice likes sci-fi books.", "importance": 4},
        {"content": "Bob dislikes early mornings.", "importance": 3},
    ]

    ranked = retrieve_relevant_memories("Do you still read sci-fi books?", memories, limit=1)

    assert ranked == [{"content": "Alice likes sci-fi books.", "importance": 4}]


def test_retrieve_relevant_memories_supports_chinese_keyword_overlap() -> None:
    memories = [
        {"content": "Alice likes \u706b\u9505.", "importance": 4},
        {"content": "Bob dislikes \u65e9\u8d77.", "importance": 3},
    ]

    ranked = retrieve_relevant_memories("\u4f60\u8fd8\u559c\u6b22\u706b\u9505\u5417", memories, limit=1)

    assert ranked == [{"content": "Alice likes \u706b\u9505.", "importance": 4}]


def test_retrieve_relevant_memories_omits_high_importance_zero_overlap_fallback() -> None:
    ranked = retrieve_relevant_memories(
        "What happened with the Shanghai plan?",
        [{"content": "Alice likes hotpot.", "importance": 5}],
        limit=3,
    )

    assert ranked == []


def test_retrieve_relevant_history_prefers_exact_chinese_phrase_without_unrelated_fallback() -> None:
    messages = [
        {"plain_text": "\u6885\u897f\u62ff\u8fc7\u4e16\u754c\u676f\u51a0\u519b\u3002", "id": 1},
        {"plain_text": "\u4eca\u665a\u5403\u706b\u9505\u3002", "id": 2},
        {"plain_text": "\u6211\u89c9\u5f97\u6885\u897f\u5728\u5df4\u8428\u65f6\u671f\u6700\u5f3a\u3002", "id": 3},
    ]

    ranked = retrieve_relevant_history("\u6885\u897f\u5728\u5df4\u8428\u600e\u4e48\u6837", messages, limit=2)

    assert [message["id"] for message in ranked] == [3, 1]


def test_retrieve_relevant_history_returns_empty_when_nothing_matches() -> None:
    messages = [{"plain_text": "\u4eca\u665a\u5403\u706b\u9505\u3002", "id": 1}]

    assert retrieve_relevant_history("\u6885\u897f\u5728\u5df4\u8428\u600e\u4e48\u6837", messages, limit=3) == []


def test_tokenize_text_normalizes_case_and_punctuation() -> None:
    assert tokenize_text("Sci-fi, Books!  ") == ["sci", "fi", "books"]


def test_tokenize_text_keeps_chinese_characters() -> None:
    assert tokenize_text("\u706b\u9505, Hotpot!") == ["\u706b", "\u9505", "hotpot"]


def test_hashed_text_embedding_is_deterministic_and_normalized() -> None:
    first = hashed_text_embedding("Alice 喜欢火锅")
    second = hashed_text_embedding("Alice 喜欢火锅")

    assert first == second
    assert len(first) == 256
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_extract_structured_memory_candidates_keeps_explicit_plan_with_source() -> None:
    candidates = extract_structured_memory_candidates(
        scope_id="10001",
        source_msg_id="m-plan",
        lines=["Alice\uff1a\u6211\u4e0b\u5468\u51c6\u5907\u53bb\u4e0a\u6d77\u3002"],
    )

    assert candidates == [
        {
            "scope_type": "group",
            "scope_id": "10001",
            "subject_type": "user",
            "subject_id": "Alice",
            "memory_kind": "plan",
            "content": "Alice: \u6211\u4e0b\u5468\u51c6\u5907\u53bb\u4e0a\u6d77.",
            "importance": 3,
            "confidence": 0.7,
            "source_msg_id": "m-plan",
            "valid_from": None,
        }
    ]


def test_is_history_detail_query_detects_temporal_followup() -> None:
    assert is_history_detail_query("\u4ed6\u4eec\u4e4b\u524d\u5230\u5e95\u51b3\u5b9a\u4e86\u4ec0\u4e48\uff1f") is True
    assert is_history_detail_query("\u4eca\u5929\u5403\u4ec0\u4e48\uff1f") is False


def test_extract_structured_memory_candidates_marks_explicit_cancellation_for_supersession() -> None:
    candidates = extract_structured_memory_candidates(
        scope_id="10001",
        source_msg_id="m-cancel",
        lines=["Alice\uff1a\u6211\u53d6\u6d88\u4e0b\u5468\u53bb\u4e0a\u6d77\u7684\u8ba1\u5212\u3002"],
    )

    assert candidates[0]["memory_kind"] == "plan"
    assert candidates[0]["supersedes_kind"] == "plan"
    assert candidates[0]["confidence"] == 0.8


def test_extract_structured_personal_decision_stays_with_speaker() -> None:
    candidates = extract_structured_memory_candidates(
        scope_id="10001",
        source_msg_id="m-decision",
        lines=["Alice: I decided to resign next month."],
    )

    assert candidates[0]["subject_type"] == "user"
    assert candidates[0]["subject_id"] == "Alice"
