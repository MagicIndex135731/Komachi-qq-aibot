from app.core.memory_engine import (
    extract_memory_candidates,
    retrieve_relevant_history,
    retrieve_relevant_memories,
)
from app.providers.embeddings import tokenize_text


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
