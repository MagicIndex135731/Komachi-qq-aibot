from app.core.memory_engine import extract_memory_candidates, retrieve_relevant_memories
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


def test_tokenize_text_normalizes_case_and_punctuation() -> None:
    assert tokenize_text("Sci-fi, Books!  ") == ["sci", "fi", "books"]


def test_tokenize_text_keeps_chinese_characters() -> None:
    assert tokenize_text("\u706b\u9505, Hotpot!") == ["\u706b", "\u9505", "hotpot"]
