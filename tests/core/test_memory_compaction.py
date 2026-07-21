from dataclasses import asdict

import pytest

from app.core.memory_compaction import (
    MemoryFact,
    build_memory_compaction_prompt,
    canonical_key,
    parse_memory_compaction_response,
    structured_digest,
)


def _fact(*, source_ids: list[str] | None = None, content: str = "Alice likes hotpot.") -> dict:
    return {
        "kind": "preference",
        "subject_id": "Alice",
        "predicate": "likes",
        "object_text": "hotpot",
        "content": content,
        "importance": 4,
        "confidence": 0.8,
        "source_msg_ids": source_ids or ["m-1"],
        "valid_until": None,
        "ignored_by_parser": "do not persist",
    }


def test_parser_filters_fields_and_deduplicates_source_backed_facts() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice prefers hotpot.",
            "facts": [
                _fact(source_ids=["m-1"]),
                _fact(source_ids=["m-2"], content="Alice enjoys hotpot."),
            ],
            "unsafe_top_level": True,
        },
        allowed_source_msg_ids={"m-1", "m-2"},
    )

    assert result.summary == "Alice prefers hotpot."
    assert len(result.facts) == 1
    assert asdict(result.facts[0]) == {
        "kind": "preference",
        "subject_id": "Alice",
        "predicate": "likes",
        "object_text": "hotpot",
        "content": "Alice likes hotpot.",
        "importance": 4,
        "confidence": 0.8,
        "source_msg_ids": ("m-1", "m-2"),
        "valid_until": None,
    }


def test_parser_discards_hallucinated_source_ids_and_uses_summary_fallback() -> None:
    result = parse_memory_compaction_response(
        '{"summary":"unsafe summary","facts":[{"kind":"fact","subject_id":"group","predicate":"meeting","object_text":"Saturday","content":"Meeting is Saturday.","importance":4,"confidence":0.9,"source_msg_ids":["invented"],"valid_until":null}]}',
        allowed_source_msg_ids={"m-1"},
        fallback_text="Recent chat: no verified memory.",
    )

    assert result.summary == "unsafe summary"
    assert result.facts == ()


def test_parser_bad_json_returns_safe_summary_only_fallback() -> None:
    result = parse_memory_compaction_response("not json", allowed_source_msg_ids={"m-1"}, fallback_text="Recent chat: hello")

    assert result.summary == "Recent chat: hello"
    assert result.facts == ()


def test_parser_discards_unknown_subject_ids() -> None:
    result = parse_memory_compaction_response(
        {"summary": "x", "facts": [_fact()]},
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"10001", "group"},
    )

    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_parser_rejects_user_fact_citing_another_authors_message() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "A statement was made.",
            "facts": [
                {
                    "kind": "preference",
                    "subject_id": "42",
                    "predicate": "likes",
                    "object_text": "hotpot",
                    "content": "Alice likes hotpot.",
                    "importance": 4,
                    "confidence": 0.9,
                    "source_msg_ids": ["m-bob"],
                    "valid_until": None,
                }
            ],
        },
        allowed_source_msg_ids={"m-bob"},
        allowed_subject_ids={"42", "43", "group"},
        source_subject_ids={"m-bob": "43"},
    )

    assert result.facts == ()


def test_strict_parser_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response("not json", strict=True)


def test_strict_parser_raises_on_incomplete_schema() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response({"summary": "missing facts"}, strict=True)


def test_strict_parser_raises_on_blank_summary() -> None:
    with pytest.raises(ValueError):
        parse_memory_compaction_response({"summary": "   ", "facts": []}, strict=True)


def test_strict_parser_rejects_invalid_fact_without_losing_valid_summary() -> None:
    result = parse_memory_compaction_response(
        {"summary": "source-backed window summary", "facts": [{"kind": "preference"}]},
        strict=True,
    )

    assert result.summary == "source-backed window summary"
    assert result.facts == ()
    assert result.rejected_fact_count == 1


def test_parser_safely_normalizes_missing_content_and_blank_valid_until() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice has a current plan.",
            "facts": [
                {
                    "kind": "current", "subject_id": "42", "predicate": "plans",
                    "object_text": "visit Shanghai", "importance": 4, "confidence": 0.9,
                    "source_msg_ids": ["m-1"], "valid_until": "",
                }
            ],
        },
        allowed_source_msg_ids={"m-1"},
        allowed_subject_ids={"42"},
        source_subject_ids={"m-1": "42"},
        strict=True,
    )

    assert result.facts[0].content == "42: plans visit Shanghai"
    assert result.facts[0].valid_until is None


def test_parser_normalizes_noncritical_model_format_variance() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "An event occurred.",
            "facts": [
                {
                    "kind": "event", "subject_id": "42", "predicate": "attended",
                    "object_text": "meeting", "content": "Alice attended the meeting.",
                    "importance": 7.2, "confidence": 1.4, "source_msg_ids": [123],
                    "valid_until": "unknown",
                }
            ],
        },
        allowed_source_msg_ids={"123"},
        allowed_subject_ids={"42"},
        source_subject_ids={"123": "42"},
        strict=True,
    )

    assert result.facts[0].importance == 5
    assert result.facts[0].confidence == 1.0
    assert result.facts[0].source_msg_ids == ("123",)
    assert result.facts[0].valid_until is None


def test_parser_rejects_single_author_personal_fact_mislabeled_as_group() -> None:
    result = parse_memory_compaction_response(
        {
            "summary": "Alice made a decision.",
            "facts": [
                {
                    "kind": "decision",
                    "subject_id": "group",
                    "predicate": "decided",
                    "object_text": "resign",
                    "content": "Alice decided to resign.",
                    "importance": 4,
                    "confidence": 0.9,
                    "source_msg_ids": ["m-alice"],
                    "valid_until": None,
                }
            ],
        },
        allowed_source_msg_ids={"m-alice"},
        allowed_subject_ids={"42", "group"},
        source_subject_ids={"m-alice": "42"},
    )

    assert result.facts == ()


def test_canonical_key_normalizes_case_spacing_and_unicode() -> None:
    assert canonical_key("Preference", " A\u3000lice ", "LIKES", "Hotpot") == canonical_key(
        "preference", "a lice", "likes", " hotpot "
    )


def test_prompt_builder_has_localized_schema_and_citable_messages() -> None:
    chinese = build_memory_compaction_prompt(
        language="zh",
        previous_digest="Rolling group memory: old detail",
        messages=[{"platform_msg_id": "m-1", "plain_text": "Alice 喜欢火锅"}],
    )
    english = build_memory_compaction_prompt(
        language="en",
        messages=[{"message_id": "m-2", "content": "Alice likes hotpot"}],
    )

    assert "write summary and fact content in Chinese" in chinese
    assert "If any field is uncertain, omit that fact" in chinese
    assert "or expired as kind" in chinese
    assert "[m-1] Alice 喜欢火锅" in chinese
    assert "Rolling group memory:" not in chinese
    assert "Output exactly one compact JSON object" in english
    assert "[m-2] Alice likes hotpot" in english


def test_structured_digest_is_deterministic_and_never_nests_rolling_labels() -> None:
    fact = MemoryFact(
        kind="preference",
        subject_id="Alice",
        predicate="likes",
        object_text="hotpot",
        content="Alice likes hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-1",),
    )

    digest = structured_digest("Rolling group memory: Rolling group memory: Alice prefers hotpot.", [fact, fact])

    assert digest == (
        "Memory digest:\n"
        "summary: Alice prefers hotpot.\n"
        "facts:\n"
        "- preference | Alice | likes | hotpot | Alice likes hotpot. | sources=m-1 | valid_until=null"
    )


def test_structured_digest_can_compact_its_own_output_without_promoting_old_facts_into_summary() -> None:
    first = structured_digest("Rolling group memory: Alice prefers hotpot.")

    assert structured_digest(first) == "Memory digest:\nsummary: Alice prefers hotpot.\nfacts:\n- (none)"


def test_structured_digest_is_stable_for_source_order_and_equivalent_fact_order() -> None:
    first = MemoryFact(
        kind="Preference",
        subject_id=" Alice ",
        predicate="LIKES",
        object_text=" hotpot ",
        content="Alice likes hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-2", "m-1"),
    )
    second = MemoryFact(
        kind="preference",
        subject_id="Alice",
        predicate="likes",
        object_text="hotpot",
        content="Alice Likes Hotpot.",
        importance=4,
        confidence=0.8,
        source_msg_ids=("m-1", "m-2"),
    )

    assert structured_digest("summary", [first, second]) == structured_digest("summary", [second, first])


def test_parser_rejects_fact_with_blank_or_non_string_source_id() -> None:
    result = parse_memory_compaction_response(
        {"summary": "test", "facts": [_fact(source_ids=["m-1", " "]), _fact(source_ids=["m-1", None])]},
        allowed_source_msg_ids={"m-1"},
    )

    assert result.facts == ()


def test_structured_digest_sorts_sources_for_a_single_direct_fact() -> None:
    unordered = MemoryFact(
        kind="fact",
        subject_id="group",
        predicate="meeting",
        object_text="Saturday",
        content="Meeting is Saturday.",
        importance=4,
        confidence=0.9,
        source_msg_ids=("m-2", "m-1"),
    )
    ordered = MemoryFact(
        kind="fact",
        subject_id="group",
        predicate="meeting",
        object_text="Saturday",
        content="Meeting is Saturday.",
        importance=4,
        confidence=0.9,
        source_msg_ids=("m-1", "m-2"),
    )

    assert structured_digest("summary", [unordered]) == structured_digest("summary", [ordered])


def test_structured_digest_breaks_complete_canonical_ties_stably() -> None:
    first = MemoryFact("Fact", "same", "P", "O", "same", 4, 0.8, ("m-1",))
    second = MemoryFact("fact", "same", "p", "o", "same", 4, 0.8, ("m-1",))

    assert structured_digest("summary", [first, second]) == structured_digest("summary", [second, first])
