from __future__ import annotations

from app.core.member_memory_backfill import parse_review_output


def test_parse_review_output_extracts_drop_set() -> None:
    text = (
        "```json\n"
        '{"drop": ["他的外公保有记忆"], '
        '"reasons": {"他的外公保有记忆": "从失忆梗反推"}}\n'
        "```"
    )

    assert parse_review_output(text) == {"他的外公保有记忆"}


def test_parse_review_output_tolerates_missing_fence() -> None:
    assert parse_review_output('{"drop": ["x", "y"]}') == {"x", "y"}
