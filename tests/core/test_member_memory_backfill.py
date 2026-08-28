from __future__ import annotations

from app.core.member_memory_backfill import build_slices, parse_review_output


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


def test_build_slices_overlaps_boundaries() -> None:
    slices = build_slices(
        ["一一一一一一", "二二二二二二二二", "三三三三三三三三", "四四四四四四四四"],
        slice_chars=8,
        overlap_lines=2,
    )

    assert slices[-1][:2] == ["二二二二二二二二", "三三三三三三三三"]
    assert "四四四四四四四四" in slices[-1]
