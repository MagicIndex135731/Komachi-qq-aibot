from __future__ import annotations

import pytest

from app.core.memory_tools import (
    MEMORY_TOOL_CONTENT_LIMIT,
    MEMORY_TOOL_KINDS,
    memory_tool_schemas,
    validate_memory_search_args,
    validate_memory_write_args,
)


def test_memory_tool_schemas_expose_three_strict_functions() -> None:
    schemas = memory_tool_schemas()
    assert [schema["name"] for schema in schemas] == [
        "memory_search",
        "memory_read",
        "memory_write",
    ]
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["parameters"]["additionalProperties"] is False
    search_params = schemas[0]["parameters"]["properties"]
    assert search_params["layer"]["enum"] == ["all", "facts", "raw", "summaries"]
    assert search_params["limit"]["maximum"] == 20
    write_params = schemas[2]["parameters"]
    assert set(write_params["required"]) == {
        "kind",
        "subject",
        "predicate",
        "object_text",
        "content",
        "source_msg_ids",
    }
    assert write_params["properties"]["kind"]["enum"] == sorted(MEMORY_TOOL_KINDS)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"query": ""}, "query must be a non-empty string"),
        ({"query": "x", "layer": "magic"}, "layer must be one of raw, facts, summaries, all"),
        ({"query": "x", "limit": 0}, "limit must be between 1 and 20"),
        ({"query": "x", "limit": "5"}, "limit must be an integer"),
        ({"query": "x", "member": 42}, "member must be a string"),
    ],
)
def test_validate_memory_search_args_rejects_invalid(arguments, error) -> None:
    assert validate_memory_search_args(arguments) == error


def test_validate_memory_search_args_accepts_valid() -> None:
    assert validate_memory_search_args(
        {"query": "冰美式", "layer": "facts", "member": "阿渣", "limit": 3}
    ) is None


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"kind": "expired"}, "kind is not in the allowed set"),
        (
            {"kind": "preference", "subject": ""},
            "subject must be a non-empty string",
        ),
        (
            {"kind": "preference", "subject": "99", "predicate": ""},
            "predicate must be a non-empty string",
        ),
        (
            {
                "kind": "preference",
                "subject": "99",
                "predicate": "likes",
                "object_text": "",
            },
            "object_text must be a non-empty string",
        ),
        (
            {
                "kind": "preference",
                "subject": "99",
                "predicate": "likes",
                "object_text": "x",
                "content": "",
            },
            "content must be a non-empty string",
        ),
        (
            {
                "kind": "preference",
                "subject": "99",
                "predicate": "likes",
                "object_text": "x",
                "content": "y" * (MEMORY_TOOL_CONTENT_LIMIT + 1),
            },
            f"content exceeds {MEMORY_TOOL_CONTENT_LIMIT} characters",
        ),
        (
            {
                "kind": "preference",
                "subject": "99",
                "predicate": "likes",
                "object_text": "x",
                "content": "y",
                "source_msg_ids": [],
            },
            "source_msg_ids must be a non-empty list of strings",
        ),
    ],
)
def test_validate_memory_write_args_rejects_invalid(arguments, error) -> None:
    assert validate_memory_write_args(arguments) == error


def test_validate_memory_write_args_accepts_valid() -> None:
    assert validate_memory_write_args(
        {
            "kind": "preference",
            "subject": "99",
            "predicate": "likes",
            "object_text": "冰美式",
            "content": "提问者喜欢喝冰美式",
            "source_msg_ids": ["tool-query"],
        }
    ) is None
