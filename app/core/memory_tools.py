"""Memory tool schemas and argument validation for the function-calling loop."""

from __future__ import annotations

from typing import Any


MEMORY_TOOL_KINDS = frozenset(
    {"fact", "preference", "taboo", "plan", "decision", "profile"}
)
MEMORY_TOOL_LAYERS = frozenset({"raw", "facts", "summaries", "all"})
MEMORY_TOOL_CONTENT_LIMIT = 400
MEMORY_TOOL_LIMIT_MIN = 1
MEMORY_TOOL_LIMIT_MAX = 20


def memory_tool_schemas() -> list[dict[str, Any]]:
    """Return the OpenAI-compatible function schemas exposed to the model."""
    return [
        {
            "type": "function",
            "name": "memory_search",
            "description": (
                "Search this group's memory across raw original messages, "
                "structured facts, or episode summaries. Results are always "
                "scoped to the current group."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or a natural-language question.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": sorted(MEMORY_TOOL_LAYERS),
                        "default": "all",
                    },
                    "member": {
                        "type": "string",
                        "description": "Optional member nickname or QQ ID to restrict results.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": MEMORY_TOOL_LIMIT_MIN,
                        "maximum": MEMORY_TOOL_LIMIT_MAX,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "memory_read",
            "description": (
                "Read a member's profile facts (preferences, taboos, "
                "relationships) and recent activity in this group."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member": {
                        "type": "string",
                        "description": "Member nickname or QQ ID.",
                    },
                },
                "required": ["member"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "memory_write",
            "description": (
                "Persist a source-backed fact about the current user or the "
                "group. Only facts explicitly stated in the current "
                "conversation may be written; every source message must be a "
                "real message from this group in this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": sorted(MEMORY_TOOL_KINDS),
                    },
                    "subject": {
                        "type": "string",
                        "description": "Current user's QQ ID or the literal 'group'.",
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Short predicate such as 'likes' or 'prefers'.",
                    },
                    "object_text": {
                        "type": "string",
                        "description": "Short object of the fact.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Full fact sentence, at most "
                            f"{MEMORY_TOOL_CONTENT_LIMIT} characters."
                        ),
                    },
                    "source_msg_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Real platform message IDs from this conversation.",
                    },
                },
                "required": [
                    "kind",
                    "subject",
                    "predicate",
                    "object_text",
                    "content",
                    "source_msg_ids",
                ],
                "additionalProperties": False,
            },
        },
    ]


def validate_memory_search_args(arguments: dict[str, Any]) -> str | None:
    """Return an error message for invalid memory_search arguments."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return "query must be a non-empty string"
    layer = str(arguments.get("layer") or "all").strip()
    if layer not in MEMORY_TOOL_LAYERS:
        return "layer must be one of raw, facts, summaries, all"
    limit = arguments.get("limit", MEMORY_TOOL_LIMIT_MAX)
    if not isinstance(limit, int) or isinstance(limit, bool):
        return "limit must be an integer"
    if not MEMORY_TOOL_LIMIT_MIN <= limit <= MEMORY_TOOL_LIMIT_MAX:
        return f"limit must be between {MEMORY_TOOL_LIMIT_MIN} and {MEMORY_TOOL_LIMIT_MAX}"
    member = arguments.get("member")
    if member is not None and not isinstance(member, str):
        return "member must be a string"
    return None


def validate_memory_write_args(arguments: dict[str, Any]) -> str | None:
    """Return an error message for invalid memory_write arguments."""
    kind = arguments.get("kind")
    if kind not in MEMORY_TOOL_KINDS:
        return "kind is not in the allowed set"
    subject = arguments.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        return "subject must be a non-empty string"
    for field in ("predicate", "object_text", "content"):
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"{field} must be a non-empty string"
    if len(str(arguments.get("content") or "")) > MEMORY_TOOL_CONTENT_LIMIT:
        return f"content exceeds {MEMORY_TOOL_CONTENT_LIMIT} characters"
    source_ids = arguments.get("source_msg_ids")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(value, str) or not value.strip() for value in source_ids)
    ):
        return "source_msg_ids must be a non-empty list of strings"
    return None
