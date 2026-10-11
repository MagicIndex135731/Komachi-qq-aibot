"""Content-free structure and usage metrics for native model requests.

These are counts, not a token estimator or a context packing policy. Missing
provider fields remain unknown here; the existing usage recorder keeps its
compatibility contract independently.
"""
from __future__ import annotations

from typing import Any


def request_context_counts(body: dict[str, Any], *, protocol: str) -> dict[str, int]:
    items = body.get("input" if protocol == "openai-responses" else "messages", [])
    entries = items if isinstance(items, list) else [items] if isinstance(items, (str, dict)) else []
    counts = {"context_items": len(entries), "context_text_chars": 0, "context_images": 0,
        "context_tools": len(body["tools"]) if isinstance(body.get("tools"), list) else 0,
        "context_function_calls": 0, "context_tool_results": 0}

    def content(value):
        if isinstance(value, str):
            counts["context_text_chars"] += len(value)
        elif isinstance(value, list):
            for part in value:
                content(part)
        elif isinstance(value, dict):
            kind = value.get("type")
            if kind in {"image", "input_image", "image_url"}:
                counts["context_images"] += 1
            elif kind in {"text", "input_text", "output_text", "reasoning_text", "summary_text"}:
                content(value.get("text"))

    content(body.get("instructions"))
    for item in entries:
        if isinstance(item, str):
            content(item)
        elif isinstance(item, dict):
            if item.get("type") == "function_call":
                counts["context_function_calls"] += 1
                content(item.get("arguments"))
            elif item.get("type") == "function_call_output":
                counts["context_tool_results"] += 1
                content(item.get("output"))
            else:
                content(item.get("content"))
                content(item.get("summary"))
                if item.get("role") == "tool":
                    counts["context_tool_results"] += 1
                calls = item.get("tool_calls")
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            counts["context_function_calls"] += 1
                            function = call.get("function")
                            if isinstance(function, dict):
                                content(function.get("arguments"))
    return counts


def usage_token_metrics(usage: Any, *, protocol: str) -> dict[str, Any]:
    metrics = {"usage_status": "unknown", "input_tokens": None, "output_tokens": None,
        "cached_input_tokens": None, "cache_token_ratio": None}
    if usage is None:
        return metrics
    if not isinstance(usage, dict):
        return {**metrics, "usage_status": "invalid"}
    responses = protocol == "openai-responses"
    details = usage.get("input_tokens_details" if responses else "prompt_tokens_details")
    if details is not None and not isinstance(details, dict):
        return {**metrics, "usage_status": "invalid"}
    values = (usage.get("input_tokens" if responses else "prompt_tokens"),
        usage.get("output_tokens" if responses else "completion_tokens"),
        details.get("cached_tokens") if isinstance(details, dict) else None)
    if any(value is not None and (type(value) is not int or value < 0) for value in values):
        return {**metrics, "usage_status": "invalid"}
    input_tokens, output_tokens, cached_tokens = values
    if input_tokens is not None and cached_tokens is not None and cached_tokens > input_tokens:
        return {**metrics, "usage_status": "invalid"}
    return {"usage_status": "reported" if all(value is not None for value in values) else "partial",
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_input_tokens": cached_tokens,
        "cache_token_ratio": round(cached_tokens / input_tokens, 6)
            if input_tokens and cached_tokens is not None else None}
