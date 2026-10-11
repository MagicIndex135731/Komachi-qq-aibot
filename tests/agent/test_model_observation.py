"""Synthetic request metrics never retain provider or conversation content."""
import json

import pytest

from app.agent.model_observation import request_context_counts, usage_token_metrics


@pytest.mark.parametrize("protocol,body", [
    ("openai-responses", {"instructions": "rules", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"},
            {"type": "input_image", "image_url": "https://private.invalid/image?secret=value"}]},
        {"type": "function_call", "arguments": "{}"},
        {"type": "function_call_output", "output": "result"},
    ], "tools": [{"name": "read"}]}),
    ("openai-completions", {"messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret"}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": "{}"}}]},
        {"role": "tool", "content": "result"},
    ], "tools": [{"function": {"name": "read"}}]}),
])
def test_context_counts_support_both_native_protocols_without_content(protocol, body):
    metrics = request_context_counts(body, protocol=protocol)
    assert metrics["context_items"] == (3 if protocol == "openai-responses" else 4)
    assert metrics["context_text_chars"] == 18
    assert metrics["context_images"] == metrics["context_tools"] == 1
    assert metrics["context_function_calls"] == metrics["context_tool_results"] == 1
    serialized = json.dumps(metrics)
    for forbidden in ("rules", "hello", '"result"', "secret", "https", "data:"):
        assert forbidden not in serialized


@pytest.mark.parametrize("protocol,usage", [
    ("openai-responses", {"input_tokens": 20, "output_tokens": 4,
        "input_tokens_details": {"cached_tokens": 12}}),
    ("openai-completions", {"prompt_tokens": 20, "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 12}}),
])
def test_usage_observation_distinguishes_reported_zero_and_cache_ratio(protocol, usage):
    metrics = usage_token_metrics(usage, protocol=protocol)
    assert metrics == {"usage_status": "reported", "input_tokens": 20, "output_tokens": 4,
        "cached_input_tokens": 12, "cache_token_ratio": 0.6}
    usage["input_tokens_details" if protocol == "openai-responses" else "prompt_tokens_details"]["cached_tokens"] = 0
    assert usage_token_metrics(usage, protocol=protocol)["cache_token_ratio"] == 0.0


@pytest.mark.parametrize("usage,expected_status", [(None, "unknown"), ({}, "partial"),
    ({"input_tokens": 20, "output_tokens": 4}, "partial"),
    ({"input_tokens": "secret-url", "output_tokens": True}, "invalid"),
    ({"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 20}}, "invalid")])
def test_missing_and_invalid_usage_never_becomes_false_zero(usage, expected_status):
    metrics = usage_token_metrics(usage, protocol="openai-responses")
    assert metrics["usage_status"] == expected_status
    assert metrics["cache_token_ratio"] is None
    if usage is None or not usage or expected_status == "invalid":
        assert metrics["input_tokens"] is None
    assert metrics["cached_input_tokens"] is None
    assert "secret-url" not in json.dumps(metrics)
