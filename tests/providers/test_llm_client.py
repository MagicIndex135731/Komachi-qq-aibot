import json
import base64
from pathlib import Path

import httpx
import pytest

from app.providers.llm_client import LlmClient
from app.core.message_content import ImageAttachment


def _responses_stream_body(
    *,
    response_id: str,
    text: str,
    usage: dict | None = None,
) -> str:
    completed_response = {
        "id": response_id,
        "object": "response",
        "status": "completed",
    }
    if usage is not None:
        completed_response["usage"] = usage
    return (
        f'event: response.created\n'
        f'data: {json.dumps({"type": "response.created", "response": {"id": response_id}})}\n\n'
        f'event: response.output_item.added\n'
        f'data: {json.dumps({"type": "response.output_item.added", "item": {"id": "msg_1"}})}\n\n'
        f'event: response.content_part.added\n'
        f'data: {json.dumps({"type": "response.content_part.added", "part": {"type": "output_text"}})}\n\n'
        f'event: response.output_text.delta\n'
        f'data: {json.dumps({"type": "response.output_text.delta", "delta": text})}\n\n'
        f'event: response.output_text.done\n'
        f'data: {json.dumps({"type": "response.output_text.done", "text": text})}\n\n'
        f'event: response.completed\n'
        f'data: {json.dumps({"type": "response.completed", "response": completed_response})}\n\n'
    )


def test_llm_client_posts_to_chat_completions_endpoint_with_bearer_auth() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello from model",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        [
            "System persona: You are Mira.",
            "Safety rules: Stay safe.",
            "Group policy: Speak only in allowlisted groups.",
            "Reply style: Talk like a real person in chat.",
            "Recent messages:\nAlice: hi\nMira: hello",
            "Target message: Alice: hi",
        ]
    )

    assert text == "hello from model"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gpt-5.4"
    assert captured["payload"]["messages"] == [
        {
            "role": "system",
            "content": (
                "System persona: You are Mira.\n\n"
                "Safety rules: Stay safe.\n\n"
                "Group policy: Speak only in allowlisted groups.\n\n"
                "Reply style: Talk like a real person in chat."
            ),
        },
        {
            "role": "user",
            "content": "Recent messages:\nAlice: hi\nMira: hello\n\nTarget message: Alice: hi",
        },
    ]


def test_llm_client_accepts_streamed_chat_completions_response() -> None:
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"hello "}}]}',
                'data: {"choices":[{"delta":{"content":"from stream"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "hello from stream"
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"
    assert recorded[0].input_tokens == 11
    assert recorded[0].output_tokens == 3


def test_llm_client_posts_cc_models_to_anthropic_messages_endpoint() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["anthropic_version"] = request.headers.get("anthropic-version")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": "hello from cc model"},
                ],
                "usage": {
                    "input_tokens": 101,
                    "output_tokens": 19,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    recorded = []
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(
        [
            "System persona: You are Mira.",
            "Safety rules: Stay safe.",
            "Reply style: Talk like a real person in chat.",
            "Recent messages:\nAlice: hi\nMira: hello",
            "Target message: Alice: hi",
        ]
    )

    assert text == "hello from cc model"
    assert captured["url"] == "https://api.example.test/v1/messages"
    assert captured["x_api_key"] == "test-key"
    assert captured["anthropic_version"] == "2023-06-01"
    assert captured["payload"] == {
        "model": "cc-gpt-5.4",
        "system": (
            "System persona: You are Mira.\n\n"
            "Safety rules: Stay safe.\n\n"
            "Reply style: Talk like a real person in chat."
        ),
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Recent messages:\nAlice: hi\nMira: hello\n\nTarget message: Alice: hi",
            }
        ],
    }
    assert len(recorded) == 1
    assert recorded[0].endpoint == "anthropic_messages"
    assert recorded[0].input_tokens == 101
    assert recorded[0].cached_input_tokens == 0
    assert recorded[0].output_tokens == 19


def test_llm_client_records_usage_from_anthropic_messages_cache_reads() -> None:
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [{"type": "text", "text": "hello from cc model"}],
                "usage": {
                    "input_tokens": 885,
                    "cache_read_input_tokens": 3840,
                    "output_tokens": 53,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "hello from cc model"
    assert len(recorded) == 1
    assert recorded[0].endpoint == "anthropic_messages"
    assert recorded[0].input_tokens == 4725
    assert recorded[0].cached_input_tokens == 3840
    assert recorded[0].output_tokens == 53


def test_llm_client_falls_back_to_chat_completions_when_cc_messages_transport_fails() -> None:
    captured_urls: list[str] = []
    anthropic_attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        if str(request.url).endswith("/messages"):
            anthropic_attempts["count"] += 1
            raise httpx.ConnectError("tls eof", request=request)
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reply from fallback chat completions",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 33,
                        "completion_tokens": 7,
                    },
                },
            )
        raise AssertionError(f"unexpected url: {request.url}")

    transport = httpx.MockTransport(handler)
    recorded = []
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert anthropic_attempts["count"] == client.REQUEST_MAX_ATTEMPTS
    assert text == "reply from fallback chat completions"
    assert captured_urls == ["https://api.example.test/v1/messages"] * client.REQUEST_MAX_ATTEMPTS + [
        "https://api.example.test/v1/chat/completions"
    ]
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"


def test_llm_client_falls_back_to_configured_chat_model_when_cc_messages_transport_fails() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/messages"):
            raise httpx.ConnectError("tls eof", request=request)
        if str(request.url).endswith("/chat/completions"):
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reply from gpt fallback model",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 5,
                    },
                },
            )
        raise AssertionError(f"unexpected url: {request.url}")

    transport = httpx.MockTransport(handler)
    recorded = []
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        fallback_model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply from gpt fallback model"
    assert captured_payloads == [
        {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Target message: Alice: hi"}],
        }
    ]
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"
    assert recorded[0].model == "gpt-5.4"


def test_llm_client_reads_nested_chat_completions_content_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "nested response text",
                                }
                            ],
                        },
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "nested response text"


def test_llm_client_reads_text_from_unexpected_sse_chat_completions_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = (
            'data: {"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null,"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop","index":0}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "hello world"


def test_llm_client_makes_single_chat_completions_request() -> None:
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "single request reply",
                        },
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "single request reply"
    assert captured_urls == ["https://api.example.test/v1/chat/completions"]


def test_llm_client_retries_chat_completions_when_success_response_has_invalid_json() -> None:
    captured_urls: list[str] = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                200,
                content=b"",
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "reply after retry",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply after retry"
    assert captured_urls == [
        "https://api.example.test/v1/chat/completions",
        "https://api.example.test/v1/chat/completions",
    ]


def test_llm_client_retries_chat_completions_when_request_times_out() -> None:
    captured_urls: list[str] = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "reply after timeout retry",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply after timeout retry"
    assert captured_urls == [
        "https://api.example.test/v1/chat/completions",
        "https://api.example.test/v1/chat/completions",
    ]


def test_llm_client_retries_chat_completions_when_server_returns_502() -> None:
    captured_urls: list[str] = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "reply after status retry",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply after status retry"
    assert captured_urls == [
        "https://api.example.test/v1/chat/completions",
        "https://api.example.test/v1/chat/completions",
    ]


def test_llm_client_falls_back_to_configured_chat_model_when_primary_chat_model_returns_http_error() -> None:
    captured_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_models.append(payload["model"])
        if payload["model"] == "gpt-5.4-mini":
            return httpx.Response(400, json={"error": "upstream bad response"})
        if payload["model"] == "gpt-4o-mini":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reply from fallback chat model",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                    },
                },
            )
        raise AssertionError(f"unexpected model: {payload['model']}")

    transport = httpx.MockTransport(handler)
    recorded = []
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4-mini",
        fallback_model="gpt-4o-mini",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply from fallback chat model"
    assert captured_models == ["gpt-5.4-mini", "gpt-4o-mini"]
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"
    assert recorded[0].model == "gpt-4o-mini"


def test_llm_client_switches_to_fallback_chat_model_after_first_primary_timeout() -> None:
    captured_models: list[str] = []
    primary_attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = payload["model"]
        captured_models.append(model)
        if model == "gpt-5.4-mini":
            primary_attempts["count"] += 1
            raise httpx.ReadTimeout("timed out", request=request)
        if model == "gpt-4o-mini":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reply from timeout fallback chat model",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                    },
                },
            )
        raise AssertionError(f"unexpected model: {model}")

    transport = httpx.MockTransport(handler)
    recorded = []
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4-mini",
        fallback_model="gpt-4o-mini",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply from timeout fallback chat model"
    assert primary_attempts["count"] == 1
    assert captured_models == ["gpt-5.4-mini", "gpt-4o-mini"]
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"
    assert recorded[0].model == "gpt-4o-mini"


def test_llm_client_treats_reply_style_as_system_instruction() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    client.generate_text(
        [
            "System persona: You are Mira.",
            "Safety rules: Stay safe.",
            "Group policy: Speak only in allowlisted groups.",
            "Reply style: Talk like a real person in chat.",
            "Target message: Alice: hi",
        ]
    )

    assert "Reply style: Talk like a real person in chat." in captured["payload"]["messages"][0]["content"]


def test_llm_client_propagates_non_retryable_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    try:
        client.generate_text(["Target message: Alice: hi"])
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected HTTPStatusError")


def test_llm_client_raises_value_error_when_chat_payload_has_no_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                }
                            ],
                        },
                    },
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    try:
        client.generate_text(["Target message: Alice: hi"])
    except ValueError as exc:
        assert str(exc) == "model response did not include output text"
    else:
        raise AssertionError("expected ValueError")


def test_llm_client_records_usage_from_chat_completions() -> None:
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello from model"}}],
                "usage": {
                    "prompt_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens": 45,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "hello from model"
    assert len(recorded) == 1
    assert recorded[0].endpoint == "chat_completions"
    assert recorded[0].input_tokens == 120
    assert recorded[0].cached_input_tokens == 20
    assert recorded[0].output_tokens == 45


def test_llm_client_skips_usage_record_when_chat_completions_has_no_usage() -> None:
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "reply without usage"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "reply without usage"
    assert recorded == []


def test_llm_client_raises_value_error_when_anthropic_payload_has_no_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    try:
        client.generate_text(["Target message: Alice: hi"])
    except ValueError as exc:
        assert str(exc) == "model response did not include output text"
    else:
        raise AssertionError("expected ValueError")


def test_llm_client_posts_cc_models_with_image_blocks_to_anthropic_messages_endpoint() -> None:
    captured = {}
    attempts = {"image_get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert str(request.url) == "https://img.example.test/cat.png"
            attempts["image_get"] += 1
            if attempts["image_get"] == 1:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(
                200,
                content=b"png-bytes",
                headers={"content-type": "image/png"},
            )

        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [{"type": "text", "text": "saw the image"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png")],
    )

    assert text == "saw the image"
    assert attempts["image_get"] == 2
    assert captured["url"] == "https://api.example.test/v1/messages"
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target message: Alice: look at this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(b"png-bytes").decode("ascii"),
                    },
                },
            ],
        }
    ]


def test_llm_client_posts_chat_completions_with_image_data_urls_when_images_provided() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert str(request.url) == "https://img.example.test/cat.png"
            return httpx.Response(
                200,
                content=b"jpeg-bytes",
                headers={"content-type": "image/jpeg"},
            )

        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "saw the image",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png")],
    )

    assert text == "saw the image"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target message: Alice: look at this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
                    },
                },
            ],
        }
    ]


def test_llm_client_prefers_local_image_path_over_remote_url_when_available(tmp_path) -> None:
    captured = {}
    cached_image = tmp_path / "cached.png"
    cached_image.write_bytes(b"cached-png-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            raise AssertionError("remote image download should be skipped when local cache exists")

        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [{"type": "text", "text": "saw the cached image"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[
            ImageAttachment(
                url="https://img.example.test/expired.png",
                file_id="expired.png",
                local_path=str(cached_image),
            )
        ],
    )

    assert text == "saw the cached image"
    assert captured["url"] == "https://api.example.test/v1/messages"
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target message: Alice: look at this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(b"cached-png-bytes").decode("ascii"),
                    },
                },
            ],
        }
    ]


def test_llm_client_uses_responses_stream_model_for_text_when_configured() -> None:
    captured = {}
    recorded = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            text=_responses_stream_body(
                response_id="resp_test_1",
                text="hello from responses stream",
                usage={
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 45,
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        fallback_model="gpt-5.4",
        responses_model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
        usage_recorder=recorded.append,
    )

    text = client.generate_text(
        [
            "System persona: You are Mira.",
            "Safety rules: Stay safe.",
            "Reply style: Talk like a real person in chat.",
            "Recent messages:\nAlice: hi\nMira: hello",
            "Target message: Alice: hi",
        ]
    )

    assert text == "hello from responses stream"
    assert captured["url"] == "https://api.example.test/v1/responses"
    assert captured["payload"] == {
        "model": "gpt-5.4",
        "stream": True,
        "instructions": (
            "System persona: You are Mira.\n\n"
            "Safety rules: Stay safe.\n\n"
            "Reply style: Talk like a real person in chat."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Recent messages:\nAlice: hi\nMira: hello\n\nTarget message: Alice: hi",
                    }
                ],
            }
        ],
    }
    assert len(recorded) == 1
    assert recorded[0].endpoint == "responses"
    assert recorded[0].model == "gpt-5.4"
    assert recorded[0].input_tokens == 120
    assert recorded[0].cached_input_tokens == 20
    assert recorded[0].output_tokens == 45


def test_llm_client_does_not_send_previous_response_id_on_http_responses_endpoint() -> None:
    captured_payloads: list[dict] = []
    responses = iter(
        (
            _responses_stream_body(response_id="resp_prev_1", text="first reply"),
            _responses_stream_body(response_id="resp_prev_2", text="second reply"),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            text=next(responses),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        fallback_model="gpt-5.4",
        responses_model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    first = client.generate_text(["Target message: Alice: first"], conversation_key="session-1")
    second = client.generate_text(["Target message: Alice: second"], conversation_key="session-1")

    assert first == "first reply"
    assert second == "second reply"
    assert captured_payloads[0] == {
        "model": "gpt-5.4",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Target message: Alice: first"}],
            }
        ],
    }
    assert captured_payloads[1] == {
        "model": "gpt-5.4",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Target message: Alice: second"}],
            }
        ],
    }


def test_llm_client_routes_images_to_responses_with_input_image_when_configured() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                content=b"png-bytes",
                headers={"content-type": "image/png"},
            )
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            text=_responses_stream_body(response_id="resp_img_1", text="responses image reply"),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="cc-gpt-5.4",
        fallback_model="gpt-5.4",
        responses_model="gpt-5.4",
        compat_model="cc-gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png")],
        conversation_key="session-1",
    )

    assert text == "responses image reply"
    assert captured["url"] == "https://api.example.test/v1/responses"
    assert captured["payload"] == {
        "model": "gpt-5.4",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Target message: Alice: look at this"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii"),
                    },
                ],
            }
        ],
    }


def test_llm_client_routes_images_to_dedicated_vision_model_when_configured() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                content=b"png-bytes",
                headers={"content-type": "image/png"},
            )
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "vision reply",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4-mini",
        fallback_model="gpt-4o-mini",
        vision_model="gpt-4o",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png")],
        conversation_key="session-1",
    )

    assert text == "vision reply"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-4o"


def test_llm_client_adds_minimal_reasoning_effort_for_gpt_5_nano() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "cheap fallback reply",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5-nano",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(["Target message: Alice: hi"])

    assert text == "cheap fallback reply"
    assert captured["payload"]["model"] == "gpt-5-nano"
    assert captured["payload"]["reasoning_effort"] == "minimal"


def test_llm_client_uses_string_image_url_for_codexzh_proxy_chat_payload() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                content=b"jpeg-bytes",
                headers={"content-type": "image/jpeg"},
            )
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "proxy image reply",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.codexzh.com/v1",
        api_key="test-key",
        model="gpt-5.4",
        compat_model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    text = client.generate_text(
        ["Target message: Alice: look at this"],
        images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png")],
    )

    assert text == "proxy image reply"
    assert captured["url"] == "https://api.codexzh.com/v1/chat/completions"
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target message: Alice: look at this"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii"),
                },
            ],
        }
    ]


def test_llm_client_posts_to_images_generations_endpoint() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "created": 123,
                "data": [{"b64_json": "abc"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        size="1024x1024",
    )

    assert captured["url"] == "https://api.example.test/v1/images/generations"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "model": "gpt-image-2",
        "prompt": "draw a cat",
        "n": 1,
        "response_format": "url",
        "size": "1024x1024",
    }
    assert result.created == 123
    assert result.images == [{"b64_json": "abc"}]


def test_llm_client_retries_images_generations_when_server_returns_502() -> None:
    captured_urls: list[str] = []
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(
            200,
            json={
                "created": 123,
                "data": [{"url": "https://img.example.test/generated.png"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    result = client.generate_image(prompt="draw a cat", model="gpt-image-2")

    assert result.images == [{"url": "https://img.example.test/generated.png"}]
    assert captured_urls == [
        "https://api.example.test/v1/images/generations",
        "https://api.example.test/v1/images/generations",
    ]


def test_llm_client_posts_to_images_edits_endpoint_with_reference_images(tmp_path) -> None:
    captured = {}
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"png-bytes")

    class FakeHttpClient:
        def post(self, url, *, headers=None, data=None, files=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            captured["files"] = files
            captured["timeout"] = timeout
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers),
                json={
                    "created": 123,
                    "data": [{"b64_json": "abc"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=FakeHttpClient(),
    )

    result = client.edit_image(
        prompt="turn this into watercolor",
        model="gpt-image-2",
        images=[
            ImageAttachment(
                url="https://img.example.test/source.png",
                file_id="source.png",
                local_path=str(source_path),
            )
        ],
        size="1024x1024",
        quality="low",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        timeout_seconds=60.0,
    )

    assert result.images == [{"b64_json": "abc"}]
    assert captured["url"] == "https://api.example.test/v1/images/edits"
    assert captured["headers"] == {"Authorization": "Bearer test-key"}
    assert captured["data"] == {
        "model": "gpt-image-2",
        "prompt": "turn this into watercolor",
        "n": "1",
        "size": "1024x1024",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": "70",
        "moderation": "low",
    }
    assert captured["timeout"] == 60.0
    assert len(captured["files"]) == 1
    assert captured["files"][0][0] == "image"
    assert captured["files"][0][1][0] == "source.png"
    assert captured["files"][0][1][1] == b"png-bytes"
    assert captured["files"][0][1][2] == "image/png"


def test_llm_client_posts_to_configured_images_generations_endpoint() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "created": 123,
                "data": [{"b64_json": "abc"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        image_generations_endpoint="/v1/images/generations",
        http_client=httpx.Client(transport=transport),
    )

    result = client.generate_image(prompt="draw a cat", model="gpt-image-2")

    assert result.images == [{"b64_json": "abc"}]
    assert captured["url"] == "https://api.example.test/v1/images/generations"


def test_llm_client_posts_to_configured_images_edits_endpoint(tmp_path) -> None:
    captured = {}
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"png-bytes")

    class FakeHttpClient:
        def post(self, url, *, headers=None, data=None, files=None, timeout=None):
            captured["url"] = url
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers),
                json={
                    "created": 123,
                    "data": [{"b64_json": "abc"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        image_edits_endpoint="/v1/images/edits",
        http_client=FakeHttpClient(),
    )

    result = client.edit_image(
        prompt="turn this into watercolor",
        model="gpt-image-2",
        images=[
            ImageAttachment(
                url="https://img.example.test/source.png",
                file_id="source.png",
                local_path=str(source_path),
            )
        ],
    )

    assert result.images == [{"b64_json": "abc"}]
    assert captured["url"] == "https://api.example.test/v1/images/edits"


def test_llm_client_generate_image_respects_max_attempts_override() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        attempts["count"] += 1
        raise httpx.ReadTimeout("timed out")

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    with pytest.raises(ValueError, match="images generations request failed after retries") as exc_info:
        client.generate_image(prompt="draw a cat", model="gpt-image-2", max_attempts=1)

    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
    assert attempts["count"] == 1


def test_llm_client_generate_image_passes_timeout_override_to_http_client() -> None:
    captured = {}

    class FakeHttpClient:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            captured["json"] = kwargs.get("json")
            captured["has_timeout_key"] = "timeout" in kwargs
            captured["timeout"] = kwargs.get("timeout")
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=kwargs.get("headers"), json=kwargs.get("json")),
                json={
                    "created": 123,
                    "data": [{"b64_json": "abc"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=FakeHttpClient(),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        timeout_seconds=60.0,
    )

    assert result.images == [{"b64_json": "abc"}]
    assert captured["has_timeout_key"] is True
    assert captured["timeout"] == 60.0


def test_llm_client_generate_image_can_disable_http_timeout_per_request() -> None:
    captured = {}

    class FakeHttpClient:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            captured["json"] = kwargs.get("json")
            captured["has_timeout_key"] = "timeout" in kwargs
            captured["timeout"] = kwargs.get("timeout")
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=kwargs.get("headers"), json=kwargs.get("json")),
                json={
                    "created": 123,
                    "data": [{"b64_json": "abc"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=FakeHttpClient(),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        timeout_seconds=None,
    )

    assert result.images == [{"b64_json": "abc"}]
    assert captured["has_timeout_key"] is True
    assert captured["timeout"] is None


def test_llm_client_generate_image_includes_compression_and_moderation_fields() -> None:
    captured = {}

    class FakeHttpClient:
        def post(self, url, *, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers, json=json),
                json={
                    "created": 123,
                    "data": [{"b64_json": "abc"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=FakeHttpClient(),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        quality="low",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
    )

    assert result.images == [{"b64_json": "abc"}]
    assert captured["json"] == {
        "model": "gpt-image-2",
        "prompt": "draw a cat",
        "n": 1,
        "response_format": "url",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 70,
        "moderation": "low",
    }


def test_llm_client_generate_image_omits_compression_for_png() -> None:
    captured = {}

    class FakeHttpClient:
        def post(self, url, *, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers, json=json),
                json={
                    "created": 123,
                    "data": [{"url": "https://img.example.test/generated.png"}],
                },
            )

    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=FakeHttpClient(),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        quality="high",
        output_format="png",
        output_compression=70,
        moderation="low",
    )

    assert result.images == [{"url": "https://img.example.test/generated.png"}]
    assert captured["json"] == {
        "model": "gpt-image-2",
        "prompt": "draw a cat",
        "n": 1,
        "response_format": "url",
        "quality": "high",
        "output_format": "png",
        "moderation": "low",
    }


def test_llm_client_falls_back_to_url_image_response_format_when_b64_json_is_rejected() -> None:
    captured_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        if payload["response_format"] == "b64_json":
            return httpx.Response(
                400,
                request=request,
                json={"error": {"message": "response_format b64_json is not supported"}},
            )
        if payload["response_format"] == "url":
            return httpx.Response(
                200,
                request=request,
                json={
                    "created": 123,
                    "data": [{"url": "https://img.example.test/generated.png"}],
                },
            )
        raise AssertionError(f"unexpected response_format: {payload['response_format']}")

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        model="gpt-5.4",
        http_client=httpx.Client(transport=transport),
    )

    result = client.generate_image(
        prompt="draw a cat",
        model="gpt-image-2",
        response_format="b64_json",
    )

    assert [payload["response_format"] for payload in captured_payloads] == ["b64_json", "url"]
    assert result.images == [{"url": "https://img.example.test/generated.png"}]
