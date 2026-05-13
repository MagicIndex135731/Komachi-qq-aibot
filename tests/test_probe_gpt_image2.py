from __future__ import annotations

import httpx

from scripts.probe_gpt_image2 import classify_probe_result, probe_gpt_image_2


def test_probe_gpt_image_2_reports_listed_and_generation_success() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if str(request.url).endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-5.4"},
                        {"id": "gpt-image-2"},
                    ]
                },
            )
        if str(request.url).endswith("/images/generations"):
            return httpx.Response(
                200,
                json={"created": 123, "data": [{"b64_json": "abc"}]},
            )
        raise AssertionError(f"unexpected url: {request.url}")

    result = probe_gpt_image_2(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["model_check"]["path"] == "/models"
    assert result["model_check"]["gpt_image_2_listed"] is True
    assert result["generation_check"]["path"] == "/images/generations"
    assert result["generation_check"]["usable"] is True
    assert result["generation_check"]["image_count"] == 1
    assert calls == [
        ("GET", "https://api.example.test/v1/models"),
        ("POST", "https://api.example.test/v1/images/generations"),
    ]


def test_probe_gpt_image_2_falls_back_to_v1_paths_and_classifies_connect_error() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        raise httpx.ConnectError("connection failed", request=request)

    result = probe_gpt_image_2(
        base_url="https://api.example.test",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["model_check"]["status"] == "connect_error"
    assert result["generation_check"]["status"] == "connect_error"
    assert calls[:2] == [
        ("GET", "https://api.example.test/models"),
        ("GET", "https://api.example.test/v1/models"),
    ]
    assert calls[2:7] == [("POST", "https://api.example.test/images/generations")] * 5
    assert calls[7:12] == [("POST", "https://api.example.test/v1/images/generations")] * 5


def test_probe_gpt_image_2_does_not_append_v1_twice_when_base_url_already_has_v1() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if str(request.url) == "https://api.example.test/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-5.4"},
                        {"id": "gpt-image-2"},
                    ]
                },
            )
        if str(request.url) == "https://api.example.test/v1/images/generations":
            raise httpx.ConnectError("connection failed", request=request)
        raise AssertionError(f"unexpected url: {request.url}")

    result = probe_gpt_image_2(
        base_url="https://api.example.test/v1",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["model_check"]["path"] == "/models"
    assert result["generation_check"]["path"] == "/images/generations"
    assert result["generation_check"]["status"] == "connect_error"
    assert calls[0] == ("GET", "https://api.example.test/v1/models")
    assert calls[1:] == [("POST", "https://api.example.test/v1/images/generations")] * 5


def test_classify_probe_result_marks_winerror_10013_as_outbound_blocked() -> None:
    result = {
        "base_url": "https://api.example.test/v1",
        "model_check": {"status": "connect_error", "error_detail": "ConnectError: [WinError 10013] blocked"},
        "generation_check": {"status": "connect_error", "error_detail": "ConnectError: [WinError 10013] blocked"},
    }

    classified = classify_probe_result(result)

    assert classified["diagnosis"] == "local_outbound_https_blocked"


def test_classify_probe_result_does_not_claim_machine_block_when_models_endpoint_worked() -> None:
    result = {
        "base_url": "https://api.example.test/v1",
        "model_check": {"status": 200, "gpt_image_2_listed": True},
        "generation_check": {"status": "connect_error", "error_detail": "ConnectError: [WinError 10013] blocked"},
    }

    classified = classify_probe_result(result)

    assert classified["diagnosis"] == "image_generation_transport_failed"
