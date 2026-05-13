from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AppSettings
from app.providers.llm_client import LlmClient


def _extract_error_summary(payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        message = error.get("message")
        return (
            str(error_type) if error_type is not None else None,
            str(message)[:160] if message is not None else None,
        )
    if error is not None:
        return None, str(error)[:160]
    return None, None


def _candidate_paths(base_url: str, *, primary: str, fallback: str) -> tuple[str, ...]:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return (primary,)
    return (primary, fallback)


def _model_check(base_url: str, api_key: str, http_client: httpx.Client) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    last_result: dict[str, Any] = {"kind": "model_check"}
    for path in _candidate_paths(base_url, primary="/models", fallback="/v1/models"):
        url = f"{base_url}{path}"
        try:
            response = http_client.get(url, headers=headers)
            payload = response.json()
        except httpx.ConnectError as exc:
            last_result = {
                "kind": "model_check",
                "path": path,
                "status": "connect_error",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except httpx.TimeoutException as exc:
            last_result = {
                "kind": "model_check",
                "path": path,
                "status": "timeout",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except httpx.TransportError as exc:
            last_result = {
                "kind": "model_check",
                "path": path,
                "status": type(exc).__name__,
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except ValueError:
            payload = None
            last_result = {
                "kind": "model_check",
                "path": path,
                "status": response.status_code,
                "invalid_json": True,
            }
        else:
            listed = None
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list):
                    listed = any(isinstance(item, dict) and item.get("id") == "gpt-image-2" for item in data)
            last_result = {
                "kind": "model_check",
                "path": path,
                "status": response.status_code,
                "gpt_image_2_listed": listed,
            }
        if response.status_code < 400:
            return last_result
    return last_result


def _generation_check(base_url: str, api_key: str, http_client: httpx.Client, *, prompt: str) -> dict[str, Any]:
    last_result: dict[str, Any] = {"kind": "generation_check"}
    for path in _candidate_paths(base_url, primary="/images/generations", fallback="/v1/images/generations"):
        candidate_base_url = base_url if path == "/images/generations" else f"{base_url}/v1"
        client = LlmClient(
            base_url=candidate_base_url.rstrip("/"),
            api_key=api_key,
            model="gpt-5.4",
            http_client=http_client,
        )
        try:
            result = client.generate_image(prompt=prompt, model="gpt-image-2")
        except httpx.ConnectError as exc:
            last_result = {
                "kind": "generation_check",
                "path": path,
                "status": "connect_error",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except httpx.TimeoutException as exc:
            last_result = {
                "kind": "generation_check",
                "path": path,
                "status": "timeout",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except httpx.TransportError as exc:
            last_result = {
                "kind": "generation_check",
                "path": path,
                "status": type(exc).__name__,
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            continue
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json() if exc.response is not None else None
            except ValueError:
                payload = None
            error_type, error_message = _extract_error_summary(payload)
            last_result = {
                "kind": "generation_check",
                "path": path,
                "status": exc.response.status_code if exc.response is not None else 0,
            }
            if error_type is not None:
                last_result["error_type"] = error_type
            if error_message is not None:
                last_result["error_message"] = error_message
        except ValueError as exc:
            cause = exc.__cause__ or exc
            if isinstance(cause, httpx.ConnectError):
                last_result = {
                    "kind": "generation_check",
                    "path": path,
                    "status": "connect_error",
                    "error_detail": f"{type(cause).__name__}: {cause}",
                }
            elif isinstance(cause, httpx.TimeoutException):
                last_result = {
                    "kind": "generation_check",
                    "path": path,
                    "status": "timeout",
                    "error_detail": f"{type(cause).__name__}: {cause}",
                }
            elif isinstance(cause, httpx.TransportError):
                last_result = {
                    "kind": "generation_check",
                    "path": path,
                    "status": type(cause).__name__,
                    "error_detail": f"{type(cause).__name__}: {cause}",
                }
            else:
                last_result = {
                    "kind": "generation_check",
                    "path": path,
                    "status": "invalid_response_shape",
                }
        else:
            last_result = {
                "kind": "generation_check",
                "path": path,
                "status": 200,
                "usable": bool(result.images),
                "image_count": len(result.images),
            }
        status = last_result["status"]
        if isinstance(status, int):
            if status >= 500:
                continue
            return last_result
        if status in {"connect_error", "timeout"} or status.endswith("Error") or status.endswith("Timeout"):
            continue
        return last_result
    return last_result


def probe_gpt_image_2(
    *,
    base_url: str,
    api_key: str,
    http_client: httpx.Client | None = None,
    prompt: str = "A tiny test image with a plain background.",
) -> dict[str, Any]:
    client = http_client or httpx.Client(timeout=20.0)
    try:
        return {
            "base_url": base_url,
            "model_check": _model_check(base_url, api_key, client),
            "generation_check": _generation_check(base_url, api_key, client, prompt=prompt),
        }
    finally:
        if http_client is None:
            client.close()


def classify_probe_result(result: dict[str, Any]) -> dict[str, Any]:
    model_check = result.get("model_check", {})
    generation_check = result.get("generation_check", {})
    model_status = model_check.get("status")
    generation_status = generation_check.get("status")
    if model_status == 200 and generation_status in {
        "connect_error",
        "timeout",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }:
        result["diagnosis"] = "image_generation_transport_failed"
        return result
    error_detail = " ".join(
        str(value)
        for value in (
            model_check.get("error_detail"),
            generation_check.get("error_detail"),
        )
        if value
    ).lower()
    if "winerror 10013" in error_detail:
        result["diagnosis"] = "local_outbound_https_blocked"
        return result
    if model_check.get("status") == "connect_error" and generation_check.get("status") == "connect_error":
        result["diagnosis"] = "api_unreachable"
        return result
    if generation_check.get("usable") is True:
        result["diagnosis"] = "gpt_image_2_usable"
        return result
    result["diagnosis"] = "inconclusive"
    return result


def main() -> int:
    settings = AppSettings()
    result = classify_probe_result(probe_gpt_image_2(
        base_url=settings.llm_base_url.rstrip("/"),
        api_key=settings.llm_api_key,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
