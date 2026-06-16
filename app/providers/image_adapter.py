from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


DEFAULT_IMAGE_RESPONSE_FORMAT = "url"
FALLBACK_IMAGE_RESPONSE_FORMAT = "url"


@dataclass(slots=True)
class ImageArtifact:
    b64_json: str | None = None
    url: str | None = None
    output_format: str | None = None


@dataclass(slots=True)
class ImageGenerationResult:
    created: int | None
    images: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ImageArtifact] = field(default_factory=list)


def normalize_image_response_format(response_format: str | None) -> str:
    normalized = str(response_format or "").strip().lower()
    if normalized in {"url", "b64_json"}:
        return normalized
    return DEFAULT_IMAGE_RESPONSE_FORMAT


def build_image_generation_payload(
    *,
    model: str,
    prompt: str,
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    output_format: str | None = None,
    output_compression: int | None = None,
    moderation: str | None = None,
    response_format: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "response_format": normalize_image_response_format(response_format),
    }
    if size:
        payload["size"] = size
    if quality:
        payload["quality"] = quality
    if background:
        payload["background"] = background
    if output_format:
        payload["output_format"] = output_format
    normalized_output_format = (output_format or "").strip().lower()
    if output_compression is not None and normalized_output_format in {"jpeg", "jpg", "webp"}:
        payload["output_compression"] = int(output_compression)
    if moderation:
        payload["moderation"] = moderation
    return payload


def build_image_edit_data(
    *,
    model: str,
    prompt: str,
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    output_format: str | None = None,
    output_compression: int | None = None,
    moderation: str | None = None,
    response_format: str | None = None,
) -> dict[str, str]:
    data: dict[str, str] = {
        "model": model,
        "prompt": prompt,
        "n": "1",
    }
    explicit_response_format = str(response_format or "").strip()
    if explicit_response_format:
        data["response_format"] = normalize_image_response_format(explicit_response_format)
    if size:
        data["size"] = size
    if quality:
        data["quality"] = quality
    if background:
        data["background"] = background
    if output_format:
        data["output_format"] = output_format
    normalized_output_format = (output_format or "").strip().lower()
    if output_compression is not None and normalized_output_format in {"jpeg", "jpg", "webp"}:
        data["output_compression"] = str(int(output_compression))
    if moderation:
        data["moderation"] = moderation
    return data


def parse_image_generation_result(response_data: dict[str, Any]) -> ImageGenerationResult:
    data = response_data.get("data")
    if not isinstance(data, list):
        raise ValueError("image generation response did not include data list")
    images = [item for item in data if isinstance(item, dict)]
    return ImageGenerationResult(
        created=int(response_data["created"]) if "created" in response_data and response_data["created"] is not None else None,
        images=images,
        artifacts=[coerce_image_artifact(item) for item in images if coerce_image_artifact(item) is not None],
    )


def coerce_image_artifact(item: Any) -> ImageArtifact | None:
    if isinstance(item, ImageArtifact):
        return item

    if isinstance(item, dict):
        b64_json = str(item.get("b64_json", "") or "").strip() or None
        url = str(item.get("url", "") or "").strip() or None
        output_format = str(item.get("output_format", "") or "").strip() or None
    else:
        b64_json = str(getattr(item, "b64_json", "") or "").strip() or None
        url = str(getattr(item, "url", "") or "").strip() or None
        output_format = str(getattr(item, "output_format", "") or "").strip() or None

    if not any((b64_json, url, output_format)):
        return None
    return ImageArtifact(b64_json=b64_json, url=url, output_format=output_format)


def is_unsupported_b64_response_format_error(exc: Exception) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    response = exc.response
    if response is None or response.status_code not in {400, 415, 422}:
        return False
    try:
        payload = response.json()
    except ValueError:
        payload = None

    error_parts: list[str] = []
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("type", "code", "message", "param"):
                value = error.get(key)
                if value is not None:
                    error_parts.append(str(value))
        elif error is not None:
            error_parts.append(str(error))
    if not error_parts:
        error_parts.append(response.text[:400])

    haystack = " ".join(error_parts).lower()
    mentions_response_format = "response_format" in haystack and "b64_json" in haystack
    looks_unsupported = any(
        token in haystack
        for token in ("not supported", "unsupported", "invalid", "not available", "unknown")
    )
    return mentions_response_format and looks_unsupported
