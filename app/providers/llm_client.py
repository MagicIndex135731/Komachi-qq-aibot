from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.message_content import ImageAttachment
from app.providers.image_adapter import (
    FALLBACK_IMAGE_RESPONSE_FORMAT,
    ImageArtifact,
    ImageGenerationResult,
    build_image_edit_data,
    build_image_generation_payload,
    is_unsupported_b64_response_format_error,
    normalize_image_response_format,
    parse_image_generation_result,
)

logger = logging.getLogger(__name__)


INSTRUCTION_PREFIXES = (
    "System persona:",
    "Safety rules:",
    "Group policy:",
    "Reply style:",
)
PROXY_CHAT_IMAGE_STRING_HOSTS = {"api.codexzh.com"}
USE_CLIENT_DEFAULT_TIMEOUT = object()


@dataclass(slots=True)
class LlmUsage:
    timestamp: datetime
    model: str
    endpoint: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(slots=True)
class ResponsesStreamResult:
    text: str | None
    response_id: str | None
    usage: LlmUsage | None


@dataclass(slots=True)
class ResponsesImageResult:
    result: ImageGenerationResult
    response_id: str | None
    usage: LlmUsage | None


class LlmClient:
    ANTHROPIC_MAX_TOKENS = 1024
    REQUEST_MAX_ATTEMPTS = 5
    IMAGE_DOWNLOAD_MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        vision_model: str | None = None,
        responses_model: str | None = None,
        image_responses_model: str | None = None,
        compat_model: str | None = None,
        image_generations_endpoint: str = "/images/generations",
        image_edits_endpoint: str = "/images/edits",
        builtin_web_search: bool = False,
        web_search_context_size: str = "high",
        reasoning_effort: str = "",
        http_client: httpx.Client | None = None,
        usage_recorder=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback_model = (fallback_model or "").strip()
        self.vision_model = (vision_model or "").strip()
        self.responses_model = (responses_model or "").strip()
        self.image_responses_model = (image_responses_model or "").strip()
        self.compat_model = (compat_model or model).strip() or model
        self.image_generations_endpoint = self._normalize_endpoint(
            image_generations_endpoint,
            default="/images/generations",
        )
        self.image_edits_endpoint = self._normalize_endpoint(
            image_edits_endpoint,
            default="/images/edits",
        )
        self.builtin_web_search = bool(builtin_web_search)
        self.web_search_context_size = self._normalize_web_search_context_size(web_search_context_size)
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.http_client = http_client or httpx.Client(timeout=30.0, trust_env=False)
        self.usage_recorder = usage_recorder
        self._conversation_response_ids: dict[str, str] = {}
        self._base_host = (urlparse(self.base_url).hostname or "").lower()

    def _normalize_endpoint(self, endpoint: str, *, default: str) -> str:
        normalized = (endpoint or "").strip()
        if not normalized:
            normalized = default
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        if self.base_url.endswith("/v1") and normalized.startswith("/v1/"):
            normalized = normalized[3:]
        return normalized

    def _normalize_web_search_context_size(self, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return "high"

    def _normalize_reasoning_effort(self, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized in {"minimal", "low", "medium", "high"}:
            return normalized
        return ""

    def _uses_anthropic_messages_api(self, *, model: str | None = None) -> bool:
        active_model = (model or self.model).strip()
        return active_model.startswith("cc-")

    def _responses_enabled(self) -> bool:
        return bool(self.responses_model)

    def _chat_image_url_uses_string_shape(self) -> bool:
        return self._base_host in PROXY_CHAT_IMAGE_STRING_HOSTS

    def _responses_previous_response_id(self, *, conversation_key: str | None) -> str | None:
        del conversation_key
        return None

    def _remember_response_id(self, *, conversation_key: str | None, response_id: str | None) -> None:
        del conversation_key, response_id

    def _split_prompt_lines(self, prompt_lines: list[str]) -> tuple[list[str], list[str]]:
        instructions: list[str] = []
        input_lines: list[str] = []

        collecting_instructions = True
        for line in prompt_lines:
            if collecting_instructions and line.startswith(INSTRUCTION_PREFIXES):
                instructions.append(line)
                continue

            collecting_instructions = False
            input_lines.append(line)

        return instructions, input_lines

    def _build_responses_payload(
        self,
        *,
        model: str,
        instructions: list[str],
        input_lines: list[str],
        images: list[ImageAttachment] | None = None,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "\n\n".join(input_lines),
            }
        ]
        image_parts = self._load_input_images(images or [])
        for image_part in image_parts:
            data_url = f"data:{image_part['media_type']};base64,{image_part['data']}"
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            )
        payload: dict[str, Any] = {
            "model": model,
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.builtin_web_search:
            payload["tools"] = [
                {
                    "type": "web_search",
                    "search_context_size": self.web_search_context_size,
                }
            ]
        return payload

    def _build_responses_image_payload(
        self,
        *,
        prompt: str,
        images: list[ImageAttachment] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_responses_payload(
            model=self.image_responses_model,
            instructions=[],
            input_lines=[prompt],
            images=images,
        )
        payload["tools"] = [{"type": "image_generation"}]
        return payload

    def _build_chat_completions_payload(
        self,
        *,
        instructions: list[str],
        input_lines: list[str],
        images: list[ImageAttachment] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": "\n\n".join(instructions)})
        user_text = "\n\n".join(input_lines)
        image_parts = self._load_input_images(images or [])
        if image_parts:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for image_part in image_parts:
                data_url = f"data:{image_part['media_type']};base64,{image_part['data']}"
                image_url_value: str | dict[str, str]
                if self._chat_image_url_uses_string_shape():
                    image_url_value = data_url
                else:
                    image_url_value = {"url": data_url}
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url_value,
                    }
                )
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_text})
        return {
            "model": model or self.model,
            "messages": messages,
        }

    def _apply_model_specific_chat_payload_options(self, *, payload: dict[str, Any], model: str) -> dict[str, Any]:
        normalized_model = str(model or "").strip().lower()
        updated_payload = dict(payload)
        if normalized_model == "gpt-5-nano":
            updated_payload["reasoning_effort"] = "minimal"
        return updated_payload

    def _build_anthropic_messages_payload(
        self,
        *,
        model: str,
        instructions: list[str],
        input_lines: list[str],
        images: list[ImageAttachment] | None = None,
    ) -> dict[str, Any]:
        user_text = "\n\n".join(input_lines)
        image_parts = self._load_input_images(images or [])
        if image_parts:
            user_content: str | list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            user_content.extend(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_part["media_type"],
                        "data": image_part["data"],
                    },
                }
                for image_part in image_parts
            )
        else:
            user_content = user_text
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": user_content}],
        }
        if instructions:
            payload["system"] = "\n\n".join(instructions)
        return payload

    def _guess_image_media_type(self, *, url: str, response: httpx.Response) -> str | None:
        content_type = response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
        if content_type.startswith("image/"):
            return content_type
        guessed_type, _encoding = mimetypes.guess_type(url)
        if guessed_type and guessed_type.startswith("image/"):
            return guessed_type
        return None

    def _is_retryable_status_code(self, status_code: int) -> bool:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    def _sleep_before_retry(self, *, attempt: int, max_attempts: int) -> None:
        if attempt >= max_attempts:
            return
        time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))

    def _download_input_image(self, *, image_url: str, file_id: str | None) -> httpx.Response | None:
        for attempt in range(1, self.IMAGE_DOWNLOAD_MAX_ATTEMPTS + 1):
            try:
                response = self.http_client.get(image_url)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if attempt >= self.IMAGE_DOWNLOAD_MAX_ATTEMPTS or not self._is_retryable_status_code(status_code):
                    logger.exception("llm_input_image_download_failed url=%s file_id=%s", image_url, file_id)
                    return None
                logger.warning(
                    "llm_input_image_download_retry attempt=%s status=%s url=%s file_id=%s",
                    attempt,
                    status_code,
                    image_url,
                    file_id,
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=self.IMAGE_DOWNLOAD_MAX_ATTEMPTS)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.IMAGE_DOWNLOAD_MAX_ATTEMPTS:
                    logger.exception("llm_input_image_download_failed url=%s file_id=%s", image_url, file_id)
                    return None
                logger.warning(
                    "llm_input_image_download_retry attempt=%s reason=%s url=%s file_id=%s",
                    attempt,
                    type(exc).__name__,
                    image_url,
                    file_id,
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=self.IMAGE_DOWNLOAD_MAX_ATTEMPTS)
            except Exception:
                logger.exception("llm_input_image_download_failed url=%s file_id=%s", image_url, file_id)
                return None
        return None

    def _load_local_input_image(
        self,
        *,
        local_path: str,
        image_url: str,
        file_id: str | None,
    ) -> dict[str, str] | None:
        try:
            image_bytes = Path(local_path).read_bytes()
        except OSError:
            logger.exception("llm_input_image_local_read_failed local_path=%s file_id=%s", local_path, file_id)
            return None

        media_type, _encoding = mimetypes.guess_type(local_path)
        if media_type is None or not media_type.startswith("image/"):
            media_type, _encoding = mimetypes.guess_type(image_url)
        if media_type is None or not media_type.startswith("image/"):
            logger.warning("llm_input_image_local_unsupported_media_type local_path=%s file_id=%s", local_path, file_id)
            return None
        if not image_bytes:
            logger.warning("llm_input_image_local_empty_body local_path=%s file_id=%s", local_path, file_id)
            return None

        return {
            "media_type": media_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }

    def _load_input_images(self, images: list[ImageAttachment]) -> list[dict[str, str]]:
        encoded_images: list[dict[str, str]] = []
        for image in images:
            local_path = (image.local_path or "").strip()
            if local_path:
                local_image = self._load_local_input_image(
                    local_path=local_path,
                    image_url=image.url,
                    file_id=image.file_id,
                )
                if local_image is not None:
                    encoded_images.append(local_image)
                    continue

            image_url = image.url.strip()
            if not image_url:
                logger.warning("llm_input_image_missing_url file_id=%s", image.file_id)
                continue
            response = self._download_input_image(image_url=image_url, file_id=image.file_id)
            if response is None:
                continue

            media_type = self._guess_image_media_type(url=image_url, response=response)
            if media_type is None:
                logger.warning(
                    "llm_input_image_unsupported_media_type url=%s content_type=%s",
                    image_url,
                    response.headers.get("content-type"),
                )
                continue
            if not response.content:
                logger.warning("llm_input_image_empty_body url=%s file_id=%s", image_url, image.file_id)
                continue

            encoded_images.append(
                {
                    "media_type": media_type,
                    "data": base64.b64encode(response.content).decode("ascii"),
                }
            )
        return encoded_images

    def _input_image_filename(self, *, image: ImageAttachment, media_type: str, index: int) -> str:
        extension = mimetypes.guess_extension(media_type) or ".png"
        if extension == ".jpe":
            extension = ".jpg"

        candidates = [
            (image.file_id or "").strip(),
            Path((image.local_path or "").strip()).name if (image.local_path or "").strip() else "",
            Path(urlparse(image.url).path).name,
        ]
        for candidate in candidates:
            name = Path(candidate).name.strip()
            if not name:
                continue
            if Path(name).suffix:
                return name
            return f"{name}{extension}"
        return f"image-{index}{extension}"

    def _load_edit_input_images(self, images: list[ImageAttachment]) -> list[tuple[str, tuple[str, bytes, str]]]:
        multipart_images: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, image in enumerate(images, start=1):
            local_path = (image.local_path or "").strip()
            if local_path:
                local_image = self._load_local_input_image(
                    local_path=local_path,
                    image_url=image.url,
                    file_id=image.file_id,
                )
                if local_image is not None:
                    multipart_images.append(
                        (
                            "image",
                            (
                                self._input_image_filename(
                                    image=image,
                                    media_type=local_image["media_type"],
                                    index=index,
                                ),
                                base64.b64decode(local_image["data"]),
                                local_image["media_type"],
                            ),
                        )
                    )
                    continue

            image_url = image.url.strip()
            if not image_url:
                logger.warning("llm_input_image_missing_url file_id=%s", image.file_id)
                continue
            response = self._download_input_image(image_url=image_url, file_id=image.file_id)
            if response is None:
                continue

            media_type = self._guess_image_media_type(url=image_url, response=response)
            if media_type is None:
                logger.warning(
                    "llm_input_image_unsupported_media_type url=%s content_type=%s",
                    image_url,
                    response.headers.get("content-type"),
                )
                continue
            if not response.content:
                logger.warning("llm_input_image_empty_body url=%s file_id=%s", image_url, image.file_id)
                continue

            multipart_images.append(
                (
                    "image",
                    (
                        self._input_image_filename(
                            image=image,
                            media_type=media_type,
                            index=index,
                        ),
                        response.content,
                        media_type,
                    ),
                )
            )
        return multipart_images

    def _extract_responses_text(self, payload: dict[str, Any]) -> str | None:
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text

        for output_item in payload.get("output", []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    return text
        return None

    def _extract_chat_completions_text(self, payload: dict[str, Any]) -> str | None:
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        return text
        return None

    def _extract_anthropic_messages_text(self, payload: dict[str, Any]) -> str | None:
        pieces: list[str] = []
        for item in payload.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
        if not pieces:
            return None
        return "".join(pieces)

    def _extract_chat_completions_text_from_sse(self, response_text: str) -> str | None:
        payload = self._extract_chat_completions_payload_from_sse(response_text)
        return self._extract_chat_completions_text(payload) if payload is not None else None

    def _extract_chat_completions_payload_from_sse(self, response_text: str) -> dict[str, Any] | None:
        pieces: list[str] = []
        usage: dict[str, Any] | None = None
        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                payload = json.loads(payload_text)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            for choice in payload.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                message = choice.get("message")
                if isinstance(delta, dict):
                    content = delta.get("content")
                elif isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = choice.get("text")
                if isinstance(content, str) and content:
                    pieces.append(content)
        if not pieces:
            return {"choices": [], "usage": usage} if usage is not None else None
        payload: dict[str, Any] = {"choices": [{"message": {"content": "".join(pieces)}}]}
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _log_responses_tool_event(self, payload: dict[str, Any], *, response_id: str | None) -> None:
        payload_type = payload.get("type")
        if not isinstance(payload_type, str) or not payload_type:
            return

        item = payload.get("item")
        item_type = item.get("type") if isinstance(item, dict) else None
        is_tool_event = (
            "web_search_call" in payload_type
            or "tool_call" in payload_type
            or (isinstance(item_type, str) and (item_type.endswith("_call") or item_type == "web_search_call"))
        )
        if not is_tool_event:
            return

        item_id = payload.get("item_id")
        status = payload.get("status")
        query = payload.get("query")
        title = payload.get("title")
        url = payload.get("url")
        if isinstance(item, dict):
            item_id = item_id or item.get("id")
            status = status or item.get("status")
            query = query or item.get("query")
            title = title or item.get("title")
            url = url or item.get("url")

        logger.info(
            "responses_tool_event response_id=%s event=%s item_id=%s item_type=%s status=%s query=%r title=%r url=%r",
            response_id or "",
            payload_type,
            item_id or "",
            item_type or "",
            status or "",
            self._truncate_log_value(query),
            self._truncate_log_value(title),
            self._truncate_log_value(url),
        )

    def _truncate_log_value(self, value: Any, *, limit: int = 240) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _extract_responses_result_from_sse(
        self,
        response_text: str,
        *,
        model: str,
    ) -> ResponsesStreamResult:
        pieces: list[str] = []
        fallback_done_text: str | None = None
        response_id: str | None = None
        usage_payload: dict[str, Any] | None = None

        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                payload = json.loads(payload_text)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue

            response = payload.get("response")
            if response_id is None and isinstance(response, dict):
                maybe_response_id = response.get("id")
                if isinstance(maybe_response_id, str) and maybe_response_id:
                    response_id = maybe_response_id
            if response_id is None:
                maybe_response_id = payload.get("id")
                if isinstance(maybe_response_id, str) and maybe_response_id:
                    response_id = maybe_response_id

            payload_type = payload.get("type")
            self._log_responses_tool_event(payload, response_id=response_id)
            if payload_type == "response.output_text.delta":
                delta = payload.get("delta")
                if isinstance(delta, str) and delta:
                    pieces.append(delta)
                    continue
            if payload_type == "response.output_text.done" and fallback_done_text is None:
                done_text = payload.get("text")
                if isinstance(done_text, str) and done_text:
                    fallback_done_text = done_text
                    continue
            if payload_type == "response.completed" and isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    usage_payload = usage

        extracted_text = "".join(pieces) if pieces else fallback_done_text
        usage = self._extract_responses_usage({"usage": usage_payload}, model=model) if usage_payload else None
        return ResponsesStreamResult(text=extracted_text, response_id=response_id, usage=usage)

    def _extract_responses_image_artifacts(self, payload: Any) -> list[ImageArtifact]:
        artifacts: list[ImageArtifact] = []
        seen: set[tuple[str, str]] = set()

        def append_result(value: Any) -> None:
            if isinstance(value, str):
                normalized = value.strip()
                if not normalized:
                    return
                if normalized.startswith(("http://", "https://", "data:")):
                    artifact = ImageArtifact(url=normalized, output_format="png")
                    key = ("url", normalized)
                else:
                    artifact = ImageArtifact(b64_json=normalized, output_format="png")
                    key = ("b64", normalized)
                if key not in seen:
                    seen.add(key)
                    artifacts.append(artifact)
                return
            if isinstance(value, dict):
                b64_value = str(value.get("b64_json", "") or "").strip()
                url_value = str(value.get("url", "") or "").strip()
                if b64_value:
                    append_result(b64_value)
                elif url_value:
                    append_result(url_value)
                return
            if isinstance(value, list):
                for item in value:
                    append_result(item)

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            if value.get("type") == "image_generation_call":
                append_result(value.get("result"))
            item = value.get("item")
            if isinstance(item, dict):
                visit(item)
            response = value.get("response")
            if isinstance(response, dict):
                visit(response.get("output"))
            output = value.get("output")
            if isinstance(output, list):
                visit(output)

        visit(payload)
        return artifacts

    def _build_responses_image_result(
        self,
        *,
        artifacts: list[ImageArtifact],
        response_id: str | None,
        usage: LlmUsage | None,
    ) -> ResponsesImageResult:
        if not artifacts:
            raise ValueError("responses image generation did not include an image result")
        images: list[dict[str, Any]] = []
        for artifact in artifacts:
            if artifact.b64_json:
                images.append({"b64_json": artifact.b64_json})
            elif artifact.url:
                images.append({"url": artifact.url})
        return ResponsesImageResult(
            result=ImageGenerationResult(created=None, images=images, artifacts=artifacts),
            response_id=response_id,
            usage=usage,
        )

    def _extract_responses_image_result_from_sse(
        self,
        response_text: str,
        *,
        model: str,
    ) -> ResponsesImageResult:
        artifacts: list[ImageArtifact] = []
        response_id: str | None = None
        usage_payload: dict[str, Any] | None = None

        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                payload = json.loads(payload_text)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            response = payload.get("response")
            if response_id is None and isinstance(response, dict):
                maybe_response_id = response.get("id")
                if isinstance(maybe_response_id, str) and maybe_response_id:
                    response_id = maybe_response_id
            self._log_responses_tool_event(payload, response_id=response_id)
            artifacts.extend(self._extract_responses_image_artifacts(payload))
            if payload.get("type") == "response.completed" and isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    usage_payload = usage

        deduped: list[ImageArtifact] = []
        seen: set[tuple[str | None, str | None]] = set()
        for artifact in artifacts:
            key = (artifact.b64_json, artifact.url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(artifact)
        usage = self._extract_responses_usage({"usage": usage_payload}, model=model) if usage_payload else None
        return self._build_responses_image_result(
            artifacts=deduped,
            response_id=response_id,
            usage=usage,
        )

    def _record_usage(self, usage: LlmUsage | None) -> None:
        if usage is None or self.usage_recorder is None:
            return
        try:
            self.usage_recorder(usage)
        except Exception:
            logger.exception("Failed to record llm usage")

    def _extract_responses_usage(self, payload: dict[str, Any], *, model: str | None = None) -> LlmUsage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        details = usage.get("input_tokens_details")
        cached_input_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        return LlmUsage(
            timestamp=datetime.now().astimezone(),
            model=model or self.model,
            endpoint="responses",
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )

    def _extract_chat_completions_usage(self, payload: dict[str, Any], *, model: str | None = None) -> LlmUsage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        cached_input_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        return LlmUsage(
            timestamp=datetime.now().astimezone(),
            model=model or self.model,
            endpoint="chat_completions",
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )

    def _extract_anthropic_messages_usage(self, payload: dict[str, Any], *, model: str | None = None) -> LlmUsage | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = int(usage.get("input_tokens") or 0)
        cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return LlmUsage(
            timestamp=datetime.now().astimezone(),
            model=model or self.model,
            endpoint="anthropic_messages",
            input_tokens=input_tokens + cache_read_input_tokens + cache_creation_input_tokens,
            cached_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
        )

    def _chat_fallback_model(self) -> str:
        return self.fallback_model or self.model

    def _distinct_chat_fallback_model(self, *, primary_model: str) -> str:
        fallback_model = (self.fallback_model or "").strip()
        if not fallback_model or fallback_model == primary_model:
            return ""
        return fallback_model

    def _image_chat_model(self, *, default_model: str) -> str:
        vision_model = (self.vision_model or "").strip()
        return vision_model or default_model

    def _request_chat_completions_json(
        self,
        *,
        chat_payload: dict[str, Any],
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_limit = max(1, int(max_attempts or self.REQUEST_MAX_ATTEMPTS))

        for attempt in range(1, attempt_limit + 1):
            try:
                response = self.http_client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=chat_payload,
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("chat_completions_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("chat_completions_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("chat_completions_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue

            sse_payload = self._extract_chat_completions_payload_from_sse(response.text)
            if sse_payload is not None and self._extract_chat_completions_text(sse_payload) is not None:
                logger.warning(
                    "chat_completions_unexpected_sse attempt=%s content_type=%s",
                    attempt,
                    response.headers.get("content-type"),
                )
                return sse_payload
            try:
                return response.json()
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "chat_completions_invalid_json attempt=%s status=%s content_type=%s body_prefix=%r",
                    attempt,
                    response.status_code,
                    response.headers.get("content-type"),
                    response.text[:200],
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)

        if last_error is None:
            raise ValueError("chat completions request failed without a captured exception")
        raise ValueError("chat completions request failed after retries") from last_error

    def _request_anthropic_messages_json(self, *, messages_payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, self.REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = self.http_client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=messages_payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("anthropic_messages_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("anthropic_messages_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("anthropic_messages_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except ValueError as exc:
                last_error = exc
                logger.warning("anthropic_messages_invalid_json attempt=%s", attempt)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)

        if last_error is None:
            raise ValueError("anthropic messages request failed without a captured exception")
        raise ValueError("anthropic messages request failed after retries") from last_error

    def _request_responses_stream_result(
        self,
        *,
        responses_payload: dict[str, Any],
        model: str,
    ) -> ResponsesStreamResult:
        last_error: Exception | None = None

        for attempt in range(1, self.REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = self.http_client.post(
                    f"{self.base_url}/responses",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=responses_payload,
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("responses_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("responses_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("responses_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                return self._extract_responses_result_from_sse(response.text, model=model)

            try:
                response_data = response.json()
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "responses_invalid_json attempt=%s status=%s content_type=%s body_prefix=%r",
                    attempt,
                    response.status_code,
                    content_type,
                    response.text[:200],
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue

            usage = self._extract_responses_usage(response_data, model=model)
            text = self._extract_responses_text(response_data)
            response_id = response_data.get("id") if isinstance(response_data, dict) else None
            return ResponsesStreamResult(text=text, response_id=response_id, usage=usage)

        if last_error is None:
            raise ValueError("responses request failed without a captured exception")
        raise ValueError("responses request failed after retries") from last_error

    def _request_responses_image_result(
        self,
        *,
        responses_payload: dict[str, Any],
        model: str,
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> ResponsesImageResult:
        last_error: Exception | None = None
        attempt_limit = max(1, int(max_attempts or 1))

        for attempt in range(1, attempt_limit + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "headers": {"Authorization": f"Bearer {self.api_key}"},
                    "json": responses_payload,
                }
                if timeout_seconds is not USE_CLIENT_DEFAULT_TIMEOUT:
                    request_kwargs["timeout"] = timeout_seconds
                response = self.http_client.post(
                    f"{self.base_url}/responses",
                    **request_kwargs,
                )
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "responses_image_transport_retry attempt=%s reason=%s",
                    attempt,
                    type(exc).__name__,
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("responses_image_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue

            content_type = response.headers.get("content-type", "")
            try:
                if "text/event-stream" in content_type:
                    return self._extract_responses_image_result_from_sse(response.text, model=model)
                response_data = response.json()
                if not isinstance(response_data, dict):
                    raise ValueError("responses image generation returned a non-object response")
                usage = self._extract_responses_usage(response_data, model=model)
                response_id = response_data.get("id")
                return self._build_responses_image_result(
                    artifacts=self._extract_responses_image_artifacts(response_data),
                    response_id=response_id if isinstance(response_id, str) else None,
                    usage=usage,
                )
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "responses_image_invalid_result attempt=%s status=%s content_type=%s",
                    attempt,
                    response.status_code,
                    content_type,
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)

        if last_error is None:
            raise ValueError("responses image request failed without a captured exception")
        raise ValueError("responses image request failed after retries") from last_error

    def _request_images_generations_json(
        self,
        *,
        payload: dict[str, Any],
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_limit = max(1, int(max_attempts or self.REQUEST_MAX_ATTEMPTS))

        for attempt in range(1, attempt_limit + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "headers": {"Authorization": f"Bearer {self.api_key}"},
                    "json": payload,
                }
                if timeout_seconds is not USE_CLIENT_DEFAULT_TIMEOUT:
                    request_kwargs["timeout"] = timeout_seconds
                response = self.http_client.post(
                    f"{self.base_url}{self.image_generations_endpoint}",
                    **request_kwargs,
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("images_generations_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("images_generations_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("images_generations_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue

            try:
                return response.json()
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "images_generations_invalid_json attempt=%s status=%s content_type=%s body_prefix=%r",
                    attempt,
                    response.status_code,
                    response.headers.get("content-type"),
                    response.text[:200],
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)

        if last_error is None:
            raise ValueError("images generations request failed without a captured exception")
        raise ValueError("images generations request failed after retries") from last_error

    def _request_images_edits_json(
        self,
        *,
        data: dict[str, str],
        files: list[tuple[str, tuple[str, bytes, str]]],
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_limit = max(1, int(max_attempts or self.REQUEST_MAX_ATTEMPTS))

        for attempt in range(1, attempt_limit + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "headers": {"Authorization": f"Bearer {self.api_key}"},
                    "data": data,
                    "files": files,
                }
                if timeout_seconds is not USE_CLIENT_DEFAULT_TIMEOUT:
                    request_kwargs["timeout"] = timeout_seconds
                response = self.http_client.post(
                    f"{self.base_url}{self.image_edits_endpoint}",
                    **request_kwargs,
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("images_edits_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("images_edits_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("images_edits_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)
                continue

            try:
                return response.json()
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "images_edits_invalid_json attempt=%s status=%s content_type=%s body_prefix=%r",
                    attempt,
                    response.status_code,
                    response.headers.get("content-type"),
                    response.text[:200],
                )
                self._sleep_before_retry(attempt=attempt, max_attempts=attempt_limit)

        if last_error is None:
            raise ValueError("images edits request failed without a captured exception")
        raise ValueError("images edits request failed after retries") from last_error

    def _parse_image_generation_result(self, response_data: dict[str, Any]) -> ImageGenerationResult:
        return parse_image_generation_result(response_data)

    def _request_images_generations_with_response_format_fallback(
        self,
        *,
        payload: dict[str, Any],
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        try:
            return self._request_images_generations_json(
                payload=payload,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
        except httpx.HTTPStatusError as exc:
            if (
                normalize_image_response_format(str(payload.get("response_format", "")))
                != "b64_json"
                or not is_unsupported_b64_response_format_error(exc)
            ):
                raise
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = FALLBACK_IMAGE_RESPONSE_FORMAT
            logger.warning(
                "images_generations_response_format_fallback from=%s to=%s status=%s",
                payload.get("response_format"),
                fallback_payload["response_format"],
                exc.response.status_code if exc.response is not None else 0,
            )
            return self._request_images_generations_json(
                payload=fallback_payload,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )

    def _request_images_edits_with_response_format_fallback(
        self,
        *,
        data: dict[str, str],
        files: list[tuple[str, tuple[str, bytes, str]]],
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        try:
            return self._request_images_edits_json(
                data=data,
                files=files,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
        except httpx.HTTPStatusError as exc:
            if (
                normalize_image_response_format(data.get("response_format"))
                != "b64_json"
                or not is_unsupported_b64_response_format_error(exc)
            ):
                raise
            fallback_data = dict(data)
            fallback_data["response_format"] = FALLBACK_IMAGE_RESPONSE_FORMAT
            logger.warning(
                "images_edits_response_format_fallback from=%s to=%s status=%s",
                data.get("response_format"),
                fallback_data["response_format"],
                exc.response.status_code if exc.response is not None else 0,
            )
            return self._request_images_edits_json(
                data=fallback_data,
                files=files,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )

    def _generate_text_without_responses(
        self,
        *,
        instructions: list[str],
        input_lines: list[str],
        images: list[ImageAttachment] | None = None,
    ) -> str:
        compat_model = self.compat_model or self.model
        active_model = self._image_chat_model(default_model=compat_model) if images else compat_model

        if self._uses_anthropic_messages_api(model=active_model):
            messages_payload = self._build_anthropic_messages_payload(
                model=active_model,
                instructions=instructions,
                input_lines=input_lines,
                images=images,
            )
            try:
                response_data = self._request_anthropic_messages_json(messages_payload=messages_payload)
            except ValueError as exc:
                logger.warning(
                    "anthropic_messages_fallback_to_chat_completions reason=%s",
                    type(exc.__cause__ or exc).__name__,
                )
                chat_payload = self._build_chat_completions_payload(
                    instructions=instructions,
                    input_lines=input_lines,
                    images=images,
                    model=self._chat_fallback_model(),
                )
                response_data = self._request_chat_completions_json(chat_payload=chat_payload)
                self._record_usage(
                    self._extract_chat_completions_usage(response_data, model=chat_payload["model"])
                )
                text = self._extract_chat_completions_text(response_data)
            else:
                self._record_usage(self._extract_anthropic_messages_usage(response_data, model=active_model))
                text = self._extract_anthropic_messages_text(response_data)
        else:
            chat_payload = self._build_chat_completions_payload(
                instructions=instructions,
                input_lines=input_lines,
                images=images,
                model=active_model,
            )
            chat_payload = self._apply_model_specific_chat_payload_options(
                payload=chat_payload,
                model=active_model,
            )
            response_model = chat_payload["model"]
            fallback_model = "" if images and self.vision_model else self._distinct_chat_fallback_model(primary_model=response_model)
            try:
                response_data = self._request_chat_completions_json(
                    chat_payload=chat_payload,
                    max_attempts=1 if fallback_model else None,
                )
            except Exception as exc:
                if not fallback_model:
                    raise
                logger.warning(
                    "chat_completions_model_fallback primary_model=%s fallback_model=%s reason=%s",
                    response_model,
                    fallback_model,
                    type(exc.__cause__ or exc).__name__,
                )
                fallback_payload = self._build_chat_completions_payload(
                    instructions=instructions,
                    input_lines=input_lines,
                    images=images,
                    model=fallback_model,
                )
                fallback_payload = self._apply_model_specific_chat_payload_options(
                    payload=fallback_payload,
                    model=fallback_model,
                )
                response_data = self._request_chat_completions_json(chat_payload=fallback_payload)
                response_model = fallback_payload["model"]
            self._record_usage(self._extract_chat_completions_usage(response_data, model=response_model))
            text = self._extract_chat_completions_text(response_data)

        if text is not None:
            return text
        raise ValueError("model response did not include output text")

    def generate_text(
        self,
        prompt_lines: list[str],
        *,
        images: list[ImageAttachment] | None = None,
        conversation_key: str | None = None,
    ) -> str:
        instructions, input_lines = self._split_prompt_lines(prompt_lines)

        if self._responses_enabled():
            responses_model = self._image_chat_model(default_model=self.responses_model) if images else self.responses_model
            responses_payload = self._build_responses_payload(
                model=responses_model,
                instructions=instructions,
                input_lines=input_lines,
                images=images,
                previous_response_id=self._responses_previous_response_id(conversation_key=conversation_key),
            )
            try:
                responses_result = self._request_responses_stream_result(
                    responses_payload=responses_payload,
                    model=responses_model,
                )
            except ValueError as exc:
                logger.warning(
                    "responses_fallback_to_compat reason=%s",
                    type(exc.__cause__ or exc).__name__,
                )
            else:
                self._remember_response_id(
                    conversation_key=conversation_key,
                    response_id=responses_result.response_id,
                )
                self._record_usage(responses_result.usage)
                if responses_result.text is not None:
                    return responses_result.text
                raise ValueError("model response did not include output text")

        return self._generate_text_without_responses(
            instructions=instructions,
            input_lines=input_lines,
            images=images,
        )

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        moderation: str | None = None,
        response_format: str | None = None,
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> ImageGenerationResult:
        if self.image_responses_model:
            del model, size, quality, background, output_format, output_compression, moderation, response_format
            responses_result = self._request_responses_image_result(
                responses_payload=self._build_responses_image_payload(prompt=prompt),
                model=self.image_responses_model,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
            self._record_usage(responses_result.usage)
            return responses_result.result

        payload = build_image_generation_payload(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            moderation=moderation,
            response_format=response_format,
        )

        response_data = self._request_images_generations_with_response_format_fallback(
            payload=payload,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
        )
        return self._parse_image_generation_result(response_data)

    def edit_image(
        self,
        *,
        prompt: str,
        model: str,
        images: list[ImageAttachment],
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        moderation: str | None = None,
        response_format: str | None = None,
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> ImageGenerationResult:
        if self.image_responses_model:
            del model, size, quality, background, output_format, output_compression, moderation, response_format
            if not images:
                raise ValueError("image edit request did not include a usable input image")
            responses_result = self._request_responses_image_result(
                responses_payload=self._build_responses_image_payload(prompt=prompt, images=images),
                model=self.image_responses_model,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
            self._record_usage(responses_result.usage)
            return responses_result.result

        files = self._load_edit_input_images(images)
        if not files:
            raise ValueError("image edit request did not include a usable input image")

        data = build_image_edit_data(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            moderation=moderation,
            response_format=response_format,
        )

        response_data = self._request_images_edits_with_response_format_fallback(
            data=data,
            files=files,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
        )
        return self._parse_image_generation_result(response_data)
