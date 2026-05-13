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
class ImageGenerationResult:
    created: int | None
    images: list[dict[str, Any]]


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
        responses_model: str | None = None,
        compat_model: str | None = None,
        http_client: httpx.Client | None = None,
        usage_recorder=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback_model = (fallback_model or "").strip()
        self.responses_model = (responses_model or "").strip()
        self.compat_model = (compat_model or model).strip() or model
        self.http_client = http_client or httpx.Client(timeout=30.0)
        self.usage_recorder = usage_recorder
        self._conversation_response_ids: dict[str, str] = {}
        self._base_host = (urlparse(self.base_url).hostname or "").lower()

    def _uses_anthropic_messages_api(self, *, model: str | None = None) -> bool:
        active_model = (model or self.model).strip()
        return active_model.startswith("cc-")

    def _responses_enabled(self) -> bool:
        return bool(self.responses_model)

    def _chat_image_url_uses_string_shape(self) -> bool:
        return self._base_host in PROXY_CHAT_IMAGE_STRING_HOSTS

    def _responses_previous_response_id(self, *, conversation_key: str | None) -> str | None:
        if not conversation_key:
            return None
        return self._conversation_response_ids.get(conversation_key)

    def _remember_response_id(self, *, conversation_key: str | None, response_id: str | None) -> None:
        if not conversation_key or not response_id:
            return
        self._conversation_response_ids[conversation_key] = response_id

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
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "\n\n".join(input_lines),
                        }
                    ],
                }
            ],
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
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
        pieces: list[str] = []
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
            for choice in payload.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    pieces.append(content)
        if not pieces:
            return None
        return "".join(pieces)

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

    def _request_chat_completions_json(self, *, chat_payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, self.REQUEST_MAX_ATTEMPTS + 1):
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
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("chat_completions_transport_retry attempt=%s reason=%s", attempt, type(exc).__name__)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if not self._is_retryable_status_code(status_code):
                    raise
                last_error = exc
                logger.warning("chat_completions_status_retry attempt=%s status=%s", attempt, status_code)
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)
                continue

            sse_text = self._extract_chat_completions_text_from_sse(response.text)
            if sse_text is not None:
                logger.warning(
                    "chat_completions_unexpected_sse attempt=%s content_type=%s",
                    attempt,
                    response.headers.get("content-type"),
                )
                return {"choices": [{"message": {"content": sse_text}}]}
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
                self._sleep_before_retry(attempt=attempt, max_attempts=self.REQUEST_MAX_ATTEMPTS)

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
                    f"{self.base_url}/images/generations",
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

    def _generate_text_without_responses(
        self,
        *,
        instructions: list[str],
        input_lines: list[str],
        images: list[ImageAttachment] | None = None,
    ) -> str:
        compat_model = self.compat_model or self.model

        if self._uses_anthropic_messages_api(model=compat_model):
            messages_payload = self._build_anthropic_messages_payload(
                model=compat_model,
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
                self._record_usage(self._extract_anthropic_messages_usage(response_data, model=compat_model))
                text = self._extract_anthropic_messages_text(response_data)
        else:
            chat_payload = self._build_chat_completions_payload(
                instructions=instructions,
                input_lines=input_lines,
                images=images,
                model=compat_model,
            )
            response_data = self._request_chat_completions_json(chat_payload=chat_payload)
            self._record_usage(self._extract_chat_completions_usage(response_data, model=chat_payload["model"]))
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

        if not images and self._responses_enabled():
            responses_payload = self._build_responses_payload(
                model=self.responses_model,
                instructions=instructions,
                input_lines=input_lines,
                previous_response_id=self._responses_previous_response_id(conversation_key=conversation_key),
            )
            try:
                responses_result = self._request_responses_stream_result(
                    responses_payload=responses_payload,
                    model=self.responses_model,
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
        max_attempts: int | None = None,
        timeout_seconds: float | None | object = USE_CLIENT_DEFAULT_TIMEOUT,
    ) -> ImageGenerationResult:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "url",
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

        response_data = self._request_images_generations_json(
            payload=payload,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
        )
        data = response_data.get("data")
        if not isinstance(data, list):
            raise ValueError("image generation response did not include data list")
        images = [item for item in data if isinstance(item, dict)]
        return ImageGenerationResult(
            created=int(response_data["created"]) if "created" in response_data and response_data["created"] is not None else None,
            images=images,
        )
