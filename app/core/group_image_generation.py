from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from app.adapters.sender import OutboundMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GroupImageGenerationRequest:
    group_id: int
    trigger_message_id: str
    prompt: str
    requester_user_id: int


@dataclass(slots=True)
class GroupImageGenerationEnqueueResult:
    accepted: bool
    queue_position: int | None = None
    reason: str = ""


class GroupImageGenerationService:
    def __init__(
        self,
        *,
        llm_client,
        sender,
        output_dir: Path,
        model: str,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        moderation: str | None = None,
        max_slots: int = 3,
        image_max_attempts: int = 1,
        image_timeout_seconds: float | None = None,
        failure_reply_text: str = "这张没跑出来，你换个说法试试",
    ) -> None:
        self.llm_client = llm_client
        self.sender = sender
        self.output_dir = output_dir
        self.model = model
        self.size = size or None
        self.quality = quality or None
        self.background = background or None
        self.output_format = output_format or None
        self.output_compression = output_compression if output_compression is not None else None
        self.moderation = moderation or None
        self.max_slots = max(1, int(max_slots))
        self.image_max_attempts = max(1, int(image_max_attempts))
        self.image_timeout_seconds = None if image_timeout_seconds is None else float(image_timeout_seconds)
        self.failure_reply_text = failure_reply_text
        self.http_client = getattr(llm_client, "http_client", None) or httpx.Client(timeout=30.0)
        self._queue: deque[GroupImageGenerationRequest] = deque()
        self._running = 0
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    async def enqueue(self, request: GroupImageGenerationRequest) -> GroupImageGenerationEnqueueResult:
        async with self._lock:
            occupied_slots = self._running + len(self._queue)
            if occupied_slots >= self.max_slots:
                return GroupImageGenerationEnqueueResult(accepted=False, reason="queue_full")
            self._queue.append(request)
            self._idle_event.clear()
            self._ensure_worker_locked()
            return GroupImageGenerationEnqueueResult(
                accepted=True,
                queue_position=occupied_slots + 1,
            )

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()
        worker_task = self._worker_task
        if worker_task is not None:
            await asyncio.gather(worker_task, return_exceptions=True)

    def _ensure_worker_locked(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        try:
            while True:
                async with self._lock:
                    if not self._queue:
                        if self._running == 0:
                            self._idle_event.set()
                        return
                    request = self._queue.popleft()
                    self._running += 1
                try:
                    await self._process_request(request)
                except Exception:
                    logger.exception(
                        "group_image_worker_request_failed group_id=%s msg_id=%s",
                        request.group_id,
                        request.trigger_message_id,
                    )
                finally:
                    async with self._lock:
                        self._running = max(0, self._running - 1)
                        if not self._queue and self._running == 0:
                            self._idle_event.set()
        finally:
            async with self._lock:
                if not self._queue and self._running == 0:
                    self._idle_event.set()

    async def _process_request(self, request: GroupImageGenerationRequest) -> None:
        try:
            image_path = await asyncio.to_thread(self._generate_and_store_image, request)
            await self.sender.send_group_image(group_id=request.group_id, image_file=str(image_path))
        except Exception as exc:
            logger.exception(
                "group_image_generation_failed group_id=%s msg_id=%s",
                request.group_id,
                request.trigger_message_id,
            )
            await self._send_group_notice(
                request=request,
                text=self._failure_reply_for_exception(exc),
                log_event="group_image_failure_notice_failed",
            )
        else:
            await self._send_group_notice(
                request=request,
                text="\u56fe\u597d\u4e86",
                log_event="group_image_success_notice_failed",
            )

    async def _send_group_notice(
        self,
        *,
        request: GroupImageGenerationRequest,
        text: str,
        log_event: str,
    ) -> None:
        try:
            await self.sender.send_group_text(
                OutboundMessage(
                    group_id=request.group_id,
                    text=self._build_requester_notice_text(
                        requester_user_id=request.requester_user_id,
                        text=text,
                    ),
                )
            )
        except Exception:
            logger.exception(
                "%s group_id=%s msg_id=%s requester_user_id=%s",
                log_event,
                request.group_id,
                request.trigger_message_id,
                request.requester_user_id,
            )

    def _generate_and_store_image(self, request: GroupImageGenerationRequest) -> Path:
        result = self.llm_client.generate_image(
            prompt=request.prompt,
            model=self.model,
            size=self.size,
            quality=self.quality,
            background=self.background,
            output_format=self.output_format,
            output_compression=self.output_compression,
            moderation=self.moderation,
            max_attempts=self.image_max_attempts,
            timeout_seconds=self.image_timeout_seconds,
        )
        image_bytes, suffix = self._extract_image_bytes(result.images)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        image_path = self.output_dir / f"{uuid4().hex}{suffix}"
        image_path.write_bytes(image_bytes)
        return image_path

    def _extract_image_bytes(self, images: list[dict[str, Any]]) -> tuple[bytes, str]:
        for item in images:
            b64_value = str(item.get("b64_json", "")).strip()
            if b64_value:
                return self._decode_b64_image_bytes(b64_value), self._preferred_suffix(item=item)

            url_value = str(item.get("url", "")).strip()
            if url_value:
                response = self.http_client.get(url_value)
                response.raise_for_status()
                return response.content, self._preferred_suffix(item=item, url=url_value)

        raise ValueError("image generation response did not include a usable image")

    def _decode_b64_image_bytes(self, value: str) -> bytes:
        normalized = "".join(value.split())
        if not normalized:
            raise ValueError("image generation response included an empty base64 image")
        normalized += "=" * (-len(normalized) % 4)
        return base64.b64decode(normalized)

    def _preferred_suffix(self, *, item: dict[str, Any], url: str | None = None) -> str:
        if self.output_format:
            return self.output_format
        response_format = str(item.get("output_format", "")).strip()
        if response_format:
            return response_format
        if url:
            suffix = Path(urlparse(url).path).suffix
            if suffix:
                return suffix.lstrip(".")
        return "png"

    def _failure_reply_for_exception(self, exc: Exception) -> str:
        cause = exc.__cause__ or exc
        if isinstance(cause, (httpx.ReadTimeout, httpx.RemoteProtocolError)):
            return "这张超时了，你把描述再收短点，少点人物我再画"
        return self.failure_reply_text

    def _build_requester_notice_text(self, *, requester_user_id: int, text: str) -> str:
        clean_text = text.strip()
        if clean_text:
            return f"[CQ:at,qq={requester_user_id}] {clean_text}"
        return f"[CQ:at,qq={requester_user_id}]"
