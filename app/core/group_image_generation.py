from __future__ import annotations

import asyncio
import base64
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from sqlalchemy.engine import Engine

from app.adapters.sender import OutboundMessage, OutboundPrivateMessage
from app.core.message_content import ImageAttachment
from app.storage.db import session_scope
from app.storage.repositories import JobRepository

logger = logging.getLogger(__name__)

AUTO_IMAGE_SIZE = "auto"
LANDSCAPE_IMAGE_SIZE = "3840x2160"
PORTRAIT_IMAGE_SIZE = "2160x3840"
LANDSCAPE_IMAGE_SIZE_KEYWORDS = ("横图", "横版")
PORTRAIT_IMAGE_SIZE_KEYWORDS = ("竖图", "竖版")


@dataclass(slots=True)
class GroupImageGenerationRequest:
    group_id: int
    trigger_message_id: str
    prompt: str
    requester_user_id: int
    reference_images: list[ImageAttachment] = field(default_factory=list)
    web_search_query: str | None = None


@dataclass(slots=True)
class PrivateImageGenerationRequest:
    user_id: int
    trigger_message_id: str
    prompt: str
    reference_images: list[ImageAttachment] = field(default_factory=list)
    web_search_query: str | None = None
    dev_task_id: int | None = None


@dataclass(slots=True)
class GroupImageGenerationEnqueueResult:
    accepted: bool
    queue_position: int | None = None
    reason: str = ""


@dataclass(slots=True)
class ImageJobResult:
    success: bool
    notice_text: str
    image_path: Path | None = None
    failure_reason: str = ""


class GroupImageGenerationService:
    def __init__(
        self,
        *,
        llm_client,
        sender,
        web_search_client=None,
        output_dir: Path,
        model: str,
        engine: Engine | None = None,
        job_type: str = "group_image_generation",
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
        self.web_search_client = web_search_client
        self.output_dir = output_dir
        self.model = model
        self.engine = engine
        self.job_type = job_type
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

    async def start(self) -> None:
        if self.engine is None:
            return
        recovered_active_jobs = 0
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            jobs.requeue_running_jobs(job_type=self.job_type)
            recovered_active_jobs = jobs.count_active_jobs(job_type=self.job_type)
        if recovered_active_jobs <= 0:
            return
        async with self._lock:
            self._idle_event.clear()
            self._ensure_worker_locked()

    async def stop(self) -> None:
        worker_task = self._worker_task
        if worker_task is not None and worker_task.done():
            await asyncio.gather(worker_task, return_exceptions=True)

    async def enqueue(self, request: GroupImageGenerationRequest) -> GroupImageGenerationEnqueueResult:
        if self.engine is None:
            return await self._enqueue_in_memory(request)
        async with self._lock:
            occupied_slots = self._active_persisted_job_count()
            if occupied_slots >= self.max_slots:
                return GroupImageGenerationEnqueueResult(accepted=False, reason="queue_full")
            with session_scope(self.engine) as session:
                JobRepository(session).add_job(
                    job_type=self.job_type,
                    payload_json=self._serialize_request(request),
                    run_at=datetime.now(UTC),
                    status="queued",
                )
            self._idle_event.clear()
            self._ensure_worker_locked()
            return GroupImageGenerationEnqueueResult(
                accepted=True,
                queue_position=occupied_slots + 1,
            )

    async def _enqueue_in_memory(self, request: GroupImageGenerationRequest) -> GroupImageGenerationEnqueueResult:
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
        if self.engine is None:
            await self._worker_loop_in_memory()
            return
        await self._worker_loop_persistent()

    async def _worker_loop_in_memory(self) -> None:
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
                        "group_image_worker_request_failed job_type=%s trigger_message_id=%s",
                        self.job_type,
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

    async def _worker_loop_persistent(self) -> None:
        try:
            while True:
                async with self._lock:
                    claimed_job = self._claim_next_persisted_job()
                    if claimed_job is None:
                        if self._running == 0 and self._active_persisted_job_count() == 0:
                            self._idle_event.set()
                        return
                    self._running += 1
                request = self._deserialize_request(claimed_job.payload_json)
                try:
                    result = await self._process_request(request)
                except Exception as exc:
                    logger.exception(
                        "group_image_worker_request_failed job_type=%s job_id=%s",
                        self.job_type,
                        claimed_job.id,
                    )
                    result = ImageJobResult(
                        success=False,
                        notice_text=self._failure_reply_for_exception(exc),
                        failure_reason=str(exc),
                    )
                self._mark_persisted_job_result(job_id=claimed_job.id, request=request, result=result)
                await self._handle_job_result(request=request, result=result)
                async with self._lock:
                    self._running = max(0, self._running - 1)
                    if self._running == 0 and self._active_persisted_job_count() == 0:
                        self._idle_event.set()
        finally:
            async with self._lock:
                if self._running == 0 and self._active_persisted_job_count() == 0:
                    self._idle_event.set()

    def _active_persisted_job_count(self) -> int:
        if self.engine is None:
            return self._running + len(self._queue)
        with session_scope(self.engine) as session:
            return JobRepository(session).count_active_jobs(job_type=self.job_type)

    def _claim_next_persisted_job(self):
        if self.engine is None:
            return None
        with session_scope(self.engine) as session:
            return JobRepository(session).claim_oldest_queued_job(
                job_type=self.job_type,
                now=datetime.now(UTC),
            )

    def _mark_persisted_job_result(self, *, job_id: int, request, result: ImageJobResult) -> None:
        if self.engine is None:
            return
        payload = self._serialize_request(request)
        payload["last_notice_text"] = result.notice_text
        if result.image_path is not None:
            payload["image_path"] = str(result.image_path)
        if result.failure_reason:
            payload["failure_reason"] = result.failure_reason
        with session_scope(self.engine) as session:
            JobRepository(session).mark_job_status(
                job_id=job_id,
                status="completed" if result.success else "failed",
                payload_json=payload,
            )

    async def _process_request(self, request: GroupImageGenerationRequest) -> ImageJobResult:
        try:
            image_path = await asyncio.to_thread(self._generate_and_store_image, request)
            await self.sender.send_group_image(group_id=request.group_id, image_file=str(image_path))
        except Exception as exc:
            logger.exception(
                "group_image_generation_failed group_id=%s msg_id=%s",
                request.group_id,
                request.trigger_message_id,
            )
            failure_text = self._failure_reply_for_exception(exc)
            await self._send_group_notice(
                request=request,
                text=failure_text,
                log_event="group_image_failure_notice_failed",
            )
            return ImageJobResult(success=False, notice_text=failure_text, failure_reason=str(exc))

        success_text = "图好了"
        await self._send_group_notice(
            request=request,
            text=success_text,
            log_event="group_image_success_notice_failed",
        )
        return ImageJobResult(success=True, notice_text=success_text, image_path=image_path)

    async def _handle_job_result(self, *, request, result: ImageJobResult) -> None:
        del request
        del result

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
        reference_images = self._resolve_reference_images(request)
        resolved_size = self._resolve_image_size(prompt=request.prompt)
        if reference_images:
            result = self.llm_client.edit_image(
                prompt=request.prompt,
                model=self.model,
                images=reference_images,
                size=resolved_size,
                quality=self.quality,
                background=self.background,
                output_format=self.output_format,
                output_compression=self.output_compression,
                moderation=self.moderation,
                max_attempts=self.image_max_attempts,
                timeout_seconds=self.image_timeout_seconds,
            )
        else:
            result = self.llm_client.generate_image(
                prompt=request.prompt,
                model=self.model,
                size=resolved_size,
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

    def _resolve_image_size(self, *, prompt: str) -> str:
        normalized_prompt = str(prompt or "").strip()
        has_landscape = any(keyword in normalized_prompt for keyword in LANDSCAPE_IMAGE_SIZE_KEYWORDS)
        has_portrait = any(keyword in normalized_prompt for keyword in PORTRAIT_IMAGE_SIZE_KEYWORDS)
        if has_landscape and not has_portrait:
            return LANDSCAPE_IMAGE_SIZE
        if has_portrait and not has_landscape:
            return PORTRAIT_IMAGE_SIZE
        if self.size:
            return self.size
        return AUTO_IMAGE_SIZE

    def _resolve_reference_images(self, request: GroupImageGenerationRequest) -> list[ImageAttachment]:
        combined = list(request.reference_images)
        query = str(getattr(request, "web_search_query", "") or "").strip()
        if not query:
            return combined
        if self.web_search_client is None or not hasattr(self.web_search_client, "image_search"):
            raise RuntimeError("web reference image search is not configured")
        search_results = self.web_search_client.image_search(query=query, max_results=3)
        if not search_results:
            raise RuntimeError("web reference image search returned no usable images")
        combined.extend(search_results)
        return self._deduplicate_reference_images(combined)

    def _deduplicate_reference_images(self, images: list[ImageAttachment]) -> list[ImageAttachment]:
        deduped: list[ImageAttachment] = []
        seen: set[tuple[str, str, str]] = set()
        for image in images:
            key = (
                str(getattr(image, "url", "") or "").strip(),
                str(getattr(image, "file_id", "") or "").strip(),
                str(getattr(image, "local_path", "") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(image)
        return deduped

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
        message = str(cause).strip().lower()
        if message == "web reference image search returned no usable images":
            return "我去网上找这组参考图，但这次没找到能直接拿来出图的图片"
        if message == "web reference image search is not configured":
            return "这次自动上网找参考图的链路还没接好"
        if isinstance(cause, (httpx.ReadTimeout, httpx.RemoteProtocolError)):
            return "这张超时了，你把描述再收短点，少点人物我再画"
        return self.failure_reply_text

    def _build_requester_notice_text(self, *, requester_user_id: int, text: str) -> str:
        clean_text = text.strip()
        if clean_text:
            return f"[CQ:at,qq={requester_user_id}] {clean_text}"
        return f"[CQ:at,qq={requester_user_id}]"

    def _serialize_request(self, request) -> dict[str, Any]:
        payload = {
            "trigger_message_id": request.trigger_message_id,
            "prompt": request.prompt,
            "reference_images": [self._serialize_image_attachment(image) for image in request.reference_images],
            "web_search_query": request.web_search_query,
        }
        if hasattr(request, "group_id"):
            payload["group_id"] = request.group_id
            payload["requester_user_id"] = request.requester_user_id
        if hasattr(request, "user_id"):
            payload["user_id"] = request.user_id
        if hasattr(request, "dev_task_id"):
            payload["dev_task_id"] = request.dev_task_id
        return payload

    def _deserialize_request(self, payload: dict[str, Any]) -> GroupImageGenerationRequest:
        return GroupImageGenerationRequest(
            group_id=int(payload["group_id"]),
            trigger_message_id=str(payload.get("trigger_message_id", "")).strip(),
            prompt=str(payload.get("prompt", "")).strip(),
            requester_user_id=int(payload["requester_user_id"]),
            reference_images=[
                self._deserialize_image_attachment(item)
                for item in payload.get("reference_images", [])
                if isinstance(item, dict)
            ],
            web_search_query=self._optional_payload_text(payload.get("web_search_query")),
        )

    def _serialize_image_attachment(self, image: ImageAttachment) -> dict[str, str]:
        return {
            "url": str(image.url or "").strip(),
            "file_id": str(image.file_id or "").strip(),
            "local_path": str(image.local_path or "").strip(),
        }

    def _deserialize_image_attachment(self, payload: dict[str, Any]) -> ImageAttachment:
        return ImageAttachment(
            url=str(payload.get("url", "") or "").strip(),
            file_id=str(payload.get("file_id", "") or "").strip() or None,
            local_path=str(payload.get("local_path", "") or "").strip() or None,
        )

    def _optional_payload_text(self, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class PrivateImageGenerationService(GroupImageGenerationService):
    def __init__(
        self,
        *,
        task_result_callback: Callable[[int, ImageJobResult], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(job_type="private_image_generation", **kwargs)
        self.task_result_callback = task_result_callback

    async def _process_request(self, request: PrivateImageGenerationRequest) -> ImageJobResult:
        try:
            image_path = await asyncio.to_thread(self._generate_and_store_image, request)
            await self.sender.send_private_image(user_id=request.user_id, image_file=str(image_path))
        except Exception as exc:
            logger.exception(
                "private_image_generation_failed user_id=%s msg_id=%s",
                request.user_id,
                request.trigger_message_id,
            )
            failure_text = self._failure_reply_for_exception(exc)
            await self._send_private_notice(
                request=request,
                text=failure_text,
                log_event="private_image_failure_notice_failed",
            )
            return ImageJobResult(success=False, notice_text=failure_text, failure_reason=str(exc))

        success_text = "图好了"
        await self._send_private_notice(
            request=request,
            text=success_text,
            log_event="private_image_success_notice_failed",
        )
        return ImageJobResult(success=True, notice_text=success_text, image_path=image_path)

    async def _handle_job_result(self, *, request, result: ImageJobResult) -> None:
        if self.task_result_callback is None:
            return
        dev_task_id = getattr(request, "dev_task_id", None)
        if not isinstance(dev_task_id, int) or dev_task_id <= 0:
            return
        self.task_result_callback(dev_task_id, result)

    async def _send_private_notice(
        self,
        *,
        request: PrivateImageGenerationRequest,
        text: str,
        log_event: str,
    ) -> None:
        try:
            await self.sender.send_private_text(OutboundPrivateMessage(user_id=request.user_id, text=text))
        except Exception:
            logger.exception(
                "%s user_id=%s msg_id=%s",
                log_event,
                request.user_id,
                request.trigger_message_id,
            )

    def _deserialize_request(self, payload: dict[str, Any]) -> PrivateImageGenerationRequest:
        return PrivateImageGenerationRequest(
            user_id=int(payload["user_id"]),
            trigger_message_id=str(payload.get("trigger_message_id", "")).strip(),
            prompt=str(payload.get("prompt", "")).strip(),
            reference_images=[
                self._deserialize_image_attachment(item)
                for item in payload.get("reference_images", [])
                if isinstance(item, dict)
            ],
            web_search_query=self._optional_payload_text(payload.get("web_search_query")),
            dev_task_id=int(payload["dev_task_id"]) if payload.get("dev_task_id") else None,
        )

    def pending_dev_task_ids(self) -> set[int]:
        if self.engine is None:
            return set()
        with session_scope(self.engine) as session:
            jobs = JobRepository(session).list_jobs(job_type=self.job_type, statuses=["queued", "running"])
        task_ids: set[int] = set()
        for job in jobs:
            payload = job.payload_json if isinstance(job.payload_json, dict) else {}
            task_id = payload.get("dev_task_id")
            if isinstance(task_id, int) and task_id > 0:
                task_ids.add(task_id)
        return task_ids
