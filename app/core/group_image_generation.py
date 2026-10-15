from __future__ import annotations

import asyncio
import base64
import binascii
from collections import deque
from collections.abc import Callable
from contextvars import Context, ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.adapters.sender import (
    OutboundMessage, OutboundPrivateMessage, QQMessageDeliveryUncertainError,
)
from app.adapters.outbound_guard import GroupDispatchScope, current_group_dispatch, group_dispatch_scope
from app.runtime.blocking import AgentModelBusyError, BoundedBlockingRunner
from app.core.message_content import ImageAttachment
from app.providers.image_adapter import coerce_image_artifact
from app.storage.db import session_scope
from app.storage.repositories import JobRepository, MessageRepository
from app.storage.models import Job, Message
from app.core.memory_source_guard import is_usable_memory_source

logger = logging.getLogger(__name__)

AUTO_IMAGE_SIZE = "auto"
LANDSCAPE_IMAGE_SIZE = "1536x1024"
PORTRAIT_IMAGE_SIZE = "1024x1536"
LANDSCAPE_IMAGE_SIZE_KEYWORDS = ("横图", "横版")
PORTRAIT_IMAGE_SIZE_KEYWORDS = ("竖图", "竖版")

_image_job_id: ContextVar[int | None] = ContextVar("qq_image_job_id", default=None)


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
    agent_source_message_id: str | None = None
    agent_reference_source_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class GroupImageGenerationEnqueueResult:
    accepted: bool
    queue_position: int | None = None
    reason: str = ""
    job_id: int | None = None


@dataclass(slots=True)
class ImageJobResult:
    success: bool
    notice_text: str
    image_path: Path | None = None
    failure_reason: str = ""
    delivery_state: str = "sent"


@dataclass(slots=True)
class ImageReferenceSearchPlan:
    subject: str
    query: str


class GroupImageGenerationService:
    def __init__(
        self,
        *,
        llm_client,
        sender,
        web_search_client=None,
        image_reference_planner_client=None,
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
        image_max_attempts: int = 5,
        image_timeout_seconds: float | None = None,
        failure_reply_text: str = "这张没跑出来，你换个说法试试",
    ) -> None:
        self.llm_client = llm_client
        self.sender = sender
        self.web_search_client = web_search_client
        self.image_reference_planner_client = image_reference_planner_client
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
        self._stopping = False
        self._generation_runner = BoundedBlockingRunner(max_workers=1)
        self.group_source_validator: Callable[[GroupImageGenerationRequest], bool] | None = None

    async def start(self) -> None:
        self._stopping = False
        if self.engine is None:
            async with self._lock:
                if self._queue:
                    self._idle_event.clear()
                    self._ensure_worker_locked()
            return
        recovered_active_jobs = 0
        with session_scope(self.engine) as session:
            jobs = JobRepository(session)
            for job in jobs.list_jobs(job_type=self.job_type, statuses=["running"]):
                payload = dict(job.payload_json or {})
                if payload.get("delivery_state") in {"dispatching", "uncertain"}:
                    payload["delivery_state"] = "uncertain"
                    jobs.mark_job_status(job_id=job.id, status="failed", payload_json=payload)
                elif payload.get("delivery_state") == "sent":
                    jobs.mark_job_status(job_id=job.id, status="completed", payload_json=payload)
                else:
                    job.run_at = datetime.now(UTC)
                    jobs.mark_job_status(job_id=job.id, status="queued")
            recovered_active_jobs = jobs.count_active_jobs(job_type=self.job_type)
        if recovered_active_jobs <= 0:
            return
        async with self._lock:
            self._idle_event.clear()
            self._ensure_worker_locked()

    async def stop(self) -> None:
        self._stopping = True
        worker_task = self._worker_task
        if worker_task is not None:
            if not worker_task.done():
                worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)

    async def enqueue(self, request: GroupImageGenerationRequest) -> GroupImageGenerationEnqueueResult:
        expected_type = PrivateImageGenerationRequest if isinstance(self, PrivateImageGenerationService) else GroupImageGenerationRequest
        if not isinstance(request, expected_type):
            raise ValueError("image request does not belong to this chat surface")
        if self._stopping:
            return GroupImageGenerationEnqueueResult(accepted=False, reason="service_stopping")
        if isinstance(request, GroupImageGenerationRequest) and not self._group_source_usable(request):
            return GroupImageGenerationEnqueueResult(accepted=False, reason="source_unusable")
        if isinstance(request, PrivateImageGenerationRequest) and not self._agent_private_source_usable(request):
            return GroupImageGenerationEnqueueResult(accepted=False, reason="source_unusable")
        if self.engine is None:
            return await self._enqueue_in_memory(request)
        async with self._lock:
            occupied_slots = self._active_persisted_job_count()
            if occupied_slots >= self.max_slots:
                return GroupImageGenerationEnqueueResult(accepted=False, reason="queue_full")
            with session_scope(self.engine) as session:
                queued_job = JobRepository(session).add_job(
                    job_type=self.job_type,
                    payload_json=self._serialize_request(request),
                    run_at=datetime.now(UTC),
                    status="queued",
                )
                queued_job_id = queued_job.id
            self._idle_event.clear()
            self._ensure_worker_locked()
            return GroupImageGenerationEnqueueResult(
                accepted=True,
                queue_position=occupied_slots + 1,
                job_id=queued_job_id,
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
            # Accepted image jobs outlive the short conversational turn. They
            # must not inherit its dispatch fence, working note or group scope.
            self._worker_task = asyncio.create_task(self._worker_loop(), context=Context())

    def _group_source_usable(self, request: GroupImageGenerationRequest) -> bool:
        if self._stopping:
            return False
        if self.engine is not None:
            with session_scope(self.engine) as session:
                source = MessageRepository(session).get_by_platform_msg_id(request.trigger_message_id)
                if not is_usable_memory_source(source, group_id=request.group_id, author_id=request.requester_user_id):
                    return False
                if (source.raw_json or {}).get("direction") == "outbound":
                    return False
                from app.storage.models import ContextReset
                floor = session.scalar(select(ContextReset.source_id).where(
                    ContextReset.scope == f'group:{request.group_id}').order_by(ContextReset.source_id.desc()).limit(1)) or 0
                if source.id <= floor:
                    return False
        return self.group_source_validator is None or bool(self.group_source_validator(request))

    def _update_delivery_state(self, state: str, *, receipt_stage: str | None = None) -> None:
        job_id = _image_job_id.get()
        if self.engine is None or job_id is None:
            return
        with session_scope(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise RuntimeError("image job lost its persistent delivery record")
            payload = dict(job.payload_json or {})
            payload["delivery_state"] = state
            if receipt_stage is not None and "group_id" in payload:
                scope = current_group_dispatch(int(payload["group_id"]))
                if scope is not None:
                    prior = set(payload.get("image_receipt_ids", [])) if receipt_stage == "notice" else set()
                    payload[f"{receipt_stage}_receipt_ids"] = [
                        receipt for receipt in dict.fromkeys(scope.receipts) if receipt not in prior
                    ][-20:]
            job.payload_json = payload
            session.add(job)

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
                job_token = _image_job_id.set(claimed_job.id)
                try:
                    try:
                        request = self._deserialize_request(claimed_job.payload_json)
                        result = await self._process_request(request)
                    except asyncio.CancelledError:
                        self._settle_cancelled_job(claimed_job.id)
                        raise
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
                            delivery_state="failed",
                        )
                        request = None
                    self._mark_persisted_job_result(job_id=claimed_job.id, result=result)
                    if request is not None:
                        try:
                            await self._handle_job_result(request=request, result=result)
                        except Exception:
                            logger.exception("image_job_result_callback_failed job_id=%s", claimed_job.id)
                finally:
                    _image_job_id.reset(job_token)
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

    def _settle_cancelled_job(self, job_id: int) -> None:
        if self.engine is None:
            return
        with session_scope(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            payload = dict(job.payload_json or {})
            state = payload.get("delivery_state")
            if state in {"dispatching", "uncertain"}:
                payload["delivery_state"] = "uncertain"
                JobRepository(session).mark_job_status(job_id=job_id, status="failed", payload_json=payload)
            elif state == "sent":
                JobRepository(session).mark_job_status(job_id=job_id, status="completed", payload_json=payload)
            # Generation has no QQ side effect. Leave it recoverable as running.

    def _mark_persisted_job_result(self, *, job_id: int, result: ImageJobResult) -> None:
        if self.engine is None:
            return
        with session_scope(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise RuntimeError("image job lost its persistent result record")
            if job.last_error_code == 'context_reset':
                return
            payload = dict(job.payload_json or {})
            payload["last_notice_text"] = result.notice_text
            payload["delivery_state"] = result.delivery_state
            if result.image_path is not None:
                payload["image_path"] = str(result.image_path)
            if result.failure_reason:
                payload["failure_reason"] = result.failure_reason
            JobRepository(session).mark_job_status(
                job_id=job_id,
                status="completed" if result.success else "failed",
                payload_json=payload,
            )

    async def _process_request(self, request: GroupImageGenerationRequest) -> ImageJobResult:
        if not self._group_source_usable(request):
            return ImageJobResult(False, "", failure_reason="source_unusable", delivery_state="blocked")
        scope = GroupDispatchScope(
            group_id=request.group_id,
            can_send=lambda: self._group_source_usable(request),
            on_dispatch=lambda: self._update_delivery_state("dispatching"),
        )
        with group_dispatch_scope(scope):
            return await self._process_group_request(request)

    async def _process_group_request(self, request: GroupImageGenerationRequest) -> ImageJobResult:
        try:
            image_path = await self._generation_runner.run(self._generate_and_store_image, request)
            if not self._group_source_usable(request):
                return ImageJobResult(False, "", image_path, "source_unusable", "blocked")
            self._update_delivery_state("dispatching")
            await self.sender.send_group_image(group_id=request.group_id, image_file=str(image_path))
            self._update_delivery_state("sent", receipt_stage="image")
        except AgentModelBusyError:
            return ImageJobResult(False, "", failure_reason="generation_capacity_unavailable", delivery_state="failed")
        except QQMessageDeliveryUncertainError as exc:
            self._update_delivery_state("uncertain")
            return ImageJobResult(False, "", failure_reason=str(exc), delivery_state="uncertain")
        except Exception as exc:
            logger.exception(
                "group_image_generation_failed group_id=%s msg_id=%s",
                request.group_id,
                request.trigger_message_id,
            )
            failure_text = self._failure_reply_for_exception(exc)
            notice_state = await self._send_group_notice(
                request=request,
                text=failure_text,
                log_event="group_image_failure_notice_failed",
            )
            return ImageJobResult(False, failure_text, failure_reason=str(exc), delivery_state=notice_state)

        success_text = "图好了"
        notice_state = await self._send_group_notice(
            request=request,
            text=success_text,
            log_event="group_image_success_notice_failed",
        )
        return ImageJobResult(
            success=notice_state != "uncertain", notice_text=success_text,
            image_path=image_path, delivery_state="uncertain" if notice_state == "uncertain" else "sent",
        )

    async def _handle_job_result(self, *, request, result: ImageJobResult) -> None:
        del request
        del result

    async def _send_group_notice(
        self,
        *,
        request: GroupImageGenerationRequest,
        text: str,
        log_event: str,
    ) -> str:
        if not self._group_source_usable(request):
            return "blocked"
        try:
            self._update_delivery_state("dispatching")
            await self.sender.send_group_text(
                OutboundMessage(
                    group_id=request.group_id,
                    text=self._build_requester_notice_text(
                        requester_user_id=request.requester_user_id,
                        text=text,
                    ),
                )
            )
            self._update_delivery_state("sent", receipt_stage="notice")
            return "sent"
        except QQMessageDeliveryUncertainError:
            self._update_delivery_state("uncertain")
            return "uncertain"
        except Exception:
            logger.exception(
                "%s group_id=%s msg_id=%s requester_user_id=%s",
                log_event,
                request.group_id,
                request.trigger_message_id,
                request.requester_user_id,
            )
            return "failed"

    def _generate_and_store_image(self, request: GroupImageGenerationRequest) -> Path:
        reference_images = self._resolve_reference_images(request)
        logger.info(
            "image_reference_images_resolved scope_id=%s msg_id=%s count=%s searched=%s",
            getattr(request, "group_id", getattr(request, "user_id", "")),
            request.trigger_message_id,
            len(reference_images),
            bool(str(getattr(request, "web_search_query", "") or "").strip()),
        )
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
        image_items = list(getattr(result, "artifacts", []) or [])
        if not image_items:
            image_items = list(getattr(result, "images", []) or [])
        image_bytes, suffix = self._extract_image_bytes(image_items)
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
        planner = self.image_reference_planner_client
        if planner is None and not query:
            return combined
        if self.web_search_client is None or not hasattr(self.web_search_client, "image_search"):
            if not query:
                return combined
            raise RuntimeError("web reference image search is not configured")
        should_search, search_plans = self._plan_reference_search_queries(request=request, query=query)
        if not should_search:
            return combined
        search_results: list[ImageAttachment] = []
        per_query_limit = 2 if planner is not None else 3
        for plan in search_plans:
            search_results.extend(
                replace(image, reference_subject=plan.subject or None)
                for image in self.web_search_client.image_search(query=plan.query, max_results=per_query_limit)
            )
            if len(search_results) >= 4:
                break
        search_results = self._deduplicate_reference_images(search_results)[:4]
        if not search_results:
            raise RuntimeError("web reference image search returned no usable images")
        combined.extend(search_results)
        return self._deduplicate_reference_images(combined)

    def _plan_reference_search_queries(
        self,
        *,
        request: GroupImageGenerationRequest,
        query: str,
    ) -> tuple[bool, list[ImageReferenceSearchPlan]]:
        planner = self.image_reference_planner_client
        if planner is None or not hasattr(planner, "generate_text"):
            return (bool(query), [ImageReferenceSearchPlan(subject="", query=query)] if query else [])

        prompt_lines = [
            "Decide whether this image request needs internet image references, then plan searches.",
            'Return JSON only: {"should_search": true|false, "references": [{"subject": "角色名", "queries": ["..."]}]}.',
            "Set should_search=false for ordinary creative generation when no real person/character reference is requested.",
            "Set should_search=true when the user asks to search/find online images, use a real person/character as reference, or requests current/external visual facts.",
            "Prefer one query per named person or character, including role/official reference terms when useful.",
            "Do not include instructions for image generation and do not add commentary.",
            f"Possible explicit search subject: {query or '(none)' }",
            f"Generation request: {request.prompt}",
        ]
        try:
            raw = str(planner.generate_text(prompt_lines)).strip()
            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return (bool(query), [ImageReferenceSearchPlan(subject="", query=query)] if query else [])
            should_search_value = parsed.get("should_search", False)
            if not isinstance(should_search_value, bool):
                raise ValueError("image reference planner returned a non-boolean should_search")
            should_search = should_search_value
            plans: list[ImageReferenceSearchPlan] = []
            references = parsed.get("references")
            seen: set[tuple[str, str]] = set()
            if isinstance(references, list):
                for reference in references:
                    if not isinstance(reference, dict):
                        continue
                    raw_subject = reference.get("subject", "")
                    subject = " ".join(raw_subject.split())[:80] if isinstance(raw_subject, str) else ""
                    planned_queries = reference.get("queries")
                    if not isinstance(planned_queries, list):
                        continue
                    for item in planned_queries:
                        if not isinstance(item, str):
                            continue
                        value = item.strip()
                        key = (subject, value)
                        if not value or key in seen or len(value) > 160:
                            continue
                        seen.add(key)
                        plans.append(ImageReferenceSearchPlan(subject=subject, query=value))
            # Accept the previous flat-array response during rolling upgrades.
            if not plans and isinstance(parsed.get("queries"), list):
                for item in parsed["queries"]:
                    if not isinstance(item, str):
                        continue
                    value = item.strip()
                    if value and len(value) <= 160 and not any(plan.query == value for plan in plans):
                        plans.append(ImageReferenceSearchPlan(subject="", query=value))
            if should_search and not plans and query:
                plans = [ImageReferenceSearchPlan(subject="", query=query)]
            return (should_search and bool(plans), plans[:3])
        except Exception:
            logger.warning("image_reference_query_planning_failed", exc_info=True)
            return (bool(query), [ImageReferenceSearchPlan(subject="", query=query)] if query else [])

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

    def _extract_image_bytes(self, images: list[Any]) -> tuple[bytes, str]:
        for item in images:
            artifact = coerce_image_artifact(item)
            if artifact is None:
                continue

            b64_value = str(artifact.b64_json or "").strip()
            if b64_value:
                return self._decode_b64_image_bytes(b64_value), self._preferred_suffix(item=item)

            url_value = str(artifact.url or "").strip()
            if url_value:
                if url_value.lower().startswith("data:"):
                    # Some image providers honor ``response_format=url`` by
                    # returning a data URI. Passing that multi-megabyte URI
                    # to httpx raises ``URL too long``; decode it locally.
                    return self._decode_data_image_url(url_value), self._preferred_suffix(item=item, url=url_value)
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

    def _decode_data_image_url(self, value: str) -> bytes:
        header, separator, encoded = value.partition(",")
        if not separator or header.lower() not in {
            "data:image/png;base64", "data:image/jpeg;base64",
            "data:image/webp;base64", "data:image/gif;base64",
        }:
            raise ValueError("image generation response included an unsupported data URL")
        # Reject oversized inline results before decoding; decoded bytes also
        # remain inside the 8 MiB media budget used by the current adapters.
        if len(encoded) > 12 * 1024 * 1024:
            raise ValueError("image generation response exceeded its size budget")
        try:
            raw = unquote_to_bytes(encoded).decode("ascii")
            normalized = "".join(raw.split())
            if not normalized:
                raise ValueError("empty data URL")
            normalized += "=" * (-len(normalized) % 4)
            contents = base64.b64decode(normalized, validate=True)
        except (UnicodeError, ValueError, binascii.Error):
            raise ValueError("image generation response included invalid data URL encoding") from None
        if len(contents) > 8 * 1024 * 1024:
            raise ValueError("image generation response exceeded its size budget")
        return contents

    def _preferred_suffix(self, *, item: Any, url: str | None = None) -> str:
        if self.output_format:
            return self.output_format
        artifact = coerce_image_artifact(item)
        if artifact is not None and artifact.output_format:
            return artifact.output_format
        if url:
            if url.lower().startswith("data:image/"):
                return url.split(";", 1)[0].split("/", 1)[1].lower()
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
        if "could not download any usable reference image" in message:
            return "参考图找到了，但图片下载失败了，我先不拿空参考图硬画"
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
        if isinstance(request, PrivateImageGenerationRequest) and request.agent_source_message_id:
            payload["agent_source_message_id"] = request.agent_source_message_id
            payload["agent_reference_source_ids"] = list(request.agent_reference_source_ids)
        return payload

    def _deserialize_request(self, payload: dict[str, Any]) -> GroupImageGenerationRequest:
        if "user_id" in payload:
            raise ValueError("private image request cannot enter a group worker")
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
            "fallback_url": str(image.fallback_url or "").strip(),
            "reference_subject": str(image.reference_subject or "").strip(),
        }

    def _deserialize_image_attachment(self, payload: dict[str, Any]) -> ImageAttachment:
        return ImageAttachment(
            url=str(payload.get("url", "") or "").strip(),
            file_id=str(payload.get("file_id", "") or "").strip() or None,
            local_path=str(payload.get("local_path", "") or "").strip() or None,
            fallback_url=str(payload.get("fallback_url", "") or "").strip() or None,
            reference_subject=str(payload.get("reference_subject", "") or "").strip() or None,
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

    def _agent_private_source_usable(self, request) -> bool:
        permission = getattr(self, "agent_user_allowed", None)
        if permission is not None and not permission(request.user_id):
            return False
        source_id = request.agent_source_message_id
        if self.engine is None:
            return not source_id
        with session_scope(self.engine) as session:
            from app.storage.models import ContextReset
            floor = session.scalar(select(ContextReset.source_id).where(
                ContextReset.scope == f'private:{request.user_id}').order_by(ContextReset.source_id.desc()).limit(1)) or 0
            if not source_id and not floor:
                return True
            row = MessageRepository(session).get_by_platform_msg_id(source_id or
                f'private-inbound-{request.user_id}-{request.trigger_message_id}')
            if not bool(row is not None and row.group_id is None
                and row.id > floor
                and is_usable_memory_source(row, author_id=request.user_id)
                and isinstance(row.raw_json, dict) and row.raw_json.get("direction") != "outbound"):
                return False
            refs = request.agent_reference_source_ids
            if not isinstance(refs, tuple) or len(refs) > 24 or any(type(value) is not int or value <= 0 for value in refs):
                return False
            for reference_id in refs:
                reference = session.get(Message, reference_id)
                if reference is None or reference.group_id is not None or not is_usable_memory_source(reference, author_id=request.user_id):
                    return False
            return True

    def _record_private_receipts(self, stage, receipts):
        job_id = _image_job_id.get()
        if self.engine is None or job_id is None:
            return
        with session_scope(self.engine) as session:
            row = session.get(Job, job_id)
            if row and row.job_type == self.job_type:
                payload = dict(row.payload_json)
                payload[f"{stage}_receipt_ids"] = list(receipts)
                row.payload_json = payload

    async def _process_request(self, request: PrivateImageGenerationRequest) -> ImageJobResult:
        if self._stopping or not self._agent_private_source_usable(request):
            return ImageJobResult(False, "", failure_reason="service_stopping" if self._stopping else "source_unavailable", delivery_state="blocked")
        image_path = None
        try:
            image_path = await self._generation_runner.run(self._generate_and_store_image, request)
            if self._stopping or not self._agent_private_source_usable(request):
                return ImageJobResult(False, "", image_path, "service_stopping" if self._stopping else "source_unavailable", "blocked")
            self._update_delivery_state("dispatching")
            receipt = await self.sender.send_private_image(user_id=request.user_id, image_file=str(image_path))
            if request.agent_source_message_id and (not isinstance(receipt, str) or not receipt):
                raise QQMessageDeliveryUncertainError("private_image_receipt_missing")
            self._record_private_receipts("image", (receipt,) if isinstance(receipt, str) and receipt else ())
            self._update_delivery_state("sent")
        except AgentModelBusyError:
            return ImageJobResult(False, "", failure_reason="generation_capacity_unavailable", delivery_state="failed")
        except QQMessageDeliveryUncertainError as exc:
            self._update_delivery_state("uncertain")
            return ImageJobResult(False, "", image_path=image_path, failure_reason=str(exc), delivery_state="uncertain")
        except Exception as exc:
            logger.exception(
                "private_image_generation_failed user_id=%s msg_id=%s",
                request.user_id,
                request.trigger_message_id,
            )
            failure_text = self._failure_reply_for_exception(exc)
            notice_state = await self._send_private_notice(
                request=request,
                text=failure_text,
                log_event="private_image_failure_notice_failed",
            )
            return ImageJobResult(False, failure_text, failure_reason=str(exc), delivery_state=notice_state)

        success_text = "图好了"
        notice_state = await self._send_private_notice(
            request=request,
            text=success_text,
            log_event="private_image_success_notice_failed",
        )
        return ImageJobResult(
            success=notice_state != "uncertain", notice_text=success_text,
            image_path=image_path, delivery_state="uncertain" if notice_state == "uncertain" else "sent",
        )

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
    ) -> str:
        if self._stopping or not self._agent_private_source_usable(request):
            return "blocked"
        try:
            self._update_delivery_state("dispatching")
            receipts = await self.sender.send_private_text(OutboundPrivateMessage(user_id=request.user_id, text=text))
            valid_receipts = tuple(value for value in receipts if isinstance(value, str) and value) if isinstance(receipts, (list, tuple)) else ()
            if request.agent_source_message_id and not valid_receipts:
                raise QQMessageDeliveryUncertainError("private_image_notice_receipt_missing")
            self._record_private_receipts("notice", valid_receipts)
            self._update_delivery_state("sent")
            return "sent"
        except QQMessageDeliveryUncertainError:
            self._update_delivery_state("uncertain")
            return "uncertain"
        except Exception:
            logger.exception(
                "%s user_id=%s msg_id=%s",
                log_event,
                request.user_id,
                request.trigger_message_id,
            )
            return "failed"

    def _deserialize_request(self, payload: dict[str, Any]) -> PrivateImageGenerationRequest:
        if "group_id" in payload:
            raise ValueError("group image request cannot enter a private worker")
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
            agent_source_message_id=self._optional_payload_text(payload.get("agent_source_message_id")),
            agent_reference_source_ids=tuple(payload.get("agent_reference_source_ids", ())),
        )

