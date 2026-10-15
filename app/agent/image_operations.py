"""One scoped implementation for foreground and durable background image work."""
from __future__ import annotations

import base64
import io
import ipaddress
import logging
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image
from app.agent.tool_agent import ToolOutcome
from app.core.message_content import ImageAttachment
from app.providers.image_adapter import coerce_image_artifact, image_failure_details

logger = logging.getLogger(__name__)


def image_operations(*, service, guard, runner, store, settings, artifact_path, reserve_retry):
    def download_image_bytes(url):
        guard.require_valid()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError("image_url_not_public")
        addresses = socket.getaddrinfo(parsed.hostname, 443)
        if not addresses or any(not ipaddress.ip_address(addr[4][0]).is_global for addr in addresses):
            raise ValueError("image_url_not_public")
        guard.require_valid()
        chunks, size = [], 0
        with service.http_client.stream("GET", url, follow_redirects=False, timeout=10) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                guard.require_valid()
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise ValueError("image_size_budget")
                chunks.append(chunk)
        return b"".join(chunks)

    def save_image_pixels(content, origin, model=""):
        guard.require_valid()
        if not content or len(content) > 8 * 1024 * 1024:
            raise ValueError("no_valid_image")
        with Image.open(io.BytesIO(content)) as picture:
            if picture.width * picture.height > 16_000_000 or picture.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("image_pixel_budget")
            picture.load()
            output = io.BytesIO()
            picture.convert("RGB").save(output, "PNG")
        content = output.getvalue()
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("image_size_budget")
        guard.require_valid()
        store.root.mkdir(parents=True, exist_ok=True)
        if origin == 'web_reference_pixels':
            from app.core.downloaded_media import DownloadedMediaCache
            cache = DownloadedMediaCache.for_store(store, settings)
            with cache.reserve(max(65536, len(content) + 32768)):
                path = store.root / ('.web-download-' + uuid4().hex + '.png')
                try:
                    path.write_bytes(content)
                    return cache.register(store, path, origin)
                finally:
                    path.unlink(missing_ok=True)
        path = store.root / (".image-" + uuid4().hex + ".png")
        try:
            path.write_bytes(content)
            return store.add(path, origin, model=model)
        finally:
            path.unlink(missing_ok=True)

    async def generate(args):
        guard.require_valid()
        attempts = 0
        reference_ids = args.get("reference_artifact_ids", [])
        subjects = args.get("reference_subjects")
        if (len(reference_ids) > 4 or len(set(reference_ids)) != len(reference_ids)
                or subjects is not None and len(subjects) != len(reference_ids)):
            return ToolOutcome({"error": "reference_ids_or_subjects_invalid"}, "blocked")
        def create():
            nonlocal attempts
            guard.require_valid()
            references = []
            for index, artifact_id in enumerate(reference_ids):
                path = artifact_path(artifact_id)
                if path.suffix not in {".png", ".jpg", ".webp", ".gif"}:
                    raise ValueError("reference_artifact_not_image")
                label = subjects[index].strip() if subjects is not None else f"当前范围的真实参考产物 {artifact_id}"
                references.append(ImageAttachment(url="", local_path=str(path), reference_subject=label or None))
            size = service._resolve_image_size(prompt=args["prompt"]) if callable(getattr(service, "_resolve_image_size", None)) else service.size
            options = {"prompt": args["prompt"], "model": service.model, "size": size,
                "quality": service.quality, "output_format": service.output_format, "response_format": "b64_json",
                "max_attempts": 1, "timeout_seconds": min(float(service.image_timeout_seconds or 120), 120)}
            for name in ("background", "output_compression", "moderation"):
                value = getattr(service, name, None)
                if value is not None:
                    options[name] = value
            edit = getattr(service.llm_client, 'edit_image', None)
            if references and not callable(edit):
                raise ValueError('image_edit_not_supported')
            deadline = time.monotonic() + options['timeout_seconds']
            # One bounded retry of an explicit provider error, with the
            # same model/references and the original total time budget.
            for attempt in range(2):
                guard.require_valid()
                options['timeout_seconds'] = max(0.001, deadline - time.monotonic())
                attempts += 1
                try:
                    result = edit(**options, images=references) if references else service.llm_client.generate_image(**options)
                    break
                except Exception as exc:
                    failure = image_failure_details(exc)
                    if attempt or not failure['retryable'] or deadline - time.monotonic() < 5:
                        raise
                    if not reserve_retry():
                        raise
                    logger.warning('agent_image_provider_retry attempt=2 http_status=%s', failure['provider_http_status'])
            items = getattr(result, "artifacts", None) or getattr(result, "images", [])
            content = None
            for item in items[:2]:
                artifact = coerce_image_artifact(item)
                if artifact is None:
                    continue
                if artifact.b64_json:
                    if len(artifact.b64_json) > 12 * 1024 * 1024:
                        raise ValueError("image_size_budget")
                    content = base64.b64decode(artifact.b64_json, validate=True)
                elif artifact.url:
                    content = download_image_bytes(artifact.url)
                if content:
                    break
            resolve_model = getattr(service.llm_client, "resolved_image_model", None)
            actual_model = resolve_model(edit=bool(references)) if callable(resolve_model) else service.model
            return save_image_pixels(content, "autonomous_generated", actual_model)
        try:
            image_id = await runner.run(create)
        except ValueError as exc:
            codes = {"unowned_or_revoked_artifact": "reference_artifact_unavailable",
                "artifact_source_budget_exhausted": "source_budget_exhausted",
                "reference_artifact_not_image": "reference_artifact_not_image",
                "image_edit_not_supported": "image_edit_not_supported"}
            failure = image_failure_details(exc)
            if str(exc) in codes:
                failure.update(error=codes[str(exc)], result_state='blocked')
            elif failure['error'] == 'image_result_invalid' and reference_ids:
                failure['error'] = 'image_reference_failed'
            return ToolOutcome({**failure, 'provider_attempts': attempts,
                "reference_edit_requested": bool(reference_ids), "text_generation_fallback": False},
                "blocked" if str(exc) in codes else "failed")
        except Exception as exc:
            return ToolOutcome({**image_failure_details(exc), 'provider_attempts': attempts,
                "reference_edit_requested": bool(reference_ids), "text_generation_fallback": False}, "failed")
        guard.require_valid()
        path = artifact_path(image_id)
        attachment = ImageAttachment(url="", local_path=str(path), reference_subject=f"小町本轮实际生成作品 {image_id}")
        return ToolOutcome({"artifact_id": image_id, "origin": "actual_generated_pixels", "ready": True}, images=(attachment,))

    return generate, download_image_bytes, save_image_pixels
