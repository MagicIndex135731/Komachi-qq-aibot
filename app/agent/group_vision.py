"""Bounded, source-scoped pixels for autonomous group observations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
import tempfile
import time

import httpx
from PIL import Image
from sqlalchemy import select

from app.agent.social_images import (
    FETCH_SECONDS, FORMATS, MAX_IMAGES, MAX_PIXELS, MAX_TOTAL_BYTES,
    _fetch_pixels, allowed_media_url,
)
from app.core.image_optimizer import optimize_image_bytes
from app.core.memory_source_guard import is_usable_memory_source
from app.core.message_content import ImageAttachment
from app.storage.db import session_scope
from app.storage.models import Message


logger = logging.getLogger(__name__)
MAX_SOURCES = 24
MAX_SOURCE_IDS = 200


@dataclass(frozen=True, slots=True)
class _ImageSource:
    platform_id: str
    author_id: int
    candidates: tuple[dict, ...]


def _cancelled(cancelled) -> bool:
    return cancelled is not None and cancelled.is_set()


def _canonical_sources(engine, group_id: int, source_message_ids: tuple[int, ...]) -> list[_ImageSource]:
    # Observations can append old memory evidence after recent chat sources.
    # Bound the candidate set, then let canonical DB order choose recency.
    ids = tuple(dict.fromkeys(value for value in source_message_ids[:MAX_SOURCE_IDS] if type(value) is int and value > 0))
    if type(group_id) is not int or group_id <= 0 or not ids:
        return []
    with session_scope(engine) as session:
        rows = session.scalars(select(Message).where(
            Message.group_id == group_id, Message.id.in_(ids),
        ).order_by(Message.id.desc()).limit(MAX_SOURCES))
        sources = []
        for row in rows:
            if not is_usable_memory_source(row, group_id=group_id):
                continue
            raw = row.raw_json or {}
            if isinstance(raw, str):
                raw = json.loads(raw)
            items = raw.get("message", raw.get("raw_message", []))
            if not isinstance(items, list):
                continue
            candidates = []
            for item in items:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                data = item.get("data")
                if not isinstance(data, dict):
                    continue
                candidates.append({"local_path": data.get("local_path"), "url": data.get("url")})
                if len(candidates) >= MAX_IMAGES:
                    break
            if candidates:
                sources.append(_ImageSource(row.platform_msg_id, row.user_id, tuple(candidates)))
        return sources


def _local_pixels(value, *, group_cache: Path, budget: list[int], deadline: float):
    if not isinstance(value, str) or not value or time.monotonic() >= deadline or budget[0] <= 0:
        return None
    try:
        path = Path(value).resolve(strict=True)
        # Canonical payloads can still contain paths supplied by a bridge or
        # message sender. Only this group's cache can provide local pixels.
        if not path.is_relative_to(group_cache) or not path.is_file():
            return None
        with path.open("rb") as stream:
            data = stream.read(budget[0] + 1)
        budget[0] -= len(data)
        if budget[0] < 0 or time.monotonic() >= deadline:
            return None
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format not in FORMATS or picture.width * picture.height > MAX_PIXELS:
                return None
            media_type, suffix = FORMATS[picture.format]
            picture.verify()
        with Image.open(io.BytesIO(data)) as picture:
            picture.load()
        optimized = optimize_image_bytes(data, media_type=media_type)
        if optimized is not None:
            data, media_type = optimized
            suffix = ".png" if media_type == "image/png" else ".jpg"
        if time.monotonic() >= deadline:
            return None
        return data, media_type, suffix
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


@contextmanager
def canonical_group_image_inputs(
    engine, group_id: int, source_message_ids: tuple[int, ...], *, cache_dir: Path,
    cancelled=None, http_client: httpx.Client | None = None,
):
    """Yield at most two validated images tied to actual canonical sources.

    ``source_message_ids`` contains internal Message IDs, never platform IDs.
    Cached originals survive; temporary provider snapshots exist until the
    consumer finishes. URLs and fallback paths never reach the provider.
    """
    sources = [] if _cancelled(cancelled) else _canonical_sources(engine, group_id, source_message_ids)
    if not sources:
        yield []
        return
    cache_root = Path(cache_dir).resolve()
    group_cache = cache_root / str(group_id)
    resolved_group_cache = group_cache.resolve()
    if group_cache.is_symlink() or resolved_group_cache != group_cache or not resolved_group_cache.is_relative_to(cache_root):
        yield []
        return
    group_cache = resolved_group_cache
    group_cache.mkdir(parents=True, exist_ok=True)
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=5.0, follow_redirects=False, trust_env=False)
    try:
        with tempfile.TemporaryDirectory(prefix="agent-vision-", dir=group_cache) as directory:
            attachments, seen, budget = [], set(), [MAX_TOTAL_BYTES]
            deadline = time.monotonic() + FETCH_SECONDS
            # The first image of each newest source takes priority over a
            # single author's album, retaining useful multi-author context.
            for offset in range(MAX_IMAGES):
                for source in sources:
                    if len(attachments) >= MAX_IMAGES or _cancelled(cancelled) or time.monotonic() >= deadline or budget[0] <= 0:
                        break
                    if offset >= len(source.candidates):
                        continue
                    candidate = source.candidates[offset]
                    try:
                        pixels = _local_pixels(candidate.get("local_path"), group_cache=group_cache, budget=budget, deadline=deadline)
                        url = candidate.get("url")
                        if pixels is None and allowed_media_url(url) and not _cancelled(cancelled):
                            pixels = _fetch_pixels(client, url, budget=budget, deadline=deadline)
                        if pixels is None or _cancelled(cancelled) or time.monotonic() >= deadline:
                            continue
                        data, _, suffix = pixels
                        digest = hashlib.sha256(data).hexdigest()
                        key = (source.platform_id, digest)
                        if key in seen:
                            continue
                        seen.add(key)
                        path = Path(directory) / (digest + suffix)
                        path.write_bytes(data)
                        escaped_id = json.dumps(source.platform_id, ensure_ascii=False)[1:-1]
                        attachments.append(ImageAttachment(
                            url="", local_path=str(path.resolve()), source_user_id=source.author_id,
                            source_message_id=source.platform_id,
                            reference_subject=f"QQ observed image group_id={group_id} author_id={source.author_id} message_id={escaped_id}",
                        ))
                    except Exception as exc:
                        # Log a category only: media URLs may contain tokens,
                        # and local paths can contain private identifiers.
                        logger.warning("agent_group_media_failed error=%s", type(exc).__name__)
            yield [] if _cancelled(cancelled) else attachments
    finally:
        if owns_client:
            client.close()


@contextmanager
def group_image_inputs(batch, *, engine, cache_dir: Path, cancelled=None, http_client=None):
    """Translate an actual batch to canonical IDs, checking its author scope."""
    events = {event.platform_msg_id: event.user_id for event in batch.events[-MAX_SOURCES:] if event.group_id == batch.group_id}
    ids = []
    if events and not _cancelled(cancelled):
        with session_scope(engine) as session:
            for source in session.scalars(select(Message).where(
                Message.group_id == batch.group_id, Message.platform_msg_id.in_(tuple(events)),
            ).order_by(Message.id.desc()).limit(MAX_SOURCES)):
                if is_usable_memory_source(source, group_id=batch.group_id, author_id=events[source.platform_msg_id]):
                    ids.append(source.id)
    with canonical_group_image_inputs(
        engine, batch.group_id, tuple(ids), cache_dir=cache_dir, cancelled=cancelled, http_client=http_client,
    ) as images:
        yield images
