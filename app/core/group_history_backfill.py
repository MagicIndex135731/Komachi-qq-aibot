from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import select

from app.adapters.onebot_models import parse_group_message_event
from app.storage.db import session_scope
from app.storage.models import Message

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_BATCHES = 8


async def backfill_recent_group_history(
    *,
    router,
    gateway,
    bot_qq: int,
    bot_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int = DEFAULT_MAX_BATCHES,
) -> int:
    runtime = getattr(router, "runtime", None)
    if runtime is None or not hasattr(runtime, "group_policy"):
        return 0
    if gateway is None or not hasattr(gateway, "call_api"):
        return 0

    group_ids = _allowed_group_ids(runtime_group_policy=runtime.group_policy)
    if not group_ids:
        return 0

    persisted_count = 0
    for group_id in group_ids:
        persisted_count += await _backfill_group_history(
            router=router,
            gateway=gateway,
            group_id=group_id,
            bot_qq=bot_qq,
            bot_name=bot_name,
            batch_size=batch_size,
            max_batches=max_batches,
        )
    return persisted_count


def _allowed_group_ids(*, runtime_group_policy: dict) -> list[int]:
    groups = runtime_group_policy.get("groups", {})
    allowed: list[int] = []
    for group_id_text, group_config in groups.items():
        if not bool(group_config.get("enabled", False)):
            continue
        try:
            allowed.append(int(group_id_text))
        except (TypeError, ValueError):
            continue
    return sorted(set(allowed))


async def _backfill_group_history(
    *,
    router,
    gateway,
    group_id: int,
    bot_qq: int,
    bot_name: str,
    batch_size: int,
    max_batches: int,
) -> int:
    last_known_id = _last_known_group_message_id(engine=router.engine, group_id=group_id)
    collected_payloads: list[dict] = []
    next_message_seq: int | None = None

    for _ in range(max(1, int(max_batches))):
        response = await gateway.call_api(
            "get_group_msg_history",
            _build_history_request_params(
                group_id=group_id,
                count=batch_size,
                message_seq=next_message_seq,
            ),
        )
        payloads = _extract_history_messages(response)
        if not payloads:
            break
        collected_payloads.extend(payloads)
        if last_known_id and any(str(payload.get("message_id", "")).strip() == last_known_id for payload in payloads):
            break
        if len(payloads) < batch_size:
            break
        next_candidate = _extract_oldest_message_seq(payloads)
        if next_candidate is None or next_candidate == next_message_seq:
            break
        next_message_seq = next_candidate

    persisted_count = 0
    for payload in _sort_payloads_ascending(collected_payloads):
        message_id = str(payload.get("message_id", "")).strip()
        if not message_id or (last_known_id and message_id == last_known_id):
            continue
        normalized_payload = _normalize_group_history_payload(payload=payload, group_id=group_id)
        try:
            event = parse_group_message_event(
                normalized_payload,
                bot_qq=bot_qq,
                bot_name=bot_name,
            )
        except Exception:
            logger.exception("group_history_backfill_parse_failed group_id=%s message_id=%s", group_id, message_id)
            continue
        if router.ingest_historical_group_message(event):
            persisted_count += 1
    return persisted_count


def _last_known_group_message_id(*, engine, group_id: int) -> str | None:
    with session_scope(engine) as session:
        stmt = (
            select(Message.platform_msg_id)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
            .limit(1)
        )
        last_known_id = session.execute(stmt).scalar_one_or_none()
    return str(last_known_id).strip() if last_known_id is not None else None


def _build_history_request_params(*, group_id: int, count: int, message_seq: int | None) -> dict:
    params = {"group_id": group_id, "count": max(1, int(count))}
    if message_seq is not None:
        params["message_seq"] = message_seq
    return params


def _extract_history_messages(response: dict) -> list[dict]:
    data = response.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    messages = data.get("messages", data.get("data", data.get("list")))
    if isinstance(messages, list):
        return [item for item in messages if isinstance(item, dict)]
    return []


def _extract_oldest_message_seq(payloads: Iterable[dict]) -> int | None:
    sequences: list[int] = []
    for payload in payloads:
        raw_seq = payload.get("message_seq", payload.get("messageSeq"))
        if raw_seq is None:
            continue
        try:
            sequences.append(int(raw_seq))
        except (TypeError, ValueError):
            continue
    return min(sequences) if sequences else None


def _sort_payloads_ascending(payloads: Iterable[dict]) -> list[dict]:
    unique_payloads: dict[str, dict] = {}
    for payload in payloads:
        message_id = str(payload.get("message_id", "")).strip()
        if not message_id:
            continue
        unique_payloads[message_id] = payload
    return sorted(
        unique_payloads.values(),
        key=lambda payload: (
            _payload_timestamp(payload),
            str(payload.get("message_id", "")),
        ),
    )


def _payload_timestamp(payload: dict) -> datetime:
    raw_time = payload.get("time")
    try:
        return datetime.fromtimestamp(int(raw_time), tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.fromtimestamp(0, tz=UTC)


def _normalize_group_history_payload(*, payload: dict, group_id: int) -> dict:
    normalized = dict(payload)
    normalized.setdefault("post_type", "message")
    normalized.setdefault("message_type", "group")
    normalized["group_id"] = int(normalized.get("group_id", group_id) or group_id)
    if "message" not in normalized and "raw_message" in normalized:
        normalized["message"] = normalized["raw_message"]
    return normalized
