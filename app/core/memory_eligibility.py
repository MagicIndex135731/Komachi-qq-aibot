"""Pure post-retrieval eligibility checks for raw memory evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, Sequence

from app.core.memory_query_resolver import MemoryQueryPlan
from app.core.time_utils import ASIA_SHANGHAI


class MemorySource(Protocol):
    """Minimal source shape; concrete rows may expose compatible aliases."""

    group_id: int
    user_id: int | str | None
    sent_at: datetime
    source_msg_id: str


_INELIGIBLE_DELIVERY_STATES = frozenset({"reserved", "blocked", "uncertain", "deleted"})


def eligible(source_message: object | None, plan: MemoryQueryPlan) -> bool:
    """Return whether one raw source satisfies every hard query boundary.

    Message timestamps are stored as a naive Shanghai clock face, so a naive
    source timestamp is interpreted as Asia/Shanghai. Query-plan boundaries
    are always normalized to aware UTC values by the resolver.
    """

    if source_message is None or plan.group_id is None:
        return False
    if _integer_attr(source_message, "group_id") != plan.group_id:
        return False
    if not _stable_source_id(source_message):
        return False
    if _is_ineligible(source_message):
        return False

    sent_at = _source_time(source_message)
    if sent_at is None:
        return False
    if plan.start_at_utc is not None and sent_at < _as_utc(plan.start_at_utc):
        return False
    if plan.end_at_utc is not None and sent_at >= _as_utc(plan.end_at_utc):
        return False

    subject_ids = plan.subject_ids
    if subject_ids is None:
        if plan.answer_mode == "mention":
            return False
        return True
    if not subject_ids:
        return False

    allowed = frozenset(subject_ids)
    if plan.answer_mode == "mention":
        mentioned = _string_sequence_attr(
            source_message,
            "mentioned_uins",
            "mentioned_user_ids",
            "mention_uins",
        )
        return bool(allowed.intersection(mentioned))
    author_id = _string_attr(source_message, "user_id", "speaker_id", "sender_uin")
    if author_id in allowed:
        return True
    return False


def is_memory_source_eligible(source_message: object | None, plan: MemoryQueryPlan) -> bool:
    """Descriptive compatibility alias for callers that avoid bare predicates."""

    return eligible(source_message, plan)


def _integer_attr(source: object, name: str) -> int | None:
    value = getattr(source, name, None)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_source_id(source: object) -> str | None:
    return _string_attr(source, "source_msg_id", "platform_msg_id", "message_id")


def _string_attr(source: object, *names: str) -> str | None:
    for name in names:
        value = getattr(source, name, None)
        if value is None or isinstance(value, bool):
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _string_sequence_attr(source: object, *names: str) -> tuple[str, ...]:
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(
                normalized
                for item in value
                if item is not None
                and not isinstance(item, bool)
                and (normalized := str(item).strip())
            )
    return ()


def _is_ineligible(source: object) -> bool:
    if any(
        bool(getattr(source, name, False))
        for name in ("blocked", "is_blocked", "reserved", "is_reserved", "deleted", "is_deleted", "ineligible")
    ):
        return True
    if getattr(source, "eligible", True) is False:
        return True
    delivery_state = str(
        getattr(source, "delivery_state", "") or ""
    ).strip().casefold()
    if delivery_state in _INELIGIBLE_DELIVERY_STATES:
        return True
    raw_json = getattr(source, "raw_json", None)
    if isinstance(raw_json, dict):
        delivery_state = str(raw_json.get("delivery_state", "") or "").strip().casefold()
        if delivery_state in _INELIGIBLE_DELIVERY_STATES:
            return True
    return False


def _source_time(source: object) -> datetime | None:
    for name in ("sent_at", "timestamp", "event_time"):
        value = getattr(source, name, None)
        if isinstance(value, datetime):
            return _as_utc(value)
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=ASIA_SHANGHAI).astimezone(UTC)
    return value.astimezone(UTC)
