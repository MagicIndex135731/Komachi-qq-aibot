"""Shared provenance eligibility for memory and persona derivations."""

from __future__ import annotations

from collections.abc import Mapping
import json


INELIGIBLE_SOURCE_STATES = frozenset({"reserved", "blocked", "uncertain", "deleted"})


def _field(source: object, key: str, default=None):
    return (
        source.get(key, default)
        if isinstance(source, Mapping)
        else getattr(source, key, default)
    )


def memory_provenance_ids(memory: object) -> tuple[str, ...]:
    values = _field(memory, "source_msg_ids") or (_field(memory, "source_msg_id"),)
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in values
            if isinstance(item, str) and item.strip()
        )
    )


def is_usable_memory_source(
    source: object | None,
    *,
    group_id: int | None = None,
    author_id: int | str | None = None,
) -> bool:
    """Accept only real, confirmed, non-recalled sources in the given scope.

    Handles ORM rows and raw SQLite mappings without treating serialized
    raw_json as an empty payload. Time and query-subject policy remain the
    responsibility of the caller.
    """
    if source is None:
        return False
    source_id = _field(source, "source_msg_id") or _field(source, "platform_msg_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return False
    source_group = _field(source, "group_id")
    try:
        if group_id is not None and (
            isinstance(source_group, bool) or int(source_group) != int(group_id)
        ):
            return False
        if author_id is not None and str(_field(source, "user_id")) != str(author_id):
            return False
    except (TypeError, ValueError):
        return False
    if any(
        bool(_field(source, flag, False))
        for flag in (
            "blocked",
            "reserved",
            "deleted",
            "is_blocked",
            "is_reserved",
            "is_deleted",
            "ineligible",
        )
    ):
        return False
    if _field(source, "eligible", True) is False:
        return False
    raw_json = _field(source, "raw_json", {})
    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return False
    if raw_json is not None and not isinstance(raw_json, Mapping):
        return False
    states = (
        _field(source, "delivery_state", ""),
        (raw_json or {}).get("delivery_state", ""),
    )
    return not any(
        str(state or "").strip().casefold() in INELIGIBLE_SOURCE_STATES
        for state in states
    )
