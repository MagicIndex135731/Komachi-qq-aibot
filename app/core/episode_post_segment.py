from __future__ import annotations

import json
import logging
import re
from typing import Protocol, Sequence


logger = logging.getLogger(__name__)


class PostSegmentMessage(Protocol):
    platform_msg_id: str
    user_id: int
    plain_text: str


def build_post_segment_prompt(
    messages: Sequence[PostSegmentMessage],
    *,
    max_chars_per_message: int = 120,
) -> list[str]:
    lines = [
        f"{index}. {str(message.plain_text or '')[:max_chars_per_message]}"
        for index, message in enumerate(messages, start=1)
        if str(message.plain_text or "").strip()
    ]
    return [
        "System persona: You split a QQ group chat log into topic segments.",
        "Safety rules: Reply with exactly one JSON object, no markdown: "
        '{"segments": [{"start": 1, "end": 5, "topic": "short label"}, ...]}. '
        "Every message must belong to exactly one segment; segments must be contiguous and ordered.",
        "Group policy: Split when the conversation clearly moves to a new topic (different subject, "
        "new question unrelated to the previous flow, or a hard scene change). Brief jokes, "
        "back-and-forth and single-line reactions stay in the current segment.",
        "Chat log:",
        *lines,
    ]


def parse_post_segment_boundaries(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    data: object | None = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
    if not isinstance(data, dict):
        return []
    segments = data.get("segments")
    if not isinstance(segments, list):
        return []
    parsed: list[tuple[int, int]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start") or 0)
            end = int(item.get("end") or 0)
        except (TypeError, ValueError):
            continue
        if start >= 1 and end >= start:
            parsed.append((start, end))
    return parsed


def split_messages(
    messages: Sequence[PostSegmentMessage],
    boundaries: Sequence[tuple[int, int]],
) -> list[list[PostSegmentMessage]]:
    if not boundaries:
        return [list(messages)]
    pieces: list[list[PostSegmentMessage]] = []
    cursor = 0
    for start, end in sorted(boundaries):
        start = max(1, start)
        end = min(len(messages), end)
        if start > cursor + 1:
            pieces.append(list(messages[cursor : start - 1]))
        if start <= end:
            pieces.append(list(messages[start - 1 : end]))
        cursor = max(cursor, end)
    if cursor < len(messages):
        pieces.append(list(messages[cursor:]))
    return [piece for piece in pieces if piece]


def post_segment_episode(
    *,
    client,
    messages: Sequence[PostSegmentMessage],
    min_messages: int = 25,
    max_chars_per_message: int = 120,
) -> list[list[PostSegmentMessage]]:
    """Split an episode into topic pieces with the upstream model.

    Returns a single piece (the whole episode) when segmentation is not
    applicable or the model output cannot be parsed, so callers keep the
    existing behavior as the safe fallback.
    """
    if len(messages) < max(1, int(min_messages)):
        return [list(messages)]
    try:
        raw = client.generate_text(
            build_post_segment_prompt(
                messages,
                max_chars_per_message=max_chars_per_message,
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("episode_post_segment_call_failed", exc_info=True)
        return [list(messages)]
    boundaries = parse_post_segment_boundaries(raw)
    if not boundaries:
        logger.info(
            "episode_post_segment_no_boundaries messages=%s",
            len(messages),
        )
        return [list(messages)]
    pieces = split_messages(messages, boundaries)
    if len(pieces) <= 1:
        return [list(messages)]
    logger.info(
        "episode_post_segment_split messages=%s pieces=%s",
        len(messages),
        len(pieces),
    )
    return pieces
