from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImageAttachment:
    url: str
    file_id: str | None = None
    local_path: str | None = None
    source_message_id: str | None = None
    source_user_id: int | None = None
    source_nickname: str | None = None
    source_group_card: str | None = None


def extract_images_from_message(
    message: list[dict[str, Any]] | str,
    *,
    source_message_id: str | None = None,
    source_user_id: int | None = None,
    source_nickname: str | None = None,
    source_group_card: str | None = None,
) -> list[ImageAttachment]:
    if isinstance(message, str):
        return []
    images: list[ImageAttachment] = []
    for item in message:
        if item.get("type") != "image":
            continue
        data = item.get("data", {})
        images.append(
            ImageAttachment(
                url=str(data.get("url", "")).strip(),
                file_id=str(data.get("file", "")).strip() or None,
                local_path=str(data.get("local_path", "")).strip() or None,
                source_message_id=source_message_id,
                source_user_id=source_user_id,
                source_nickname=source_nickname,
                source_group_card=source_group_card,
            )
        )
    return images


def extract_images_from_raw_payload(raw_payload: dict[str, Any]) -> list[ImageAttachment]:
    message = raw_payload.get("message", raw_payload.get("raw_message", ""))
    sender = raw_payload.get("sender", {})
    if not isinstance(sender, dict):
        sender = {}
    raw_user_id = sender.get("user_id", raw_payload.get("user_id"))
    try:
        source_user_id = int(raw_user_id) if raw_user_id is not None else None
    except (TypeError, ValueError):
        source_user_id = None
    return extract_images_from_message(
        message,
        source_message_id=str(raw_payload.get("message_id", "")).strip() or None,
        source_user_id=source_user_id,
        source_nickname=str(sender.get("nickname", "")).strip() or None,
        source_group_card=str(sender.get("card", "")).strip() or None,
    )


def extract_reply_to_msg_id(message: list[dict[str, Any]] | str) -> str | None:
    if isinstance(message, str):
        return None
    for item in message:
        if item.get("type") != "reply":
            continue
        reply_id = str(item.get("data", {}).get("id", "")).strip()
        if reply_id:
            return reply_id
    return None
