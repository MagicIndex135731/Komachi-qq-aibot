from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImageAttachment:
    url: str
    file_id: str | None = None
    local_path: str | None = None


def extract_images_from_message(message: list[dict[str, Any]] | str) -> list[ImageAttachment]:
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
            )
        )
    return images


def extract_images_from_raw_payload(raw_payload: dict[str, Any]) -> list[ImageAttachment]:
    message = raw_payload.get("message", raw_payload.get("raw_message", ""))
    return extract_images_from_message(message)


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
