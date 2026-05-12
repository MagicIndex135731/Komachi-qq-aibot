from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.message_content import ImageAttachment, extract_images_from_message, extract_reply_to_msg_id


@dataclass(slots=True)
class GroupMessageEvent:
    platform_msg_id: str
    group_id: int
    user_id: int
    nickname: str
    group_card: str
    plain_text: str
    raw_payload: dict[str, Any]
    timestamp: datetime
    msg_type: str
    images: list[ImageAttachment]
    mentioned_bot: bool
    reply_to_msg_id: str | None


@dataclass(slots=True)
class PrivateMessageEvent:
    platform_msg_id: str
    user_id: int
    nickname: str
    plain_text: str
    raw_payload: dict[str, Any]
    timestamp: datetime
    msg_type: str = "text"
    images: list[ImageAttachment] = field(default_factory=list)
    reply_to_msg_id: str | None = None


def _flatten_message(message: list[dict[str, Any]] | str) -> str:
    if isinstance(message, str):
        return message
    parts: list[str] = []
    for item in message:
        if item.get("type") == "text":
            parts.append(item.get("data", {}).get("text", ""))
    return "".join(parts).strip()


def _contains_explicit_mention(message: list[dict[str, Any]] | str, *, bot_qq: int, bot_name: str) -> bool:
    if isinstance(message, str):
        normalized_message = message.lstrip()
        return normalized_message.startswith(f"@{bot_name}") or normalized_message.startswith(f"@{bot_qq}")
    for item in message:
        if item.get("type") != "at":
            continue
        data = item.get("data", {})
        if str(data.get("qq", "")) == str(bot_qq):
            return True
    plain_text = _flatten_message(message)
    return plain_text.startswith(f"@{bot_name}") or plain_text.startswith(f"@{bot_qq}")
def _classify_message_type(*, plain_text: str, images: list[ImageAttachment]) -> str:
    if images and plain_text:
        return "mixed"
    if images:
        return "image"
    return "text"


def parse_group_message_event(payload: dict[str, Any], *, bot_qq: int, bot_name: str) -> GroupMessageEvent:
    message = payload.get("message", payload.get("raw_message", ""))
    plain_text = _flatten_message(message)
    images = extract_images_from_message(message)
    reply_to_msg_id = extract_reply_to_msg_id(message)
    mentioned_bot = _contains_explicit_mention(message, bot_qq=bot_qq, bot_name=bot_name)
    return GroupMessageEvent(
        platform_msg_id=str(payload["message_id"]),
        group_id=int(payload["group_id"]),
        user_id=int(payload["user_id"]),
        nickname=payload.get("sender", {}).get("nickname", ""),
        group_card=payload.get("sender", {}).get("card", ""),
        plain_text=plain_text,
        raw_payload=payload,
        timestamp=datetime.fromtimestamp(payload["time"], tz=UTC),
        msg_type=_classify_message_type(plain_text=plain_text, images=images),
        images=images,
        mentioned_bot=mentioned_bot,
        reply_to_msg_id=reply_to_msg_id,
    )


def parse_private_message_event(payload: dict[str, Any]) -> PrivateMessageEvent:
    message = payload.get("message", payload.get("raw_message", ""))
    plain_text = _flatten_message(message)
    images = extract_images_from_message(message)
    return PrivateMessageEvent(
        platform_msg_id=str(payload["message_id"]),
        user_id=int(payload["user_id"]),
        nickname=payload.get("sender", {}).get("nickname", ""),
        plain_text=plain_text,
        raw_payload=payload,
        timestamp=datetime.fromtimestamp(payload["time"], tz=UTC),
        msg_type=_classify_message_type(plain_text=plain_text, images=images),
        images=images,
        reply_to_msg_id=extract_reply_to_msg_id(message),
    )
