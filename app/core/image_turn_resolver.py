from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re

from app.core.message_content import ImageAttachment, extract_images_from_raw_payload


IMAGE_INTENT_KEYWORDS = (
    "这张",
    "这图",
    "这个图",
    "图片",
    "照片",
    "表情",
    "看图",
    "看这个",
    "看看这个",
    "看我发的",
    "我发的图",
    "我发的这张图",
    "刚发的图",
    "帮我看",
    "识别",
    "认一下",
    "什么内容",
    "啥内容",
    "上面那个",
    "上图",
)
PRIVATE_IMAGE_FOLLOWUP_HINTS = (
    "这是",
    "这就是",
    "这个是",
    "这个角色",
    "这个人物",
    "这角色",
    "她是",
    "他是",
    "来自",
    "出自",
    "游戏里",
    "作品里",
    "番里",
    "哪部",
    "哪作",
    "名字",
)
NORMALIZER = re.compile(r"[\s\u3000`~!@#$%^&*()_+\-=\[\]{}\\|;:'\",<.>/?，。！？：；、“”‘’（）《》【】]+")


@dataclass(slots=True)
class ResolvedImageTurn:
    images: list[ImageAttachment]
    source_msg_id: str
    source_kind: str
    followup_from_prior_prompt: bool = False


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _normalize_text(value: str) -> str:
    return NORMALIZER.sub("", value).lower()


def _contains_bot_name(text: str, bot_names: set[str]) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False
    for bot_name in bot_names:
        normalized_name = _normalize_text(bot_name)
        if normalized_name and normalized_name in normalized_text:
            return True
    return False


def _strip_bot_names(text: str, bot_names: set[str]) -> str:
    normalized_text = _normalize_text(text)
    stripped_text = normalized_text
    for bot_name in sorted(bot_names, key=len, reverse=True):
        normalized_name = _normalize_text(bot_name)
        if normalized_name:
            stripped_text = stripped_text.replace(normalized_name, "")
    return stripped_text


def _has_image_intent_keywords(text: str) -> bool:
    normalized_text = _normalize_text(text)
    return any(keyword in normalized_text for keyword in IMAGE_INTENT_KEYWORDS)


def _looks_like_contextual_private_image_followup(text: str) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text or len(normalized_text) > 32:
        return False
    return any(keyword in normalized_text for keyword in PRIVATE_IMAGE_FOLLOWUP_HINTS)


def _message_is_addressed(*, plain_text: str, mentioned_bot: bool, bot_names: set[str]) -> bool:
    return mentioned_bot or _contains_bot_name(plain_text, bot_names)


def _message_opens_image_session(
    *,
    plain_text: str,
    mentioned_bot: bool,
    bot_names: set[str],
    reply_to_msg_id: str | None,
    has_images: bool,
) -> bool:
    if not _message_is_addressed(plain_text=plain_text, mentioned_bot=mentioned_bot, bot_names=bot_names):
        return False
    if has_images or reply_to_msg_id is not None:
        return True
    remaining_text = _strip_bot_names(plain_text, bot_names)
    if not remaining_text:
        return True
    return _has_image_intent_keywords(remaining_text)


def _extract_message_images(message) -> list[ImageAttachment]:
    raw_json = getattr(message, "raw_json", {})
    if not isinstance(raw_json, dict):
        return []
    return extract_images_from_raw_payload(raw_json)


def _recent_user_messages_before_event(*, event, messages, window: timedelta) -> list[object]:
    since = _normalize_timestamp(event.timestamp) - window
    recent_messages = messages.list_recent_group_messages_for_user_since(
        group_id=event.group_id,
        user_id=event.user_id,
        since=since,
        limit=20,
    )
    return [
        message
        for message in recent_messages
        if message.platform_msg_id != event.platform_msg_id
        and _normalize_timestamp(message.timestamp) <= _normalize_timestamp(event.timestamp)
    ]


def _recent_group_messages_before_event(*, event, messages, window: timedelta) -> list[object]:
    since = _normalize_timestamp(event.timestamp) - window
    recent_messages = messages.list_recent_group_messages(group_id=event.group_id, limit=200)
    return [
        message
        for message in recent_messages
        if message.platform_msg_id != event.platform_msg_id
        and since <= _normalize_timestamp(message.timestamp) <= _normalize_timestamp(event.timestamp)
    ]


def _recent_private_messages_before_event(*, event, messages, window: timedelta) -> list[object]:
    since = _normalize_timestamp(event.timestamp) - window
    recent_messages = messages.list_recent_private_messages_for_user_since(
        user_id=event.user_id,
        since=since,
        limit=20,
    )
    current_platform_msg_ids = {
        str(getattr(event, "platform_msg_id", "")).strip(),
        f"private-inbound-{event.user_id}-{str(getattr(event, 'platform_msg_id', '')).strip()}",
    }
    return [
        message
        for message in recent_messages
        if message.platform_msg_id not in current_platform_msg_ids
        and since <= _normalize_timestamp(message.timestamp) <= _normalize_timestamp(event.timestamp)
    ]


def _find_recent_image_message(*, prior_messages: list[object]) -> object | None:
    for message in reversed(prior_messages):
        if _extract_message_images(message):
            return message
        if str(getattr(message, "plain_text", "")).strip():
            return None
    return None


def _find_recent_group_image_message(*, prior_messages: list[object]) -> object | None:
    for message in reversed(prior_messages):
        if _extract_message_images(message):
            return message
    return None


def _find_recent_open_prompt_before_event(*, prior_messages: list[object], bot_names: set[str]) -> object | None:
    for message in reversed(prior_messages):
        message_images = _extract_message_images(message)
        if message_images:
            continue

        if _message_opens_image_session(
            plain_text=str(getattr(message, "plain_text", "")),
            mentioned_bot=bool(getattr(message, "mentioned_bot", False)),
            bot_names=bot_names,
            reply_to_msg_id=getattr(message, "reply_to_msg_id", None),
            has_images=False,
        ):
            return message

        if str(getattr(message, "plain_text", "")).strip():
            return None
    return None


def resolve_images_for_turn(
    *,
    event,
    addressed_turn: bool,
    bot_names: set[str],
    messages,
    quoted_raw_payload: dict | None = None,
    lookback_window: timedelta = timedelta(minutes=3),
    allow_recent_image_without_intent: bool = False,
) -> ResolvedImageTurn | None:
    prior_messages = _recent_user_messages_before_event(
        event=event,
        messages=messages,
        window=lookback_window,
    )

    if event.images:
        if addressed_turn:
            return ResolvedImageTurn(
                images=event.images,
                source_msg_id=event.platform_msg_id,
                source_kind="current",
            )

        open_prompt = _find_recent_open_prompt_before_event(
            prior_messages=prior_messages,
            bot_names=bot_names,
        )
        if open_prompt is None:
            return None
        return ResolvedImageTurn(
            images=event.images,
            source_msg_id=event.platform_msg_id,
            source_kind="current",
            followup_from_prior_prompt=True,
        )

    opens_image_session = _message_opens_image_session(
        plain_text=event.plain_text,
        mentioned_bot=addressed_turn,
        bot_names=bot_names,
        reply_to_msg_id=event.reply_to_msg_id,
        has_images=False,
    )
    if not opens_image_session and not (allow_recent_image_without_intent and addressed_turn and event.reply_to_msg_id is None):
        return None

    if event.reply_to_msg_id is not None:
        quoted_message = messages.get_by_platform_msg_id(event.reply_to_msg_id)
        if quoted_message is not None:
            quoted_images = _extract_message_images(quoted_message)
            if quoted_images:
                return ResolvedImageTurn(
                    images=quoted_images,
                    source_msg_id=quoted_message.platform_msg_id,
                    source_kind="quoted",
                )
            return None

        if isinstance(quoted_raw_payload, dict):
            quoted_images = extract_images_from_raw_payload(quoted_raw_payload)
            if quoted_images:
                source_msg_id = str(quoted_raw_payload.get("message_id", "")).strip() or event.reply_to_msg_id
                return ResolvedImageTurn(
                    images=quoted_images,
                    source_msg_id=source_msg_id,
                    source_kind="quoted_remote",
                )
            return ResolvedImageTurn(
                images=[],
                source_msg_id=event.reply_to_msg_id,
                source_kind="quoted_remote",
            )
        return None

    recent_image_message = _find_recent_image_message(prior_messages=prior_messages)
    if recent_image_message is None:
        return None
    return ResolvedImageTurn(
        images=_extract_message_images(recent_image_message),
        source_msg_id=recent_image_message.platform_msg_id,
        source_kind="recent",
    )


def resolve_private_images_for_turn(
    *,
    event,
    messages,
    quoted_raw_payload: dict | None = None,
    lookback_window: timedelta = timedelta(minutes=3),
) -> ResolvedImageTurn | None:
    if event.images:
        return ResolvedImageTurn(
            images=event.images,
            source_msg_id=event.platform_msg_id,
            source_kind="current",
        )

    if event.reply_to_msg_id is not None:
        quoted_platform_ids = [
            f"private-inbound-{event.user_id}-{event.reply_to_msg_id}",
            str(event.reply_to_msg_id),
        ]
        for quoted_platform_id in quoted_platform_ids:
            quoted_message = messages.get_by_platform_msg_id(quoted_platform_id)
            if quoted_message is None:
                continue
            quoted_images = _extract_message_images(quoted_message)
            if quoted_images:
                return ResolvedImageTurn(
                    images=quoted_images,
                    source_msg_id=quoted_message.platform_msg_id,
                    source_kind="quoted",
                )
            return None
        if isinstance(quoted_raw_payload, dict):
            quoted_images = extract_images_from_raw_payload(quoted_raw_payload)
            if quoted_images:
                source_msg_id = str(quoted_raw_payload.get("message_id", "")).strip() or str(event.reply_to_msg_id)
                return ResolvedImageTurn(
                    images=quoted_images,
                    source_msg_id=source_msg_id,
                    source_kind="quoted_remote",
                )
            return ResolvedImageTurn(
                images=[],
                source_msg_id=str(event.reply_to_msg_id),
                source_kind="quoted_remote",
            )
        return None

    plain_text = str(getattr(event, "plain_text", ""))
    prior_messages = _recent_private_messages_before_event(
        event=event,
        messages=messages,
        window=lookback_window,
    )
    recent_image_message = _find_recent_image_message(prior_messages=prior_messages)
    if recent_image_message is None:
        return None
    if not _has_image_intent_keywords(plain_text):
        latest_prior_message = prior_messages[-1] if prior_messages else None
        if latest_prior_message is None:
            return None
        if latest_prior_message.platform_msg_id != recent_image_message.platform_msg_id:
            return None
        if not _looks_like_contextual_private_image_followup(plain_text):
            return None
    return ResolvedImageTurn(
        images=_extract_message_images(recent_image_message),
        source_msg_id=recent_image_message.platform_msg_id,
        source_kind="recent",
    )
