from datetime import UTC, datetime

from app.adapters.onebot_models import parse_group_message_event
from app.core.image_turn_resolver import resolve_images_for_turn
from app.core.time_utils import ASIA_SHANGHAI
from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


BOT_QQ = 123456789
GROUP_ID = 10001
USER_ID = 20001


def _image_payload(*, message_id: str, timestamp: datetime) -> dict:
    return {
        "post_type": "message",
        "message_type": "group",
        "message_id": message_id,
        "group_id": GROUP_ID,
        "user_id": USER_ID,
        "sender": {"user_id": USER_ID, "nickname": "Maple", "card": ""},
        "message": [
            {
                "type": "image",
                "data": {
                    "file": f"{message_id}.png",
                    "url": f"https://img.example.test/{message_id}.png",
                },
            }
        ],
        "time": int(timestamp.astimezone(UTC).timestamp()),
    }


def _text_event(*, timestamp: datetime, text: str = "这张图怎么样") -> object:
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": "q-1",
        "group_id": GROUP_ID,
        "user_id": USER_ID,
        "sender": {"user_id": USER_ID, "nickname": "Maple", "card": ""},
        "message": [
            {"type": "at", "data": {"qq": str(BOT_QQ)}},
            {"type": "text", "data": {"text": f" {text}"}},
        ],
        "time": int(timestamp.astimezone(UTC).timestamp()),
    }
    return parse_group_message_event(payload, bot_qq=BOT_QQ, bot_name="小町")


def _seed_image(tmp_path, *, message_id: str, timestamp: datetime) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=GROUP_ID, group_name="10001", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=USER_ID, nickname="Maple", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id=message_id,
            group_id=GROUP_ID,
            user_id=USER_ID,
            timestamp=timestamp,
            plain_text="",
            raw_json=_image_payload(message_id=message_id, timestamp=timestamp),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
    return engine


def test_old_image_outside_three_minute_window_is_not_attached(tmp_path) -> None:
    engine = _seed_image(
        tmp_path,
        message_id="old-image",
        timestamp=datetime(2026, 8, 9, 23, 58, 22, tzinfo=ASIA_SHANGHAI),
    )
    event = _text_event(timestamp=datetime(2026, 8, 10, 0, 51, 13, tzinfo=ASIA_SHANGHAI))

    with session_scope(engine) as session:
        result = resolve_images_for_turn(
            event=event,
            addressed_turn=True,
            bot_names={"小町"},
            messages=MessageRepository(session),
        )

    assert result is None


def test_fresh_image_within_window_is_attached(tmp_path) -> None:
    engine = _seed_image(
        tmp_path,
        message_id="fresh-image",
        timestamp=datetime(2026, 8, 10, 0, 50, 30, tzinfo=ASIA_SHANGHAI),
    )
    event = _text_event(timestamp=datetime(2026, 8, 10, 0, 51, 13, tzinfo=ASIA_SHANGHAI))

    with session_scope(engine) as session:
        result = resolve_images_for_turn(
            event=event,
            addressed_turn=True,
            bot_names={"小町"},
            messages=MessageRepository(session),
        )

    assert result is not None
    assert result.images[0].file_id == "fresh-image.png"
    assert result.images[0].source_message_id == "fresh-image"
    assert result.images[0].source_user_id == USER_ID
    assert result.images[0].source_nickname == "Maple"
