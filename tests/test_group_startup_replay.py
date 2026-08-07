from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import app.group_main as group_main
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    UserRepository,
)


def _seed(sqlite_engine) -> None:
    now = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=100000001,
            group_name="g",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="A",
            group_card="",
        )
        UserRepository(session).upsert_user(
            user_id=123456789,
            nickname="bot",
            group_card="",
        )
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="before-1",
            group_id=100000001,
            user_id=20001,
            timestamp=now,
            plain_text="旧的",
            raw_json={"sender": {"nickname": "A", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )
        messages.add_group_message(
            platform_msg_id="window-1",
            group_id=100000001,
            user_id=20001,
            timestamp=now + __import__("datetime").timedelta(minutes=1),
            plain_text="启动窗口 @提问",
            raw_json={"sender": {"nickname": "A", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )
        messages.add_group_message(
            platform_msg_id="window-2",
            group_id=100000001,
            user_id=20001,
            timestamp=now + __import__("datetime").timedelta(minutes=2),
            plain_text="没 @ 的",
            raw_json={"sender": {"nickname": "A", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="window-3",
            group_id=100000001,
            user_id=123456789,
            timestamp=now + __import__("datetime").timedelta(minutes=3),
            plain_text="机器人自己的",
            raw_json={"sender": {"nickname": "bot", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )


def _watermark(sqlite_engine) -> int:
    with session_scope(sqlite_engine) as session:
        row = session.execute(
            __import__("sqlalchemy").text(
                "SELECT id FROM messages WHERE platform_msg_id='before-1'"
            )
        ).scalar_one()
        return int(row)


def test_write_group_ready_marker_writes_fresh_json(tmp_path) -> None:
    group_main._write_group_ready_marker(log_dir=tmp_path, state="ready")

    payload = json.loads((tmp_path / "group.ready.json").read_text(encoding="utf-8"))
    assert payload["state"] == "ready"
    assert payload["pid"] > 0
    assert "updated_at" in payload


def test_write_group_ready_marker_refreshes_connected_state(tmp_path) -> None:
    group_main._write_group_ready_marker(log_dir=tmp_path, state="connected")
    first = json.loads((tmp_path / "group.ready.json").read_text(encoding="utf-8"))

    group_main._write_group_ready_marker(log_dir=tmp_path, state="ready")
    second = json.loads((tmp_path / "group.ready.json").read_text(encoding="utf-8"))

    assert first["state"] == "connected"
    assert second["state"] == "ready"
    assert second["updated_at"] >= first["updated_at"]


def test_startup_window_mention_rows_filters(sqlite_engine) -> None:
    _seed(sqlite_engine)
    rows = group_main._startup_window_mention_rows(
        sqlite_engine,
        watermark_message_id=_watermark(sqlite_engine),
        enabled_group_ids=(100000001,),
        bot_qq=123456789,
    )
    platform_ids = [row.platform_msg_id for row in rows]
    assert platform_ids == ["window-1"]


def test_replay_startup_window_mentions_calls_router(sqlite_engine) -> None:
    _seed(sqlite_engine)

    class StubRouter:
        def __init__(self) -> None:
            self.calls = []

        async def _handle_persisted_group_message(self, event) -> None:
            self.calls.append(
                (
                    event.group_id,
                    event.platform_msg_id,
                    event.mentioned_bot,
                    event.plain_text,
                )
            )

    router = StubRouter()
    settings = SimpleNamespace(bot_qq=123456789)
    runtime = SimpleNamespace(
        group_policy={
            "groups": {
                "100000001": {"enabled": True, "speak": True},
            }
        },
        persona={"name": "小町"},
    )

    asyncio.run(
        group_main._replay_startup_window_mentions(
            engine=sqlite_engine,
            router=router,
            settings=settings,
            runtime=runtime,
            watermark_message_id=_watermark(sqlite_engine),
        )
    )

    assert len(router.calls) == 1
    group_id, platform_id, mentioned_bot, plain_text = router.calls[0]
    assert group_id == 100000001
    assert platform_id == "window-1"
    assert mentioned_bot is True
    assert plain_text == "启动窗口 @提问"
