import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.message_archive import append_group_message_archive, sync_group_message_archives_from_db
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


def test_append_group_message_archive_writes_identity_snapshot_and_text(tmp_path) -> None:
    archive_path = append_group_message_archive(
        history_dir=tmp_path,
        group_id=10001,
        timestamp=datetime(2026, 5, 9, 12, 34, 56, tzinfo=UTC),
        platform_msg_id="msg-1",
        user_id=10001,
        nickname="不知道叫什么",
        group_card="群友甲",
        plain_text="今天吃什么",
        msg_type="text",
        mentioned_bot=True,
        reply_to_msg_id="quoted-1",
        direction="inbound",
        image_local_paths=[],
    )

    assert archive_path == tmp_path / "group-10001" / "2026-05-09.jsonl"
    lines = archive_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record == {
        "timestamp": "2026-05-09T12:34:56+00:00",
        "group_id": 10001,
        "platform_msg_id": "msg-1",
        "user_id": 10001,
        "nickname": "不知道叫什么",
        "group_card": "群友甲",
        "plain_text": "今天吃什么",
        "msg_type": "text",
        "mentioned_bot": True,
        "reply_to_msg_id": "quoted-1",
        "direction": "inbound",
        "image_local_paths": [],
    }


def test_append_group_message_archive_skips_duplicate_platform_msg_id(tmp_path) -> None:
    kwargs = dict(
        history_dir=tmp_path,
        group_id=10001,
        timestamp=datetime(2026, 5, 9, 12, 34, 56, tzinfo=UTC),
        platform_msg_id="msg-dup-1",
        user_id=10001,
        nickname="Alice",
        group_card="",
        plain_text="hello",
        msg_type="text",
        mentioned_bot=False,
        reply_to_msg_id=None,
        direction="inbound",
        image_local_paths=[],
    )

    archive_path = append_group_message_archive(**kwargs)
    append_group_message_archive(**kwargs)

    assert archive_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {
                "timestamp": "2026-05-09T12:34:56+00:00",
                "group_id": 10001,
                "platform_msg_id": "msg-dup-1",
                "user_id": 10001,
                "nickname": "Alice",
                "group_card": "",
                "plain_text": "hello",
                "msg_type": "text",
                "mentioned_bot": False,
                "reply_to_msg_id": None,
                "direction": "inbound",
                "image_local_paths": [],
            },
            ensure_ascii=False,
        )
    ]


def test_sync_group_message_archives_from_db_rebuilds_allowed_group_history_from_local_database(
    sqlite_engine,
    tmp_path,
) -> None:
    existing_archive = tmp_path / "group-10001" / "2026-05-09.jsonl"
    existing_archive.parent.mkdir(parents=True, exist_ok=True)
    existing_archive.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-09T12:00:01+00:00",
                        "group_id": 10001,
                        "platform_msg_id": "m-1",
                        "user_id": 20001,
                        "nickname": "Old Alice",
                        "group_card": "",
                        "plain_text": "first",
                        "msg_type": "text",
                        "mentioned_bot": False,
                        "reply_to_msg_id": None,
                        "direction": "inbound",
                        "image_local_paths": [],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-09T12:00:01+00:00",
                        "group_id": 10001,
                        "platform_msg_id": "m-1",
                        "user_id": 20001,
                        "nickname": "Old Alice",
                        "group_card": "",
                        "plain_text": "first",
                        "msg_type": "text",
                        "mentioned_bot": False,
                        "reply_to_msg_id": None,
                        "direction": "inbound",
                        "image_local_paths": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="test", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="A-card")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="m-1",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
            plain_text="first",
            raw_json={
                "sender": {"nickname": "Alice", "card": "A-card"},
                "message": [{"type": "text", "data": {"text": "first"}}],
            },
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="bot-reply-m-1",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 12, 0, 2, tzinfo=UTC),
            plain_text="reply",
            raw_json={"direction": "outbound", "delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id="m-1",
            mentioned_bot=False,
        )

    summary = sync_group_message_archives_from_db(
        engine=sqlite_engine,
        history_dir=tmp_path,
        allowed_group_ids={10001},
    )

    assert summary == {10001: 2}
    lines = existing_archive.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["platform_msg_id"] == "m-1"
    assert first["nickname"] == "Alice"
    assert first["group_card"] == "A-card"
    assert second["platform_msg_id"] == "bot-reply-m-1"
    assert second["direction"] == "outbound"
