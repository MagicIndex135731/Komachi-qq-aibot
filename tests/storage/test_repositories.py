from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import text

from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import (
    GroupRepository,
    DevSessionRepository,
    DevTaskRepository,
    MemoryRepository,
    MessageRepository,
    SummaryRepository,
    UsageRepository,
    UserRepository,
)


def test_repositories_store_groups_users_and_messages(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)

        groups.upsert_group(group_id=10001, group_name="test-group", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="Alice card")
        messages.add_group_message(
            platform_msg_id="m-1",
            group_id=10001,
            user_id=20001,
            timestamp=datetime.now(UTC),
            plain_text="@bot hi",
            raw_json={"self_id": 123456789},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=True,
        )

        recent = messages.list_recent_group_messages(group_id=10001, limit=10)
        message_count = messages.count_group_messages(group_id=10001)

    assert recent[0].plain_text == "@bot hi"
    assert recent[0].mentioned_bot is True
    assert message_count == 1


def test_message_repository_lists_all_delivered_group_messages_chronologically(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="test-group", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="late",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 9, 12, 1, tzinfo=UTC),
            plain_text="later",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="early-bot",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
            plain_text="earlier bot reply",
            raw_json={"direction": "outbound", "delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="reserved",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 12, 2, tzinfo=UTC),
            plain_text="not delivered",
            raw_json={"direction": "outbound", "delivery_state": "reserved"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

        history = messages.list_group_messages_chronological(
            group_id=10001,
            exclude_platform_msg_id="late",
        )

    assert [message.platform_msg_id for message in history] == ["early-bot"]


def test_dev_repositories_create_owner_session_and_queue_task(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)

        owner_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=owner_session.id,
            requested_by_qq=10001,
            raw_request_text="check logs",
            intent_type="log_investigation",
        )

        queued = tasks.list_tasks_by_status("queued")

    assert owner_session.owner_qq == 10001
    assert task.status == "queued"
    assert [item.raw_request_text for item in queued] == ["check logs"]


def test_dev_repositories_can_start_new_owner_session_and_pick_latest(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        sessions = DevSessionRepository(session)
        first_session = sessions.create_owner_session(owner_qq=10001)
        second_session = sessions.create_owner_session(owner_qq=10001)
        latest_session = sessions.get_latest_owner_session(owner_qq=10001)

    assert second_session.id > first_session.id
    assert latest_session is not None
    assert latest_session.id == second_session.id


def test_create_all_backfills_dev_session_mode_for_existing_sqlite_db(tmp_path) -> None:
    engine = build_engine(tmp_path / "legacy-dev-sessions.db")

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE dev_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_qq INTEGER NOT NULL,
                    started_at DATETIME NOT NULL,
                    last_active_at DATETIME NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    last_task_id INTEGER NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO dev_sessions (owner_qq, started_at, last_active_at, summary, last_task_id)
                VALUES (10001, '2026-05-11 00:00:00', '2026-05-11 00:00:00', '', NULL)
                """
            )
        )

    create_all(engine)

    with engine.connect() as connection:
        columns = [row[1] for row in connection.execute(text("PRAGMA table_info(dev_sessions)"))]
        session_mode = connection.execute(text("select session_mode from dev_sessions where id = 1")).scalar_one()

    assert "session_mode" in columns
    assert session_mode == "project"


def test_repositories_return_recent_group_summaries_and_memories(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        summaries = SummaryRepository(session)
        memories = MemoryRepository(session)

        summaries.add_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="daily",
            start_at=datetime(2026, 5, 7, tzinfo=UTC),
            end_at=datetime(2026, 5, 7, 12, tzinfo=UTC),
            content="summary-1",
            source_count=2,
        )
        summaries.add_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="daily",
            start_at=datetime(2026, 5, 8, tzinfo=UTC),
            end_at=datetime(2026, 5, 8, 12, tzinfo=UTC),
            content="summary-2",
            source_count=3,
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="Alice likes hotpot.",
            importance=5,
            confidence=0.9,
            source_msg_id="m-1",
        )

        recent_summaries = summaries.list_recent_group_summaries(scope_id="10001", limit=10)
        group_memories = memories.list_group_memories(scope_id="10001", limit=10)

    assert recent_summaries == ["summary-1", "summary-2"]
    assert [memory.content for memory in group_memories] == ["Alice likes hotpot."]


def test_build_engine_enables_foreign_keys_for_new_connections(tmp_path) -> None:
    sqlite_path = tmp_path / "fk-regression.db"
    engine = build_engine(sqlite_path)

    engine.dispose()

    with engine.connect() as connection:
        foreign_keys = connection.execute(text("PRAGMA foreign_keys;")).scalar_one()

    assert foreign_keys == 1
    assert sqlite_path.exists()


def test_usage_repository_summarizes_usage_window(tmp_path) -> None:
    engine = build_engine(tmp_path / "usage.db")
    create_all(engine)

    with session_scope(engine) as session:
        usage = UsageRepository(session)
        usage.add_usage(
            timestamp=datetime(2026, 5, 9, 1, 0, tzinfo=UTC),
            model="gpt-5.4",
            endpoint="responses",
            input_tokens=100,
            cached_input_tokens=10,
            output_tokens=20,
        )
        usage.add_usage(
            timestamp=datetime(2026, 5, 9, 2, 0, tzinfo=UTC),
            model="gpt-5.4",
            endpoint="chat_completions",
            input_tokens=50,
            cached_input_tokens=0,
            output_tokens=30,
        )
        usage.add_usage(
            timestamp=datetime(2026, 5, 8, 23, 59, tzinfo=UTC),
            model="gpt-5.4",
            endpoint="responses",
            input_tokens=999,
            cached_input_tokens=0,
            output_tokens=999,
        )

        summary = usage.summarize_usage(
            start_at=datetime(2026, 5, 9, 0, 0, tzinfo=UTC),
            end_at=datetime(2026, 5, 9, 23, 59, 59, tzinfo=UTC),
            model="gpt-5.4",
        )

    assert summary == {
        "call_count": 2,
        "input_tokens": 150,
        "cached_input_tokens": 10,
        "output_tokens": 50,
    }


def test_usage_repository_normalizes_local_timezone_timestamps_to_utc_window(tmp_path) -> None:
    engine = build_engine(tmp_path / "usage-local.db")
    create_all(engine)
    china = timezone(timedelta(hours=8))

    with session_scope(engine) as session:
        usage = UsageRepository(session)
        usage.add_usage(
            timestamp=datetime(2026, 5, 9, 15, 5, tzinfo=china),
            model="gpt-5.4",
            endpoint="chat_completions",
            input_tokens=200,
            cached_input_tokens=0,
            output_tokens=50,
        )

        summary = usage.summarize_usage(
            start_at=datetime(2026, 5, 9, 7, 0, tzinfo=UTC),
            end_at=datetime(2026, 5, 9, 7, 10, tzinfo=UTC),
            model="gpt-5.4",
        )

    assert summary == {
        "call_count": 1,
        "input_tokens": 200,
        "cached_input_tokens": 0,
        "output_tokens": 50,
    }


def test_message_repository_lists_group_messages_since_for_weekly_report(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-report.db")
    create_all(engine)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)

        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        groups.upsert_group(group_id=10002, group_name="group-2", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="Alice")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")

        messages.add_group_message(
            platform_msg_id="m-old",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 1, tzinfo=UTC),
            plain_text="too old",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-keep",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 14, tzinfo=UTC),
            plain_text="这条要进周报",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-bot",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 14, 1, tzinfo=UTC),
            plain_text="bot self message",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-blank",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 14, 2, tzinfo=UTC),
            plain_text="   ",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-other-group",
            group_id=10002,
            user_id=20001,
            timestamp=datetime(2026, 5, 14, 3, tzinfo=UTC),
            plain_text="other group",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-reserved",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 14, 4, tzinfo=UTC),
            plain_text="reserved outbound",
            raw_json={"delivery_state": "reserved"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

        kept = messages.list_group_messages_since(
            group_id=10001,
            since=datetime(2026, 5, 8, tzinfo=UTC),
            bot_user_id=123456789,
            limit=50,
        )

    assert [message.platform_msg_id for message in kept] == ["m-keep"]
