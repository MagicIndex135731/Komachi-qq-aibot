from datetime import UTC, datetime, timedelta, timezone

import pytest

from sqlalchemy import text

from app.storage.db import build_engine, create_all, session_scope
from app.storage.models import RetrievalDocumentMessage
from app.storage.repositories import (
    GroupRepository,
    DevSessionRepository,
    DevTaskRepository,
    MemoryRepository,
    MessageRepository,
    EpisodeRepository,
    RetrievalDocumentRepository,
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


def test_qq_blocked_reply_stays_in_context_but_not_memory_compaction_sources(tmp_path) -> None:
    engine = build_engine(tmp_path / "bot.db")
    create_all(engine)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="test-group", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        blocked = messages.add_group_message(
            platform_msg_id="blocked-1",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
            plain_text="blocked sensitive reply\n\n[system delivery note]",
            raw_json={
                "direction": "outbound",
                "delivery_state": "blocked",
                "failure_kind": "qq_sensitive_content",
            },
            msg_type="text",
            reply_to_msg_id="inbound-1",
            mentioned_bot=False,
        )
        session.flush()

        recent = messages.list_recent_group_messages(group_id=10001, limit=10)
        summary_recent = messages.list_recent_group_messages_for_summarization(group_id=10001, limit=10)
        chronological = messages.list_group_messages_chronological(group_id=10001)
        compaction_windows = messages.list_recent_group_message_windows(
            group_id=10001,
            batch_size=1,
            limit_windows=10,
        )
        compaction_range = messages.list_group_messages_by_id_range(
            group_id=10001,
            start_id=blocked.id,
            end_id=blocked.id,
        )

    assert [message.platform_msg_id for message in recent] == ["blocked-1"]
    assert summary_recent == []
    assert [message.platform_msg_id for message in chronological] == ["blocked-1"]
    assert compaction_windows == []
    assert compaction_range == []


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
        messages.add_group_message(
            platform_msg_id="m-blocked",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 14, 5, tzinfo=UTC),
            plain_text="blocked outbound",
            raw_json={
                "direction": "outbound",
                "delivery_state": "blocked",
                "failure_kind": "qq_sensitive_content",
            },
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m-future",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 16, tzinfo=UTC),
            plain_text="outside requested weekly window",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

        kept = messages.list_group_messages_since(
            group_id=10001,
            since=datetime(2026, 5, 8, tzinfo=UTC),
            bot_user_id=123456789,
            limit=None,
            until=datetime(2026, 5, 15, tzinfo=UTC),
            exclude_qq_blocked_outbound=True,
        )

    assert [message.platform_msg_id for message in kept] == ["m-keep"]


def test_message_repository_limited_weekly_messages_keep_latest_rows_in_order(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-latest-limit.db")
    create_all(engine)
    start_at = datetime(2026, 5, 8, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        for index in range(250):
            messages.add_group_message(
                platform_msg_id=f"m-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=start_at + timedelta(minutes=index),
                plain_text=f"weekly source {index}",
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

        kept = messages.list_group_messages_since(
            group_id=10001,
            since=start_at,
            until=start_at + timedelta(days=1),
            bot_user_id=123456789,
            limit=200,
            exclude_qq_blocked_outbound=True,
        )

    assert [message.platform_msg_id for message in kept] == [
        f"m-{index}" for index in range(50, 250)
    ]


def _add_weekly_source_message(
    messages: MessageRepository,
    *,
    group_id: int,
    user_id: int,
    platform_msg_id: str,
    timestamp: datetime,
    raw_json: dict | None = None,
):
    message = messages.add_group_message(
        platform_msg_id=platform_msg_id,
        group_id=group_id,
        user_id=user_id,
        timestamp=timestamp,
        plain_text=f"source {platform_msg_id}",
        raw_json=raw_json or {},
        msg_type="text",
        reply_to_msg_id=None,
        mentioned_bot=False,
    )
    messages.session.flush()
    return message


def _add_weekly_episode_summary(
    session,
    *,
    group_id: int,
    source_messages,
    start_at: datetime,
    end_at: datetime,
    content_hash: str,
    status: str = "active",
    document_kind: str = "episode_summary",
    episode_status: str = "processed",
    compaction_version: str = "compact-v2",
    metadata_generation: str | None = None,
):
    episode = EpisodeRepository(session).create_episode(
        group_id=group_id,
        start_message_id=source_messages[0].id,
        started_at=start_at,
        segmentation_version="segment-v2",
        status=episode_status,
    )
    episode.compaction_version = compaction_version
    summary = SummaryRepository(session).upsert_summary(
        scope_type="group",
        scope_id=str(group_id),
        summary_level="episode",
        summary_key=content_hash,
        start_at=start_at,
        end_at=end_at,
        content=f"summary {content_hash}",
        source_count=len(source_messages),
        source_start_msg_id=source_messages[0].platform_msg_id,
        source_end_msg_id=source_messages[-1].platform_msg_id,
    )
    session.flush()
    return RetrievalDocumentRepository(session).upsert_document(
        scope_type="group",
        scope_id=str(group_id),
        group_id=group_id,
        episode_id=episode.id,
        document_kind=document_kind,
        source_table="summaries",
        source_id=str(summary.id),
        start_at=start_at,
        end_at=end_at,
        content=f"summary {content_hash}",
        metadata_json={
            "compaction_generation": (
                compaction_version
                if metadata_generation is None
                else metadata_generation
            )
        },
        content_hash=content_hash,
        source_message_ids=[message.id for message in source_messages],
        status=status,
    )


def test_retrieval_documents_list_weekly_episode_summaries_scoped_by_window_status_and_kind(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-summaries.db")
    create_all(engine)
    window_start = datetime(2026, 7, 17, 12, tzinfo=UTC)
    window_end = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        groups.upsert_group(group_id=10002, group_name="group-2", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")

        left_edge = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-left-edge",
            timestamp=window_start - timedelta(days=1),
        )
        inside = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-inside",
            timestamp=window_start + timedelta(days=1),
        )
        other = _add_weekly_source_message(
            messages,
            group_id=10002,
            user_id=20002,
            platform_msg_id="weekly-other-group",
            timestamp=window_start + timedelta(days=1),
        )
        left_edge_document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[left_edge],
            start_at=window_start - timedelta(days=1),
            end_at=window_start,
            content_hash="weekly-left-edge",
        )
        expected = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[inside],
            start_at=window_start + timedelta(days=1),
            end_at=window_end - timedelta(days=1),
            content_hash="weekly-inside",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[inside],
            start_at=window_start + timedelta(days=2),
            end_at=window_start + timedelta(days=3),
            content_hash="weekly-inactive",
            status="inactive",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[inside],
            start_at=window_start + timedelta(days=3),
            end_at=window_start + timedelta(days=4),
            content_hash="weekly-other-kind",
            document_kind="episode",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10002,
            source_messages=[other],
            start_at=window_start + timedelta(days=1),
            end_at=window_end - timedelta(days=1),
            content_hash="weekly-other-group",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[left_edge],
            start_at=window_start - timedelta(days=3),
            end_at=window_start - timedelta(microseconds=1),
            content_hash="weekly-before-window",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[inside],
            start_at=window_end + timedelta(microseconds=1),
            end_at=window_end + timedelta(days=1),
            content_hash="weekly-after-window",
        )

        documents = RetrievalDocumentRepository(session).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=window_start,
            end_at=window_end,
        )

    assert [document.document_id for document in documents] == [
        left_edge_document.id,
        expected.id,
    ]
    assert [document.content for document in documents] == [
        "summary weekly-left-edge",
        "summary weekly-inside",
    ]
    assert documents[1].episode_id == expected.episode_id
    assert documents[1].source_msg_ids == ()


def test_retrieval_documents_weekly_summary_requires_a_current_episode(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-current-episode.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        source = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-not-current-source",
            timestamp=now,
        )
        document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[source],
            start_at=now,
            end_at=now,
            content_hash="weekly-not-current",
        )
        episode = EpisodeRepository(session).get_episode(document.episode_id)
        assert episode is not None
        episode.is_current = False
        session.add(episode)

        documents = RetrievalDocumentRepository(session).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []


def test_retrieval_documents_weekly_summary_query_is_bounded(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-summary-limit.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        for index in range(4):
            source = _add_weekly_source_message(
                messages,
                group_id=10001,
                user_id=20001,
                platform_msg_id=f"weekly-limited-summary-{index}",
                timestamp=now - timedelta(minutes=4 - index),
            )
            _add_weekly_episode_summary(
                session,
                group_id=10001,
                source_messages=[source],
                start_at=source.timestamp,
                end_at=source.timestamp,
                content_hash=f"weekly-limited-summary-{index}",
            )

        repository = RetrievalDocumentRepository(session)
        documents = repository.list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
            limit=3,
        )
        with pytest.raises(ValueError, match="limit"):
            repository.list_active_episode_summaries_for_window(
                group_id=10001,
                start_at=now - timedelta(days=7),
                end_at=now,
                limit=0,
            )

    assert len(documents) == 3
    assert [document.content for document in documents] == [
        "summary weekly-limited-summary-0",
        "summary weekly-limited-summary-1",
        "summary weekly-limited-summary-2",
    ]


def test_retrieval_documents_weekly_summary_rejects_missing_provenance(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-missing-provenance.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        source = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-missing-provenance-source",
            timestamp=now,
        )
        episode = EpisodeRepository(session).create_episode(
            group_id=10001,
            start_message_id=source.id,
            started_at=now,
            segmentation_version="segment-v2",
            status="processed",
        )
        session.flush()
        RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=episode.id,
            document_kind="episode_summary",
            source_table="summaries",
            source_id="weekly-missing-provenance",
            start_at=now,
            end_at=now,
            content="summary missing provenance",
            metadata_json={},
            content_hash="weekly-missing-provenance",
            source_message_ids=[],
        )

        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []


def test_retrieval_documents_weekly_summary_rejects_cross_group_or_dangling_provenance(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-invalid-provenance.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        groups.upsert_group(group_id=10002, group_name="group-2", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        source = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-valid-source",
            timestamp=now,
        )
        other = _add_weekly_source_message(
            messages,
            group_id=10002,
            user_id=20002,
            platform_msg_id="weekly-cross-group-source",
            timestamp=now,
        )
        document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[source],
            start_at=now,
            end_at=now,
            content_hash="weekly-cross-group-provenance",
        )
        document_id = document.id
        other_message_id = other.id

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
        connection.execute(
            text(
                "UPDATE retrieval_document_messages "
                "SET group_id = :group_id, message_id = :message_id "
                "WHERE document_id = :document_id"
            ),
            {
                "group_id": 10002,
                "message_id": other_message_id,
                "document_id": document_id,
            },
        )
        connection.commit()

    with session_scope(engine) as session:
        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
        connection.execute(
            text(
                "UPDATE retrieval_document_messages "
                "SET group_id = :group_id, message_id = :message_id "
                "WHERE document_id = :document_id"
            ),
            {
                "group_id": 10001,
                "message_id": 999999,
                "document_id": document_id,
            },
        )
        connection.commit()

    with session_scope(engine) as session:
        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []


def test_retrieval_documents_weekly_summary_preserves_source_provenance_order(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-source-order.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="group-1", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        first = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-source-first",
            timestamp=now - timedelta(minutes=2),
        )
        second = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-source-second",
            timestamp=now - timedelta(minutes=1),
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[second, first],
            start_at=now - timedelta(minutes=2),
            end_at=now,
            content_hash="weekly-source-order",
        )

        repository = RetrievalDocumentRepository(session)
        documents = repository.list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )
        message_groups = repository.list_weekly_document_message_groups(
            group_id=10001,
            document_ids=[documents[0].document_id],
            start_at=now - timedelta(days=7),
            end_at=now,
            bot_user_id=123456789,
        )

    assert [message.platform_msg_id for message in message_groups[0]] == [
        "weekly-source-second",
        "weekly-source-first",
    ]


def test_retrieval_documents_weekly_summary_requires_processed_matching_generation(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-terminal-generation.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        sources = [
            _add_weekly_source_message(
                messages,
                group_id=10001,
                user_id=20001,
                platform_msg_id=f"weekly-state-{index}",
                timestamp=now - timedelta(minutes=4 - index),
            )
            for index in range(4)
        ]
        expected = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[sources[0]],
            start_at=sources[0].timestamp,
            end_at=sources[0].timestamp,
            content_hash="weekly-processed-current",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[sources[1]],
            start_at=sources[1].timestamp,
            end_at=sources[1].timestamp,
            content_hash="weekly-processing",
            episode_status="processing",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[sources[2]],
            start_at=sources[2].timestamp,
            end_at=sources[2].timestamp,
            content_hash="weekly-failed",
            episode_status="failed",
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[sources[3]],
            start_at=sources[3].timestamp,
            end_at=sources[3].timestamp,
            content_hash="weekly-stale-generation",
            compaction_version="compact-current",
            metadata_generation="compact-stale",
        )

        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert [document.document_id for document in documents] == [expected.id]


@pytest.mark.parametrize(
    "raw_json",
    [
        {"delivery_state": "reserved"},
        {
            "delivery_state": "blocked",
            "failure_kind": "qq_sensitive_content",
        },
    ],
)
def test_retrieval_documents_weekly_summary_excludes_unsafe_source_provenance(
    tmp_path,
    raw_json,
) -> None:
    engine = build_engine(tmp_path / f"weekly-unsafe-{raw_json['delivery_state']}.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        source = _add_weekly_source_message(
            MessageRepository(session),
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-unsafe-source",
            timestamp=now,
            raw_json=raw_json,
        )
        _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[source],
            start_at=now,
            end_at=now,
            content_hash="weekly-unsafe-document",
        )

        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
            bot_user_id=123456789,
        )

    assert documents == []


def test_retrieval_documents_weekly_summary_only_accepts_source_role(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-source-role.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        source = _add_weekly_source_message(
            MessageRepository(session),
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-context-only",
            timestamp=now,
        )
        document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[source],
            start_at=now,
            end_at=now,
            content_hash="weekly-context-document",
        )
        session.flush()
        provenance = session.get(
            RetrievalDocumentMessage,
            (document.id, source.id, "source"),
        )
        assert provenance is not None
        session.delete(provenance)
        session.flush()
        session.add(
            RetrievalDocumentMessage(
                document_id=document.id,
                message_id=source.id,
                role="context",
                group_id=10001,
                ordinal=0,
            )
        )
        session.flush()

        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []


def test_retrieval_documents_weekly_summary_excludes_partially_deleted_provenance(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-partial-provenance.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        first = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-partial-first",
            timestamp=now - timedelta(minutes=1),
        )
        second = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-partial-second",
            timestamp=now,
        )
        document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[first, second],
            start_at=first.timestamp,
            end_at=second.timestamp,
            content_hash="weekly-partial-document",
        )
        session.flush()
        provenance = session.get(
            RetrievalDocumentMessage,
            (document.id, second.id, "source"),
        )
        assert provenance is not None
        session.delete(provenance)
        session.flush()

        documents = RetrievalDocumentRepository(
            session
        ).list_active_episode_summaries_for_window(
            group_id=10001,
            start_at=now - timedelta(days=7),
            end_at=now,
        )

    assert documents == []


def test_weekly_document_evidence_rechecks_window_and_document_status(tmp_path) -> None:
    engine = build_engine(tmp_path / "weekly-evidence-recheck.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)
    window_start = now - timedelta(days=7)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        before = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-before-window-source",
            timestamp=window_start - timedelta(seconds=1),
        )
        inside = _add_weekly_source_message(
            messages,
            group_id=10001,
            user_id=20001,
            platform_msg_id="weekly-inside-window-source",
            timestamp=window_start,
        )
        document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[before, inside],
            start_at=window_start - timedelta(minutes=1),
            end_at=window_start,
            content_hash="weekly-boundary-evidence",
        )
        repository = RetrievalDocumentRepository(session)
        groups = repository.list_weekly_document_message_groups(
            group_id=10001,
            document_ids=[document.id],
            start_at=window_start,
            end_at=now,
            bot_user_id=123456789,
        )
        assert [message.platform_msg_id for message in groups[0]] == [
            "weekly-inside-window-source"
        ]

        document.status = "inactive"
        session.add(document)
        session.flush()
        with pytest.raises(ValueError, match="no longer valid"):
            repository.list_weekly_document_message_groups(
                group_id=10001,
                document_ids=[document.id],
                start_at=window_start,
                end_at=now,
                bot_user_id=123456789,
            )


def test_weekly_uncovered_messages_are_safe_bounded_and_ignore_nonterminal_coverage(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "weekly-uncovered-limit.db")
    create_all(engine)
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)
    window_start = now - timedelta(days=7)

    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group-1",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="",
        )
        messages = MessageRepository(session)
        rows = [
            _add_weekly_source_message(
                messages,
                group_id=10001,
                user_id=20001,
                platform_msg_id=f"weekly-uncovered-{index}",
                timestamp=window_start + timedelta(minutes=index),
            )
            for index in range(802)
        ]
        covered_document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[rows[0]],
            start_at=rows[0].timestamp,
            end_at=rows[0].timestamp,
            content_hash="weekly-covered",
        )
        nonterminal_document = _add_weekly_episode_summary(
            session,
            group_id=10001,
            source_messages=[rows[1]],
            start_at=rows[1].timestamp,
            end_at=rows[1].timestamp,
            content_hash="weekly-processing-coverage",
            episode_status="processing",
        )

        uncovered = RetrievalDocumentRepository(
            session
        ).list_uncovered_group_messages_for_weekly_report(
            group_id=10001,
            start_at=window_start,
            end_at=now,
            bot_user_id=123456789,
            covered_document_ids=[
                covered_document.id,
                nonterminal_document.id,
            ],
            limit=801,
        )

    uncovered_ids = [message.platform_msg_id for message in uncovered]
    assert len(uncovered_ids) == 801
    assert "weekly-uncovered-0" not in uncovered_ids
    assert "weekly-uncovered-1" in uncovered_ids
    assert uncovered_ids[-1] == "weekly-uncovered-801"
