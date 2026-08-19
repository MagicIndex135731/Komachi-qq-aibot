from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    create_retrieval_vector_generation,
    create_all,
    ensure_retrieval_vector_generation,
    find_active_retrieval_vector_generation,
    mark_retrieval_vector_embeddings_failed,
    refresh_retrieval_vector_generation,
    resume_retrieval_vector_generation,
    rollback_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import EpisodeMessage, RetrievalDocument
from app.storage.repositories import (
    EpisodeRepository,
    GroupRepository,
    JobRepository,
    MemoryRepository,
    MemoryBackfillRunRepository,
    RetrievalDocumentRepository,
    RetrievalIndexStateRepository,
    UserRepository,
    MessageRepository,
    _initial_vector_fetch_limit,
    _next_vector_fetch_limit,
    _vector_fetch_ceiling,
)


def test_memory_document_rejects_cross_group_source_memory(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        source_message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="memory-source-group-one",
        )
        other_message = _seed_group_message(
            session,
            group_id=10002,
            user_id=20002,
            platform_msg_id="memory-source-group-two",
        )
        memory = MemoryRepository(session).upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            canonical_key="preference|20001|likes|animation",
            predicate="likes",
            object_text="animation",
            content="Alice likes animation.",
            importance=4,
            confidence=0.9,
            source_msg_ids=[source_message.platform_msg_id],
            valid_from=source_message.timestamp,
        )
        session.flush()

        with pytest.raises(ValueError, match="source scope mismatch"):
            RetrievalDocumentRepository(session).upsert_document(
                scope_type="group",
                scope_id="10002",
                group_id=10002,
                episode_id=None,
                document_kind="memory",
                source_table="memory_items",
                source_id=str(memory.id),
                start_at=other_message.timestamp,
                end_at=other_message.timestamp,
                content=memory.content,
                metadata_json={},
                content_hash="cross-group-memory-document",
                source_message_ids=[other_message.id],
            )


def _seed_group_message(
    session,
    *,
    group_id: int,
    user_id: int,
    platform_msg_id: str,
    minute: int = 0,
):
    groups = GroupRepository(session)
    users = UserRepository(session)
    groups.upsert_group(
        group_id=group_id,
        group_name=f"group-{group_id}",
        enabled=True,
        speak_enabled=True,
    )
    users.upsert_user(user_id=user_id, nickname=f"user-{user_id}", group_card="")
    message = MessageRepository(session).add_group_message(
        platform_msg_id=platform_msg_id,
        group_id=group_id,
        user_id=user_id,
        timestamp=datetime(2026, 7, 23, 8, minute, tzinfo=UTC),
        plain_text=f"message {platform_msg_id}",
        raw_json={},
        msg_type="text",
        reply_to_msg_id=None,
        mentioned_bot=False,
    )
    session.flush()
    return message


def test_raw_message_v3_projection_is_idempotent_and_revokes_uncertain_source(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-idempotent",
        )
        message.plain_text = "即时检索原始消息"
        documents = RetrievalDocumentRepository(session)
        first = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        second = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=7,
        )
        assert first is not None
        assert second is not None
        assert second.id == first.id
        assert second.embedding_generation == 7
        assert second.embedding_status == "pending"
        assert documents.list_source_message_ids(
            document_id=first.id,
            group_id=10001,
        ) == [message.id]
        assert [
            hit.document_id
            for hit in documents.search_group_documents_fts_hits(
                group_id=10001,
                query="即时检索",
                limit=10,
                document_kinds=("raw_message_v3",),
            )
        ] == [first.id]

        message.raw_json = {
            "delivery_state": "uncertain",
            "failure_kind": "delivery_result_unknown",
        }
        assert (
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=message.id,
                embedding_generation=7,
            )
            is None
        )
        assert documents.search_group_documents_fts_hits(
            group_id=10001,
            query="即时检索",
            limit=10,
            document_kinds=("raw_message_v3",),
        ) == []


def test_fts_fallback_matches_separated_short_chinese_topic_terms(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-short-topic",
        )
        message.plain_text = "我最近在杭州，准备安排周末旅行。"
        document = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        assert document is not None

        hits = RetrievalDocumentRepository(session).search_group_documents_fts_hits(
            group_id=10001,
            query="说说杭州旅行",
            limit=10,
            document_kinds=("raw_message_v3",),
        )

    assert [hit.document_id for hit in hits] == [document.id]


def test_deleted_raw_message_is_revoked_and_filtered_by_every_active_channel(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-deleted",
        )
        message.plain_text = "deleted marker evidence"
        documents = RetrievalDocumentRepository(session)
        document = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        assert document is not None
        message.raw_json = {"delivery_state": "deleted"}
        session.flush()

        assert documents.search_group_documents_fts_hits(
            group_id=10001,
            query="deleted marker",
            limit=10,
            document_kinds=("raw_message_v3",),
        ) == []
        assert documents.search_group_documents_temporal_hits(
            group_id=10001,
            start_at=None,
            end_at=None,
            limit=10,
            document_kinds=("raw_message_v3",),
            allow_unbounded=True,
        ) == []
        assert documents.search_group_documents_entity_hits(
            group_id=10001,
            entities=("deleted marker",),
            speaker_ids=("20001",),
            limit=10,
            document_kinds=("raw_message_v3",),
        ) == []
        assert documents.search_group_reference_hits(
            group_id=10001,
            reference_msg_ids=(message.platform_msg_id,),
            include_replies=False,
            limit=10,
            document_kinds=("raw_message_v3",),
        ) == []
        assert (
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=message.id,
            )
            is None
        )


def test_raw_message_v3_fts_fallback_enforces_kind_time_speaker_and_safety(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        allowed = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-allowed",
            minute=1,
        )
        wrong_speaker = _seed_group_message(
            session,
            group_id=10001,
            user_id=20002,
            platform_msg_id="raw-v3-wrong-speaker",
            minute=2,
        )
        blocked = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-blocked",
            minute=3,
        )
        for message in (allowed, wrong_speaker, blocked):
            message.plain_text = "共同检索文本"
        blocked.raw_json = {
            "delivery_state": "blocked",
            "failure_kind": "qq_sensitive_content",
        }
        documents = RetrievalDocumentRepository(session)
        allowed_document = documents.project_raw_message_v3(
            group_id=10001,
            message_id=allowed.id,
            embedding_generation=None,
        )
        documents.project_raw_message_v3(
            group_id=10001,
            message_id=wrong_speaker.id,
            embedding_generation=None,
        )
        assert (
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=blocked.id,
                embedding_generation=None,
            )
            is None
        )
        session.execute(text("DROP TABLE IF EXISTS retrieval_documents_fts"))

        hits = documents.search_group_documents_fts_hits(
            group_id=10001,
            query="共同检索",
            limit=10,
            document_kinds=("raw_message_v3",),
            start_at=datetime(2026, 7, 23, 8, 1, tzinfo=UTC),
            end_at=datetime(2026, 7, 23, 8, 2, tzinfo=UTC),
            speaker_ids=("20001",),
        )
        assert allowed_document is not None
        assert [hit.document_id for hit in hits] == [allowed_document.id]


def test_raw_message_v3_mention_filter_requires_real_onebot_at_segment(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        mentioned = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-mentioned-requester",
            minute=1,
        )
        topical_only = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-not-mentioned",
            minute=2,
        )
        for message in (mentioned, topical_only):
            message.plain_text = "他们最近有没有叫我"
        mentioned.raw_json = {
            "message": [
                {"type": "at", "data": {"qq": "30001"}},
                {"type": "text", "data": {"text": "来看看"}},
            ]
        }
        topical_only.raw_json = {
            "message": [
                {"type": "text", "data": {"text": "30001 来看看"}},
            ]
        }
        documents = RetrievalDocumentRepository(session)
        expected = documents.project_raw_message_v3(
            group_id=10001,
            message_id=mentioned.id,
        )
        documents.project_raw_message_v3(
            group_id=10001,
            message_id=topical_only.id,
        )

        hits = documents.search_group_documents_fts_hits(
            group_id=10001,
            query="他们最近",
            limit=10,
            document_kinds=("raw_message_v3",),
            mentioned_user_ids=("30001",),
        )
        assert expected is not None
        assert [hit.document_id for hit in hits] == [expected.id]

        temporal_hits = documents.search_group_documents_temporal_hits(
            group_id=10001,
            start_at=None,
            end_at=None,
            limit=10,
            document_kinds=("raw_message_v3",),
            mentioned_user_ids=("30001",),
            allow_unbounded=True,
            sample_time_coverage=False,
        )
        assert [hit.document_id for hit in temporal_hits] == [expected.id]


def test_raw_message_v3_unbounded_temporal_coverage_samples_full_history(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        expected_source_ids: list[str] = []
        for minute in range(5):
            message = _seed_group_message(
                session,
                group_id=10001,
                user_id=20001,
                platform_msg_id=f"raw-v3-coverage-{minute}",
                minute=minute,
            )
            message.plain_text = f"coverage {minute}"
            document = documents.project_raw_message_v3(
                group_id=10001,
                message_id=message.id,
            )
            assert document is not None
            if minute in {0, 2, 4}:
                expected_source_ids.append(message.platform_msg_id)

        hits = documents.search_group_documents_temporal_hits(
            group_id=10001,
            start_at=None,
            end_at=None,
            limit=3,
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
            allow_unbounded=True,
        )
        assert [hit.source_msg_ids[0] for hit in hits] == expected_source_ids


def test_raw_message_v3_filters_preserve_none_empty_and_exact_states(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-three-state",
        )
        message.plain_text = "三态边界检索文本"
        message.raw_json = {
            "message": [{"type": "at", "data": {"qq": "30001"}}],
        }
        documents = RetrievalDocumentRepository(session)
        document = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        assert document is not None

        assert documents.search_group_documents_fts(
            group_id=10001,
            query="三态边界",
            limit=10,
            document_kinds=("raw_message_v3",),
            speaker_ids=None,
            mentioned_user_ids=None,
        ) == [document]
        assert documents.search_group_documents_fts(
            group_id=10001,
            query="三态边界",
            limit=10,
            document_kinds=("raw_message_v3",),
            speaker_ids=(),
        ) == []
        assert documents.search_group_documents_fts(
            group_id=10001,
            query="三态边界",
            limit=10,
            document_kinds=("raw_message_v3",),
            mentioned_user_ids=(),
        ) == []
        assert documents.search_group_documents_fts_hits(
            group_id=10001,
            query="三态边界",
            limit=10,
            subject_ids=(),
            document_kinds=("raw_message_v3",),
        ) == []
        assert documents.search_group_documents_temporal_hits(
            group_id=10001,
            start_at=None,
            end_at=None,
            limit=10,
            subject_ids=(),
            document_kinds=("raw_message_v3",),
            allow_unbounded=True,
        ) == []
        assert documents.search_group_documents_entity_hits(
            group_id=10001,
            entities=("三态边界",),
            speaker_ids=(),
            limit=10,
            subject_ids=(),
            document_kinds=("raw_message_v3",),
        ) == []
        assert documents.search_group_reference_hits(
            group_id=10001,
            reference_msg_ids=(message.platform_msg_id,),
            include_replies=False,
            limit=10,
            subject_ids=(),
            document_kinds=("raw_message_v3",),
        ) == []


def test_raw_entity_and_reference_prefilter_legacy_noise_before_limit(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-prefilter-target",
        )
        message.plain_text = "稀疏目标作者关键发言"
        documents = RetrievalDocumentRepository(session)
        raw = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        assert raw is not None
        for index in range(12):
            documents.upsert_document(
                scope_type="group",
                scope_id="10001",
                group_id=10001,
                episode_id=None,
                document_kind="episode",
                source_table="conversation_episodes",
                source_id=f"legacy-noise-{index}",
                start_at=message.timestamp,
                end_at=message.timestamp,
                content="稀疏目标作者关键发言",
                metadata_json={},
                content_hash=f"legacy-noise-{index}",
                source_message_ids=[message.id],
            )

        entity_hits = documents.search_group_documents_entity_hits(
            group_id=10001,
            entities=("稀疏目标作者",),
            speaker_ids=("20001",),
            limit=1,
            document_kinds=("raw_message_v3",),
        )
        reference_hits = documents.search_group_reference_hits(
            group_id=10001,
            reference_msg_ids=(message.platform_msg_id,),
            include_replies=False,
            limit=1,
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        )

        assert [hit.document_id for hit in entity_hits] == [raw.id]
        assert [hit.document_id for hit in reference_hits] == [raw.id]


def test_raw_temporal_prefilters_sparse_author_before_limit(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        target = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-sparse-author",
            minute=0,
        )
        expected = documents.project_raw_message_v3(
            group_id=10001,
            message_id=target.id,
        )
        assert expected is not None
        for minute in range(1, 10):
            noise = _seed_group_message(
                session,
                group_id=10001,
                user_id=20002,
                platform_msg_id=f"raw-v3-busy-noise-{minute}",
                minute=minute,
            )
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=noise.id,
            )

        hits = documents.search_group_documents_temporal_hits(
            group_id=10001,
            start_at=datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 23, 9, 0, tzinfo=UTC),
            limit=1,
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        )

        assert [hit.document_id for hit in hits] == [expected.id]


def test_memory_v2_schema_is_additive_and_repeated_initialization_preserves_messages(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "memory-v2.db")
    create_all(engine)
    with session_scope(engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="original-1",
        )
        original_id = message.id

    create_all(engine)
    create_all(engine)

    with engine.connect() as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
            )
        }
        original = connection.execute(
            text(
                "SELECT id, platform_msg_id, group_id, plain_text "
                "FROM messages WHERE platform_msg_id = 'original-1'"
            )
        ).one()

    assert {
        "conversation_episodes",
        "episode_messages",
        "retrieval_documents",
        "retrieval_document_messages",
        "retrieval_index_state",
        "memory_backfill_runs",
        "memory_late_arrival_preparations",
    } <= tables
    assert tuple(original) == (original_id, "original-1", 10001, "message original-1")


def test_memory_v2_migrates_a_v1_database_without_rewriting_raw_messages(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "legacy-v1.db")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE groups ("
                "group_id INTEGER PRIMARY KEY, group_name VARCHAR(255) NOT NULL DEFAULT '', "
                "enabled BOOLEAN NOT NULL DEFAULT 1, speak_enabled BOOLEAN NOT NULL DEFAULT 0, "
                "reply_mode VARCHAR(32) NOT NULL DEFAULT 'balanced', "
                "cooldown_seconds INTEGER NOT NULL DEFAULT 90, "
                "persona_variant VARCHAR(64) NOT NULL DEFAULT 'default')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE users ("
                "user_id INTEGER PRIMARY KEY, nickname VARCHAR(255) NOT NULL DEFAULT '', "
                "group_card VARCHAR(255) NOT NULL DEFAULT '', first_seen_at DATETIME NULL, "
                "last_seen_at DATETIME NULL, profile_summary TEXT NOT NULL DEFAULT '')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, platform_msg_id VARCHAR(128) UNIQUE NOT NULL, "
                "group_id INTEGER NULL REFERENCES groups(group_id), "
                "user_id INTEGER NOT NULL REFERENCES users(user_id), timestamp DATETIME NOT NULL, "
                "raw_json JSON NOT NULL, plain_text TEXT NOT NULL DEFAULT '', "
                "msg_type VARCHAR(32) NOT NULL DEFAULT 'text', "
                "reply_to_msg_id VARCHAR(128) NULL, mentioned_bot BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE jobs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, job_type VARCHAR(64) NOT NULL, "
                "job_key VARCHAR(255) NOT NULL DEFAULT '', payload_json JSON NOT NULL, "
                "status VARCHAR(32) NOT NULL, run_at DATETIME NOT NULL)"
            )
        )
        connection.execute(text("INSERT INTO groups(group_id) VALUES (10001)"))
        connection.execute(text("INSERT INTO users(user_id) VALUES (20001)"))
        connection.execute(
            text(
                "INSERT INTO messages("
                "platform_msg_id, group_id, user_id, timestamp, raw_json, plain_text"
                ") VALUES ('legacy-raw', 10001, 20001, '2026-07-23 08:00:00', "
                "'{\"message\":\"不可变原文\"}', '不可变原文')"
            )
        )

    create_all(engine)
    with engine.connect() as connection:
        raw = connection.execute(
            text(
                "SELECT platform_msg_id, group_id, user_id, CAST(raw_json AS TEXT), plain_text "
                "FROM messages WHERE platform_msg_id = 'legacy-raw'"
            )
        ).one()
        job_columns = {
            str(row[1]) for row in connection.execute(text("PRAGMA table_info(jobs)"))
        }
        assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"

    assert tuple(raw) == (
        "legacy-raw",
        10001,
        20001,
        '{"message":"不可变原文"}',
        "不可变原文",
    )
    assert {
        "requested_generation",
        "processed_generation",
        "claimed_generation",
        "locked_at",
        "lease_until",
    } <= job_columns


def test_create_all_is_safe_for_two_concurrent_entry_points(tmp_path) -> None:
    database_path = tmp_path / "concurrent-memory-v2.db"
    initial = build_engine(database_path)
    create_all(initial)
    with session_scope(initial) as session:
        _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="before-concurrent-migration",
        )
    initial.dispose()

    barrier = Barrier(2)
    engines = [build_engine(database_path), build_engine(database_path)]

    def initialize(engine) -> None:
        try:
            barrier.wait()
            create_all(engine)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(initialize, engines))

    check = create_engine(f"sqlite:///{database_path}", future=True)
    with check.connect() as connection:
        assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
        assert (
            connection.execute(text("SELECT count(*) FROM messages")).scalar_one() == 1
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM conversation_episodes")
            ).scalar_one()
            == 0
        )
    check.dispose()


def test_create_all_is_safe_for_two_concurrent_fresh_database_entry_points(
    tmp_path,
) -> None:
    database_path = tmp_path / "concurrent-fresh-memory-v2.db"
    barrier = Barrier(2)
    engines = [build_engine(database_path), build_engine(database_path)]

    def initialize(engine) -> None:
        try:
            barrier.wait()
            create_all(engine)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(initialize, engines))

    check = create_engine(f"sqlite:///{database_path}", future=True)
    with check.connect() as connection:
        assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'conversation_episodes'"
                )
            ).scalar_one()
            == 1
        )
    check.dispose()


def test_episode_membership_and_documents_reject_cross_group_links(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        first = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="group-one",
        )
        second = _seed_group_message(
            session,
            group_id=10002,
            user_id=20002,
            platform_msg_id="group-two",
        )
        episodes = EpisodeRepository(session)
        episode = episodes.create_episode(
            group_id=10001,
            start_message_id=first.id,
            started_at=first.timestamp,
            segmentation_version="v2",
        )
        session.flush()
        episodes.add_message(
            episode_id=episode.id,
            group_id=10001,
            message_id=first.id,
            ordinal=0,
            estimated_tokens=3,
        )
        session.flush()
        episode_id = episode.id
        first_id = first.id
        second_id = second.id

    with session_scope(sqlite_engine) as session:
        session.add(
            EpisodeMessage(
                episode_id=episode_id,
                group_id=10001,
                message_id=second_id,
                ordinal=1,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()

    with session_scope(sqlite_engine) as session:
        first = MessageRepository(session).get_by_platform_msg_id("group-one")
        second = MessageRepository(session).get_by_platform_msg_id("group-two")
        assert first is not None and second is not None
        episodes = EpisodeRepository(session)
        episode = episodes.get_open_episode(group_id=10001)
        assert episode is not None
        assert [
            message.id
            for message in episodes.list_episode_messages(
                episode_id=episode.id,
                group_id=10001,
            )
        ] == [first.id]
        assert (
            episodes.list_episode_messages(
                episode_id=episode.id,
                group_id=10002,
            )
            == []
        )
        documents = RetrievalDocumentRepository(session)
        with pytest.raises(IntegrityError):
            documents.upsert_document(
                scope_type="group",
                scope_id="10001",
                group_id=10001,
                episode_id=episode.id,
                document_kind="episode",
                source_table="conversation_episodes",
                source_id=str(episode.id),
                start_at=first.timestamp,
                end_at=first.timestamp,
                content="cross-group provenance must fail",
                metadata_json={},
                content_hash="cross-group",
                source_message_ids=[second.id],
            )
            session.flush()
        session.rollback()
        session.rollback()


def test_one_message_has_one_episode_and_group_has_one_open_episode(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        first = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="unique-membership",
        )
        episodes = EpisodeRepository(session)
        open_episode = episodes.create_episode(
            group_id=10001,
            start_message_id=first.id,
            started_at=first.timestamp,
            segmentation_version="v2",
        )
        session.flush()
        first_id = first.id

    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        with pytest.raises(IntegrityError):
            episodes.create_episode(
                group_id=10001,
                start_message_id=first_id,
                started_at=datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
                segmentation_version="v2-next",
            )
            session.flush()
        session.rollback()

    with session_scope(sqlite_engine) as session:
        first = MessageRepository(session).get_by_platform_msg_id("unique-membership")
        assert first is not None
        episodes = EpisodeRepository(session)
        open_episode = episodes.get_open_episode(group_id=10001)
        assert open_episode is not None
        episodes.add_message(
            episode_id=open_episode.id,
            group_id=10001,
            message_id=first.id,
            ordinal=0,
            estimated_tokens=2,
        )
        episodes.close_episode(
            episode_id=open_episode.id,
            ended_at=first.timestamp,
            end_message_id=first.id,
            boundary_reason="idle",
            content_hash="closed",
        )
        replacement = episodes.create_episode(
            group_id=10001,
            start_message_id=first.id,
            started_at=first.timestamp,
            segmentation_version="v3",
        )
        session.flush()
        with pytest.raises(IntegrityError):
            episodes.add_message(
                episode_id=replacement.id,
                group_id=10001,
                message_id=first.id,
                ordinal=0,
                estimated_tokens=2,
            )
            session.flush()
        session.rollback()


def test_replayed_episode_boundary_gets_a_derived_generation(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="replayed-boundary",
        )
        episodes = EpisodeRepository(session)
        original = episodes.create_episode(
            group_id=10001,
            start_message_id=message.id,
            started_at=message.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        original_id = original.id
        assert (
            episodes.close_episode(
                episode_id=original_id,
                ended_at=message.timestamp,
                end_message_id=message.id,
                boundary_reason="idle",
                content_hash="same-content",
            )
            is not None
        )
        assert episodes.supersede_episode(
            episode_id=original_id,
            group_id=10001,
        )
        replay = episodes.create_episode(
            group_id=10001,
            start_message_id=message.id,
            started_at=message.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        replay_id = replay.id

        closed = episodes.close_episode(
            episode_id=replay_id,
            ended_at=message.timestamp,
            end_message_id=message.id,
            boundary_reason="idle",
            content_hash="same-content",
        )
        session.flush()

        assert closed is not None
        assert closed.status == "closed"
        assert closed.segmentation_version == f"segment-v2:late:{replay_id}"


def test_guarded_episode_append_rejects_superseded_target(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="guarded-append",
        )
        episodes = EpisodeRepository(session)
        stale = episodes.create_episode(
            group_id=10001,
            start_message_id=message.id,
            started_at=message.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        stale_id = stale.id
        message_id = message.id
        assert episodes.supersede_episode(
            episode_id=stale_id,
            group_id=10001,
        )

    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        assert not episodes.add_message_if_current(
            episode_id=stale_id,
            group_id=10001,
            message_id=message_id,
            estimated_tokens=7,
        )
        assert episodes.list_episode_messages(
            episode_id=stale_id,
            group_id=10001,
        ) == []
        message = MessageRepository(session).get_by_platform_msg_id(
            "guarded-append"
        )
        assert message is not None
        current = episodes.create_episode(
            group_id=10001,
            start_message_id=message.id,
            started_at=message.timestamp,
            segmentation_version="segment-v2:late:1",
        )
        session.flush()
        assert episodes.add_message_if_current(
            episode_id=current.id,
            group_id=10001,
            message_id=message.id,
            estimated_tokens=7,
        )
        session.flush()
        current = episodes.get_episode(current.id)
        assert current is not None
        assert current.message_count == 1
        assert current.token_count == 7


def test_coalescing_job_generation_and_completion_cas_do_not_lose_wakeups(
    sqlite_engine,
) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        first = jobs.enqueue_coalescing_job(
            job_type="episode_allocate",
            job_key="group:10001:v2",
            payload_json={"group_id": 10001},
            run_at=now,
        )
        assert first.requested_generation == 1
        claimed = jobs.claim_coalescing_job(
            job_type="episode_allocate",
            worker_id="worker-a",
            now=now,
            lease_seconds=30,
        )
        assert claimed is not None
        claimed_generation = claimed.claimed_generation
        assert claimed_generation == 1

    with session_scope(sqlite_engine) as session:
        running = JobRepository(session).enqueue_coalescing_job(
            job_type="episode_allocate",
            job_key="group:10001:v2",
            payload_json={"group_id": 10001, "dirty": True},
            run_at=now,
        )
        assert running.status == "running"
        assert running.requested_generation == 2

    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        completed = jobs.complete_coalescing_job(
            job_id=running.id,
            worker_id="worker-a",
            claimed_generation=claimed_generation,
            now=now,
        )
        assert completed is not None
        assert completed.status == "queued"
        assert completed.processed_generation == 1
        reclaimed = jobs.claim_coalescing_job(
            job_type="episode_allocate",
            worker_id="worker-b",
            now=now,
            lease_seconds=30,
        )
        assert reclaimed is not None
        assert reclaimed.claimed_generation == 2
        assert (
            jobs.claim_coalescing_job(
                job_type="episode_allocate",
                worker_id="worker-c",
                now=now,
                lease_seconds=30,
            )
            is None
        )
        terminal = jobs.complete_coalescing_job(
            job_id=reclaimed.id,
            worker_id="worker-b",
            claimed_generation=2,
            now=now,
        )
        assert terminal is not None
        assert terminal.status == "completed"
        assert terminal.processed_generation == 2

    with session_scope(sqlite_engine) as session:
        rearmed = JobRepository(session).enqueue_coalescing_job(
            job_type="episode_allocate",
            job_key="group:10001:v2",
            payload_json={"group_id": 10001, "after_completion": True},
            run_at=now,
        )
        assert rearmed.status == "queued"
        assert rearmed.requested_generation == 3


def test_stale_coalescing_lease_is_recovered(sqlite_engine) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        jobs.enqueue_coalescing_job(
            job_type="episode_allocate",
            job_key="group:10001:v2",
            payload_json={"group_id": 10001},
            run_at=now,
        )
        assert jobs.claim_coalescing_job(
            job_type="episode_allocate",
            worker_id="crashed-worker",
            now=now,
            lease_seconds=5,
        )

    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        assert (
            jobs.requeue_stale_coalescing_jobs(
                job_type="episode_allocate",
                now=now + timedelta(seconds=6),
            )
            == 1
        )
        recovered = jobs.claim_coalescing_job(
            job_type="episode_allocate",
            worker_id="recovery-worker",
            now=now + timedelta(seconds=6),
            lease_seconds=5,
        )
        assert recovered is not None
        assert recovered.claimed_generation == 1


def test_two_workers_cannot_claim_the_same_coalescing_generation(sqlite_engine) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        JobRepository(session).enqueue_coalescing_job(
            job_type="episode_allocate",
            job_key="group:10001:v2",
            payload_json={"group_id": 10001},
            run_at=now,
        )

    barrier = Barrier(2)

    def claim(worker_id: str) -> int | None:
        barrier.wait()
        with session_scope(sqlite_engine) as session:
            claimed = JobRepository(session).claim_coalescing_job(
                job_type="episode_allocate",
                worker_id=worker_id,
                now=now,
                lease_seconds=30,
            )
            return claimed.id if claimed is not None else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ["worker-a", "worker-b"]))

    assert sum(claimed_id is not None for claimed_id in claims) == 1


def test_retrieval_document_fts_is_group_scoped_and_falls_back_to_like(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        first = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="fts-group-one",
        )
        second = _seed_group_message(
            session,
            group_id=10002,
            user_id=20002,
            platform_msg_id="fts-group-two",
        )
        documents = RetrievalDocumentRepository(session)
        target = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(first.id),
            start_at=first.timestamp,
            end_at=first.timestamp,
            content="Alice plans a 杭州旅行 next week",
            metadata_json={},
            content_hash="target",
            source_message_ids=[first.id],
        )
        documents.upsert_document(
            scope_type="group",
            scope_id="10002",
            group_id=10002,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(second.id),
            start_at=second.timestamp,
            end_at=second.timestamp,
            content="Alice plans a 杭州旅行 next week",
            metadata_json={},
            content_hash="other",
            source_message_ids=[second.id],
        )
        found = documents.search_group_documents_fts(
            group_id=10001,
            query="杭州旅行",
            limit=10,
        )
        assert [document.id for document in found] == [target.id]
        assert documents.list_source_message_ids(
            document_id=target.id,
            group_id=10001,
        ) == [first.id]
        assert (
            documents.list_source_message_ids(
                document_id=target.id,
                group_id=10002,
            )
            == []
        )
        session.execute(text("DROP TABLE IF EXISTS retrieval_documents_fts"))
        fallback = documents.search_group_documents_fts(
            group_id=10001,
            query="杭州",
            limit=10,
        )
        assert [document.id for document in fallback] == [target.id]


def test_retrieval_index_generation_activation_uses_cas(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        states = RetrievalIndexStateRepository(session)
        first = states.upsert_generation(
            channel="test-fts",
            generation=1,
            physical_table="retrieval_documents_fts",
            provider="sqlite",
            model="trigram",
            dimensions=None,
            version="1",
            status="ready",
            total_documents=5,
            indexed_documents=5,
        )
        assert states.activate_generation(
            channel="test-fts",
            generation=1,
            expected_active_generation=None,
        )
        second = states.upsert_generation(
            channel="test-fts",
            generation=2,
            physical_table="retrieval_documents_fts_v2",
            provider="sqlite",
            model="trigram",
            dimensions=None,
            version="2",
            status="ready",
            total_documents=6,
            indexed_documents=6,
        )
        assert not states.activate_generation(
            channel="test-fts",
            generation=2,
            expected_active_generation=None,
        )
        assert states.activate_generation(
            channel="test-fts",
            generation=2,
            expected_active_generation=first.generation,
        )
        active = states.get_active_generation(channel="test-fts")
        assert active is not None
        assert active.generation == second.generation


def test_memory_backfill_run_persists_frozen_group_watermarks_and_generations(
    sqlite_engine,
) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        runs = MemoryBackfillRunRepository(session)
        created = runs.create_run(
            run_key="memory-v2-20260723",
            snapshot_watermarks={"10001": 123, "10002": 456},
            manifest={"format_version": 1, "digest": "sha256:abc"},
            segmentation_generation="segment-v2",
            compaction_generation="compact-v2",
            index_generation="fts-1",
            created_at=now,
        )
        session.flush()
        run_id = created.id

    with session_scope(sqlite_engine) as session:
        runs = MemoryBackfillRunRepository(session)
        same = runs.create_run(
            run_key="memory-v2-20260723",
            snapshot_watermarks={"10001": 999},
            manifest={"format_version": 99},
            segmentation_generation="ignored",
            compaction_generation="ignored",
            index_generation="ignored",
            created_at=now + timedelta(minutes=1),
        )
        assert same.id == run_id
        assert same.snapshot_watermarks_json == {"10001": 123, "10002": 456}
        completed = runs.update_status(
            run_id=run_id,
            status="completed",
            completed_at=now + timedelta(minutes=2),
            last_error_code="",
        )
        assert completed is not None
        loaded = runs.get_run(run_id=run_id)
        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.segmentation_generation == "segment-v2"
        assert loaded.compaction_generation == "compact-v2"
        assert loaded.index_generation == "fts-1"


def test_episode_worker_queries_are_scoped_watermarked_and_cas_guarded(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        first = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="allocator-first",
            minute=1,
        )
        second = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="allocator-second",
            minute=2,
        )
        other = _seed_group_message(
            session,
            group_id=10002,
            user_id=20002,
            platform_msg_id="allocator-other",
            minute=3,
        )
        episodes = EpisodeRepository(session)
        episode = episodes.create_episode(
            group_id=10001,
            start_message_id=first.id,
            started_at=first.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        episodes.add_message(
            episode_id=episode.id,
            group_id=10001,
            message_id=first.id,
            ordinal=0,
            estimated_tokens=1,
        )
        episodes.close_episode(
            episode_id=episode.id,
            ended_at=first.timestamp,
            end_message_id=first.id,
            boundary_reason="idle",
            content_hash="episode-one",
        )
        episode_id = episode.id
        second_id = second.id
        other_id = other.id

    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        assert [
            row.id
            for row in episodes.list_unassigned_messages(
                group_id=10001,
                watermark_message_id=second_id,
            )
        ] == [second_id]
        assert [
            row.id
            for row in episodes.list_unassigned_messages(
                watermark_message_id=other_id,
            )
        ] == [second_id, other_id]
        assert [
            row.id
            for row in episodes.list_processable_episodes(
                group_id=10001,
                statuses=("closed",),
                compaction_version="",
            )
        ] == [episode_id]
        assert episodes.compare_and_set_status(
            episode_id=episode_id,
            group_id=10001,
            expected_statuses=("closed",),
            new_status="processing",
            compaction_version="compact-v2",
        )
        assert not episodes.compare_and_set_status(
            episode_id=episode_id,
            group_id=10002,
            expected_statuses=("processing",),
            new_status="processed",
        )
        assert episodes.compare_and_set_status(
            episode_id=episode_id,
            group_id=10001,
            expected_statuses=("processing",),
            new_status="processed",
        )


def test_late_arrival_supersedes_only_the_scoped_episode_and_deactivates_documents(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="late-arrival-source",
        )
        _seed_group_message(
            session,
            group_id=10002,
            user_id=20002,
            platform_msg_id="late-arrival-other",
        )
        episodes = EpisodeRepository(session)
        episode = episodes.create_episode(
            group_id=10001,
            start_message_id=message.id,
            started_at=message.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        episodes.add_message(
            episode_id=episode.id,
            group_id=10001,
            message_id=message.id,
            ordinal=0,
            estimated_tokens=1,
        )
        episodes.close_episode(
            episode_id=episode.id,
            ended_at=message.timestamp + timedelta(minutes=5),
            end_message_id=message.id,
            boundary_reason="idle",
            content_hash="late-episode",
        )
        document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=episode.id,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id=str(episode.id),
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="late arrival derived document",
            metadata_json={},
            content_hash="late-document",
            source_message_ids=[message.id],
        )
        episode_id = episode.id
        message_id = message.id
        document_id = document.id

    with session_scope(sqlite_engine) as session:
        episodes = EpisodeRepository(session)
        affected = episodes.find_episode_for_late_arrival(
            group_id=10001,
            timestamp=datetime(2026, 7, 23, 8, 2, tzinfo=UTC),
            segmentation_version="segment-v2",
        )
        assert affected is not None and affected.id == episode_id
        assert (
            episodes.find_episode_for_late_arrival(
                group_id=10002,
                timestamp=datetime(2026, 7, 23, 8, 2, tzinfo=UTC),
                segmentation_version="segment-v2",
            )
            is None
        )
        assert episodes.prepare_late_arrival_resegment(
            group_id=10001,
            message_id=message_id,
            timestamp=datetime(2026, 7, 23, 8, 2, tzinfo=UTC),
            segmentation_version="segment-v2",
            compaction_version="compact-v3",
        ) == [message_id]
        assert not episodes.supersede_episode(
            episode_id=episode_id,
            group_id=10002,
        )
        superseded = episodes.get_episode(episode_id)
        assert superseded is not None
        assert superseded.status == "superseded"
        assert superseded.is_current is False
        assert superseded.compaction_version == "compact-v3"
        assert (
            session.execute(
                text(
                    "SELECT count(*) FROM episode_messages "
                    "WHERE episode_id = :episode_id"
                ),
                {"episode_id": episode_id},
            ).scalar_one()
            == 0
        )
        documents = RetrievalDocumentRepository(session)
        assert (
            documents.deactivate_episode_documents(
                group_id=10001,
                episode_id=episode_id,
            )
            == 0
        )
        assert session.get(type(document), document_id).status == "inactive"


def test_late_arrival_preparation_is_single_winner_under_concurrency(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        original = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="late-concurrent-original",
        )
        late = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="late-concurrent-message",
        )
        episodes = EpisodeRepository(session)
        episode = episodes.create_episode(
            group_id=10001,
            start_message_id=original.id,
            started_at=original.timestamp,
            segmentation_version="segment-v2",
        )
        session.flush()
        episodes.add_message(
            episode_id=episode.id,
            group_id=10001,
            message_id=original.id,
            ordinal=0,
            estimated_tokens=1,
        )
        episodes.close_episode(
            episode_id=episode.id,
            ended_at=original.timestamp + timedelta(minutes=5),
            end_message_id=original.id,
            boundary_reason="idle",
            content_hash="late-concurrent-episode",
        )
        late_id = int(late.id)
        late_timestamp = original.timestamp + timedelta(minutes=2)

    barrier = Barrier(2)

    def prepare_once() -> list[int]:
        barrier.wait()
        with session_scope(sqlite_engine) as session:
            return EpisodeRepository(session).prepare_late_arrival_resegment(
                group_id=10001,
                message_id=late_id,
                timestamp=late_timestamp,
                segmentation_version="segment-v2",
                compaction_version="compact-v3",
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: prepare_once(), (1, 2)))

    assert sorted(bool(result) for result in results) == [False, True]
    with sqlite_engine.connect() as connection:
        assert connection.execute(
            text(
                "SELECT COUNT(*) FROM memory_late_arrival_preparations "
                "WHERE group_id = 10001 AND message_id = :message_id "
                "AND segmentation_generation = 'segment-v2'"
            ),
            {"message_id": late_id},
        ).scalar_one() == 1


def test_coalescing_failure_is_finite_and_failed_job_can_be_explicitly_retried(
    sqlite_engine,
) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        queued = jobs.enqueue_coalescing_job(
            job_type="memory_episode_process",
            job_key="episode:1:compact-v2",
            payload_json={"episode_id": 1},
            run_at=now,
            max_attempts=2,
        )
        claimed = jobs.claim_coalescing_job(
            job_type="memory_episode_process",
            worker_id="worker-a",
            now=now,
            lease_seconds=30,
        )
        assert claimed is not None
        first_failure = jobs.fail_coalescing_job(
            job_id=claimed.id,
            worker_id="worker-a",
            claimed_generation=claimed.claimed_generation,
            error_code="provider_unavailable",
            now=now,
            retry_at=now + timedelta(seconds=5),
        )
        assert first_failure is not None
        assert first_failure.status == "queued"
        assert first_failure.attempt_count == 1

    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        claimed = jobs.claim_coalescing_job(
            job_type="memory_episode_process",
            worker_id="worker-b",
            now=now + timedelta(seconds=5),
            lease_seconds=30,
        )
        assert claimed is not None
        terminal = jobs.fail_coalescing_job(
            job_id=claimed.id,
            worker_id="worker-b",
            claimed_generation=claimed.claimed_generation,
            error_code="bad_dimensions",
            now=now + timedelta(seconds=5),
            retry_at=now + timedelta(seconds=10),
        )
        assert terminal is not None
        assert terminal.status == "failed"
        assert terminal.attempt_count == 2
        retried = jobs.retry_failed_coalescing_job(
            job_id=terminal.id,
            run_at=now + timedelta(seconds=20),
        )
        assert retried is not None
        assert retried.status == "queued"
        assert retried.last_error_code == ""


def test_stale_coalescing_owner_cannot_overwrite_new_owner_payload(
    sqlite_engine,
) -> None:
    now = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        jobs = JobRepository(session)
        queued = jobs.enqueue_coalescing_job(
            job_type="memory_shadow",
            job_key="message:shadow-1",
            payload_json={"phase": "initial"},
            run_at=now,
        )
        first = jobs.claim_coalescing_job(
            job_type="memory_shadow",
            worker_id="worker-a",
            now=now,
            lease_seconds=30,
        )
        assert first is not None
        jobs.enqueue_coalescing_job(
            job_type="memory_shadow",
            job_key="message:shadow-1",
            payload_json={"phase": "rearmed"},
            run_at=now,
        )
        jobs.complete_coalescing_job(
            job_id=queued.id,
            worker_id="worker-a",
            claimed_generation=first.claimed_generation,
            now=now,
        )
        second = jobs.claim_coalescing_job(
            job_type="memory_shadow",
            worker_id="worker-b",
            now=now,
            lease_seconds=30,
        )
        assert second is not None
        assert (
            jobs.update_coalescing_job_payload(
                job_id=second.id,
                worker_id="worker-a",
                claimed_generation=first.claimed_generation,
                payload_json={"phase": "stale"},
            )
            is None
        )
        updated = jobs.update_coalescing_job_payload(
            job_id=second.id,
            worker_id="worker-b",
            claimed_generation=second.claimed_generation,
            payload_json={"phase": "owned"},
        )
        assert updated is not None
        assert updated.payload_json == {"phase": "owned"}


def test_versioned_vector_generation_keeps_old_active_until_compatible_cas_swap(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="vector-generation-source",
        )
        document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(message.id),
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="semantic-only evidence",
            metadata_json={},
            content_hash="vector-generation-document",
            source_message_ids=[message.id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode_summary",
            source_table="summaries",
            source_id="excluded-summary",
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="explicitly excluded summary",
            metadata_json={},
            content_hash="excluded-summary-document",
            source_message_ids=[message.id],
            embedding_eligible=False,
            embedding_status="disabled",
        )
        document_id = document.id

    first_generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-v1",
        dimensions=3,
        version="v1",
    )
    assert first_generation is not None
    assert (
        mark_retrieval_vector_embeddings_failed(
            sqlite_engine,
            generation=first_generation,
            group_id=10002,
            document_ids=[document_id],
            error_code="wrong_group",
        )
        == 0
    )
    assert (
        mark_retrieval_vector_embeddings_failed(
            sqlite_engine,
            generation=first_generation,
            group_id=10001,
            document_ids=[document_id],
            error_code="provider_failure",
        )
        == 1
    )
    failed_coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=first_generation,
    )
    assert failed_coverage.failed_documents == 1
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=first_generation,
        rows=[(document_id, 10001, [1.0, 0.0, 0.0])],
    ) == 1
    first_coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=first_generation,
        mark_ready=True,
    )
    assert first_coverage.coverage == 1.0
    assert first_coverage.total_documents == 1
    assert first_coverage.failed_documents == 0
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=first_generation,
        expected_active_generation=None,
    )

    assert (
        ensure_retrieval_vector_generation(
            sqlite_engine,
            provider="fake",
            model="semantic-v1",
            dimensions=3,
            version="v1",
        )
        == first_generation
    )
    second_generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-v2",
        dimensions=4,
        version="v2",
    )
    assert second_generation is not None
    assert second_generation != first_generation
    with session_scope(sqlite_engine) as session:
        active = RetrievalIndexStateRepository(session).get_active_generation(
            channel="vector"
        )
        assert active is not None and active.generation == first_generation

    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=second_generation,
        rows=[(document_id, 10001, [0.0, 1.0, 0.0, 0.0])],
    ) == 1
    assert refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=second_generation,
        mark_ready=True,
    ).status == "ready"
    assert not activate_retrieval_vector_generation(
        sqlite_engine,
        generation=second_generation,
        expected_active_generation=9999,
    )
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=second_generation,
        expected_active_generation=first_generation,
    )
    with sqlite_engine.connect() as connection:
        table_names = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE name LIKE 'retrieval_documents_vec_g%'"
                )
            )
        }
    assert {
        f"retrieval_documents_vec_g{first_generation}",
        f"retrieval_documents_vec_g{second_generation}",
    } <= table_names


def test_raw_v3_generation_is_additive_kind_scoped_and_activated_by_cas(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-generation-source",
        )
        message.plain_text = "raw generation evidence"
        documents = RetrievalDocumentRepository(session)
        raw_document = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=1,
        )
        episode_document = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="legacy-episode",
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="legacy episode evidence",
            metadata_json={},
            content_hash="legacy-episode",
            source_message_ids=[message.id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        assert raw_document is not None
        raw_document_id = int(raw_document.id)
        episode_document_id = int(episode_document.id)

    active_generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
    )
    assert active_generation is not None
    with session_scope(sqlite_engine) as session:
        raw_document = session.get(RetrievalDocument, raw_document_id)
        assert raw_document is not None
        raw_document.embedding_generation = active_generation
        episode_document = session.get(RetrievalDocument, episode_document_id)
        assert episode_document is not None
        episode_document.embedding_generation = active_generation
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=active_generation,
        rows=[(episode_document_id, 10001, [0.0, 1.0])],
    ) == 1
    assert refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=active_generation,
        mark_ready=True,
    ).status == "ready"
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=active_generation,
        expected_active_generation=None,
    )
    assert (
        find_active_retrieval_vector_generation(
            sqlite_engine,
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            document_family="raw_message_v3",
        )
        is None
    )

    raw_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert raw_generation is not None
    assert raw_generation != active_generation
    with sqlite_engine.connect() as connection:
        state_kind = connection.execute(
            text(
                "SELECT document_family FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": raw_generation},
        ).scalar_one()
        active_before = connection.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).scalar_one()
    assert state_kind == "raw_message_v3"
    assert int(active_before) == active_generation

    with session_scope(sqlite_engine) as session:
        document = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=raw_generation,
        )
        assert document is not None
    with pytest.raises(ValueError, match="document family"):
        write_retrieval_vector_embeddings(
            sqlite_engine,
            generation=raw_generation,
            rows=[(episode_document_id, 10001, [0.0, 1.0])],
        )
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=raw_generation,
        rows=[(raw_document_id, 10001, [1.0, 0.0])],
    ) == 1
    coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        mark_ready=True,
    )
    assert coverage.status == "ready"
    assert coverage.total_documents == 1
    assert coverage.indexed_documents == 1
    assert (
        resume_retrieval_vector_generation(
            sqlite_engine,
            generation=raw_generation,
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            document_family="raw_message_v3",
        )
        == raw_generation
    )
    assert not activate_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        expected_active_generation=9999,
    )
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        expected_active_generation=None,
    )
    assert (
        find_active_retrieval_vector_generation(
            sqlite_engine,
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            document_family="raw_message_v3",
        )
        == raw_generation
    )
    assert (
        find_active_retrieval_vector_generation(
            sqlite_engine,
            provider="fake",
            model="other-model",
            dimensions=2,
            version="v1",
            document_family="raw_message_v3",
        )
        is None
    )


def test_raw_generation_cannot_be_ready_or_active_without_any_documents(
    sqlite_engine,
) -> None:
    generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-empty",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert generation is not None

    coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        mark_ready=True,
    )

    assert coverage.total_documents == 0
    assert coverage.indexed_documents == 0
    assert coverage.status == "building"
    assert not activate_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        expected_active_generation=None,
    )


def test_stale_raw_generation_write_cannot_steal_a_newer_document_claim(
    sqlite_engine,
) -> None:
    first_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-cas",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    second_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-cas",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert first_generation is not None
    assert second_generation is not None
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-generation-cas",
        )
        document = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=first_generation,
        )
        assert document is not None
        document = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=second_generation,
        )
        assert document is not None
        document_id = int(document.id)

    with pytest.raises(ValueError, match="generation claim"):
        write_retrieval_vector_embeddings(
            sqlite_engine,
            generation=first_generation,
            rows=[(document_id, 10001, [1.0, 0.0])],
        )

    with session_scope(sqlite_engine) as session:
        document = session.get(RetrievalDocument, document_id)
        assert document is not None
        assert document.embedding_generation == second_generation


def test_active_old_raw_projection_cannot_empty_a_building_new_generation(
    sqlite_engine,
) -> None:
    old_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-projection-cas",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert old_generation is not None
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-projection-generation-cas",
        )
        projected = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=old_generation,
        )
        assert projected is not None
        document_id = int(projected.id)
        message_id = int(message.id)
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=old_generation,
        rows=[(document_id, 10001, [1.0, 0.0])],
    ) == 1
    assert refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=old_generation,
        mark_ready=True,
    ).status == "ready"
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=old_generation,
        expected_active_generation=None,
    )

    new_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-projection-cas",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert new_generation is not None
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        claimed = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message_id,
            embedding_generation=new_generation,
        )
        assert claimed is not None
        replayed_by_old_worker = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message_id,
            embedding_generation=old_generation,
        )
        assert replayed_by_old_worker is not None
        assert replayed_by_old_worker.embedding_generation == new_generation

    coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=new_generation,
        mark_ready=True,
    )
    assert coverage.total_documents == 1
    assert coverage.indexed_documents == 0
    assert coverage.status == "building"


def test_raw_v3_vector_search_expands_until_scoped_hit_or_generation_exhaustion(
    sqlite_engine,
) -> None:
    generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert generation is not None
    vector_rows: list[tuple[int, int, list[float]]] = []
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        for minute in range(8):
            distractor = _seed_group_message(
                session,
                group_id=10001,
                user_id=20002,
                platform_msg_id=f"raw-v3-vector-distractor-{minute}",
                minute=minute,
            )
            distractor.plain_text = f"distractor {minute}"
            projected = documents.project_raw_message_v3(
                group_id=10001,
                message_id=distractor.id,
                embedding_generation=generation,
            )
            assert projected is not None
            vector_rows.append((projected.id, 10001, [1.0, minute / 1000]))
        target = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="raw-v3-vector-target",
            minute=9,
        )
        target.plain_text = "scoped target"
        projected_target = documents.project_raw_message_v3(
            group_id=10001,
            message_id=target.id,
            embedding_generation=generation,
        )
        assert projected_target is not None
        vector_rows.append((projected_target.id, 10001, [0.0, 1.0]))
        target_document_id = int(projected_target.id)

    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=generation,
        rows=vector_rows,
    ) == len(vector_rows)
    assert refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        mark_ready=True,
    ).status == "ready"
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        assert documents.search_group_documents_vector_hits(
            group_id=10001,
            embedding=[1.0, 0.0],
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            limit=1,
            subject_ids=("20001",),
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        ) == []
        prepared_hits = documents.search_group_documents_vector_hits(
            group_id=10001,
            embedding=[1.0, 0.0],
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            generation=generation,
            limit=1,
            subject_ids=("20001",),
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        )
        assert [hit.document_id for hit in prepared_hits] == [target_document_id]
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        expected_active_generation=None,
    )
    with session_scope(sqlite_engine) as session:
        documents = RetrievalDocumentRepository(session)
        hits = documents.search_group_documents_vector_hits(
            group_id=10001,
            embedding=[1.0, 0.0],
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            limit=1,
            subject_ids=("20001",),
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        )
        assert [hit.document_id for hit in hits] == [target_document_id]
        assert documents.search_group_documents_vector_hits(
            group_id=10001,
            embedding=[1.0, 0.0],
            provider="fake",
            model="semantic",
            dimensions=2,
            version="v1",
            limit=1,
            subject_ids=(),
            document_kinds=("raw_message_v3",),
            speaker_ids=("20001",),
        ) == []


def test_raw_v3_vector_search_expansion_respects_sqlite_vec_knn_cap() -> None:
    assert _initial_vector_fetch_limit(
        requested=600,
        available=17_704,
        has_post_filters=False,
    ) == 600
    assert _initial_vector_fetch_limit(
        requested=600,
        available=17_704,
        has_post_filters=True,
    ) == 4096
    assert _next_vector_fetch_limit(2400, requested=150, available=17_704) == 4096
    assert _next_vector_fetch_limit(4096, requested=150, available=17_704) == 4096
    assert _next_vector_fetch_limit(2400, requested=150, available=3000) == 3000
    assert _vector_fetch_ceiling(
        requested=30,
        available=17_704,
        has_sparse_post_filters=False,
        has_exclusion_filter=True,
    ) == 120
    assert _vector_fetch_ceiling(
        requested=30,
        available=17_704,
        has_sparse_post_filters=True,
        has_exclusion_filter=True,
    ) == 4096


def test_legacy_coverage_excludes_raw_and_explicit_rollback_preserves_generations(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="legacy-rollback-source",
        )
        message.plain_text = "legacy and raw evidence"
        documents = RetrievalDocumentRepository(session)
        episode = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="legacy-rollback-episode",
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="legacy episode",
            metadata_json={},
            content_hash="legacy-rollback-episode",
            source_message_ids=[message.id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        raw = documents.project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
        )
        assert raw is not None
        episode_id = int(episode.id)
        raw_id = int(raw.id)
        message_id = int(message.id)

    legacy_generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
    )
    assert legacy_generation is not None
    with session_scope(sqlite_engine) as session:
        episode = session.get(RetrievalDocument, episode_id)
        raw = session.get(RetrievalDocument, raw_id)
        assert episode is not None and raw is not None
        episode.embedding_generation = legacy_generation
        raw.embedding_generation = legacy_generation
        raw.embedding_eligible = True
        raw.embedding_status = "pending"
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=legacy_generation,
        rows=[(episode_id, 10001, [1.0, 0.0])],
    ) == 1
    with pytest.raises(ValueError, match="document family"):
        write_retrieval_vector_embeddings(
            sqlite_engine,
            generation=legacy_generation,
            rows=[(raw_id, 10001, [0.0, 1.0])],
        )
    legacy_coverage = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=legacy_generation,
        mark_ready=True,
    )
    assert legacy_coverage.status == "ready"
    assert legacy_coverage.total_documents == 1
    assert legacy_coverage.indexed_documents == 1
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=legacy_generation,
        expected_active_generation=None,
    )

    raw_generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert raw_generation is not None
    with session_scope(sqlite_engine) as session:
        projected = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message_id,
            embedding_generation=raw_generation,
        )
        assert projected is not None
    assert write_retrieval_vector_embeddings(
        sqlite_engine,
        generation=raw_generation,
        rows=[(raw_id, 10001, [0.0, 1.0])],
    ) == 1
    assert refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        mark_ready=True,
    ).status == "ready"
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        expected_active_generation=None,
    )

    assert resume_retrieval_vector_generation(
        sqlite_engine,
        generation=raw_generation,
        provider="fake",
        model="semantic",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    ) is None

    with session_scope(sqlite_engine) as session:
        new_episode = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="new-episode-after-raw-activation",
            start_at=datetime(2026, 7, 23, 9, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 23, 9, 0, tzinfo=UTC),
            content="new legacy coverage must not block emergency rollback",
            metadata_json={},
            content_hash="new-episode-after-raw-activation",
            source_message_ids=[message_id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        assert new_episode.id is not None

    with sqlite_engine.connect() as connection:
        legacy_state = connection.execute(
            text(
                "SELECT is_active, document_family FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": legacy_generation},
        ).one()
        raw_state = connection.execute(
            text(
                "SELECT is_active, document_family, physical_table "
                "FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": raw_generation},
        ).one()
        raw_table_exists = connection.execute(
            text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = :physical_table"
            ),
            {"physical_table": raw_state.physical_table},
        ).scalar_one_or_none()
    assert bool(legacy_state.is_active)
    assert legacy_state.document_family == ""
    assert bool(raw_state.is_active)
    assert raw_state.document_family == "raw_message_v3"
    assert raw_table_exists == 1


def test_resume_rejects_failed_raw_generation(sqlite_engine) -> None:
    generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-failed-resume",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert generation is not None
    with sqlite_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE retrieval_index_state SET status = 'failed' "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": generation},
        )

    assert resume_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        provider="fake",
        model="semantic-failed-resume",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    ) is None


def test_vector_generation_refuses_pending_disabled_and_failed_eligible_targets(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        message = _seed_group_message(
            session,
            group_id=10001,
            user_id=20001,
            platform_msg_id="vector-incomplete-source",
        )
        document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id="incomplete-window",
            start_at=message.timestamp,
            end_at=message.timestamp,
            content="eligible vector target",
            metadata_json={},
            content_hash="eligible-vector-target",
            source_message_ids=[message.id],
            embedding_eligible=True,
            embedding_status="pending",
        )
        document_id = int(document.id)

    generation = ensure_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="semantic-incomplete",
        dimensions=2,
        version="v1",
    )
    assert generation is not None
    pending = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        mark_ready=True,
    )
    assert pending.status == "building"
    assert not activate_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        expected_active_generation=None,
    )

    assert mark_retrieval_vector_embeddings_failed(
        sqlite_engine,
        generation=generation,
        group_id=10001,
        document_ids=[document_id],
        error_code="provider_failure",
    ) == 1
    failed = refresh_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        mark_ready=True,
    )
    assert failed.status == "failed"


def test_vector_generation_repository_rejects_unvalidated_or_mutated_identity(
    sqlite_engine,
) -> None:
    with session_scope(sqlite_engine) as session:
        states = RetrievalIndexStateRepository(session)
        with pytest.raises(ValueError):
            states.upsert_generation(
                channel="vector",
                generation=1,
                physical_table="retrieval_documents; DROP TABLE messages",
                provider="fake",
                model="semantic",
                dimensions=3,
                version="v1",
                status="building",
                total_documents=0,
                indexed_documents=0,
            )
        persisted = states.upsert_generation(
            channel="vector",
            generation=1,
            physical_table="retrieval_documents_vec_g1",
            provider="fake",
            model="semantic",
            dimensions=3,
            version="v1",
            status="building",
            total_documents=0,
            indexed_documents=0,
        )
        assert persisted.generation == 1
        with pytest.raises(ValueError):
            states.upsert_generation(
                channel="vector",
                generation=1,
                physical_table="retrieval_documents_vec_g1",
                provider="fake",
                model="semantic-v2",
                dimensions=4,
                version="v2",
                status="building",
                total_documents=0,
                indexed_documents=0,
            )


def test_missing_sqlite_vec_disables_only_vector_generation(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'without-vector.db'}", future=True)
    create_all(engine)
    assert (
        ensure_retrieval_vector_generation(
            engine,
            provider="fake",
            model="semantic",
            dimensions=3,
            version="v1",
        )
        is None
    )
    with engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM sqlite_master "
                    "WHERE name = 'retrieval_documents_fts'"
                )
            ).scalar_one()
            == 1
        )
