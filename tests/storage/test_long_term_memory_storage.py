from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import GroupRepository, MemoryRepository, MessageRepository, SummaryRepository, UserRepository


def test_current_memories_are_isolated_and_respect_validity_window(sqlite_engine) -> None:
    now = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        memories.upsert_memory(
            scope_type="group",
            scope_id="group-a",
            subject_type="user",
            subject_id="alice",
            memory_kind="plan",
            content="Alice will visit Shanghai.",
            importance=5,
            confidence=0.9,
            source_msg_id="a-1",
            valid_from=now - timedelta(hours=1),
        )
        memories.upsert_memory(
            scope_type="group",
            scope_id="group-a",
            subject_type="user",
            subject_id="alice",
            memory_kind="plan",
            content="Alice's cancelled plan.",
            importance=5,
            confidence=0.9,
            source_msg_id="a-2",
            valid_until=now,
        )
        memories.upsert_memory(
            scope_type="group",
            scope_id="group-b",
            subject_type="user",
            subject_id="alice",
            memory_kind="plan",
            content="Other group memory.",
            importance=9,
            confidence=1.0,
            source_msg_id="b-1",
        )

        current = memories.list_current_group_memories(scope_id="group-a", limit=10, as_of=now)

    assert [memory.content for memory in current] == ["Alice will visit Shanghai."]


def test_upsert_memory_is_idempotent_and_keeps_supersession_auditable(sqlite_engine) -> None:
    observed_at = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        old = memories.upsert_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="alice",
            memory_kind="preference",
            content="Alice likes hotpot.",
            importance=3,
            confidence=0.7,
            source_msg_id="m-1",
            valid_from=observed_at - timedelta(days=1),
        )
        same = memories.upsert_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="alice",
            memory_kind="preference",
            content="Alice likes hotpot.",
            importance=5,
            confidence=0.95,
            source_msg_id="m-1",
            valid_from=observed_at - timedelta(days=1),
        )
        replacement = memories.upsert_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="alice",
            memory_kind="preference",
            content="Alice prefers noodles now.",
            importance=5,
            confidence=0.9,
            source_msg_id="m-2",
            valid_from=observed_at,
            supersedes_id=old.id,
        )

        active = memories.list_current_group_memories(scope_id="10001", limit=10, as_of=observed_at)

    assert same.id == old.id
    assert old.importance == 5
    assert old.confidence == 0.95
    assert old.status == "superseded"
    assert old.superseded_by_id == replacement.id
    assert [memory.id for memory in active] == [replacement.id]


def test_upsert_summary_tracks_recursive_sources_without_duplicate_rows(sqlite_engine) -> None:
    start = datetime(2026, 7, 15, tzinfo=UTC)
    end = datetime(2026, 7, 16, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        summaries = SummaryRepository(session)
        first = summaries.upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="daily",
            summary_key="2026-07-15:music",
            start_at=start,
            end_at=end,
            content="They discussed music.",
            source_count=25,
            source_start_msg_id="m-1",
            source_end_msg_id="m-25",
            source_summary_ids=[3, 4],
        )
        second = summaries.upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="daily",
            summary_key="2026-07-15:music",
            start_at=start,
            end_at=end,
            content="They decided to share a playlist.",
            source_count=26,
            source_start_msg_id="m-1",
            source_end_msg_id="m-26",
            source_summary_ids=[3, 4, 5],
        )
        found = summaries.list_group_summaries(scope_id="10001", limit=10, summary_levels=["daily"])

    assert first.id == second.id
    assert [(item.summary_key, item.content, item.source_end_msg_id, item.source_summary_ids) for item in found] == [
        ("2026-07-15:music", "They decided to share a playlist.", "m-26", [3, 4, 5])
    ]


def test_create_all_migrates_legacy_memory_and_summary_tables_without_rebuild(tmp_path) -> None:
    engine = build_engine(tmp_path / "legacy-memory.db")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE summaries ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, scope_type VARCHAR(32) NOT NULL, "
                "scope_id VARCHAR(64) NOT NULL, summary_level VARCHAR(32) NOT NULL, "
                "start_at DATETIME NOT NULL, end_at DATETIME NOT NULL, content TEXT NOT NULL, "
                "source_count INTEGER NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE memory_items ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, scope_type VARCHAR(32) NOT NULL, "
                "scope_id VARCHAR(64) NOT NULL, subject_type VARCHAR(32) NOT NULL, "
                "subject_id VARCHAR(64) NOT NULL, memory_kind VARCHAR(32) NOT NULL, "
                "content TEXT NOT NULL, importance INTEGER NOT NULL, confidence FLOAT NOT NULL, "
                "source_msg_id VARCHAR(128) NOT NULL, expires_at DATETIME NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO memory_items "
                "(scope_type, scope_id, subject_type, subject_id, memory_kind, content, importance, confidence, source_msg_id, expires_at) "
                "VALUES ('group', '10001', 'user', 'alice', 'plan', 'legacy', 3, 0.8, 'm-1', '2026-07-20 00:00:00')"
            )
        )

    create_all(engine)

    with engine.connect() as connection:
        memory_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(memory_items)"))}
        summary_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(summaries)"))}
        migrated = connection.execute(
            text("SELECT valid_until, status FROM memory_items WHERE source_msg_id = 'm-1'")
        ).one()

    assert {"valid_from", "valid_until", "status", "supersedes_id", "superseded_by_id"} <= memory_columns
    assert {"summary_key", "source_start_msg_id", "source_end_msg_id", "source_summary_ids", "status"} <= summary_columns
    assert migrated.status == "active"
    assert migrated.valid_until is not None


def test_memory_fts_is_an_optional_group_scoped_accelerator(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        memory = memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="alice",
            memory_kind="preference",
            content="Alice likes 火锅.",
            importance=4,
            confidence=0.8,
            source_msg_id="m-1",
        )
        expired = memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="bob",
            memory_kind="preference",
            content="Bob likes 火锅.",
            importance=5,
            confidence=0.9,
            source_msg_id="m-2",
            valid_until=datetime(2026, 1, 1, tzinfo=UTC),
        )
        fts_available = session.execute(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_items_fts'")
        ).scalar_one_or_none()
        found = memories.search_group_memories_fts(scope_id="10001", query="你还喜欢火锅吗", limit=10)

    if fts_available:
        assert [item.id for item in found] == [memory.id]
    else:
        assert found == []


def test_schema_migration_is_idempotent_for_an_already_upgraded_database(sqlite_engine) -> None:
    create_all(sqlite_engine)
    create_all(sqlite_engine)

    with sqlite_engine.connect() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(memory_items)"))}

    assert {"valid_from", "valid_until", "status", "supersedes_id", "superseded_by_id"} <= columns


def test_memory_search_recalls_old_chinese_memory_even_when_high_importance_noise_fills_current_limit(sqlite_engine) -> None:
    now = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        target = memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="alice",
            memory_kind="plan",
            content="Alice: \u6211\u4e0b\u5468\u51c6\u5907\u53bb\u4e0a\u6d77\u3002",
            importance=1,
            confidence=0.8,
            source_msg_id="plan-old",
            valid_from=now,
        )
        for index in range(70):
            memories.add_memory(
                scope_type="group",
                scope_id="10001",
                subject_type="user",
                subject_id=f"noise-{index}",
                memory_kind="fact",
                content=f"\u65e0\u5173\u8bb0\u5fc6 {index}",
                importance=5,
                confidence=0.9,
                source_msg_id=f"noise-{index}",
                valid_from=now,
            )
        found = memories.search_group_memories_fts(
            scope_id="10001",
            query="\u4e4b\u524d\u51c6\u5907\u53bb\u4e0a\u6d77\u7684\u8ba1\u5212\u662f\u4ec0\u4e48",
            limit=8,
        )

    assert target.id in [memory.id for memory in found]


def test_optional_vector_index_returns_fuzzy_memory_candidates(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        hotpot = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            canonical_key="preference|42|likes|spicy hotpot",
            predicate="likes",
            object_text="spicy hotpot",
            content="Alice likes spicy hotpot for dinner.",
            importance=4,
            confidence=0.9,
            source_msg_ids=["m-hotpot"],
        )
        memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="43",
            memory_kind="preference",
            canonical_key="preference|43|likes|science fiction",
            predicate="likes",
            object_text="science fiction",
            content="Bob likes science fiction novels.",
            importance=4,
            confidence=0.9,
            source_msg_ids=["m-books"],
        )
        found = memories.search_group_memories_vector(
            scope_id="10001",
            query="Who enjoys hotpot at dinner?",
            limit=2,
        )

    assert found
    assert found[0].id == hotpot.id


def test_canonical_memory_merges_sources_and_removes_superseded_search_entries(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        first = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            canonical_key="preference|42|likes|hotpot",
            predicate="likes",
            object_text="hotpot",
            content="Alice likes hotpot.",
            importance=3,
            confidence=0.7,
            source_msg_ids=["m-1"],
        )
        merged = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            canonical_key="preference|42|likes|hotpot",
            predicate="likes",
            object_text="hotpot",
            content="Alice repeatedly says she likes hotpot.",
            importance=5,
            confidence=0.95,
            source_msg_ids=["m-2", "m-1"],
        )
        memories.mark_superseded(memory_id=merged.id)
        fts_found = memories.search_group_memories_fts(scope_id="10001", query="hotpot", limit=5)
        vector_found = memories.search_group_memories_vector(scope_id="10001", query="hotpot", limit=5)

    assert first.id == merged.id
    assert merged.source_msg_ids == ["m-1", "m-2"]
    assert merged.mention_count == 2
    assert merged.importance == 5
    assert merged.confidence == 0.95
    assert fts_found == []
    assert vector_found == []


def test_canonical_upsert_upgrades_same_source_legacy_memory(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        legacy = memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="Alice likes hotpot.",
            importance=4,
            confidence=0.8,
            source_msg_id="m-legacy",
        )
        duplicate = memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="Alice repeats that she likes hotpot.",
            importance=4,
            confidence=0.8,
            source_msg_id="m-legacy-2",
        )
        old_batch_duplicate = memories.add_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="preference", content="Alice has always liked hotpot.", importance=4,
            confidence=0.8, source_msg_id="m-old-batch",
        )
        upgraded = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            canonical_key="preference|42|likes|hotpot",
            predicate="likes",
            object_text="hotpot",
            content="Alice likes hotpot.",
            importance=4,
            confidence=0.9,
            source_msg_ids=["m-legacy", "m-legacy-2"],
        )
        later_legacy = memories.add_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="preference", content="Alice still likes hotpot.", importance=4,
            confidence=0.8, source_msg_id="m-legacy-3",
        )
        upgraded_again = memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="preference", canonical_key="preference|42|likes|hotpot",
            predicate="likes", object_text="hotpot", content="Alice likes hotpot.",
            importance=4, confidence=0.9, source_msg_ids=["m-legacy-3"],
        )

    assert upgraded.id == legacy.id
    assert upgraded_again.id == legacy.id
    assert upgraded.canonical_key == "preference|42|likes|hotpot"
    assert duplicate.status == "superseded"
    assert old_batch_duplicate.status == "superseded"
    assert later_legacy.status == "superseded"


def test_supersession_requires_matching_object_and_specific_content(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="plan", canonical_key="plan|42|plan|shanghai", predicate="plan",
            object_text="shanghai", content="Alice plans Shanghai.", importance=3, confidence=0.8,
            source_msg_ids=["m-shanghai"],
        )
        memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="plan", canonical_key="plan|42|plan|car", predicate="plan",
            object_text="car", content="Alice plans to buy a car.", importance=3, confidence=0.8,
            source_msg_ids=["m-car"],
        )
        hotpot = memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="preference", canonical_key="preference|42|likes|hotpot", predicate="likes",
            object_text="hotpot", content="Alice likes hotpot.", importance=3, confidence=0.8,
            source_msg_ids=["m-hotpot"],
        )
        skiing = memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="42",
            memory_kind="preference", canonical_key="preference|42|likes|skiing", predicate="likes",
            object_text="skiing", content="Alice likes skiing.", importance=3, confidence=0.8,
            source_msg_ids=["m-skiing"],
        )
        assert memories.find_current_memory_for_supersession(
            scope_id="10001", subject_type="user", subject_id="42", memory_kind="plan",
            replacement_content="Alice cancels the car plan.",
        ).object_text == "car"
        assert memories.find_current_memory_for_supersession(
            scope_id="10001", subject_type="user", subject_id="42", memory_kind="plan",
            replacement_content="Alice: I am not planning anymore.",
        ) is None
        assert memories.find_current_memory_for_supersession(
            scope_id="10001", subject_type="user", subject_id="42", memory_kind="plan",
            replacement_content="Alice: 我不打算了。",
        ) is None
        shanghai_trip = memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="43",
            memory_kind="plan", canonical_key="plan|43|plan|shanghai-trip", predicate="plan",
            object_text="Shanghai trip", content="Bob plans a trip to Shanghai.", importance=3,
            confidence=0.8, source_msg_ids=["m-trip-shanghai"],
        )
        tokyo_trip = memories.upsert_canonical_memory(
            scope_type="group", scope_id="10001", subject_type="user", subject_id="43",
            memory_kind="plan", canonical_key="plan|43|plan|tokyo-trip", predicate="plan",
            object_text="Tokyo trip", content="Bob plans a trip to Tokyo.", importance=3,
            confidence=0.8, source_msg_ids=["m-trip-tokyo"],
        )
        assert memories.find_current_memory_for_supersession(
            scope_id="10001", subject_type="user", subject_id="43", memory_kind="plan",
            replacement_content="Bob: cancel trip.",
        ) is None
        assert shanghai_trip.status == "active"
        assert tokyo_trip.status == "active"
        assert memories.supersede_current_memories(
            scope_id="10001", subject_id="42", predicate="likes", object_text="skiing", valid_until=datetime.now(UTC)
        ) == 1
        assert hotpot.status == "active"
        assert skiing.status == "superseded"


def test_backfill_windows_are_stable_and_inbound_count_excludes_bot(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=10001, group_name="test", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=42, nickname="Alice", group_card="Alice")
        UserRepository(session).upsert_user(user_id=99, nickname="Bot", group_card="Bot")
        messages = MessageRepository(session)
        for index in range(10):
            messages.add_group_message(
                platform_msg_id=f"user-{index}", group_id=10001, user_id=42,
                timestamp=datetime(2026, 7, 16, 1, index, tzinfo=UTC), plain_text=f"message {index}",
                raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
            )
            messages.add_group_message(
                platform_msg_id=f"bot-{index}", group_id=10001, user_id=99,
                timestamp=datetime(2026, 7, 16, 1, index, 30, tzinfo=UTC), plain_text=f"reply {index}",
                raw_json={"delivery_state": "delivered"}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
            )
        first = messages.list_recent_group_message_windows(
            group_id=10001, batch_size=10, limit_windows=2, excluded_user_ids={99}
        )
        messages.add_group_message(
            platform_msg_id="user-10", group_id=10001, user_id=42,
            timestamp=datetime(2026, 7, 16, 2, 0, tzinfo=UTC), plain_text="message 10",
            raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="reserved-other", group_id=10001, user_id=42,
            timestamp=datetime(2026, 7, 16, 2, 1, tzinfo=UTC), plain_text="reserved",
            raw_json={"delivery_state": "reserved"}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
        second = messages.list_recent_group_message_windows(
            group_id=10001, batch_size=10, limit_windows=2, excluded_user_ids={99}
        )
        recent_inbound = messages.list_recent_group_inbound_messages(
            group_id=10001, bot_user_id=99, limit=10
        )
        inbound_count = messages.count_group_inbound_messages(group_id=10001, bot_user_id=99)

    assert [row.platform_msg_id for row in first[0]] == [f"user-{index}" for index in range(10)]
    assert [row.platform_msg_id for row in second[0]] == [f"user-{index}" for index in range(10)]
    assert len(second) == 1
    assert [row.platform_msg_id for row in recent_inbound] == [f"user-{index}" for index in range(1, 11)]
    assert inbound_count == 11


def test_vector_search_filters_group_before_top_k(sqlite_engine) -> None:
    with session_scope(sqlite_engine) as session:
        memories = MemoryRepository(session)
        target = memories.upsert_canonical_memory(
            scope_type="group", scope_id="target", subject_type="user", subject_id="42",
            memory_kind="preference", canonical_key="preference|42|likes|hotpot",
            predicate="likes", object_text="hotpot", content="Alice likes hotpot.",
            importance=4, confidence=0.9, source_msg_ids=["target-message"],
        )
        for index in range(20):
            memories.upsert_canonical_memory(
                scope_type="group", scope_id="other", subject_type="user", subject_id=str(index),
                memory_kind="preference", canonical_key=f"preference|{index}|likes|hotpot",
                predicate="likes", object_text="hotpot", content="Alice likes hotpot.",
                importance=4, confidence=0.9, source_msg_ids=[f"other-{index}"],
            )
        found = memories.search_group_memories_vector(scope_id="target", query="hotpot", limit=1)

    assert [memory.id for memory in found] == [target.id]
