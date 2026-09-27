from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from app.core.persona_style_retrieval import DOCUMENT_SCHEMA, build_style_document
from scripts import replay_persona_style_retrieval as replay_module


class _FakeProvider:
    available = True
    identity = SimpleNamespace(
        provider="test",
        model="fake",
        version="v1",
        dimensions=2,
    )

    def embed_query(self, _text):
        return [1.0, 0.0]


def _create_database(path: Path) -> None:
    now = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    entry = {
        "msg_id": "style-1",
        "user_id": 22,
        "group_id": 33,
        "text": "等会就来",
        "context_before": [],
        "context_after": [{"text": "不能进入文档"}],
        "reply_target": "群友: 今晚打游戏吗",
        "timestamp": now - timedelta(days=2),
    }
    document = build_style_document(entry, expected_user_id=22, expected_group_id=33)
    assert document is not None
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE messages (
              id INTEGER PRIMARY KEY, platform_msg_id TEXT, group_id INTEGER,
              user_id INTEGER, plain_text TEXT, reply_to_msg_id TEXT, timestamp TEXT
            );
            CREATE TABLE persona_style_examples (
              msg_id TEXT PRIMARY KEY, user_id INTEGER, group_id INTEGER, text TEXT,
              context_before TEXT, context_after TEXT, reply_target TEXT, timestamp TEXT
            );
            CREATE TABLE persona_example_vectors (
              msg_id TEXT PRIMARY KEY, user_id INTEGER, group_id INTEGER,
              provider TEXT, model TEXT, embedding_version TEXT, dimensions INTEGER,
              document_schema TEXT, document_hash TEXT, vector_json TEXT
            );
            """
        )
        db.execute(
            "INSERT INTO messages VALUES (1, 'event-1', 33, 44, ?, NULL, ?)",
            ("今晚打游戏吗", now.isoformat()),
        )
        db.execute(
            "INSERT INTO persona_style_examples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["msg_id"], entry["user_id"], entry["group_id"], entry["text"],
                json.dumps(entry["context_before"], ensure_ascii=False),
                json.dumps(entry["context_after"], ensure_ascii=False),
                entry["reply_target"], entry["timestamp"].isoformat(),
            ),
        )
        db.execute(
            "INSERT INTO persona_example_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["msg_id"], 22, 33, "test", "fake", "v1", 2,
                DOCUMENT_SCHEMA, document.document_hash, json.dumps([1.0, 0.0]),
            ),
        )


def test_read_only_replay_uses_compatible_vectors_and_hides_text(tmp_path, monkeypatch) -> None:
    database = tmp_path / "bot.db"
    _create_database(database)
    settings = SimpleNamespace(
        sqlite_path=database,
        bot_qq=99,
        memory_embedding_provider="test",
        memory_embedding_device="cpu",
        memory_embedding_model="fake",
        memory_embedding_dimensions=2,
        memory_embedding_cache_dir=tmp_path / "models",
        memory_embedding_local_files_only=True,
        memory_embedding_version="v1",
        memory_embedding_base_url="",
        memory_embedding_api_key="",
        memory_embedding_timeout_seconds=1,
    )
    monkeypatch.setattr(replay_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        replay_module,
        "load_runtime_config",
        lambda _settings: SimpleNamespace(
            personas={"member": {"source_user_id": 22, "source_group_id": 33}}
        ),
    )
    monkeypatch.setattr(replay_module, "build_embedding_provider", lambda **_kwargs: _FakeProvider())
    args = SimpleNamespace(
        message_id=1,
        platform_msg_id=None,
        persona_key="member",
        db=database,
        show_text=False,
    )

    result = replay_module.replay(args)

    assert result["query_source_count"] == 1
    assert result["compatible_vectors"] == 1
    assert result["selected_count"] == 1
    assert result["selected"][0]["msg_id"] == "style-1"
    assert "reply" not in result["selected"][0]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM persona_example_vectors").fetchone()[0] == 1


def test_read_only_replay_falls_back_when_query_embedding_raises(tmp_path, monkeypatch) -> None:
    class _FailingQueryProvider(_FakeProvider):
        def embed_query(self, _text):
            raise RuntimeError("temporary failure")

    database = tmp_path / "bot.db"
    _create_database(database)
    settings = SimpleNamespace(
        sqlite_path=database,
        bot_qq=99,
        memory_embedding_provider="test",
        memory_embedding_device="cpu",
        memory_embedding_model="fake",
        memory_embedding_dimensions=2,
        memory_embedding_cache_dir=tmp_path / "models",
        memory_embedding_local_files_only=True,
        memory_embedding_version="v1",
        memory_embedding_base_url="",
        memory_embedding_api_key="",
        memory_embedding_timeout_seconds=1,
    )
    monkeypatch.setattr(replay_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        replay_module,
        "load_runtime_config",
        lambda _settings: SimpleNamespace(
            personas={"member": {"source_user_id": 22, "source_group_id": 33}}
        ),
    )
    monkeypatch.setattr(
        replay_module,
        "build_embedding_provider",
        lambda **_kwargs: _FailingQueryProvider(),
    )
    args = SimpleNamespace(
        message_id=1,
        platform_msg_id=None,
        persona_key="member",
        db=database,
        show_text=False,
    )

    result = replay_module.replay(args)

    assert result["fallback"] == "lexical"
    assert result["selected_count"] == 1
