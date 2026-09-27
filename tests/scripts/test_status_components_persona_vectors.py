from __future__ import annotations

import json
from types import SimpleNamespace
import sqlite3

from scripts.status_components import _check_persona_state


def _database() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE persona_style_sync_state (
          group_id INTEGER, user_id INTEGER, last_refresh_at TEXT, new_since_refresh INTEGER
        );
        CREATE TABLE persona_style_examples (
          msg_id TEXT PRIMARY KEY, user_id INTEGER, group_id INTEGER
        );
        CREATE TABLE persona_example_vectors (
          msg_id TEXT PRIMARY KEY, user_id INTEGER, group_id INTEGER,
          provider TEXT, model TEXT, embedding_version TEXT, dimensions INTEGER,
          document_schema TEXT, document_hash TEXT
        );
        INSERT INTO persona_style_examples VALUES ('sample-1', 22, 33);
        INSERT INTO persona_example_vectors VALUES (
          'sample-1', 22, 33, 'local', 'fake', 'v1', 2,
          'style-situation-v2', 'hash'
        );
        """
    )
    return db


def test_persona_vector_status_accepts_compatible_ready_marker(tmp_path) -> None:
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "persona.embedding.ready.json").write_text(
        json.dumps(
            {
                "state": "ready",
                "provider": "local",
                "model": "fake",
                "embedding_version": "v1",
                "dimensions": 2,
                "document_schema": "style-situation-v2",
                "coverage": [
                    {"quality_ready": 1, "vectors_ready": 1, "missing": 0, "failed": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        data_dir=data_dir,
        log_dir=log_dir,
        memory_embedding_provider="local",
        memory_embedding_model="fake",
        memory_embedding_version="v1",
        memory_embedding_dimensions=2,
    )

    with _database() as db:
        assert _check_persona_state(db, settings) is True


def test_persona_vector_status_rejects_metadata_mismatch(tmp_path) -> None:
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True)
    settings = SimpleNamespace(
        data_dir=data_dir,
        log_dir=log_dir,
        memory_embedding_provider="local",
        memory_embedding_model="changed-model",
        memory_embedding_version="v1",
        memory_embedding_dimensions=2,
    )

    with _database() as db:
        assert _check_persona_state(db, settings) is False


def test_persona_vector_status_rejects_missing_marker_when_samples_exist(tmp_path) -> None:
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True)
    settings = SimpleNamespace(
        data_dir=data_dir,
        log_dir=log_dir,
        memory_embedding_provider="local",
        memory_embedding_model="fake",
        memory_embedding_version="v1",
        memory_embedding_dimensions=2,
    )

    with _database() as db:
        assert _check_persona_state(db, settings) is False
