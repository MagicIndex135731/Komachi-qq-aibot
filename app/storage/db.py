from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import Base
from app.providers.embeddings import hashed_text_embedding


def build_engine(sqlite_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{sqlite_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()
        try:
            import sqlite_vec

            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)
        except (AttributeError, ImportError, OSError, ValueError, sqlite3.Error):
            try:
                dbapi_connection.enable_load_extension(False)
            except AttributeError:
                pass

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    return engine


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _run_schema_migrations(engine)
    _initialize_optional_memory_fts(engine)
    _initialize_optional_memory_vectors(engine)


def _run_schema_migrations(engine: Engine) -> None:
    with engine.begin() as connection:
        table_names = {
            str(row[0])
            for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }
        if "dev_sessions" in table_names:
            _add_missing_columns(
                connection,
                "dev_sessions",
                {"session_mode": "VARCHAR(32) NOT NULL DEFAULT 'project'"},
            )
            connection.execute(
                text(
                    "UPDATE dev_sessions "
                    "SET session_mode = 'project' "
                    "WHERE session_mode IS NULL OR session_mode = ''"
                )
            )

        if "summaries" in table_names:
            _add_missing_columns(
                connection,
                "summaries",
                {
                    "summary_key": "VARCHAR(255) NOT NULL DEFAULT ''",
                    "source_start_msg_id": "VARCHAR(128) NULL",
                    "source_end_msg_id": "VARCHAR(128) NULL",
                    "source_summary_ids": "JSON NOT NULL DEFAULT '[]'",
                    "status": "VARCHAR(32) NOT NULL DEFAULT 'active'",
                },
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_summaries_scope_level_key ON summaries (scope_type, scope_id, summary_level, summary_key)")
            )

        if "memory_items" in table_names:
            _add_missing_columns(
                connection,
                "memory_items",
                {
                    "canonical_key": "VARCHAR(255) NOT NULL DEFAULT ''",
                    "predicate": "VARCHAR(96) NOT NULL DEFAULT ''",
                    "object_text": "TEXT NOT NULL DEFAULT ''",
                    "source_msg_ids": "JSON NOT NULL DEFAULT '[]'",
                    "mention_count": "INTEGER NOT NULL DEFAULT 1",
                    "last_seen_at": "DATETIME NULL",
                    "valid_from": "DATETIME NULL",
                    "valid_until": "DATETIME NULL",
                    "status": "VARCHAR(32) NOT NULL DEFAULT 'active'",
                    "supersedes_id": "INTEGER NULL",
                    "superseded_by_id": "INTEGER NULL",
                },
            )
            connection.execute(
                text("UPDATE memory_items SET valid_until = expires_at WHERE valid_until IS NULL AND expires_at IS NOT NULL")
            )
            connection.execute(
                text("UPDATE memory_items SET status = 'active' WHERE status IS NULL OR status = ''")
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_memory_items_scope_status ON memory_items (scope_type, scope_id, status)")
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_memory_items_subject_status ON memory_items (scope_type, scope_id, subject_id, status)")
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_memory_items_canonical_key ON memory_items (scope_type, scope_id, canonical_key)")
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_items_active_canonical "
                    "ON memory_items (scope_type, scope_id, canonical_key) "
                    "WHERE canonical_key <> '' AND status = 'active'"
                )
            )

        if "jobs" in table_names:
            _add_missing_columns(
                connection,
                "jobs",
                {"job_key": "VARCHAR(255) NOT NULL DEFAULT ''"},
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS ix_jobs_job_key ON jobs (job_type, job_key)")
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_type_key "
                    "ON jobs (job_type, job_key) WHERE job_key <> ''"
                )
            )


def _table_columns(connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(text(f"PRAGMA table_info({table_name})"))}


def _add_missing_columns(connection, table_name: str, definitions: dict[str, str]) -> None:
    existing_columns = _table_columns(connection, table_name)
    for column_name, definition in definitions.items():
        if column_name not in existing_columns:
            try:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"))
            except SQLAlchemyError as exc:
                # Main, group, private, and worker processes may all start at
                # once. If another process won this exact ALTER race, the
                # schema is already in the requested state and migration can
                # safely continue; every other database error still aborts.
                if "duplicate column name" not in str(exc).lower():
                    raise


def _initialize_optional_memory_fts(engine: Engine) -> bool:
    """Create a best-effort FTS5 side index; SQLite builds without FTS5 still work."""
    try:
        with engine.begin() as connection:
            table_names = {
                str(row[0])
                for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            }
            if "memory_items" not in table_names:
                return False
            existing_fts_sql = connection.execute(
                text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_items_fts'")
            ).scalar_one_or_none()
            if existing_fts_sql is not None and "tokenize='trigram'" not in str(existing_fts_sql).lower():
                # The FTS table is a rebuildable accelerator, not the source of
                # truth. Rebuild old unicode61 indexes so Chinese substrings
                # can be found after an upgrade.
                connection.execute(text("DROP TABLE memory_items_fts"))
            connection.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts "
                    "USING fts5(content, scope_type UNINDEXED, scope_id UNINDEXED, memory_id UNINDEXED, tokenize='trigram')"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM memory_items_fts WHERE memory_id IN ("
                    "SELECT CAST(id AS TEXT) FROM memory_items WHERE status <> 'active'"
                    ")"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO memory_items_fts (content, scope_type, scope_id, memory_id) "
                    "SELECT content, scope_type, scope_id, CAST(id AS TEXT) FROM memory_items "
                    "WHERE status = 'active' AND NOT EXISTS ("
                    "SELECT 1 FROM memory_items_fts WHERE memory_id = CAST(memory_items.id AS TEXT)"
                    ")"
                )
            )
        return True
    except SQLAlchemyError:
        return False


def _initialize_optional_memory_vectors(engine: Engine, *, dimensions: int = 256) -> bool:
    """Create a rebuildable sqlite-vec side index when the extension is available."""
    try:
        with engine.begin() as connection:
            connection.execute(text("SELECT vec_version()"))
            connection.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_vec "
                    f"USING vec0(memory_id INTEGER PRIMARY KEY, embedding float[{int(dimensions)}])"
                )
            )
            rows = connection.execute(
                text(
                    "SELECT id, content FROM memory_items WHERE status = 'active' AND id NOT IN "
                    "(SELECT memory_id FROM memory_items_vec)"
                )
            )
            for memory_id, content in rows:
                connection.execute(
                    text("INSERT INTO memory_items_vec(memory_id, embedding) VALUES (:memory_id, :embedding)"),
                    {
                        "memory_id": int(memory_id),
                        "embedding": json.dumps(hashed_text_embedding(str(content or ""), dimensions=dimensions)),
                    },
                )
        return True
    except (SQLAlchemyError, ValueError):
        return False


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
