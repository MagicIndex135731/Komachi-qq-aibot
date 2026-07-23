from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Sequence

from sqlalchemy import bindparam, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import Base
from app.providers.embeddings import hashed_text_embedding


_RETRIEVAL_VECTOR_TABLE_RE = re.compile(r"retrieval_documents_vec_g([1-9][0-9]*)\Z")


@dataclass(frozen=True, slots=True)
class RetrievalVectorCoverage:
    generation: int
    status: str
    total_documents: int
    indexed_documents: int
    failed_documents: int

    @property
    def coverage(self) -> float:
        if self.total_documents == 0:
            return 1.0
        return self.indexed_documents / self.total_documents


def retrieval_vector_table_name(generation: int) -> str:
    resolved_generation = int(generation)
    if resolved_generation <= 0:
        raise ValueError("retrieval vector generation must be positive")
    return f"retrieval_documents_vec_g{resolved_generation}"


def validate_retrieval_vector_table_name(
    physical_table: str,
    *,
    generation: int | None = None,
) -> str:
    match = _RETRIEVAL_VECTOR_TABLE_RE.fullmatch(str(physical_table))
    if match is None:
        raise ValueError("invalid retrieval vector physical table")
    parsed_generation = int(match.group(1))
    if generation is not None and parsed_generation != int(generation):
        raise ValueError("retrieval vector table does not match generation")
    return str(physical_table)


def build_engine(sqlite_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{sqlite_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA busy_timeout=5000;")
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
    # Serialize both SQLAlchemy's check-then-create pass and compatibility
    # ALTERs. Protecting only the ALTER phase leaves a fresh-database race
    # where two processes both observe a missing table before either creates
    # it.
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            Base.metadata.create_all(connection)
            _apply_schema_migrations(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    _initialize_optional_memory_fts(engine)
    _initialize_optional_retrieval_fts(engine)
    _initialize_optional_memory_vectors(engine)


def _run_schema_migrations(engine: Engine) -> None:
    with engine.connect() as connection:
        # Startup can happen in the group/private/worker processes at nearly
        # the same time. A finite busy_timeout plus BEGIN IMMEDIATE serializes
        # additive schema inspection and ALTER statements without hiding
        # unrelated migration errors.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            _apply_schema_migrations(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _apply_schema_migrations(connection) -> None:
    table_names = {
        str(row[0])
        for row in connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
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
            text(
                "CREATE INDEX IF NOT EXISTS ix_summaries_scope_level_key ON summaries (scope_type, scope_id, summary_level, summary_key)"
            )
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
            text(
                "UPDATE memory_items SET valid_until = expires_at WHERE valid_until IS NULL AND expires_at IS NOT NULL"
            )
        )
        connection.execute(
            text(
                "UPDATE memory_items SET status = 'active' WHERE status IS NULL OR status = ''"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_memory_items_scope_status ON memory_items (scope_type, scope_id, status)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_memory_items_subject_status ON memory_items (scope_type, scope_id, subject_id, status)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_memory_items_canonical_key ON memory_items (scope_type, scope_id, canonical_key)"
            )
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
            {
                "job_key": "VARCHAR(255) NOT NULL DEFAULT ''",
                "requested_generation": "INTEGER NOT NULL DEFAULT 0",
                "processed_generation": "INTEGER NOT NULL DEFAULT 0",
                "claimed_generation": "INTEGER NOT NULL DEFAULT 0",
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "max_attempts": "INTEGER NOT NULL DEFAULT 3",
                "locked_by": "VARCHAR(128) NULL",
                "locked_at": "DATETIME NULL",
                "lease_until": "DATETIME NULL",
                "completed_at": "DATETIME NULL",
                "last_error_code": "VARCHAR(96) NOT NULL DEFAULT ''",
                "backfill_run_id": "INTEGER NULL",
                "target_generation": "VARCHAR(128) NOT NULL DEFAULT ''",
            },
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_jobs_job_key ON jobs (job_type, job_key)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_type_key "
                "ON jobs (job_type, job_key) WHERE job_key <> ''"
            )
        )

    if "messages" in table_names:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_id_group "
                "ON messages (id, group_id)"
            )
        )

    if "conversation_episodes" in table_names:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_conversation_episodes_id_group "
                "ON conversation_episodes (id, group_id)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_conversation_episodes_open_group "
                "ON conversation_episodes (group_id) "
                "WHERE status = 'open' AND is_current = 1"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_conversation_episodes_group_time "
                "ON conversation_episodes (group_id, started_at, ended_at, id)"
            )
        )

    if "retrieval_documents" in table_names:
        _add_missing_columns(
            connection,
            "retrieval_documents",
            {
                "embedding_eligible": "BOOLEAN NOT NULL DEFAULT 0",
            },
        )
        connection.execute(
            text(
                "UPDATE retrieval_documents SET embedding_eligible = 1 "
                "WHERE document_kind = 'episode' AND embedding_eligible = 0"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_retrieval_documents_id_group "
                "ON retrieval_documents (id, group_id)"
            )
        )


def _table_columns(connection, table_name: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(text(f"PRAGMA table_info({table_name})"))
    }


def _add_missing_columns(
    connection, table_name: str, definitions: dict[str, str]
) -> None:
    existing_columns = _table_columns(connection, table_name)
    for column_name, definition in definitions.items():
        if column_name not in existing_columns:
            try:
                connection.execute(
                    text(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
                    )
                )
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
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            }
            if "memory_items" not in table_names:
                return False
            existing_fts_sql = connection.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_items_fts'"
                )
            ).scalar_one_or_none()
            if (
                existing_fts_sql is not None
                and "tokenize='trigram'" not in str(existing_fts_sql).lower()
            ):
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


def _initialize_optional_retrieval_fts(engine: Engine) -> bool:
    """Create the rebuildable, group-scoped retrieval-document FTS accelerator."""
    try:
        with engine.begin() as connection:
            table_names = {
                str(row[0])
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            }
            if "retrieval_documents" not in table_names:
                return False
            existing_fts_sql = connection.execute(
                text(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'retrieval_documents_fts'"
                )
            ).scalar_one_or_none()
            if (
                existing_fts_sql is not None
                and "tokenize='trigram'" not in str(existing_fts_sql).lower()
            ):
                connection.execute(text("DROP TABLE retrieval_documents_fts"))
            connection.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_documents_fts "
                    "USING fts5(content, group_id UNINDEXED, document_id UNINDEXED, "
                    "content_hash UNINDEXED, tokenize='trigram')"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM retrieval_documents_fts WHERE document_id IN ("
                    "SELECT CAST(id AS TEXT) FROM retrieval_documents WHERE status <> 'active'"
                    ")"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO retrieval_documents_fts "
                    "(content, group_id, document_id, content_hash) "
                    "SELECT content, CAST(group_id AS TEXT), CAST(id AS TEXT), content_hash "
                    "FROM retrieval_documents "
                    "WHERE status = 'active' AND NOT EXISTS ("
                    "SELECT 1 FROM retrieval_documents_fts "
                    "WHERE document_id = CAST(retrieval_documents.id AS TEXT) "
                    "AND content_hash = retrieval_documents.content_hash"
                    ")"
                )
            )
            total_documents = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM retrieval_documents "
                        "WHERE status = 'active'"
                    )
                ).scalar_one()
                or 0
            )
            indexed_documents = int(
                connection.execute(
                    text(
                        "SELECT count(DISTINCT document_id) FROM retrieval_documents_fts"
                    )
                ).scalar_one()
                or 0
            )
            now = datetime.now(UTC).replace(tzinfo=None)
            connection.execute(
                text(
                    "INSERT INTO retrieval_index_state ("
                    "channel, generation, physical_table, provider, model, dimensions, "
                    "version, status, total_documents, indexed_documents, is_active, updated_at"
                    ") VALUES ("
                    "'fts', 1, 'retrieval_documents_fts', 'sqlite', 'trigram', NULL, "
                    "'1', 'ready', :total_documents, :indexed_documents, 0, :updated_at"
                    ") ON CONFLICT(channel, generation) DO UPDATE SET "
                    "physical_table = excluded.physical_table, provider = excluded.provider, "
                    "model = excluded.model, version = excluded.version, status = excluded.status, "
                    "total_documents = excluded.total_documents, "
                    "indexed_documents = excluded.indexed_documents, updated_at = excluded.updated_at"
                ),
                {
                    "total_documents": total_documents,
                    "indexed_documents": indexed_documents,
                    "updated_at": now,
                },
            )
            connection.execute(
                text(
                    "UPDATE retrieval_index_state "
                    "SET is_active = 1, activated_at = :updated_at "
                    "WHERE channel = 'fts' AND generation = 1 "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM retrieval_index_state AS active "
                    "WHERE active.channel = 'fts' AND active.is_active = 1"
                    ")"
                ),
                {"updated_at": now},
            )
        return True
    except SQLAlchemyError:
        _record_retrieval_fts_unavailable(engine)
        return False


def _record_retrieval_fts_unavailable(engine: Engine) -> None:
    try:
        with engine.begin() as connection:
            if (
                connection.execute(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'retrieval_index_state'"
                    )
                ).scalar_one_or_none()
                is None
            ):
                return
            connection.execute(
                text(
                    "INSERT INTO retrieval_index_state ("
                    "channel, generation, physical_table, provider, model, dimensions, "
                    "version, status, total_documents, indexed_documents, is_active, updated_at"
                    ") VALUES ("
                    "'fts', 1, 'retrieval_documents_fts', 'sqlite', 'trigram', NULL, "
                    "'1', 'unavailable', 0, 0, 0, :updated_at"
                    ") ON CONFLICT(channel, generation) DO UPDATE SET "
                    "status = 'unavailable', updated_at = excluded.updated_at"
                ),
                {"updated_at": datetime.now(UTC).replace(tzinfo=None)},
            )
    except SQLAlchemyError:
        return


def _initialize_optional_memory_vectors(
    engine: Engine, *, dimensions: int = 256
) -> bool:
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
                    text(
                        "INSERT INTO memory_items_vec(memory_id, embedding) VALUES (:memory_id, :embedding)"
                    ),
                    {
                        "memory_id": int(memory_id),
                        "embedding": json.dumps(
                            hashed_text_embedding(
                                str(content or ""), dimensions=dimensions
                            )
                        ),
                    },
                )
        return True
    except (SQLAlchemyError, ValueError):
        return False


def ensure_retrieval_vector_generation(
    engine: Engine,
    *,
    provider: str,
    model: str,
    dimensions: int,
    version: str,
) -> int | None:
    """Return a compatible vector generation, creating a building one if needed.

    The physical table is derived exclusively from the persisted integer
    generation.  Incompatible active generations remain untouched until a
    later coverage-checked CAS activation.
    """

    resolved_dimensions = int(dimensions)
    if resolved_dimensions <= 0 or resolved_dimensions > 65_536:
        raise ValueError("retrieval vector dimensions are invalid")
    identity = {
        "provider": str(provider),
        "model": str(model),
        "dimensions": resolved_dimensions,
        "version": str(version),
    }
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            connection.execute(text("SELECT vec_version()")).scalar_one()
            existing = connection.execute(
                text(
                    "SELECT generation, physical_table FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND provider = :provider "
                    "AND model = :model AND dimensions = :dimensions "
                    "AND version = :version AND status IN ('building','ready') "
                    "ORDER BY is_active DESC, generation DESC LIMIT 1"
                ),
                identity,
            ).one_or_none()
            if existing is not None:
                generation = int(existing.generation)
                validate_retrieval_vector_table_name(
                    str(existing.physical_table),
                    generation=generation,
                )
                physical_exists = connection.execute(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = :physical_table"
                    ),
                    {"physical_table": str(existing.physical_table)},
                ).scalar_one_or_none()
                if physical_exists is not None:
                    connection.commit()
                    return generation
                connection.execute(
                    text(
                        "UPDATE retrieval_index_state SET status = 'failed', "
                        "updated_at = :updated_at WHERE channel = 'vector' "
                        "AND generation = :generation AND is_active = 0"
                    ),
                    {
                        "generation": generation,
                        "updated_at": datetime.now(UTC).replace(tzinfo=None),
                    },
                )

            generation = int(
                connection.execute(
                    text(
                        "SELECT coalesce(max(generation), 0) + 1 "
                        "FROM retrieval_index_state WHERE channel = 'vector'"
                    )
                ).scalar_one()
            )
            physical_table = retrieval_vector_table_name(generation)
            connection.execute(
                text(
                    "INSERT INTO retrieval_index_state ("
                    "channel, generation, physical_table, provider, model, dimensions, "
                    "version, status, total_documents, indexed_documents, is_active, updated_at"
                    ") VALUES ("
                    "'vector', :generation, :physical_table, :provider, :model, :dimensions, "
                    ":version, 'building', 0, 0, 0, :updated_at"
                    ")"
                ),
                {
                    **identity,
                    "generation": generation,
                    "physical_table": physical_table,
                    "updated_at": datetime.now(UTC).replace(tzinfo=None),
                },
            )
            connection.execute(
                text(
                    f"CREATE VIRTUAL TABLE {physical_table} USING vec0("
                    "document_id INTEGER PRIMARY KEY, "
                    "group_id INTEGER PARTITION KEY, "
                    f"embedding float[{resolved_dimensions}])"
                )
            )
            connection.commit()
            return generation
        except (SQLAlchemyError, sqlite3.Error):
            connection.rollback()
            return None


def write_retrieval_vector_embeddings(
    engine: Engine,
    *,
    generation: int,
    rows: Sequence[tuple[int, int, Sequence[float]]],
) -> int:
    """Write a validated batch to a building generation and canonical metadata."""

    if not rows:
        return 0
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            state = connection.execute(
                text(
                    "SELECT physical_table, provider, model, dimensions, version, status "
                    "FROM retrieval_index_state WHERE channel = 'vector' "
                    "AND generation = :generation"
                ),
                {"generation": int(generation)},
            ).one_or_none()
            if state is None or str(state.status) not in {"building", "ready"}:
                raise ValueError("retrieval vector generation is not writable")
            physical_table = validate_retrieval_vector_table_name(
                str(state.physical_table),
                generation=int(generation),
            )
            dimensions = int(state.dimensions)
            normalized_rows: list[tuple[int, int, list[float]]] = []
            seen_document_ids: set[int] = set()
            for document_id, group_id, vector in rows:
                resolved_document_id = int(document_id)
                resolved_group_id = int(group_id)
                values = [float(value) for value in vector]
                if (
                    len(values) != dimensions
                    or not all(math.isfinite(value) for value in values)
                ):
                    raise ValueError("retrieval embedding dimensions or values are invalid")
                if resolved_document_id in seen_document_ids:
                    raise ValueError("retrieval embedding batch contains duplicate documents")
                seen_document_ids.add(resolved_document_id)
                canonical = connection.execute(
                    text(
                        "SELECT 1 FROM retrieval_documents "
                        "WHERE id = :document_id AND group_id = :group_id "
                        "AND status = 'active' AND embedding_eligible = 1"
                    ),
                    {
                        "document_id": resolved_document_id,
                        "group_id": resolved_group_id,
                    },
                ).scalar_one_or_none()
                if canonical is None:
                    raise ValueError("retrieval embedding provenance is not active and scoped")
                normalized_rows.append(
                    (resolved_document_id, resolved_group_id, values)
                )

            for document_id, group_id, vector in normalized_rows:
                connection.execute(
                    text(f"DELETE FROM {physical_table} WHERE document_id = :document_id"),
                    {"document_id": document_id},
                )
                connection.execute(
                    text(
                        f"INSERT INTO {physical_table} "
                        "(document_id, group_id, embedding) "
                        "VALUES (:document_id, :group_id, :embedding)"
                    ),
                    {
                        "document_id": document_id,
                        "group_id": group_id,
                        "embedding": json.dumps(vector, separators=(",", ":")),
                    },
                )
                connection.execute(
                    text(
                        "UPDATE retrieval_documents SET "
                        "embedding_provider = :provider, embedding_model = :model, "
                        "embedding_version = :version, embedding_dimensions = :dimensions, "
                        "embedding_generation = :generation, embedding_status = 'ready', "
                        "last_error_code = '', updated_at = :updated_at "
                        "WHERE id = :document_id AND group_id = :group_id "
                        "AND status = 'active' AND embedding_eligible = 1"
                    ),
                    {
                        "provider": str(state.provider),
                        "model": str(state.model),
                        "version": str(state.version),
                        "dimensions": dimensions,
                        "generation": int(generation),
                        "document_id": document_id,
                        "group_id": group_id,
                        "updated_at": datetime.now(UTC).replace(tzinfo=None),
                    },
                )
            connection.commit()
            return len(normalized_rows)
        except Exception:
            connection.rollback()
            raise


def mark_retrieval_vector_embeddings_failed(
    engine: Engine,
    *,
    generation: int,
    group_id: int,
    document_ids: Sequence[int],
    error_code: str,
) -> int:
    if not document_ids:
        return 0
    with engine.begin() as connection:
        state = connection.execute(
            text(
                "SELECT provider, model, dimensions, version "
                "FROM retrieval_index_state WHERE channel = 'vector' "
                "AND generation = :generation"
            ),
            {"generation": int(generation)},
        ).one_or_none()
        if state is None:
            raise ValueError("unknown retrieval vector generation")
        result = connection.execute(
            text(
                "UPDATE retrieval_documents SET "
                "embedding_provider = :provider, embedding_model = :model, "
                "embedding_version = :version, embedding_dimensions = :dimensions, "
                "embedding_generation = :generation, embedding_status = 'failed', "
                "last_error_code = :error_code, updated_at = :updated_at "
                "WHERE status = 'active' AND group_id = :group_id "
                "AND embedding_eligible = 1 "
                "AND id IN :document_ids"
            ).bindparams(bindparam("document_ids", expanding=True)),
            {
                "provider": str(state.provider),
                "model": str(state.model),
                "version": str(state.version),
                "dimensions": int(state.dimensions),
                "generation": int(generation),
                "group_id": int(group_id),
                "document_ids": tuple(int(value) for value in document_ids),
                "error_code": str(error_code or "")[:96],
                "updated_at": datetime.now(UTC).replace(tzinfo=None),
            },
        )
        return int(result.rowcount or 0)


def refresh_retrieval_vector_generation(
    engine: Engine,
    *,
    generation: int,
    mark_ready: bool = False,
) -> RetrievalVectorCoverage:
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            state = connection.execute(
                text(
                    "SELECT physical_table, provider, model, dimensions, version, status "
                    "FROM retrieval_index_state WHERE channel = 'vector' "
                    "AND generation = :generation"
                ),
                {"generation": int(generation)},
            ).one_or_none()
            if state is None:
                raise ValueError("unknown retrieval vector generation")
            physical_table = validate_retrieval_vector_table_name(
                str(state.physical_table),
                generation=int(generation),
            )
            total_documents = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM retrieval_documents "
                        "WHERE status = 'active' AND embedding_eligible = 1"
                    )
                ).scalar_one()
                or 0
            )
            indexed_documents = int(
                connection.execute(
                    text(
                        f"SELECT count(*) FROM {physical_table} AS v "
                        "JOIN retrieval_documents AS d "
                        "ON d.id = v.document_id AND d.group_id = v.group_id "
                        "WHERE d.status = 'active' AND d.embedding_eligible = 1"
                    )
                ).scalar_one()
                or 0
            )
            failed_documents = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM retrieval_documents "
                        "WHERE status = 'active' AND embedding_generation = :generation "
                        "AND embedding_eligible = 1 "
                        "AND embedding_status = 'failed' "
                        "AND embedding_provider = :provider AND embedding_model = :model "
                        "AND embedding_version = :version "
                        "AND embedding_dimensions = :dimensions"
                    ),
                    {
                        "generation": int(generation),
                        "provider": str(state.provider),
                        "model": str(state.model),
                        "version": str(state.version),
                        "dimensions": int(state.dimensions),
                    },
                ).scalar_one()
                or 0
            )
            status = str(state.status)
            if mark_ready:
                if failed_documents:
                    status = "failed"
                elif indexed_documents == total_documents:
                    status = "ready"
                else:
                    status = "building"
            connection.execute(
                text(
                    "UPDATE retrieval_index_state SET status = :status, "
                    "total_documents = :total_documents, "
                    "indexed_documents = :indexed_documents, updated_at = :updated_at "
                    "WHERE channel = 'vector' AND generation = :generation"
                ),
                {
                    "status": status,
                    "total_documents": total_documents,
                    "indexed_documents": indexed_documents,
                    "updated_at": datetime.now(UTC).replace(tzinfo=None),
                    "generation": int(generation),
                },
            )
            connection.commit()
            return RetrievalVectorCoverage(
                generation=int(generation),
                status=status,
                total_documents=total_documents,
                indexed_documents=indexed_documents,
                failed_documents=failed_documents,
            )
        except Exception:
            connection.rollback()
            raise


def activate_retrieval_vector_generation(
    engine: Engine,
    *,
    generation: int,
    expected_active_generation: int | None,
) -> bool:
    coverage = refresh_retrieval_vector_generation(
        engine,
        generation=int(generation),
        mark_ready=True,
    )
    if coverage.status != "ready":
        return False
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            target = connection.execute(
                text(
                    "SELECT physical_table FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND generation = :generation "
                    "AND status = 'ready' "
                    "AND total_documents = indexed_documents"
                ),
                {"generation": int(generation)},
            ).one_or_none()
            if target is None:
                connection.rollback()
                return False
            validate_retrieval_vector_table_name(
                str(target.physical_table),
                generation=int(generation),
            )
            active_generation = connection.execute(
                text(
                    "SELECT generation FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND is_active = 1"
                )
            ).scalar_one_or_none()
            if active_generation is not None and int(active_generation) == int(generation):
                connection.commit()
                return expected_active_generation in {
                    int(generation),
                    int(active_generation),
                }
            if expected_active_generation is None:
                if active_generation is not None:
                    connection.rollback()
                    return False
            elif active_generation is None or int(active_generation) != int(
                expected_active_generation
            ):
                connection.rollback()
                return False

            if active_generation is not None:
                deactivated = connection.execute(
                    text(
                        "UPDATE retrieval_index_state SET is_active = 0 "
                        "WHERE channel = 'vector' AND generation = :generation "
                        "AND is_active = 1"
                    ),
                    {"generation": int(active_generation)},
                )
                if int(deactivated.rowcount or 0) != 1:
                    connection.rollback()
                    return False
            activated_at = datetime.now(UTC).replace(tzinfo=None)
            activated = connection.execute(
                text(
                    "UPDATE retrieval_index_state SET is_active = 1, "
                    "activated_at = :activated_at, updated_at = :activated_at "
                    "WHERE channel = 'vector' AND generation = :generation "
                    "AND status = 'ready' AND is_active = 0"
                ),
                {
                    "generation": int(generation),
                    "activated_at": activated_at,
                },
            )
            if int(activated.rowcount or 0) != 1:
                connection.rollback()
                return False
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise


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
