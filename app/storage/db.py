from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.storage.models import Base


def build_engine(sqlite_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{sqlite_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    return engine


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _run_schema_migrations(engine)


def _run_schema_migrations(engine: Engine) -> None:
    with engine.begin() as connection:
        table_names = {
            str(row[0])
            for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }
        if "dev_sessions" not in table_names:
            return

        columns = {
            str(row[1])
            for row in connection.execute(text("PRAGMA table_info(dev_sessions)"))
        }
        if "session_mode" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE dev_sessions "
                    "ADD COLUMN session_mode VARCHAR(32) NOT NULL DEFAULT 'project'"
                )
            )
        connection.execute(
            text(
                "UPDATE dev_sessions "
                "SET session_mode = 'project' "
                "WHERE session_mode IS NULL OR session_mode = ''"
            )
        )


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
