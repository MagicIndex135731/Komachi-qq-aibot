import sqlite3

import pytest
from sqlalchemy import text

from app.storage.db import build_engine


def test_read_only_engine_keeps_pristine_snapshot_untouched(tmp_path) -> None:
    db_path = tmp_path / "frozen.db"
    connection = sqlite3.connect(str(db_path))
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.execute("INSERT INTO t VALUES (1)")
    connection.commit()
    connection.close()
    before = db_path.read_bytes()

    engine = build_engine(db_path, read_only=True)
    with engine.connect() as engine_connection:
        assert engine_connection.execute(text("SELECT COUNT(*) FROM t")).scalar() == 1
    engine.dispose()

    assert db_path.read_bytes() == before
    assert not (tmp_path / "frozen.db-wal").exists()
    assert not (tmp_path / "frozen.db-shm").exists()


def test_read_only_engine_rejects_writes(tmp_path) -> None:
    db_path = tmp_path / "frozen.db"
    connection = sqlite3.connect(str(db_path))
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.commit()
    connection.close()

    engine = build_engine(db_path, read_only=True)
    with pytest.raises(Exception):
        with engine.begin() as engine_connection:
            engine_connection.execute(text("INSERT INTO t VALUES (1)"))
    engine.dispose()


def test_read_only_engine_does_not_recreate_wal_for_wal_mode_snapshot(
    tmp_path,
) -> None:
    db_path = tmp_path / "wal.db"
    engine = build_engine(db_path)
    try:
        with engine.begin() as engine_connection:
            engine_connection.execute(text("CREATE TABLE t (id INTEGER)"))
    finally:
        engine.dispose()
    for suffix in ("-wal", "-shm"):
        path = tmp_path / f"wal.db{suffix}"
        if path.exists():
            path.unlink()
    before = db_path.read_bytes()

    read_only_engine = build_engine(db_path, read_only=True)
    try:
        with read_only_engine.connect() as engine_connection:
            assert (
                engine_connection.execute(text("SELECT COUNT(*) FROM t")).scalar()
                == 0
            )
    finally:
        read_only_engine.dispose()

    assert db_path.read_bytes() == before
    assert not (tmp_path / "wal.db-wal").exists()
    assert not (tmp_path / "wal.db-shm").exists()


def test_default_engine_still_enables_wal(tmp_path) -> None:
    db_path = tmp_path / "writable.db"
    engine = build_engine(db_path)
    try:
        with engine.connect() as engine_connection:
            assert engine_connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    finally:
        engine.dispose()
