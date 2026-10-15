"""One-shot historical noise cleanup for memory_items.

Deactivates old V1 extraction-template rows: active preference/taboo/profile
facts whose content still carries the legacy "likes"/"dislikes"/"（QQ昵称"
marker (e.g. "阿渣 likes 坐床上看动画." or "XX（QQ昵称 likes 16的.").
These rows are recoverable (valid_until set) and their retrieval documents and
persistent semantic vectors are cleaned. Clean short facts such as
"住在大新。" / "不吃香菜" (content < 6 but no template marker) are preserved.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, event as sa_event, text
from sqlalchemy.pool import NullPool

from app.storage.db import create_all, session_scope
from app.storage.repositories import MemoryRepository


NOISE_KINDS = ("preference", "taboo", "profile")


def _build_engine(database: Path):
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    @sa_event.listens_for(engine, "connect")
    def _set_busy_timeout(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    create_all(engine)
    return engine


def _candidate_ids(engine, *, limit: int | None) -> list[int]:
    kinds = ",".join(f"'{kind}'" for kind in NOISE_KINDS)
    statement = (
        "SELECT id, memory_kind, object_text, content FROM memory_items "
        f"WHERE status='active' AND memory_kind IN ({kinds}) "
        "AND ("
        "  content LIKE '%（QQ昵称 likes%' "
        "  OR content LIKE '%（QQ昵称 dislikes%' "
        "  OR content LIKE '% likes %' "
        "  OR content LIKE '% dislikes %' "
        ") "
        "ORDER BY id"
    )
    with engine.connect() as connection:
        rows = connection.execute(text(statement)).all()
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]
    return [int(row[0]) for row in rows]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deactivate historical memory-item noise facts."
    )
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    engine = _build_engine(args.database)
    try:
        ids = _candidate_ids(engine, limit=args.limit)
        if args.command == "plan" or args.dry_run:
            print(f"candidates={len(ids)}")
            return 0
        now = datetime.now(UTC)
        with session_scope(engine) as session:
            deactivated = MemoryRepository(session).deactivate_memory_items(
                ids,
                valid_until=now,
            )
        print(f"deactivated={deactivated} candidates={len(ids)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
