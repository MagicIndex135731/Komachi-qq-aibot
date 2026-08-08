"""Migrate stored naive UTC timestamps to naive Shanghai (UTC+8) clock faces.

Run once against a backed-up copy (or the production DB after taking a backup):
    python scripts/migrate_utc_to_shanghai.py --db data/bot.db --backup

Every DATETIME-typed column is shifted by +8 hours. The migration is refused
when the newest message timestamp already looks like Shanghai time, which
guards against running it twice.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


DATETIME_TYPES = {"datetime", "timestamp"}
SKIP_TABLE_SUBSTRINGS = (
    "_vec",
    "_fts",
    "_chunks",
    "_rowids",
    "_idx",
    "_data",
    "_docsize",
    "_config",
    "_content",
)


def _datetime_columns(con: sqlite3.Connection, table: str) -> list[str]:
    columns: list[str] = []
    for row in con.execute(f"PRAGMA table_info({table})"):
        declared = (row[2] or "").strip().lower()
        if any(token in declared for token in DATETIME_TYPES):
            columns.append(row[1])
    return columns


def _looks_already_migrated(con: sqlite3.Connection) -> bool:
    try:
        row = con.execute(
            "SELECT MAX(timestamp) FROM messages"
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    if not row or not row[0]:
        return False
    newest = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=UTC)
    shanghai_now = datetime.now(UTC) + timedelta(hours=8)
    return abs((shanghai_now - newest).total_seconds()) < 3600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/bot.db")
    parser.add_argument("--backup", action="store_true", help="take a backup before migrating")
    parser.add_argument("--force", action="store_true", help="skip the already-migrated guard")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        if not args.force and _looks_already_migrated(con):
            raise SystemExit("DB timestamps already look like Shanghai time; refusing (use --force to override)")
        if args.backup:
            backup_path = db_path.with_name(db_path.name + ".pre-utc-shanghai")
            backup = sqlite3.connect(backup_path)
            with backup:
                con.backup(backup)
            backup.close()
            print(f"backup written: {backup_path}")

        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if not any(token in row[0] for token in SKIP_TABLE_SUBSTRINGS)
        ]
        total = 0
        with con:
            for table in tables:
                columns = _datetime_columns(con, table)
                for column in columns:
                    cursor = con.execute(
                        f"UPDATE {table} SET {column} = datetime(datetime({column}), '+8 hours') "
                        f"WHERE {column} IS NOT NULL AND {column} != '' "
                        f"AND datetime({column}) IS NOT NULL"
                    )
                    if cursor.rowcount:
                        total += cursor.rowcount
                        print(f"{table}.{column}: {cursor.rowcount} rows")
        print(f"total updated rows: {total}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
