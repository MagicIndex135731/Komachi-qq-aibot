"""Clamp existing daily summary windows to their own Shanghai calendar day.

Older cross-midnight compaction batches stored daily summaries whose
``start_at`` pointed at the previous evening, which made the model misattribute
same-day statements to "yesterday". This repairs metadata only (content keeps
its original digest) and creates a backup before writing.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def _day_bounds(day_text: str) -> tuple[datetime, datetime] | None:
    try:
        day = datetime.strptime(day_text, "%Y-%m-%d")
    except ValueError:
        return None
    return day, day + timedelta(days=1)


def _parse_stored(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def repair_daily_summary_days(db_path: str | Path, *, dry_run: bool = False) -> dict[str, int]:
    db_path = Path(db_path)
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            """SELECT id, summary_key, start_at, end_at
               FROM summaries
               WHERE summary_level IN ('semantic_daily', 'daily')
                 AND (summary_key LIKE 'semantic-daily:%' OR summary_key LIKE 'daily:%')"""
        )
        rows = cur.fetchall()
        updates = 0
        skipped = 0
        for row_id, summary_key, start_at, end_at in rows:
            prefix, _, day_text = summary_key.rpartition(":")
            bounds = _day_bounds(day_text)
            if bounds is None:
                skipped += 1
                continue
            day_start, day_end = bounds
            parsed_start = _parse_stored(start_at)
            parsed_end = _parse_stored(end_at)
            new_start = max(parsed_start, day_start) if parsed_start else day_start
            new_end = min(parsed_end, day_end) if parsed_end else parsed_end
            if new_start == start_at and new_end == end_at:
                continue
            if not dry_run:
                cur.execute(
                    "UPDATE summaries SET start_at = ?, end_at = ? WHERE id = ?",
                    (
                        new_start.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        new_end.strftime("%Y-%m-%d %H:%M:%S.%f") if new_end is not None else None,
                        row_id,
                    ),
                )
            updates += 1
        if not dry_run:
            con.commit()
        return {"scanned": len(rows), "updated": updates, "skipped": skipped}
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not args.dry_run and not db_path.exists():
        print(f"db not found: {db_path}")
        return 1
    if not args.dry_run:
        backup_path = db_path.with_name(f"{db_path.name}.pre-daily-repair")
        if not backup_path.exists():
            source = sqlite3.connect(str(db_path))
            target = sqlite3.connect(str(backup_path))
            with target:
                source.backup(target)
            target.close()
            source.close()
            print(f"backup created: {backup_path}")
    result = repair_daily_summary_days(db_path, dry_run=args.dry_run)
    print(f"dry_run={args.dry_run} result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
