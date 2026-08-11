"""Dump recent rows from the LLBot message table (read-only)."""

from __future__ import annotations

import argparse
import sqlite3
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        cur.execute(f"SELECT * FROM message ORDER BY rowid DESC LIMIT {args.limit}")
        rows = cur.fetchall()
        print("rows", len(rows))
        for row in rows:
            print(dict(row))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
