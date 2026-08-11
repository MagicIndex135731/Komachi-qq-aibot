"""Inspect the LLBot v3 message database (read-only)."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/llbot.v3.db")
    parser.add_argument("--keyword", default="")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print("tables:", tables)
    message_tables = [t for t in tables if "msg" in t.lower() or "chat" in t.lower()]
    for table in message_tables:
        try:
            cur.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cur.fetchall()]
            print(f"--- {table} cols: {cols}")
            if not cols:
                cur.execute(f"SELECT * FROM {table} LIMIT 5")
                rows = cur.fetchall()
                print(f"--- {table} raw rows:")
                for row in rows:
                    print(row)
                continue
            text_col = next(
                (
                    c
                    for c in cols
                    if any(
                        token in c.lower()
                        for token in ("content", "msg", "text", "element", "data", "brief")
                    )
                ),
                None,
            )
            if text_col is None and cols:
                text_col = cols[1]
            if text_col:
                where = f" WHERE {text_col} LIKE ?" if args.keyword else ""
                params = (f"%{args.keyword}%",) if args.keyword else ()
                cur.execute(
                    f"SELECT * FROM {table}{where} ORDER BY rowid DESC LIMIT ?",
                    (*params, args.limit),
                )
                for row in cur.fetchall():
                    printable = {
                        k: (str(v)[:80] if k != text_col else str(v)[:120])
                        for k, v in dict(row).items()
                    }
                    print(printable)
        except Exception as exc:  # noqa: BLE001
            print(f"--- {table} error: {type(exc).__name__}: {exc}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
