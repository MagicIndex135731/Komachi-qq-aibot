"""Deactivate specific memory fact ids (reversible by id)."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--ids", required=True)
    args = parser.parse_args()
    ids = [int(value.strip()) for value in args.ids.split(",") if value.strip()]
    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)
    con.execute(
        "UPDATE memory_items SET status='inactive' WHERE id IN ({})".format(
            ",".join("?" for _ in ids)
        ),
        ids,
    )
    con.commit()
    print(f"deactivated={len(ids)}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
