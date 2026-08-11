"""Show recent decision-kind memory items (称呼/规则类指令)."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM memory_items WHERE memory_kind = 'decision'")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT subject_id, COUNT(*) AS c FROM memory_items WHERE memory_kind = 'decision' GROUP BY subject_id ORDER BY c DESC"
    )
    print("total decision items:", total)
    print("by subject:", [f"{row['subject_id']}={row['c']}" for row in cur.fetchall()])
    cur.execute(
        """SELECT id, scope_id, subject_id, memory_kind, predicate, object_text,
                  confidence, valid_from, valid_until, source_msg_id, substr(content,1,120) AS content
           FROM memory_items
           WHERE memory_kind = 'decision'
           ORDER BY id DESC LIMIT ?""",
        (args.limit,),
    )
    for r in cur.fetchall():
        print(
            f'id={r["id"]} | scope={r["scope_id"]} | subject={r["subject_id"]} | kind={r["memory_kind"]} '
            f'| predicate={r["predicate"]!r} | object={r["object_text"]!r} '
            f'| valid={r["valid_from"]}~{r["valid_until"]} | src={r["source_msg_id"]} | {r["content"]}'
        )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
