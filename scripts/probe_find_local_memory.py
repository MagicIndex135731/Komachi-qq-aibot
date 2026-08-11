"""Search memory_items/summaries for the 本地/环境 phrase and its time fields."""

from __future__ import annotations

import sqlite3


def main() -> int:
    con = sqlite3.connect("file:/workspace/data/bot.db?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """SELECT id, scope_id, subject_id, memory_kind, confidence, valid_from, valid_until, last_seen_at,
                   source_msg_id, source_msg_ids, substr(content,1,200) AS content
           FROM memory_items
           WHERE content LIKE '%本地%' OR content LIKE '%环境%' OR content LIKE '%改%'
           ORDER BY id DESC LIMIT 30"""
    )
    rows = cur.fetchall()
    print("memory_items", len(rows))
    for r in rows:
        print(
            f'id={r["id"]} | scope={r["scope_id"]} | subject={r["subject_id"]} | kind={r["memory_kind"]} '
            f'| valid={r["valid_from"]}~{r["valid_until"]} | seen={r["last_seen_at"]} '
            f'| src={r["source_msg_id"]} | {r["content"]}'
        )
    cur.execute(
        """SELECT id, scope_id, summary_level, start_at, end_at, source_start_msg_id, source_end_msg_id,
                   substr(content,1,200) AS content
           FROM summaries
           WHERE content LIKE '%本地%' OR content LIKE '%环境%' OR content LIKE '%改%'
           ORDER BY id DESC LIMIT 20"""
    )
    rows = cur.fetchall()
    print("summaries", len(rows))
    for r in rows:
        print(
            f'id={r["id"]} | scope={r["scope_id"]} | level={r["summary_level"]} '
            f'| {r["start_at"]}~{r["end_at"]} | src={r["source_start_msg_id"]}~{r["source_end_msg_id"]} '
            f'| {r["content"]}'
        )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
