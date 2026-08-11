"""Find recent bot replies mentioning 昨天/环境/本地."""

from __future__ import annotations

import sqlite3


def main() -> int:
    con = sqlite3.connect("file:/workspace/data/bot.db?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """SELECT id, platform_msg_id, group_id, user_id, msg_type, timestamp, mentioned_bot, reply_to_msg_id, plain_text
           FROM messages
           WHERE plain_text LIKE '%昨天%' OR plain_text LIKE '%本地%' OR plain_text LIKE '%环境%'
           ORDER BY id ASC"""
    )
    rows = cur.fetchall()
    print("total", len(rows))
    for r in rows:
        text = (r["plain_text"] or "")[:200].replace("\n", " ")
        print(
            f'{r["timestamp"]} | dbid={r["id"]} | g={r["group_id"]} | user={r["user_id"]} '
            f'| mention={r["mentioned_bot"]} | reply_to={r["reply_to_msg_id"]} | {text}'
        )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
