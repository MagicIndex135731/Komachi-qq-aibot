"""Print the latest messages from a bot.db (read-only)."""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT MAX(id) AS m FROM messages")
    print("max_id", cur.fetchone()["m"])
    cur.execute(
        """SELECT id, platform_msg_id, group_id, user_id, msg_type, timestamp,
                  mentioned_bot, reply_to_msg_id, plain_text
           FROM messages ORDER BY id DESC LIMIT ?""",
        (args.limit,),
    )
    for r in reversed(cur.fetchall()):
        text = (r["plain_text"] or "")[:90].replace("\n", " ")
        print(
            f'{r["timestamp"]} | dbid={r["id"]} | g={r["group_id"]} | user={r["user_id"]} '
            f'| type={r["msg_type"]} | mention={r["mentioned_bot"]} | reply_to={r["reply_to_msg_id"]} | {text}'
        )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
