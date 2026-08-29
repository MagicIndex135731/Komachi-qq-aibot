"""Extract a member's complete message corpus from the production database.

Writes two local-only files under the configured output directory:
  - group_stream.jsonl: every group message (for context windows)
  - corpus.jsonl: the target member's text messages
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from app.core.message_mentions import (
    bot_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)


def _speaker_label(raw_json: str | None, user_id: int) -> str:
    raw = raw_json or ""
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    sender = payload.get("sender") if isinstance(payload, dict) else {}
    sender = sender if isinstance(sender, dict) else {}
    card = str(sender.get("card") or "").strip()
    nickname = str(sender.get("nickname") or "").strip()
    return card or nickname or str(user_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    parser.add_argument("--bot-name", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    bot_display = collect_bot_display_names(
        row[0]
        for row in con.execute(
            "SELECT raw_json FROM messages WHERE user_id=? AND raw_json IS NOT NULL "
            "ORDER BY id DESC LIMIT 3000",
            (args.bot_qq,),
        )
    )
    bot_names = bot_mention_names(
        bot_qq=args.bot_qq,
        default_name=args.bot_name,
        display_names=bot_display,
    )
    rows = con.execute(
        "SELECT platform_msg_id, timestamp, user_id, plain_text, msg_type, "
        "reply_to_msg_id, raw_json "
        "FROM messages WHERE group_id = ? "
        "ORDER BY timestamp, id",
        (args.group_id,),
    )

    corpus_count = 0
    stream_count = 0
    seen_ids: set[str] = set()
    with (out_dir / "group_stream.jsonl").open("w", encoding="utf-8") as stream, (
        out_dir / "corpus.jsonl"
    ).open("w", encoding="utf-8") as corpus:
        for row in rows:
            platform_msg_id = str(row["platform_msg_id"])
            if platform_msg_id in seen_ids:
                continue
            seen_ids.add(platform_msg_id)
            text = str(row["plain_text"] or "").strip()
            if (
                int(row["user_id"]) == args.user_id
                and message_mentions_bot(
                    row["raw_json"],
                    bot_qq=args.bot_qq,
                    bot_names=bot_names,
                )
            ):
                # Human-to-AI turns must not enter the style corpus.
                continue
            record = {
                "timestamp": row["timestamp"],
                "platform_msg_id": platform_msg_id,
                "user_id": int(row["user_id"]),
                "speaker": _speaker_label(row["raw_json"], int(row["user_id"])),
                "text": text,
                "msg_type": row["msg_type"],
                "reply_to_msg_id": row["reply_to_msg_id"],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream_count += 1
            if int(row["user_id"]) == args.user_id and row["msg_type"] == "text" and text:
                corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
                corpus_count += 1
    con.close()
    print(
        f"stream={stream_count} corpus={corpus_count} "
        f"out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
