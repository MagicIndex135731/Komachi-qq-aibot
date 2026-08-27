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
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
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
