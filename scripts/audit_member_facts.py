"""Re-review existing imported facts and deactivate obvious junk (runs in container)."""

from __future__ import annotations

import argparse
import json
import sqlite3

from app.config import AppSettings
from app.core.member_memory_backfill import review_facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--user-ids", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = AppSettings()
    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)
    dropped_total = 0
    for user_id in args.user_ids.split(","):
        user_id = int(user_id.strip())
        rows = con.execute(
            "SELECT canonical_key, source_msg_ids FROM memory_items WHERE scope_type='group' "
            "AND scope_id=? AND subject_id=? AND memory_kind='fact' "
            "AND canonical_key<>'' AND status='active'",
            (str(args.group_id), str(user_id)),
        ).fetchall()
        facts = []
        for fact_text, source_ids in rows:
            evidence = ""
            try:
                ids = json.loads(source_ids or "[]")
            except json.JSONDecodeError:
                ids = []
            if ids:
                placeholders = ",".join("?" for _ in ids)
                texts = con.execute(
                    f"SELECT plain_text FROM messages WHERE platform_msg_id IN ({placeholders})",
                    ids,
                ).fetchall()
                evidence = " | ".join(str(t[0]) for t in texts if t[0])[:300]
            facts.append({"fact": fact_text, "evidence": evidence})
        if not facts:
            continue
        kept = review_facts(settings, facts)
        kept_texts = {str(fact["fact"]) for fact in kept}
        dropped = [fact_text for fact_text, _ in rows if fact_text not in kept_texts]
        for text in dropped:
            print(f"DROP|{user_id}|{text}")
            if not args.dry_run:
                con.execute(
                    "UPDATE memory_items SET status='inactive' "
                    "WHERE canonical_key=? AND subject_id=? AND scope_id=? AND scope_type='group'",
                    (text, str(user_id), str(args.group_id)),
                )
        dropped_total += len(dropped)
    con.commit()
    con.close()
    print(f"dropped={dropped_total} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
