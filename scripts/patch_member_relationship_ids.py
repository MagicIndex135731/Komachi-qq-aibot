"""Backfill stable member_user_id into existing persona relationship entries."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--personas-dir", default="/workspace/data/personas")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)
    labels: dict[int, str] = {}
    for user_id, nickname, card in con.execute(
        "SELECT user_id, nickname, group_card FROM users"
    ).fetchall():
        label = str(card or "").strip() or str(nickname or "").strip()
        if label:
            labels[int(user_id)] = label
    con.close()

    changed = 0
    for path in sorted(Path(args.personas_dir).glob("*.yaml")):
        if path.name.endswith(".live.yaml"):
            continue
        persona = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dirty = False
        for rel in persona.get("relationships") or []:
            if not isinstance(rel, dict) or rel.get("member_user_id"):
                continue
            member = str(rel.get("member") or "").strip()
            if member.isdigit():
                user_id = int(member)
                rel["member_user_id"] = user_id
                rel["member"] = labels.get(user_id, member)
                dirty = True
        if dirty:
            changed += 1
            if not args.dry_run:
                path.write_text(
                    yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
    print(f"changed_files={changed} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
