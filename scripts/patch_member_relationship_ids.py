"""Backfill stable member_user_id into existing persona relationship entries.

Handles both runtime member personas (data/personas/*.yaml) and static persona
profiles (configs/personas/*.yaml, e.g. azha.yaml). A relationship whose
``member`` is a bare QQ number is rewritten to the current display name while
keeping the numeric id in ``member_user_id``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml


def _split_alias_candidates(member: str) -> list[str]:
    """Return plausible alias candidates for a combined display name."""

    cleaned = member.strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    for separator in ("、", "，", ",", "&", "/", "|", "／"):
        parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
        if len(parts) > 1:
            candidates.extend(parts)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--personas-dir", default="/workspace/data/personas")
    parser.add_argument(
        "--config-dir",
        default="/workspace/configs/personas",
        help="Optional static persona profiles (e.g. azha.yaml).",
    )
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
    alias_to_user: dict[str, int] = {}
    seen: set[int] = set()
    for user_id, raw_json in con.execute(
        "SELECT user_id, raw_json FROM messages "
        "WHERE raw_json IS NOT NULL ORDER BY id DESC"
    ).fetchall():
        uid = int(user_id)
        if uid in seen:
            continue
        seen.add(uid)
        try:
            payload = json.loads(raw_json or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        sender = payload.get("sender") if isinstance(payload, dict) else {}
        sender = sender if isinstance(sender, dict) else {}
        card = str(sender.get("card") or "").strip()
        nickname = str(sender.get("nickname") or "").strip()
        label = card or nickname
        if label:
            labels.setdefault(uid, label)
            alias_to_user.setdefault(label, uid)
    con.close()

    changed = 0
    scanned = 0
    unresolved: dict[str, list[str]] = {}
    paths = sorted(Path(args.personas_dir).glob("*.yaml"))
    if args.config_dir:
        config_dir = Path(args.config_dir)
        if config_dir.is_dir():
            paths.extend(sorted(config_dir.glob("*.yaml")))
    seen_paths = set()
    for path in paths:
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        scanned += 1
        persona = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dirty = False
        for rel in persona.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            member = str(rel.get("member") or "").strip()
            user_id = rel.get("member_user_id")
            if user_id is not None:
                try:
                    user_id = int(user_id)
                except (TypeError, ValueError):
                    user_id = None
            if user_id is None:
                if member.isdigit():
                    user_id = int(member)
                else:
                    for candidate in _split_alias_candidates(member):
                        if candidate in alias_to_user:
                            user_id = alias_to_user[candidate]
                            break
            if user_id is None:
                unresolved.setdefault(str(path), []).append(member)
                continue
            label = labels.get(int(user_id), member)
            if rel.get("member_user_id") != user_id or member != label:
                rel["member_user_id"] = int(user_id)
                rel["member"] = label
                dirty = True
        if dirty:
            changed += 1
            if not args.dry_run:
                path.write_text(
                    yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
    print(
        f"scanned={scanned} changed_files={changed} "
        f"unresolved={sum(len(v) for v in unresolved.values())} dry_run={args.dry_run}"
    )
    for path, members in unresolved.items():
        print(f"unresolved {path}: {members}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
