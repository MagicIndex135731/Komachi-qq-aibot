"""Import a persona's extracted facts/external_relations into memory_items."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import yaml


def _find_source_id(con, *, user_id: int, evidence: str) -> str | None:
    text = str(evidence or "").strip()
    if not text:
        return None
    row = con.execute(
        "SELECT platform_msg_id FROM messages "
        "WHERE user_id = ? AND plain_text = ? LIMIT 1",
        (int(user_id), text),
    ).fetchone()
    if row:
        return str(row[0])
    row = con.execute(
        "SELECT platform_msg_id, plain_text FROM messages "
        "WHERE user_id = ? AND plain_text LIKE ? LIMIT 1",
        (int(user_id), f"%{text[:40]}%"),
    ).fetchone()
    return str(row[0]) if row else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--persona", default="/workspace/configs/personas/azha.yaml")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    persona = yaml.safe_load(Path(args.persona).read_text(encoding="utf-8")) or {}
    facts = persona.get("facts") or []
    relations = persona.get("external_relations") or []
    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)

    imported = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_text:
            continue
        source_id = _find_source_id(con, user_id=args.user_id, evidence=fact.get("evidence"))
        sources = [source_id] if source_id else []
        print(
            f"FACT|{fact.get('category')}|{fact_text}|source={source_id or 'canonical'}"
        )
        if args.dry_run:
            continue
        _upsert_memory(
            con,
            group_id=args.group_id,
            user_id=args.user_id,
            kind="fact",
            key=fact_text,
            predicate=str(fact.get("category") or "fact"),
            content=fact_text,
            object_text="",
            sources=sources,
        )
        imported += 1

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        name = str(relation.get("name") or "").strip()
        if not name:
            continue
        content = (
            f"{name}（{relation.get('who') or ''}）：{relation.get('relation') or ''}；"
            f"态度：{relation.get('attitude') or ''}"
        )
        source_id = None
        for evidence in relation.get("evidence") or []:
            source_id = _find_source_id(con, user_id=args.user_id, evidence=evidence)
            if source_id:
                break
        print(f"RELATION|{name}|{content}|source={source_id or 'canonical'}")
        if args.dry_run:
            continue
        _upsert_memory(
            con,
            group_id=args.group_id,
            user_id=args.user_id,
            kind="relationship",
            key=f"external:{name}",
            predicate="external_relation",
            content=content,
            object_text=f"{name}（{relation.get('who') or ''}）",
            sources=[source_id] if source_id else [],
        )
        imported += 1

    con.commit()
    con.close()
    print(f"imported={imported} dry_run={args.dry_run}")
    return 0


def _upsert_memory(con, *, group_id, user_id, kind, key, predicate, content, object_text, sources) -> None:
    row = con.execute(
        "SELECT id FROM memory_items WHERE scope_type='group' AND scope_id=? "
        "AND canonical_key=? AND status='active'",
        (str(group_id), key),
    ).fetchone()
    now = "2026-08-28 00:00:00"
    if row:
        con.execute(
            "UPDATE memory_items SET predicate=?, object_text=?, content=? WHERE id=?",
            (predicate, object_text, content, row[0]),
        )
        return
    con.execute(
        "INSERT INTO memory_items (scope_type, scope_id, subject_type, subject_id, "
        "memory_kind, canonical_key, predicate, object_text, content, importance, "
        "confidence, source_msg_id, source_msg_ids, mention_count, status) "
        "VALUES ('group', ?, 'user', ?, ?, ?, ?, ?, ?, 3, 0.8, ?, ?, ?, 'active')",
        (
            str(group_id),
            str(user_id),
            kind,
            key,
            predicate,
            object_text,
            content,
            sources[0] if sources else f"canonical:{key}",
            _json_list(sources),
            max(1, len(sources)),
        ),
    )


def _json_list(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
