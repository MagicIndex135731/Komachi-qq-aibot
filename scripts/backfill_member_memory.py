"""Retrospective fact extraction for any group member into memory_items."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3

import yaml

from app.config import AppSettings
from app.providers.llm_client import LlmClient


def _complete(settings: AppSettings, prompt: str) -> str:
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=8000,
        timeout_seconds=300.0,
        reasoning_effort="low",
    )
    return client.generate_text([prompt])


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
        "SELECT platform_msg_id FROM messages "
        "WHERE user_id = ? AND plain_text LIKE ? LIMIT 1",
        (int(user_id), f"%{text[:40]}%"),
    ).fetchone()
    return str(row[0]) if row else None


def _upsert(con, *, group_id, user_id, kind, key, predicate, content, object_text, sources) -> None:
    row = con.execute(
        "SELECT id FROM memory_items WHERE scope_type='group' AND scope_id=? "
        "AND canonical_key=? AND status='active'",
        (str(group_id), key),
    ).fetchone()
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
        "VALUES ('group', ?, 'user', ?, ?, ?, ?, ?, ?, 3, 0.75, ?, ?, ?, 'active')",
        (
            str(group_id),
            str(user_id),
            kind,
            key,
            predicate,
            object_text,
            content,
            sources[0] if sources else f"canonical:{key}",
            json.dumps(sources, ensure_ascii=False),
            max(1, len(sources)),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--slice-chars", type=int, default=16000)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)
    messages = con.execute(
        "SELECT plain_text FROM messages "
        "WHERE group_id=? AND user_id=? AND plain_text<>'' AND msg_type='text' "
        "ORDER BY id",
        (args.group_id, args.user_id),
    ).fetchall()
    if not messages:
        raise SystemExit("no messages for member")
    slices: list[list[str]] = [[]]
    used = 0
    for (text,) in messages:
        text = str(text).strip()
        if not text:
            continue
        if used + len(text) > args.slice_chars and slices[-1]:
            slices.append([])
            used = 0
        slices[-1].append(text)
        used += len(text)

    settings = AppSettings()
    all_facts: list[dict] = []
    for index, lines in enumerate(slices):
        prompt = (
            "你是人设蒸馏专家。请从下面群成员的真实发言提取有依据的持久事实，"
            '只输出一个 ```json 代码块：{"facts": [{"category": "游戏/体育/动漫/工作/生活/观点/人际关系/外部人物/其他",'
            ' "fact": "第三人称具体事实", "evidence": "逐字引用他的一句话"}]}。'
            "只提取能直接推断的事实，不要脑补；特别注意他反复转发/维护/玩梗的对象（虚拟主播、球星、up主等外部人物）也要作为事实列出。\n语料：\n"
            + "\n".join(lines)
        )
        generated = _complete(settings, prompt)
        match = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", generated, re.DOTALL)
        raw = match.group(1) if match else generated
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = yaml.safe_load(raw) or {}
        facts = data.get("facts") if isinstance(data, dict) else []
        for fact in facts:
            if isinstance(fact, dict) and fact.get("fact"):
                evidence = str(fact.get("evidence") or "").strip()
                if evidence and evidence not in "\n".join(lines):
                    continue
                all_facts.append(fact)
        print(f"slice {index + 1}/{len(slices)} facts={len(all_facts)}")

    seen: set[str] = set()
    imported = 0
    for fact in all_facts:
        key = str(fact.get("fact") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        source = _find_source_id(con, user_id=args.user_id, evidence=fact.get("evidence"))
        print(f"FACT|{fact.get('category')}|{key}|source={source or 'canonical'}")
        if args.write:
            _upsert(
                con,
                group_id=args.group_id,
                user_id=args.user_id,
                kind="fact",
                key=key,
                predicate=str(fact.get("category") or "fact"),
                content=key,
                object_text="",
                sources=[source] if source else [],
            )
            imported += 1
    con.commit()
    con.close()
    print(f"facts={len(seen)} imported={imported} write={args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
