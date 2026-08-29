"""Static health audit: personas, relationships, switch parsing, binding."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from app.config import AppSettings, load_runtime_config
from app.core.member_identity import GroupMemberIdentity
from app.core.memory_query_resolver import MemoryQueryResolver
from app.core.persona_switch import PersonaManager, parse_switch_command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--personas-dir", default="/workspace/data/personas")
    parser.add_argument("--out", default="/workspace/data/tmp/audit_static.json")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    args = parser.parse_args()

    settings = AppSettings()
    runtime = load_runtime_config(settings)
    findings: list[dict] = []

    def note(level: str, category: str, message: str, **extra) -> None:
        findings.append(
            {"level": level, "category": category, "message": message, **extra}
        )

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    member_labels: dict[int, str] = {}
    member_identities: list[GroupMemberIdentity] = []
    seen_users: set[int] = set()
    for user_id, nickname, card in con.execute(
        "SELECT user_id, nickname, group_card FROM users"
    ).fetchall():
        label = str(card or "").strip() or str(nickname or "").strip()
        if label:
            member_labels[int(user_id)] = label
    for user_id, msg_group_id, raw_json in con.execute(
        "SELECT user_id, group_id, raw_json FROM messages "
        "WHERE raw_json IS NOT NULL ORDER BY id DESC"
    ).fetchall():
        uid = int(user_id)
        if uid in seen_users:
            continue
        seen_users.add(uid)
        try:
            payload = json.loads(raw_json or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        sender = payload.get("sender") if isinstance(payload, dict) else {}
        sender = sender if isinstance(sender, dict) else {}
        label = str(sender.get("card") or "").strip() or str(
            sender.get("nickname") or ""
        ).strip()
        if label:
            member_labels.setdefault(uid, label)
        member_identities.append(
            GroupMemberIdentity(
                user_id=uid,
                nickname=str(sender.get("nickname") or "").strip(),
                group_card=str(sender.get("card") or "").strip(),
                in_scope=int(msg_group_id or 0) == int(args.group_id),
            )
        )
    member_count = con.execute(
        "SELECT COUNT(DISTINCT user_id) FROM messages WHERE group_id=? AND user_id<>?",
        (args.group_id, args.bot_qq),
    ).fetchone()[0]
    con.close()

    personas = runtime.personas
    for key, persona in personas.items():
        if key == "default":
            continue
        name = str(persona.get("name") or "").strip()
        if not name:
            note("ERROR", "persona", f"{key}: 缺 name")
        for field in ("identity", "self_concept", "speech_habits", "style_avoid"):
            if not persona.get(field):
                note("WARN", "persona", f"{key}: 缺 {field}")
        rels = persona.get("relationships") or []
        numeric_members = [
            str(rel.get("member"))
            for rel in rels
            if isinstance(rel, dict) and str(rel.get("member") or "").isdigit()
        ]
        if numeric_members:
            note(
                "ERROR",
                "relationship",
                f"{key}: relationships 里仍有 QQ 号作为 member",
                members=numeric_members[:5],
            )
        no_user_id = [
            str(rel.get("member"))
            for rel in rels
            if isinstance(rel, dict)
            and not rel.get("member_user_id")
            and not str(rel.get("member") or "").isdigit()
        ]
        if no_user_id:
            note(
                "WARN",
                "relationship",
                f"{key}: {len(no_user_id)} 条关系缺 member_user_id（改昵称后无法跟随）",
                members=no_user_id[:5],
            )
        if not rels:
            note("WARN", "relationship", f"{key}: 无 relationships")
        if not persona.get("example_bank") and not persona.get("example_lines"):
            note("WARN", "examples", f"{key}: 无示例")
        if not persona.get("burst"):
            note("WARN", "burst", f"{key}: 无 burst 配置")

    aliases = []
    for key, persona in personas.items():
        if key == "default":
            continue
        for candidate in (
            str(persona.get("name") or ""),
            *(str(a) for a in (persona.get("aliases") or [])),
        ):
            if candidate.strip():
                aliases.append((candidate.strip(), key))
    for alias, key in aliases:
        resolved = parse_switch_command(f"切换人格为:{alias}", personas)
        if resolved != key:
            note(
                "ERROR",
                "switch",
                f"切换人格为:{alias} -> {resolved}，期望 {key}",
            )

    members = member_identities
    resolver = MemoryQueryResolver()
    bind_cases = {
        "你最近面了哪些企业（'你'指阿渣本人）": "self_hint",
        "逆蝶蝶喜欢什么": "third_person",
        "如何评价我": "requester",
    }
    for query, kind in bind_cases.items():
        result = resolver.resolve(
            query,
            recent_messages=(),
            group_members=members,
            requester_id=999999,
            excluded_member_ids={int(args.bot_qq)},
        )
        if kind == "self_hint" and not result.subject_ids:
            note("ERROR", "binding", f"{query} 未绑定主体", result=result.subject_ids)
        if kind == "third_person" and not result.subject_ids:
            note("ERROR", "binding", f"{query} 未绑定主体", result=result.subject_ids)
        if kind == "requester" and result.subject_ids != ("999999",):
            note("ERROR", "binding", f"{query} 未绑定提问者", result=result.subject_ids)

    summary = {
        "personas": len(personas) - 1,
        "group_members": member_count,
        "member_labels_known": len(member_labels),
        "findings": findings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in findings:
        print(f"{item['level']}|{item['category']}|{item['message']}")
    print(f"findings={len(findings)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
