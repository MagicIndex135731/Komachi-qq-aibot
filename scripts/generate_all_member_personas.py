"""Generate an impersonation persona for every active member (runs in container)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from app.config import AppSettings
from app.core.message_mentions import (
    bot_text_mention_names,
    collect_bot_display_names,
    message_mentions_bot,
)
from app.core.style_distill import (
    assemble_persona,
    build_profile_prompt,
    build_style_samples,
    compute_relationship_map,
    compute_style_stats,
    parse_persona_yaml,
    select_examples,
)
from app.providers.llm_client import LlmClient


def _speaker_label(raw_json: str | None, user_id: int) -> str:
    try:
        payload = json.loads(raw_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    sender = payload.get("sender") if isinstance(payload, dict) else {}
    card = str(sender.get("card") or "").strip()
    nickname = str(sender.get("nickname") or "").strip()
    return card or nickname or str(user_id)


def _flatten_db_row(row) -> dict:
    raw = row["raw_json"]
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    sender = payload.get("sender") if isinstance(payload, dict) else {}
    sender = sender if isinstance(sender, dict) else {}
    card = str(sender.get("card") or "").strip()
    nickname = str(sender.get("nickname") or "").strip()
    return {
        "platform_msg_id": row["platform_msg_id"],
        "timestamp": row["timestamp"],
        "user_id": int(row["user_id"]),
        "text": row["plain_text"],
        "reply_to_msg_id": row["reply_to_msg_id"],
        "speaker": card or nickname or str(row["user_id"]),
        "group_card": card,
        "nickname": nickname,
        "raw_json": raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    parser.add_argument("--bot-name", default="")
    parser.add_argument("--min-messages", type=int, default=100)
    parser.add_argument("--out-dir", default="/workspace/data/personas")
    parser.add_argument("--skip-user-ids", default="")
    parser.add_argument("--only-user-ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    skip = {int(value.strip()) for value in args.skip_user_ids.split(",") if value.strip()}
    only = {int(value.strip()) for value in args.only_user_ids.split(",") if value.strip()}
    members = con.execute(
        "SELECT user_id FROM messages WHERE group_id=? AND user_id<>? AND plain_text<>'' "
        "GROUP BY user_id HAVING COUNT(*)>=?",
        (args.group_id, args.bot_qq, args.min_messages),
    ).fetchall()
    settings = AppSettings()
    bot_ids = {int(args.bot_qq)}
    for user_id, nickname, card in con.execute(
        "SELECT user_id, nickname, group_card FROM users"
    ):
        label = str(card or "").strip() or str(nickname or "").strip()
        if "小町" in label:
            bot_ids.add(int(user_id))
    ids_param = sorted(bot_ids)
    placeholders = ",".join("?" for _ in ids_param)
    bot_display = collect_bot_display_names(
        row[0]
        for row in con.execute(
            f"SELECT raw_json FROM messages WHERE user_id IN ({placeholders}) AND raw_json IS NOT NULL "
            "ORDER BY id DESC LIMIT 3000",
            ids_param,
        )
    )
    member_display: set[str] = set()
    for card, nickname in con.execute(
        f"SELECT DISTINCT json_extract(raw_json, '$.sender.card'), "
        f"json_extract(raw_json, '$.sender.nickname') FROM messages "
        f"WHERE raw_json IS NOT NULL AND user_id NOT IN ({placeholders})",
        ids_param,
    ):
        for value in (card, nickname):
            cleaned = str(value or "").strip()
            if cleaned:
                member_display.add(cleaned)
    bot_text_names = bot_text_mention_names(
        bot_qqs=bot_ids,
        default_name=args.bot_name,
        bot_display_names=bot_display,
        member_display_names=member_display,
    )
    for (user_id,) in members:
        user_id = int(user_id)
        if user_id in skip:
            continue
        if only and user_id not in only:
            continue
        label_row = con.execute(
            "SELECT raw_json FROM messages WHERE group_id=? AND user_id=? "
            "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 1",
            (args.group_id, user_id),
        ).fetchone()
        display_name = _speaker_label(label_row[0] if label_row else None, user_id)
        if not display_name:
            display_name = f"成员{user_id}"
        stream = []
        for row in con.execute(
            "SELECT platform_msg_id, timestamp, user_id, plain_text, msg_type, "
            "reply_to_msg_id, raw_json FROM messages WHERE group_id=? ORDER BY timestamp, id",
            (args.group_id,),
        ):
            if (
                int(row["user_id"]) == user_id
                and message_mentions_bot(
                    row["raw_json"],
                    bot_qqs=bot_ids,
                    bot_text_names=bot_text_names,
                )
            ):
                # Human-to-AI turns must not enter the style corpus.
                continue
            stream.append(_flatten_db_row(row))
        corpus = [
            {
                "text": row["text"],
                "speaker": row["speaker"],
                "platform_msg_id": row["platform_msg_id"],
                "timestamp": row["timestamp"],
            }
            for row in stream
            if int(row["user_id"]) == user_id and str(row["text"] or "").strip()
        ]
        stats = compute_style_stats(corpus)
        relationships = compute_relationship_map(
            stream, user_id=user_id, exclude_user_ids={args.bot_qq}
        )
        samples = build_style_samples(
            records=stream,
            user_id=user_id,
            context_before=6,
            context_after=3,
            max_samples=240,
        )
        if not samples:
            print(f"SKIP|{user_id}|{display_name}|no_samples")
            continue
        client = LlmClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            fallback_model=settings.llm_fallback_model,
            responses_only=True,
            responses_model=settings.llm_model,
            max_output_tokens=12000,
            timeout_seconds=300.0,
            reasoning_effort="low",
        )
        prompt = build_profile_prompt(
            samples=samples[:160],
            stats=stats,
            relationships=relationships,
            target_name=display_name,
            max_chars=30000,
        )
        try:
            generated = client.generate_text(prompt)
            profile = parse_persona_yaml(generated)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP|{user_id}|{display_name}|profile_failed={type(exc).__name__}")
            continue
        persona = assemble_persona(
            profile,
            target_name=display_name,
            group_card=display_name,
            source_user_id=user_id,
            aliases=[display_name],
        )
        persona["source_group_id"] = args.group_id
        name_to_user_id = {
            str(rel.get("member")): int(rel.get("user_id"))
            for rel in relationships
            if rel.get("member") and rel.get("user_id")
        }
        for rel in persona.get("relationships") or []:
            if not isinstance(rel, dict) or rel.get("member_user_id"):
                continue
            matched = name_to_user_id.get(str(rel.get("member") or ""))
            if matched:
                rel["member_user_id"] = matched
        persona["example_lines"] = select_examples(corpus, count=36)
        persona["example_bank"] = select_examples(corpus, count=120)
        persona.setdefault(
            "burst",
            {
                "enabled": True,
                "separator": "|",
                "max_messages": 3,
                "max_chars": 18,
                "min_delay_seconds": 0.8,
                "max_delay_seconds": 2.5,
            },
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{user_id}.yaml"
        if args.dry_run:
            print(f"DRYRUN|{user_id}|{display_name}|habits={len(persona.get('speech_habits') or [])}")
            continue
        out_path.write_text(
            yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"GENERATED|{user_id}|{display_name}|{out_path}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
