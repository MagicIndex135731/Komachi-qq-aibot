"""Generate an impersonation persona for every active member (runs in container)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from app.config import AppSettings
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
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
        stream = [
            dict(row)
            for row in con.execute(
                "SELECT platform_msg_id, timestamp, user_id, plain_text, msg_type, "
                "reply_to_msg_id, raw_json FROM messages WHERE group_id=? ORDER BY timestamp, id",
                (args.group_id,),
            )
        ]
        corpus = [
            {
                "text": row["plain_text"],
                "platform_msg_id": row["platform_msg_id"],
                "timestamp": row["timestamp"],
            }
            for row in stream
            if int(row["user_id"]) == user_id and str(row["plain_text"] or "").strip()
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
