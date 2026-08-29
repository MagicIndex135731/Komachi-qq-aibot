"""Full-history segmented persona distillation.

The whole conversation stream is split into chronological blocks; every
message of every block (with real surrounding context) is fed to luna to
produce a per-segment style summary. All summaries are then merged with
full-corpus statistics and relationship evidence into the final persona.

Usage (inside xiaomachi-bot container):
    python scripts/distill_segmented_persona.py \
        --group-id <GROUP_ID> --user-id <MEMBER_QQ> --bot-qq <BOT_QQ> \
        --target-name <NAME> --group-card <CARD> --segments 8 \
        --out /workspace/data/personas/<MEMBER_QQ>.yaml
"""

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
    build_merged_profile_prompt,
    build_segment_blocks,
    build_segment_prompt,
    build_style_samples,
    compute_relationship_map,
    compute_style_stats,
    parse_fenced_json,
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
    card = str((sender or {}).get("card") or "").strip()
    nickname = str((sender or {}).get("nickname") or "").strip()
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
        "timestamp": str(row["timestamp"] or ""),
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
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--bot-qq", type=int, required=True)
    parser.add_argument("--bot-name", default="")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--group-card", default="")
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
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
            f"SELECT raw_json FROM messages WHERE user_id IN ({placeholders}) "
            "AND raw_json IS NOT NULL ORDER BY id DESC LIMIT 3000",
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

    stream: list[dict] = []
    for row in con.execute(
        "SELECT platform_msg_id, timestamp, user_id, plain_text, msg_type, "
        "reply_to_msg_id, raw_json FROM messages WHERE group_id=? "
        "ORDER BY timestamp, id",
        (args.group_id,),
    ):
        if int(row["user_id"]) in bot_ids:
            continue
        if (
            int(row["user_id"]) == args.user_id
            and message_mentions_bot(
                row["raw_json"],
                bot_qqs=bot_ids,
                bot_text_names=bot_text_names,
            )
        ):
            continue
        stream.append(_flatten_db_row(row))
    con.close()

    corpus = [
        {"text": row["text"], "speaker": row["speaker"], "timestamp": row["timestamp"]}
        for row in stream
        if int(row["user_id"]) == args.user_id and str(row["text"] or "").strip()
    ]
    stats = compute_style_stats(corpus)
    relationships = compute_relationship_map(
        stream,
        user_id=args.user_id,
        exclude_user_ids=bot_ids,
    )
    print(
        f"stream={len(stream)} corpus={len(corpus)} "
        f"relationships={len(relationships)}",
        flush=True,
    )

    blocks = build_segment_blocks(
        stream,
        num_segments=args.segments,
        target_user_id=args.user_id,
        target_name=args.target_name,
    )
    print(
        "blocks="
        + json.dumps(
            [
                {
                    "start": block["start"],
                    "end": block["end"],
                    "messages": block["messages"],
                    "chars": block["characters"],
                }
                for block in blocks
            ],
            ensure_ascii=False,
        ),
        flush=True,
    )

    settings = AppSettings()
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=6000,
        timeout_seconds=600.0,
        reasoning_effort="low",
    )

    summaries: list[dict] = []
    for block in blocks:
        print(
            f"[segment] {block['index'] + 1}/{len(blocks)} "
            f"{block['start']} -> {block['end']} ...",
            flush=True,
        )
        if args.dry_run:
            summaries.append(
                {
                    "period": f"{block['start']} ~ {block['end']}",
                    "member_message_count": 0,
                    "identity_fragment": "(dry-run)",
                    "speech_habits": [],
                    "vocabulary": [],
                    "tone": "",
                    "topics": [],
                    "emotion_patterns": "",
                    "representative_lines": [],
                }
            )
            continue
        generated = client.generate_text(
            build_segment_prompt(
                block=block,
                target_user_id=args.user_id,
                target_name=args.target_name,
            )
        )
        try:
            summary = parse_fenced_json(generated)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[segment] {block['index'] + 1} parse failed "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return 1
        summaries.append(summary)
        print(
            f"[segment] {block['index'] + 1} ok "
            f"habits={len(summary.get('speech_habits') or [])} "
            f"lines={len(summary.get('representative_lines') or [])}",
            flush=True,
        )

    if args.dry_run:
        print("dry-run done")
        return 0

    print("[merge] generating final persona ...", flush=True)
    generated = client.generate_text(
        build_merged_profile_prompt(
            segment_summaries=summaries,
            stats=stats,
            relationships=relationships,
            target_name=args.target_name,
        )
    )
    profile = parse_persona_yaml(generated)
    persona = assemble_persona(
        profile,
        target_name=args.target_name,
        group_card=args.group_card or args.target_name,
        source_user_id=args.user_id,
        aliases=[args.target_name],
    )
    persona["source_group_id"] = args.group_id
    persona["live_refresh"] = True
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
    samples = build_style_samples(
        records=stream,
        user_id=args.user_id,
        context_before=6,
        context_after=3,
        max_samples=120,
    )
    persona["example_bank"] = [
        str(sample.get("text") or "") for sample in samples if sample.get("text")
    ]
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"PERSONA_WRITTEN|{args.user_id}|{args.target_name}|{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
