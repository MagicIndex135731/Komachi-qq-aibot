"""Evaluate a distilled persona against held-out real messages.

Replays recent target-member turns with their preceding context, generates
replies as the persona, and compares style statistics against the real corpus.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from app.config import AppSettings
from app.core.persona_engine import render_persona
from app.core.style_distill import BANNED_ADDRESS_TERMS
from app.providers.llm_client import LlmClient


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_holdout(
    stream: list[dict],
    *,
    user_id: int,
    holdout_days: int = 14,
    limit: int = 30,
    context_messages: int = 6,
) -> list[dict]:
    target = int(user_id)
    cutoff = datetime.now(UTC) - timedelta(days=holdout_days)
    non_empty = [
        row for row in stream if str(row.get("text") or "").strip()
    ]
    candidates: list[tuple[int, dict]] = []
    for index, row in enumerate(non_empty):
        if int(row.get("user_id") or 0) != target:
            continue
        ts = _parse_timestamp(row.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        context: list[str] = []
        for other in non_empty[max(0, index - context_messages) : index]:
            other_text = str(other.get("text") or "").strip()
            if other_text:
                context.append(
                    f"{other.get('speaker') or other.get('user_id')}: {other_text}"
                )
        candidates.append((index, {"real": str(row.get("text") or ""), "context": context}))
    if not candidates:
        return []
    step = max(1, len(candidates) // max(1, limit))
    return [item for _, item in candidates[::step][:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--out-dir", default="data/personas/azha/eval")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=14)
    args = parser.parse_args()

    stream = _load_jsonl(Path(args.stream))
    corpus = _load_jsonl(Path(args.corpus))
    persona = yaml.safe_load(Path(args.persona).read_text(encoding="utf-8")) or {}
    holdout = build_holdout(
        stream,
        user_id=args.user_id,
        holdout_days=args.holdout_days,
        limit=args.limit,
    )
    print(f"holdout={len(holdout)}")
    if not holdout:
        return 0

    settings = AppSettings()
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=1024,
        timeout_seconds=settings.llm_timeout_seconds,
        reasoning_effort=settings.llm_reasoning_effort,
    )
    persona_text = render_persona(persona)
    pairs: list[dict] = []
    banned_hits = 0
    for item in holdout:
        prompt_lines = [
            persona_text,
            (
                "你正在群里以这个身份接话。只输出你要发的内容本身，像真人那样短、"
                "不解释、不自我介绍、不称呼任何人为主人。最多两三条短消息，用 | 分隔。"
                "以下是聊天上下文："
            ),
            *item["context"],
        ]
        reply = client.generate_text(prompt_lines)
        hit_terms = [term for term in BANNED_ADDRESS_TERMS if term in reply]
        banned_hits += len(hit_terms)
        pairs.append(
            {
                "context": item["context"],
                "real": item["real"],
                "generated": reply,
                "banned_terms": hit_terms,
            }
        )

    real_lengths = sorted(len(str(row.get("text") or "")) for row in corpus)
    generated_lengths = sorted(
        len(str(pair["generated"])) for pair in pairs
    )
    report = {
        "holdout": len(holdout),
        "banned_term_hits": banned_hits,
        "generated_length": {
            "min": generated_lengths[0] if generated_lengths else 0,
            "median": generated_lengths[len(generated_lengths) // 2] if generated_lengths else 0,
            "max": generated_lengths[-1] if generated_lengths else 0,
        },
        "real_length": {
            "median": real_lengths[len(real_lengths) // 2] if real_lengths else 0,
            "p90": real_lengths[min(len(real_lengths) - 1, len(real_lengths) * 9 // 10)]
            if real_lengths
            else 0,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs),
        encoding="utf-8",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
