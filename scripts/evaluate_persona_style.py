"""Evaluate a distilled persona: production-model generation + blind judge.

Generation uses the production chat model (nova luna via AppSettings env), so
the measured style is exactly what the bot would actually say. Judging uses
the local deepseek proxy (this conversation's interface): each real reply and
its generated counterpart are anonymized and shuffled before scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml

from app.config import AppSettings
from app.core.persona_engine import render_persona
from app.core.style_distill import BANNED_ADDRESS_TERMS, parse_fenced_json
from app.providers.llm_client import LlmClient


_JUDGE_DIMENSIONS = ("语气", "句式", "用词", "情绪", "整体")


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
    non_empty = [row for row in stream if str(row.get("text") or "").strip()]
    candidates: list[tuple[int, dict]] = []
    for index, row in enumerate(non_empty):
        if int(row.get("user_id") or 0) != target:
            continue
        if len(str(row.get("text") or "").strip()) < 6:
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
        candidates.append(
            (index, {"real": str(row.get("text") or ""), "context": context})
        )
    if not candidates:
        return []
    step = max(1, len(candidates) // max(1, limit))
    return [item for _, item in candidates[::step][:limit]]


def _judge_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "input": prompt,
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
    }
    with httpx.Client(timeout=180) as client:
        for attempt in range(1, 3):
            response = client.post(
                f"{base_url.rstrip('/')}/responses",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            parts: list[str] = []
            for item in body.get("output") or []:
                if item.get("type") != "message":
                    continue
                direct_text = item.get("text")
                if isinstance(direct_text, str) and direct_text:
                    parts.append(direct_text)
                for content in item.get("content") or []:
                    if content.get("type") in {"output_text", "text"}:
                        text = str(content.get("text") or "")
                        if text:
                            parts.append(text)
            if parts:
                return parse_fenced_json("\n".join(parts))
            print(f"judge_empty_retry attempt={attempt}")
    raise ValueError("judge response had no text after retries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--target-name", default="阿渣")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:15721/v1")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-api-key", default="PROXY_MANAGED")
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
    generation_client = LlmClient(
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
    for index, item in enumerate(holdout):
        prompt_lines = [
            persona_text,
            (
                "你正在群里以这个身份接话。只输出你要发的内容本身，像真人那样短、"
                "不解释、不自我介绍、不称呼任何人为主人。最多两三条短消息，用 | 分隔。"
                "以下是聊天上下文："
            ),
            *item["context"],
        ]
        generated = generation_client.generate_text(prompt_lines)
        hit_terms = [term for term in BANNED_ADDRESS_TERMS if term in generated]
        banned_hits += len(hit_terms)

        # Blind, position-randomized judging.
        candidates = [
            {"label": "A", "text": generated},
            {"label": "B", "text": item["real"]},
        ]
        random.Random(index * 7919 + 13).shuffle(candidates)
        judge_prompt = (
            f"你是风格盲评评委。下面是 QQ 群聊上下文，以及两个候选回复 A/B（顺序随机）。"
            f"判断哪个更像群成员“{args.target_name}”本人会发的消息。\n"
            "对每个候选按 1-5 打分（1=完全不像，5=就是他本人）：语气一致性、句式节奏、"
            "用词习惯、情绪自然度、整体相似度。\n"
            '只输出 JSON，不要其他文字：{"more_like":"A|B|tie",'
            '"A":{"语气":int,"句式":int,"用词":int,"情绪":int,"整体":int},'
            '"B":{"语气":int,"句式":int,"用词":int,"情绪":int,"整体":int},'
            '"reason":"一句话"}。\n'
            f"上下文：\n" + "\n".join(item["context"]) + "\n"
            f"候选A：{candidates[0]['text']}\n候选B：{candidates[1]['text']}"
        )
        judge = _judge_responses(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
            prompt=judge_prompt,
        )
        mapping = {candidate["label"]: candidate["text"] for candidate in candidates}
        pairs.append(
            {
                "context": item["context"],
                "real": item["real"],
                "generated": generated,
                "banned_terms": hit_terms,
                "judge": judge,
                "generated_was": next(
                    label
                    for label, text in mapping.items()
                    if text == generated
                ),
            }
        )

    real_lengths = sorted(len(str(row.get("text") or "")) for row in corpus)
    generated_lengths = sorted(len(str(pair["generated"])) for pair in pairs)
    report = {
        "holdout": len(holdout),
        "banned_term_hits": banned_hits,
        "generated_length": {
            "min": generated_lengths[0] if generated_lengths else 0,
            "median": generated_lengths[len(generated_lengths) // 2]
            if generated_lengths
            else 0,
            "max": generated_lengths[-1] if generated_lengths else 0,
        },
        "real_length": {
            "median": real_lengths[len(real_lengths) // 2] if real_lengths else 0,
            "p90": real_lengths[min(len(real_lengths) - 1, len(real_lengths) * 9 // 10)]
            if real_lengths
            else 0,
        },
        "judge": _summarize_judgments(pairs),
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


def _summarize_judgments(pairs: list[dict]) -> dict:
    if not pairs:
        return {}
    real_dims = {dim: [] for dim in _JUDGE_DIMENSIONS}
    generated_dims = {dim: [] for dim in _JUDGE_DIMENSIONS}
    prefer_generated = 0
    prefer_real = 0
    ties = 0
    for pair in pairs:
        judge = pair.get("judge") or {}
        generated_label = pair.get("generated_was")
        real_label = "B" if generated_label == "A" else "A"
        for dim in _JUDGE_DIMENSIONS:
            for label, bucket in ((generated_label, generated_dims), (real_label, real_dims)):
                value = (judge.get(label) or {}).get(dim)
                if isinstance(value, (int, float)):
                    bucket[dim].append(float(value))
        more_like = judge.get("more_like")
        if more_like == generated_label:
            prefer_generated += 1
        elif more_like == real_label:
            prefer_real += 1
        else:
            ties += 1

    def averages(buckets: dict) -> dict:
        return {
            dim: round(sum(values) / len(values), 2)
            for dim, values in buckets.items()
            if values
        }

    return {
        "generated_scores": averages(generated_dims),
        "real_scores": averages(real_dims),
        "judge_prefers_generated": prefer_generated,
        "judge_prefers_real": prefer_real,
        "judge_ties": ties,
    }


if __name__ == "__main__":
    raise SystemExit(main())
