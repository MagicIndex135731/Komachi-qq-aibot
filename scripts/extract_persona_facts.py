"""Extract a comprehensive fact sheet about the member from their full corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from app.providers.llm_client import LlmClient


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
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


def _complete(base_url: str, api_key: str, model: str, prompt: str) -> str:
    client = LlmClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        fallback_model=model,
        responses_only=True,
        responses_model=model,
        max_output_tokens=8000,
        timeout_seconds=300.0,
        reasoning_effort="low",
    )
    return client.generate_text([prompt])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/personas/azha/persona_v2/corpus.jsonl")
    parser.add_argument("--persona", default="configs/personas/azha.yaml")
    parser.add_argument("--live", default="data/personas/azha.live.yaml")
    parser.add_argument("--target-name", default="阿渣")
    parser.add_argument("--base-url", default="http://127.0.0.1:15721/v1")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key", default="PROXY_MANAGED")
    parser.add_argument("--slice-chars", type=int, default=18000)
    args = parser.parse_args()

    corpus = _load_jsonl(Path(args.corpus))
    slices: list[list[str]] = [[]]
    used = 0
    for row in corpus:
        text = str(row.get("text") or row.get("plain_text") or "").strip()
        if not text:
            continue
        if used + len(text) > args.slice_chars and slices[-1]:
            slices.append([])
            used = 0
        slices[-1].append(text)
        used += len(text)

    all_facts: list[dict] = []
    for index, lines in enumerate(slices):
        prompt = (
            f"你是人设蒸馏专家。下面是从群成员（{args.target_name}）真实聊天记录中截取的一段。"
            "请提取关于他的、有依据的持久事实，只输出一个 ```json 代码块：\n"
            '{"facts": [{"category": "游戏/体育/动漫/工作/生活/观点/人际关系/外部人物/其他", '
            '"fact": "用第三人称写的具体事实（如：他主玩英雄联盟手游，自嘲菜逼，常聊海斗排位）", '
            '"evidence": "逐字引用他的一句话作为证据"}]}\n'
            "要求：只提取能直接推断的事实，不要脑补；尽量具体（具体到游戏名/英雄/球队/番剧/公司等）。\n"
            f"语料：\n" + "\n".join(lines)
        )
        generated = _complete(args.base_url, args.api_key, args.model, prompt)
        match = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", generated, re.DOTALL)
        raw = match.group(1) if match else generated
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = yaml.safe_load(raw) or {}
        facts = data.get("facts") if isinstance(data, dict) else None
        if isinstance(facts, list):
            all_facts.extend(item for item in facts if isinstance(item, dict))
        print(f"slice {index + 1}/{len(slices)} facts={len(all_facts)}")

    seen: set[str] = set()
    deduped: list[dict] = []
    for fact in all_facts:
        key = str(fact.get("fact") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(fact)

    persona_path = Path(args.persona)
    persona = yaml.safe_load(persona_path.read_text(encoding="utf-8")) or {}
    persona["facts"] = deduped
    persona_path.write_text(
        yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    live_path = Path(args.live)
    live = (
        yaml.safe_load(live_path.read_text(encoding="utf-8")) or {}
        if live_path.exists()
        else {}
    )
    live["facts"] = deduped
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(
        yaml.safe_dump(live, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"facts_total={len(deduped)} persona_updated={persona_path} live_updated={live_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
