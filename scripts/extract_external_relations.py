"""Extract one out-of-group figure (VTuber/star/up主) from evidence and merge
it into the persona's external_relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import yaml


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
    payload = {
        "model": model,
        "stream": False,
        "input": prompt,
        "max_output_tokens": 3000,
        "reasoning": {"effort": "low"},
    }
    with httpx.Client(timeout=180) as client:
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
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text") or "")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--corpus", default="data/personas/azha/persona_v2/corpus.jsonl")
    parser.add_argument("--stream", default="data/personas/azha/persona_v2/group_stream.jsonl")
    parser.add_argument("--persona", default="configs/personas/azha.yaml")
    parser.add_argument("--live", default="data/personas/azha.live.yaml")
    parser.add_argument("--base-url", default="http://127.0.0.1:15721/v1")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key", default="PROXY_MANAGED")
    args = parser.parse_args()

    corpus = _load_jsonl(Path(args.corpus))
    lines: list[str] = []
    seen: set[str] = set()
    for row in corpus:
        text = str(row.get("text") or row.get("plain_text") or "").strip()
        if args.keyword in text and text not in seen:
            seen.add(text)
            lines.append(text)
            if len(lines) >= 60:
                break
    if not lines:
        raise SystemExit(f"no evidence lines for {args.keyword}")

    prompt = (
        f"你是人设蒸馏专家。下面是群成员关于“{args.keyword}”的真实发言（只取与 {args.keyword} 直接相关的）。"
        "推断他与这个（群外）人物的关系，只输出一个 ```yaml 代码块：\n"
        "external_relations:\n"
        "- name: 人物常用名\n"
        "  who: 这个人的身份（虚拟主播/球星/up主等）\n"
        "  relation: 他与这个人的关系（铁粉/黑粉/纯路人等，要准确）\n"
        "  attitude: 他真实的态度（2-3 句，用他原话风格概括，不要脑补）\n"
        "  evidence:\n"
        "  - 逐字引用他的一句话\n"
        "  - 逐字引用另一句\n"
        f"证据：\n" + "\n".join(lines)
    )
    generated = _complete(args.base_url, args.api_key, args.model, prompt)
    import re

    match = re.search(r"```(?:yaml|yml)?\s*(.*?)```", generated, re.DOTALL)
    raw = match.group(1) if match else generated
    data = yaml.safe_load(raw) or {}
    entries = data.get("external_relations") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SystemExit("model did not produce external_relations")
    entry = entries[0]

    persona_path = Path(args.persona)
    persona = yaml.safe_load(persona_path.read_text(encoding="utf-8")) or {}
    current = [item for item in (persona.get("external_relations") or []) if isinstance(item, dict)]
    current = [item for item in current if item.get("name") != entry.get("name")]
    current.append(entry)
    persona["external_relations"] = current
    persona_path.write_text(
        yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    live_path = Path(args.live)
    live = yaml.safe_load(live_path.read_text(encoding="utf-8")) or {} if live_path.exists() else {}
    live_existing = [item for item in (live.get("external_relations") or []) if isinstance(item, dict)]
    live_existing = [item for item in live_existing if item.get("name") != entry.get("name")]
    live_existing.append(entry)
    live["external_relations"] = live_existing
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(
        yaml.safe_dump(live, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("entry=" + json.dumps(entry, ensure_ascii=False))
    print(f"persona_updated={persona_path} live_updated={live_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
