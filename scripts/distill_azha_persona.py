"""Distill one member's full chat persona into a local persona profile.

Two-stage pipeline:
  1. deep profile from the full corpus + mechanical style statistics +
     per-member relationship evidence
  2. validation against held-out samples and a revised profile

Usage:
  python scripts/distill_azha_persona.py \
    --stream data/personas/azha/group_stream.jsonl \
    --corpus data/personas/azha/corpus.jsonl \
    --user-id <MEMBER_QQ> \
    --target-name <DISPLAY_NAME> --group-card <GROUP_CARD>

Outputs are local-only persona artifacts (private-chat derived style):
  configs/personas/azha.yaml
  data/personas/azha/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import yaml

from app.config import AppSettings
from app.core.style_distill import (
    assemble_persona,
    build_profile_prompt,
    build_refine_prompt,
    build_style_samples,
    compute_relationship_map,
    compute_style_stats,
    parse_persona_yaml,
    select_examples,
)


def _complete_responses_nonstream(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_lines: list[str],
    max_output_tokens: int = 30000,
    reasoning_effort: str = "low",
    timeout_seconds: float = 120.0,
) -> str:
    """Call the Responses API without streaming to avoid truncated long outputs."""

    payload = {
        "model": model,
        "stream": False,
        "input": "\n\n".join(prompt_lines),
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(1, 3):
            try:
                response = client.post(
                    f"{base_url.rstrip('/')}/responses",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                break
            except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                if attempt == 2:
                    raise
                print(f"responses_retry attempt={attempt} error={type(exc).__name__}")
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                text = str(content.get("text") or "")
                if text:
                    parts.append(text)
    if not parts:
        raise ValueError("responses output did not include text")
    return "\n".join(parts)


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--group-card", default="")
    parser.add_argument("--source-group-id", type=int, default=0)
    parser.add_argument("--exclude-user-ids", default="")
    parser.add_argument("--output", default="configs/personas/azha.yaml")
    parser.add_argument("--samples", default="data/personas/azha/samples.jsonl")
    parser.add_argument("--stage1-draft", default="data/personas/azha/stage1_draft.yaml")
    parser.add_argument("--reuse-stage1", action="store_true")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--max-samples", type=int, default=700)
    parser.add_argument("--context-before", type=int, default=6)
    parser.add_argument("--context-after", type=int, default=3)
    args = parser.parse_args()

    if not args.target_name.strip() or not args.group_card.strip():
        raise SystemExit("--target-name and --group-card are required")

    stream = _load_jsonl(Path(args.stream))
    corpus = _load_jsonl(Path(args.corpus))
    exclude_user_ids = {
        int(value.strip())
        for value in args.exclude_user_ids.split(",")
        if value.strip()
    }
    stats = compute_style_stats(corpus)
    relationships = compute_relationship_map(
        stream,
        user_id=args.user_id,
        exclude_user_ids=exclude_user_ids,
    )
    samples = build_style_samples(
        records=stream,
        user_id=args.user_id,
        context_before=args.context_before,
        context_after=args.context_after,
        max_samples=args.max_samples,
    )
    if not samples:
        raise SystemExit("no usable samples extracted; aborting distillation")
    _write_jsonl(Path(args.samples), samples)
    print(
        f"corpus={stats.get('count', 0)} samples={len(samples)} "
        f"relationships={len(relationships)}"
    )

    held_out = samples[-40:]
    stage_one_samples = samples[:-40][:160]
    deterministic_examples = select_examples(corpus, count=36)
    example_bank = select_examples(corpus, count=240)
    settings = AppSettings()
    stage1_path = Path(args.stage1_draft)
    if args.reuse_stage1 and stage1_path.exists():
        draft = assemble_persona(
            parse_persona_yaml(stage1_path.read_text(encoding="utf-8")),
            target_name=args.target_name,
            group_card=args.group_card,
            source_user_id=args.user_id,
            aliases=[args.target_name],
        )
        print("stage1_profile_reused")
    else:
        stage_one_text = _complete_responses_nonstream(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            prompt_lines=build_profile_prompt(
                samples=stage_one_samples,
                stats=stats,
                relationships=relationships,
                target_name=args.target_name,
            ),
        )
        stage1_path.parent.mkdir(parents=True, exist_ok=True)
        stage1_path.write_text(stage_one_text, encoding="utf-8")
        draft = assemble_persona(
            parse_persona_yaml(stage_one_text),
            target_name=args.target_name,
            group_card=args.group_card,
            source_user_id=args.user_id,
            aliases=[args.target_name],
        )
        print("stage1_profile_generated")

    final: dict = draft
    if not args.skip_refine:
        for attempt in range(1, 3):
            try:
                stage_two_text = _complete_responses_nonstream(
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    timeout_seconds=300,
                    prompt_lines=build_refine_prompt(
                        draft_yaml=yaml.safe_dump(
                            draft, allow_unicode=True, sort_keys=False
                        ),
                        fresh_samples=held_out,
                        target_name=args.target_name,
                    ),
                )
                final = assemble_persona(
                    parse_persona_yaml(stage_two_text),
                    target_name=args.target_name,
                    group_card=args.group_card,
                    source_user_id=args.user_id,
                    aliases=[args.target_name],
                )
                break
            except (ValueError, httpx.HTTPError) as exc:
                print(
                    f"stage2_retry attempt={attempt} error={type(exc).__name__} "
                    "falling_back_to_stage1"
                )
                if attempt == 2:
                    final = draft
    final["example_lines"] = deterministic_examples
    final["example_bank"] = example_bank
    final.setdefault(
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
    existing_path = Path(args.output)
    if existing_path.exists():
        existing = yaml.safe_load(existing_path.read_text(encoding="utf-8")) or {}
        for preserved_key in ("facts", "external_relations"):
            if existing.get(preserved_key):
                final[preserved_key] = existing[preserved_key]
    if args.source_group_id > 0:
        final["source_group_id"] = args.source_group_id
    output_path = Path(args.output)
    _write_yaml(output_path, final)
    print(f"persona_written path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
