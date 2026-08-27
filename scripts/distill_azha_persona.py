"""Distill one member's chat style into a local persona profile.

Usage:
  python scripts/distill_azha_persona.py \
    --history-dir data/history --group-id <GROUP_ID> --user-id <MEMBER_QQ> \
    --target-name <DISPLAY_NAME> --group-card <GROUP_CARD> \
    [--samples-only]

Outputs are local-only persona artifacts (private-chat derived style):
  configs/personas/azha.yaml
  data/personas/azha/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from app.config import AppSettings
from app.core.style_distill import (
    assemble_persona,
    build_distill_prompt,
    build_style_samples,
    load_history_records,
    parse_persona_yaml,
)
from app.providers.llm_client import LlmClient


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
    parser.add_argument("--history-dir", default="data/history")
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--group-card", default="")
    parser.add_argument("--output", default="configs/personas/azha.yaml")
    parser.add_argument("--samples", default="data/personas/azha/samples.jsonl")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--context-before", type=int, default=6)
    parser.add_argument("--context-after", type=int, default=3)
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()

    records = load_history_records(
        history_dir=Path(args.history_dir),
        group_id=args.group_id,
    )
    samples = build_style_samples(
        records=records,
        user_id=args.user_id,
        context_before=args.context_before,
        context_after=args.context_after,
        max_samples=args.max_samples,
    )
    samples_path = Path(args.samples)
    _write_jsonl(samples_path, samples)
    print(f"samples={len(samples)} path={samples_path}")
    if args.samples_only:
        return 0
    if not samples:
        raise SystemExit("no usable samples extracted; aborting distillation")
    if not args.target_name.strip() or not args.group_card.strip():
        raise SystemExit("--target-name and --group-card are required unless --samples-only")

    settings = AppSettings()
    client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_only=True,
        responses_model=settings.llm_model,
        max_output_tokens=8192,
        timeout_seconds=settings.llm_timeout_seconds,
        reasoning_effort=settings.llm_reasoning_effort,
    )
    prompt_lines = build_distill_prompt(
        samples=samples,
        target_name=args.target_name,
    )
    generated = client.generate_text(prompt_lines)
    profile = parse_persona_yaml(generated)
    persona = assemble_persona(
        profile,
        target_name=args.target_name,
        group_card=args.group_card,
        source_user_id=args.user_id,
        aliases=[args.target_name],
    )
    output_path = Path(args.output)
    _write_yaml(output_path, persona)
    print(f"persona_written path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
