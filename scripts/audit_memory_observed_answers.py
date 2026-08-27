"""Audit historical bot replies against frozen private retrieval packets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.config import AppSettings
from scripts.memory_test_confidence import (
    _judge_samples,
    _load_latest_rows,
    _majority_decision,
    _replace_judge_answer,
)
from scripts.memory_test_fullchain import (
    DEFAULT_AUX_EFFORT,
    DEFAULT_AUX_MODEL,
    PROVIDER_ATTEMPTS,
    PROVIDER_BACKOFF_SECONDS,
    _build_eval_clients,
)
from scripts.run_memory_v3_quality_replay import ObservedResponsesTransport


CONTRACT_VERSION = "memory-observed-answer-audit-v2"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def public_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if int(row.get("valid_samples") or 0) > 0]
    grounded_correct = sum(bool(row.get("grounded_correct")) for row in valid)
    reason_codes = Counter(
        str(sample.get("decision", {}).get("reason_code") or "")
        for row in rows
        for sample in row.get("judge_samples") or ()
        if isinstance(sample.get("decision"), Mapping)
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "cases": len(rows),
        "evaluable_cases": len(valid),
        "grounded_correct": grounded_correct,
        "grounded_correct_rate": round(grounded_correct / len(valid), 6) if valid else 0.0,
        "incorrect_cases": len(valid) - grounded_correct,
        "provider_failures": sum(
            bool(sample.get("provider_error"))
            for row in rows
            for sample in row.get("judge_samples") or ()
        ),
        "protocol_failures": sum(
            bool(sample.get("protocol_error"))
            for row in rows
            for sample in row.get("judge_samples") or ()
        ),
        "judge_reason_codes": dict(sorted(reason_codes.items())),
    }


def _build_observed_audit_prompt(
    prompt_lines: Sequence[str], *, observed_answer: str
) -> list[str]:
    prompt = _replace_judge_answer(
        prompt_lines,
        answer=observed_answer,
        cited_ids=(),
        abstained=False,
    )
    prompt.append(
        "Historical observed-answer content-audit override: the original group reply did not "
        "contain structured evaluator citations. Do not penalize empty citation IDs, and do not "
        "pretend the retrieved packet was cited by the original reply. Judge answer_grounded and "
        "answer_correct only by whether every factual claim in the observed answer is directly "
        "supported by the frozen retrieved packet and category contract."
    )
    return prompt


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge historical observed answers; private output contains chat text."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--input-detail", required=True, type=Path)
    parser.add_argument("--output-private", required=True, type=Path)
    parser.add_argument("--output-public", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--judge-model", default=DEFAULT_AUX_MODEL)
    parser.add_argument("--judge-effort", default=DEFAULT_AUX_EFFORT)
    parser.add_argument("--provider-attempts", type=int, default=PROVIDER_ATTEMPTS)
    parser.add_argument("--provider-backoff", type=float, default=PROVIDER_BACKOFF_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    cases = {
        str(row["case_id"]): row
        for row in _load_jsonl(args.dataset)
        if row.get("observed_answer")
    }
    selected_case_ids = {
        value.strip() for value in args.case_ids.split(",") if value.strip()
    }
    if selected_case_ids:
        cases = {
            case_id: case
            for case_id, case in cases.items()
            if case_id in selected_case_ids
        }
    details = {str(row["case_id"]): row for row in _load_latest_rows(args.input_detail)}
    missing = sorted(set(cases).difference(details))
    if missing:
        raise ValueError(f"missing frozen detail rows: {len(missing)}")
    settings = AppSettings()
    _, judge_client = _build_eval_clients(
        settings,
        answer_model=DEFAULT_AUX_MODEL,
        answer_effort=DEFAULT_AUX_EFFORT,
        aux_model=args.judge_model,
        aux_effort=args.judge_effort,
    )
    transport = ObservedResponsesTransport(judge_client, max_attempts=args.provider_attempts)
    private_rows: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        detail = details[case_id]
        if (
            str(case.get("query") or "") != str(detail.get("query") or "")
            or str(case.get("category") or "") != str(detail.get("category") or "")
            or int(case.get("group_id") or 0) != int(detail.get("group_id") or 0)
        ):
            raise ValueError(f"dataset/detail identity mismatch for case {case_id}")
        prompt = _build_observed_audit_prompt(
            detail["judge_prompt_full"], observed_answer=str(case["observed_answer"])
        )
        samples = _judge_samples(
            prompt=prompt,
            baseline_decision=None,
            repeats=max(1, args.judge_repeats),
            axis="observed-answer|" + case_id,
            transport=transport,
            model=args.judge_model,
            effort=args.judge_effort,
            cache_dir=args.cache_dir,
            provider_attempts=args.provider_attempts,
            provider_backoff=args.provider_backoff,
        )
        majority = _majority_decision(samples)
        private_rows.append(
            {
                "case_id": case_id,
                "category": case.get("category"),
                "observed_answer": case["observed_answer"],
                "observed_answer_message_id": case.get("observed_answer_message_id"),
                "audit_citation_ids": [],
                "audit_mode": "packet_support_without_historical_citations",
                "judge_samples": samples,
                **majority,
            }
        )
    args.output_private.parent.mkdir(parents=True, exist_ok=True)
    args.output_private.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in private_rows),
        encoding="utf-8",
    )
    public = public_summary(private_rows)
    args.output_public.parent.mkdir(parents=True, exist_ok=True)
    args.output_public.write_text(
        json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not public["provider_failures"] and not public["protocol_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
