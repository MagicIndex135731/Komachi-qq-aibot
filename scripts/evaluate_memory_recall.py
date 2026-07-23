from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Literal


EVALUATION_CATEGORIES = frozenset(
    {
        "exact",
        "paraphrase",
        "vague_reference",
        "temporal",
        "multi_hop",
        "update",
        "abstention",
    }
)
EVALUATION_VARIANTS = frozenset({"v1", "v2"})
REAL_DATASET_MIN_CASES = 64
REAL_DATASET_MAX_CASES = 100
REAL_DATASET_CATEGORY_QUOTAS = {
    "exact": 10,
    "paraphrase": 10,
    "vague_reference": 10,
    "temporal": 10,
    "multi_hop": 8,
    "update": 8,
    "abstention": 8,
}
REAL_DATASET_REVIEW_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    group_id: int
    query: str
    recent_context_message_ids: tuple[str, ...]
    expected_evidence_message_ids: tuple[str, ...]
    category: str
    time_range: tuple[str, str] | None = None
    quoted_context_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_index: int
    variant: Literal["v1", "v2"]
    retrieved_evidence_message_ids: tuple[str, ...]
    packed_evidence_message_ids: tuple[str, ...]
    context_tokens: int
    latency_ms: float
    rewrite_used: bool
    retrieved_evidence_units: tuple[tuple[str, ...], ...] | None = None


def _parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected object at line {line_number}")
        records.append(value)
    if not records:
        raise ValueError("evaluation JSONL must contain at least one case")
    return records, raw


def _require_exact_fields(record: Mapping[str, Any], *, required: set[str], optional: set[str]) -> None:
    fields = set(record)
    missing = required - fields
    unexpected = fields - required - optional
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"unexpected fields: {', '.join(sorted(unexpected))}")


def _parse_message_ids(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    values = tuple(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _parse_retrieval_units(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ValueError("retrieved_evidence_units must be a list")
    units: list[tuple[str, ...]] = []
    for index, unit in enumerate(value):
        parsed = _parse_message_ids(unit, field=f"retrieved_evidence_units[{index}]")
        if not parsed:
            raise ValueError("retrieved_evidence_units must not contain empty units")
        units.append(parsed)
    return tuple(units)


def _parse_time_range(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"start_at", "end_at"}:
        raise ValueError("time_range must contain exactly start_at and end_at")
    start_at = value["start_at"]
    end_at = value["end_at"]
    if not isinstance(start_at, str) or not start_at or not isinstance(end_at, str) or not end_at:
        raise ValueError("time_range values must be non-empty strings")
    return start_at, end_at


def parse_evaluation_case(record: Mapping[str, Any]) -> EvaluationCase:
    _require_exact_fields(
        record,
        required={
            "group_id",
            "query",
            "recent_context_message_ids",
            "expected_evidence_message_ids",
            "category",
        },
        optional={"time_range", "quoted_context_message_id"},
    )
    group_id = record["group_id"]
    query = record["query"]
    category = record["category"]
    if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id <= 0:
        raise ValueError("group_id must be a positive integer")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(category, str) or category not in EVALUATION_CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(sorted(EVALUATION_CATEGORIES))}")
    time_range = _parse_time_range(record["time_range"]) if "time_range" in record else None
    quoted_context_message_id = record.get("quoted_context_message_id")
    if quoted_context_message_id is not None and (
        not isinstance(quoted_context_message_id, str)
        or not quoted_context_message_id
    ):
        raise ValueError("quoted_context_message_id must be a non-empty string")
    return EvaluationCase(
        group_id=group_id,
        query=query,
        recent_context_message_ids=_parse_message_ids(record["recent_context_message_ids"], field="recent_context_message_ids"),
        expected_evidence_message_ids=_parse_message_ids(
            record["expected_evidence_message_ids"], field="expected_evidence_message_ids"
        ),
        category=category,
        time_range=time_range,
        quoted_context_message_id=quoted_context_message_id,
    )


def load_evaluation_cases(path: Path | str) -> tuple[tuple[EvaluationCase, ...], str]:
    records, raw = _parse_jsonl(Path(path))
    return tuple(parse_evaluation_case(record) for record in records), hashlib.sha256(raw).hexdigest()


def validate_real_dataset(cases: Sequence[EvaluationCase]) -> None:
    if not REAL_DATASET_MIN_CASES <= len(cases) <= REAL_DATASET_MAX_CASES:
        raise ValueError(
            f"real dataset must contain {REAL_DATASET_MIN_CASES}..{REAL_DATASET_MAX_CASES} cases"
        )
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.category] += 1
    missing = {
        category: quota - counts[category]
        for category, quota in REAL_DATASET_CATEGORY_QUOTAS.items()
        if counts[category] < quota
    }
    if missing:
        rendered = ", ".join(
            f"{category}:{count}" for category, count in sorted(missing.items())
        )
        raise ValueError(f"real dataset category quota is not met: {rendered}")
    for case in cases:
        overlap = set(case.expected_evidence_message_ids) & set(case.recent_context_message_ids)
        if overlap:
            raise ValueError("real dataset evidence must be separate from recent context")
        if case.quoted_context_message_id in set(
            (*case.expected_evidence_message_ids, *case.recent_context_message_ids)
        ):
            raise ValueError("quoted context must be separate from evidence and recent context")
        if case.category == "vague_reference" and not case.quoted_context_message_id:
            raise ValueError("vague-reference cases require quoted context")
        if case.category == "abstention":
            if case.expected_evidence_message_ids:
                raise ValueError("abstention cases must have empty evidence")
        elif not case.expected_evidence_message_ids:
            raise ValueError("non-abstention cases require evidence")


def validate_real_dataset_review(
    cases: Sequence[EvaluationCase],
    *,
    dataset_sha256: str,
    review_path: Path | str,
    database: Path | str,
    snapshot_manifest_sha256: str | None = None,
    snapshot_watermarks: Mapping[int, int] | None = None,
) -> None:
    """Require a human-approved, dataset-bound review sidecar."""
    validate_real_dataset(cases)
    try:
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid real-dataset review sidecar") from exc
    if not isinstance(review, dict) or review.get("review_version") != REAL_DATASET_REVIEW_VERSION:
        raise ValueError("review sidecar has unsupported version")
    if review.get("dataset_sha256") != dataset_sha256:
        raise ValueError("review sidecar dataset hash does not match dataset")
    if (
        snapshot_manifest_sha256 is not None
        and review.get("snapshot_manifest_sha256") != snapshot_manifest_sha256
    ):
        raise ValueError("review sidecar snapshot manifest hash does not match backfill")
    if snapshot_watermarks is not None:
        validate_cases_within_snapshot(
            database,
            cases=cases,
            group_watermarks=snapshot_watermarks,
        )
    rows = review.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("review sidecar must contain one entry per case")
    for index, (case, row) in enumerate(zip(cases, rows)):
        if not isinstance(row, dict):
            raise ValueError(f"review entry {index} is invalid")
        if row.get("case_index") != index or row.get("approved") is not True:
            raise ValueError(f"review entry {index} is not approved")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError(f"review entry {index} has no reviewer")
        if not isinstance(row.get("reviewed_at"), str) or not row["reviewed_at"].strip():
            raise ValueError(f"review entry {index} has no review timestamp")
        if row.get("group_id") != case.group_id or row.get("category") != case.category:
            raise ValueError(f"review entry {index} does not match dataset")
        if row.get("expected_evidence_message_ids") != list(case.expected_evidence_message_ids):
            raise ValueError(f"review entry {index} evidence does not match dataset")
        evidence_sha256 = row.get("evidence_sha256")
        if (
            not isinstance(evidence_sha256, str)
            or len(evidence_sha256) != 64
            or any(character not in "0123456789abcdef" for character in evidence_sha256.lower())
        ):
            raise ValueError(f"review entry {index} has no evidence digest")
        if evidence_sha256.lower() != compute_case_evidence_sha256(database, case):
            raise ValueError(f"review entry {index} evidence digest does not match database")


def validate_cases_within_snapshot(
    database: Path | str,
    *,
    cases: Sequence[EvaluationCase],
    group_watermarks: Mapping[int, int],
) -> None:
    connection = sqlite3.connect(str(database))
    try:
        for index, case in enumerate(cases):
            watermark = group_watermarks.get(int(case.group_id))
            if watermark is None:
                raise ValueError(f"evaluation case {index} group is outside snapshot")
            source_ids = dict.fromkeys(
                (
                    *case.expected_evidence_message_ids,
                    *case.recent_context_message_ids,
                    *(
                        (case.quoted_context_message_id,)
                        if case.quoted_context_message_id
                        else ()
                    ),
                )
            )
            for source_id in source_ids:
                row = connection.execute(
                    "SELECT id, group_id FROM messages WHERE platform_msg_id = ?",
                    (str(source_id),),
                ).fetchone()
                if (
                    row is None
                    or int(row[1] or 0) != int(case.group_id)
                    or int(row[0]) > int(watermark)
                ):
                    raise ValueError(
                        f"evaluation case {index} references a message outside snapshot"
                    )
    finally:
        connection.close()


def compute_case_evidence_sha256(
    database: Path | str,
    case: EvaluationCase,
) -> str:
    contents: list[str] = []
    connection = sqlite3.connect(str(database))
    try:
        for source_id in case.expected_evidence_message_ids:
            row = connection.execute(
                "SELECT group_id, plain_text FROM messages WHERE platform_msg_id = ?",
                (str(source_id),),
            ).fetchone()
            if row is None or int(row[0] or 0) != int(case.group_id):
                raise ValueError("review evidence is missing or outside its group")
            contents.append(str(row[1] or "").strip())
    finally:
        connection.close()
    return hashlib.sha256("\n".join(contents).encode("utf-8")).hexdigest()


def parse_evaluation_result(record: Mapping[str, Any]) -> EvaluationResult:
    _require_exact_fields(
        record,
        required={
            "case_index",
            "variant",
            "retrieved_evidence_message_ids",
            "packed_evidence_message_ids",
            "context_tokens",
            "latency_ms",
            "rewrite_used",
        },
        optional={"retrieved_evidence_units"},
    )
    case_index = record["case_index"]
    variant = record["variant"]
    context_tokens = record["context_tokens"]
    latency_ms = record["latency_ms"]
    rewrite_used = record["rewrite_used"]
    if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 0:
        raise ValueError("case_index must be a non-negative integer")
    if not isinstance(variant, str) or variant not in EVALUATION_VARIANTS:
        raise ValueError("variant must be v1 or v2")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 0:
        raise ValueError("context_tokens must be a non-negative integer")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be a finite non-negative number")
    if not isinstance(rewrite_used, bool):
        raise ValueError("rewrite_used must be a boolean")
    if variant == "v2" and "retrieved_evidence_units" not in record:
        raise ValueError("v2 results require retrieved_evidence_units")
    return EvaluationResult(
        case_index=case_index,
        variant=variant,
        retrieved_evidence_message_ids=_parse_message_ids(
            record["retrieved_evidence_message_ids"], field="retrieved_evidence_message_ids"
        ),
        packed_evidence_message_ids=_parse_message_ids(
            record["packed_evidence_message_ids"], field="packed_evidence_message_ids"
        ),
        context_tokens=context_tokens,
        latency_ms=float(latency_ms),
        rewrite_used=rewrite_used,
        retrieved_evidence_units=(
            _parse_retrieval_units(record["retrieved_evidence_units"])
            if "retrieved_evidence_units" in record
            else None
        ),
    )


def load_evaluation_results(path: Path | str) -> tuple[EvaluationResult, ...]:
    records, _raw = _parse_jsonl(Path(path))
    return tuple(parse_evaluation_result(record) for record in records)


def collect_fixture_results(
    cases: Sequence[EvaluationCase],
    runner: Callable[[EvaluationCase, Literal["v1", "v2"]], EvaluationResult],
) -> tuple[EvaluationResult, ...]:
    """Collect offline adapter fixtures; production adapters are intentionally out of scope."""

    collected: list[EvaluationResult] = []
    for case_index, case in enumerate(cases):
        for variant in ("v1", "v2"):
            result = runner(case, variant)
            if result.case_index != case_index or result.variant != variant:
                raise ValueError("fixture result does not match its requested case and variant")
            collected.append(result)
    return tuple(collected)


def _per_case_metrics(case: EvaluationCase, result: EvaluationResult, *, recall_k: int) -> dict[str, float]:
    expected = set(case.expected_evidence_message_ids)
    retrieval_units = (
        result.retrieved_evidence_units[:recall_k]
        if result.retrieved_evidence_units is not None
        else tuple((source_id,) for source_id in result.retrieved_evidence_message_ids[:recall_k])
    )
    retrieved = {
        source_id
        for unit in retrieval_units
        for source_id in unit
    }
    packed = set(result.packed_evidence_message_ids)
    if not expected:
        correct_abstention = not retrieved
        return {
            "recall_at_k": float(correct_abstention),
            "mrr": float(correct_abstention),
            "ndcg": float(correct_abstention),
            "packed_evidence_hit_rate": float(not packed),
        }

    found: set[str] = set()
    ranked_gains: list[tuple[int, int]] = []
    for index, unit in enumerate(retrieval_units, start=1):
        new_hits = (set(unit) & expected) - found
        if new_hits:
            ranked_gains.append((index, len(new_hits)))
            found.update(new_hits)
    recall = len(found) / len(expected)
    mrr = 1.0 / ranked_gains[0][0] if ranked_gains else 0.0
    dcg = sum(gain / math.log2(index + 1) for index, gain in ranked_gains)
    ideal_dcg = float(len(expected))
    return {
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "packed_evidence_hit_rate": float(bool(packed & expected)),
    }


def _aggregate(cases: Sequence[EvaluationCase], results: Sequence[EvaluationResult], *, recall_k: int) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot aggregate an empty result set")
    metrics = [_per_case_metrics(cases[result.case_index], result, recall_k=recall_k) for result in results]
    count = len(results)
    sorted_latencies = sorted(result.latency_ms for result in results)
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    return {
        "case_count": count,
        "recall_at_k": sum(item["recall_at_k"] for item in metrics) / count,
        "mrr": sum(item["mrr"] for item in metrics) / count,
        "ndcg": sum(item["ndcg"] for item in metrics) / count,
        "packed_evidence_hit_rate": sum(item["packed_evidence_hit_rate"] for item in metrics) / count,
        "mean_context_tokens": sum(result.context_tokens for result in results) / count,
        "mean_latency_ms": sum(result.latency_ms for result in results) / count,
        "p95_latency_ms": sorted_latencies[p95_index],
        "rewrite_rate": sum(result.rewrite_used for result in results) / count,
    }


def evaluate(
    *,
    cases: Sequence[EvaluationCase],
    results: Sequence[EvaluationResult],
    dataset_sha256: str,
    recall_k: int = 10,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot evaluate an empty case set")
    if not isinstance(recall_k, int) or recall_k < 1:
        raise ValueError("recall_k must be a positive integer")
    if len(dataset_sha256) != 64 or any(character not in "0123456789abcdef" for character in dataset_sha256.lower()):
        raise ValueError("dataset_sha256 must be a SHA-256 hex digest")

    grouped: dict[str, list[EvaluationResult]] = {"v1": [], "v2": []}
    seen: set[tuple[int, str]] = set()
    for result in results:
        if result.case_index >= len(cases):
            raise ValueError("result case_index is outside the dataset")
        if result.variant == "v2" and result.retrieved_evidence_units is None:
            raise ValueError("v2 results require retrieved_evidence_units")
        key = (result.case_index, result.variant)
        if key in seen:
            raise ValueError("duplicate result for case and variant")
        seen.add(key)
        grouped[result.variant].append(result)
    expected = {(case_index, variant) for case_index in range(len(cases)) for variant in EVALUATION_VARIANTS}
    if seen != expected:
        raise ValueError("results must contain exactly one v1 and v2 result per case")

    variants: dict[str, Any] = {}
    for variant, variant_results in grouped.items():
        aggregate = _aggregate(cases, variant_results, recall_k=recall_k)
        category_results: dict[str, list[EvaluationResult]] = defaultdict(list)
        for result in variant_results:
            category_results[cases[result.case_index].category].append(result)
        aggregate["categories"] = {
            category: _aggregate(cases, category_group, recall_k=recall_k)
            for category, category_group in sorted(category_results.items())
        }
        group_results: dict[int, list[EvaluationResult]] = defaultdict(list)
        for result in variant_results:
            group_results[cases[result.case_index].group_id].append(result)
        aggregate["groups"] = {
            str(group_id): _aggregate(cases, group_group, recall_k=recall_k)
            for group_id, group_group in sorted(group_results.items())
        }
        variants[variant] = aggregate

    return {
        "dataset_sha256": dataset_sha256,
        "case_count": len(cases),
        "recall_k": recall_k,
        "variants": variants,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate offline V1/V2 memory recall fixtures.")
    parser.add_argument("--dataset", required=True, type=Path, help="Evaluation-case JSONL path")
    parser.add_argument("--results", required=True, type=Path, help="Fixture-result JSONL path")
    parser.add_argument("--output", type=Path, help="Optional safe aggregate report JSON path")
    parser.add_argument("--recall-k", type=int, default=10, help="Evidence recall cutoff (default: 10)")
    parser.add_argument(
        "--enforce-real-dataset",
        action="store_true",
        help="Require the 64..100 case production category quotas",
    )
    parser.add_argument("--review", type=Path, help="Human approval sidecar for a real dataset")
    parser.add_argument("--database", type=Path, help="Source database for evidence digest verification")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    if args.enforce_real_dataset:
        if args.review is None:
            raise ValueError("--review is required with --enforce-real-dataset")
        if args.database is None:
            raise ValueError("--database is required with --enforce-real-dataset")
        validate_real_dataset_review(
            cases,
            dataset_sha256=dataset_sha256,
            review_path=args.review,
            database=args.database,
        )
    report = evaluate(
        cases=cases,
        results=load_evaluation_results(args.results),
        dataset_sha256=dataset_sha256,
        recall_k=args.recall_k,
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
