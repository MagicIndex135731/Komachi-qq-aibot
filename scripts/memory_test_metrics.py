"""Pure metric functions for the memory test platform.

Kept dependency-free on the app runtime so the aggregator and its tests stay
fast. Formulas follow the existing V3 quality evaluation (citation
precision/recall, abstention P/R/F1, token and TTFT percentiles).
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    index = (len(ordered) - 1) * p / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval-level metrics from offline per-case rows.

    Each row must contain at least:
      category, kind (or tags), expected_evidence_message_ids,
      packed_source_ids, subject_expected, subject_actual,
      raw_hit, fact_hit, summary_hit, expected_layer, latency_ms.
    """
    total = len(rows)
    per_kind: dict[str, dict[str, int]] = {}
    per_layer: dict[str, dict[str, int]] = {}
    subject_correct = 0
    subject_total = 0
    cross_group_violations = 0
    latency_ms: list[float] = []
    for row in rows:
        kind = str(row.get("kind") or row.get("category") or "unknown")
        bucket = per_kind.setdefault(kind, {"hit": 0, "total": 0})
        bucket["total"] += 1
        expected = set(str(v) for v in (row.get("expected_evidence_message_ids") or ()))
        packed = set(str(v) for v in (row.get("packed_source_ids") or ()))
        if expected:
            bucket["hit"] += int(bool(expected & packed))
        else:
            bucket["hit"] += int(not packed)
        expected_layer = str(row.get("expected_layer") or "raw")
        layer_bucket = per_layer.setdefault(expected_layer, {"hit": 0, "total": 0})
        layer_bucket["total"] += 1
        layer_hit_key = f"{expected_layer}_hit"
        layer_bucket["hit"] += int(bool(row.get(layer_hit_key)))
        if row.get("subject_expected") is not None:
            subject_total += 1
            subject_correct += int(row.get("subject_actual") == row.get("subject_expected"))
        if row.get("cross_group_violation"):
            cross_group_violations += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            latency_ms.append(float(latency))
    return {
        "cases": total,
        "kind_recall": {
            kind: (
                bucket["hit"] / bucket["total"]
                if bucket["total"]
                else 0.0
            )
            for kind, bucket in sorted(per_kind.items())
        },
        "layer_hit_rate": {
            layer: (
                bucket["hit"] / bucket["total"]
                if bucket["total"]
                else 0.0
            )
            for layer, bucket in sorted(per_layer.items())
        },
        "subject_binding_accuracy": (
            subject_correct / subject_total if subject_total else None
        ),
        "cross_group_violations": cross_group_violations,
        "retrieval_latency_ms": {
            "p50": percentile(latency_ms, 50),
            "p95": percentile(latency_ms, 95),
        },
    }


def fullchain_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate model-layer metrics from full-chain per-case rows.

    Each row: answer_correct, answer_grounded, abstained, expected_abstention,
    citation_precision, citation_recall, protocol_failure_codes,
    input_tokens, output_tokens, ttft_ms, total_ms, cached.
    """
    total = len(rows)
    if not total:
        return {
            "cases": 0,
            "grounded_answer_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "abstention_f1": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "protocol_failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "ttft_ms": {"p50": 0.0, "p95": 0.0},
            "total_ms": {"p50": 0.0, "p95": 0.0},
            "cache_hits": 0,
        }
    citation_precision = [
        float(row.get("citation_precision", 0.0)) for row in rows
    ]
    citation_recall = [
        float(row.get("citation_recall", 0.0)) for row in rows
    ]
    tp = sum(
        1
        for row in rows
        if bool(row.get("expected_abstention"))
        and bool(row.get("abstained"))
    )
    fp = sum(
        1
        for row in rows
        if not bool(row.get("expected_abstention"))
        and bool(row.get("abstained"))
    )
    fn = sum(
        1
        for row in rows
        if bool(row.get("expected_abstention"))
        and not bool(row.get("abstained"))
    )
    abstention_precision = tp / (tp + fp) if tp + fp else 0.0
    abstention_recall = tp / (tp + fn) if tp + fn else 0.0
    abstention_f1 = (
        2 * abstention_precision * abstention_recall
        / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall
        else 0.0
    )
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    total_ms = [float(row["total_ms"]) for row in rows if row.get("total_ms") is not None]
    return {
        "cases": total,
        "grounded_answer_accuracy": safe_mean(
            [
                float(bool(row.get("answer_grounded")) and bool(row.get("answer_correct")))
                for row in rows
            ]
        ),
        "answer_accuracy": safe_mean(
            [float(bool(row.get("answer_correct"))) for row in rows]
        ),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "citation_precision": safe_mean(citation_precision),
        "citation_recall": safe_mean(citation_recall),
        "protocol_failures": sum(
            bool(row.get("protocol_failure_codes")) for row in rows
        ),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "ttft_ms": {
            "p50": percentile(ttft, 50),
            "p95": percentile(ttft, 95),
        },
        "total_ms": {
            "p50": percentile(total_ms, 50),
            "p95": percentile(total_ms, 95),
        },
        "cache_hits": sum(1 for row in rows if row.get("cached")),
    }


def diff_metrics(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a flat before/after diff for report rendering."""
    keys = set(baseline) | set(current)
    diff: dict[str, Any] = {}
    for key in sorted(keys):
        old = baseline.get(key)
        new = current.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            diff[key] = diff_metrics(old, new)
        elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
            diff[key] = {"before": old, "after": new, "delta": round(new - old, 6)}
        else:
            diff[key] = {"before": old, "after": new}
    return diff


def category_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category") or "unknown") for row in rows))
