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


def _abstention_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    expectation_aware: bool,
) -> tuple[float, float, float, int]:
    """Return abstention P/R/F1 and the number of labeled rows.

    Legacy rows have no ``answer_expectation`` and retain their historical
    ``expected_abstention`` behavior. New ``either`` rows are excluded only
    from the expectation-aware view because both answering and abstaining are
    valid outcomes for those cases.
    """

    labeled: list[tuple[bool, bool]] = []
    for row in rows:
        expectation = row.get("answer_expectation")
        if not expectation_aware:
            expected_abstention = _legacy_expected_abstention(row)
        elif expectation is None:
            expected_abstention = bool(row.get("expected_abstention"))
        else:
            expectation = str(expectation)
            if expectation not in {"must_answer", "must_abstain", "either"}:
                raise ValueError(f"invalid answer_expectation: {expectation!r}")
            if expectation == "either":
                continue
            expected_abstention = expectation == "must_abstain"
        labeled.append((expected_abstention, bool(row.get("abstained"))))

    tp = sum(expected and actual for expected, actual in labeled)
    fp = sum(not expected and actual for expected, actual in labeled)
    fn = sum(expected and not actual for expected, actual in labeled)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1, len(labeled)


def _legacy_expected_abstention(row: Mapping[str, Any]) -> bool:
    """Resolve the v1 abstention label without changing the v2 contract.

    New v2 rows carry ``answer_expectation=either`` for real mentions.  The
    historical empty-gold metric classified those rows as expected abstentions,
    whereas the expectation-aware metric deliberately excludes them.  Prefer
    an explicit legacy projection when a producer provides one; otherwise
    reproduce the historical empty-gold rule for ``either`` rows.
    """

    if "legacy_expected_abstention" in row:
        return bool(row.get("legacy_expected_abstention"))
    if str(row.get("answer_expectation") or "") == "either":
        return True
    return bool(row.get("expected_abstention"))


def classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval-level metrics from offline per-case rows.

    Each row must contain at least:
      category, kind (or tags), expected_evidence_message_ids,
      packed_source_ids, subject_expected, subject_actual,
      raw_hit, fact_hit, summary_hit, expected_layer, latency_ms.
    """
    total = len(rows)
    per_kind: dict[str, dict[str, int]] = {}
    legacy_per_kind: dict[str, dict[str, int]] = {}
    per_layer: dict[str, dict[str, int]] = {}
    subject_correct = 0
    subject_total = 0
    cross_group_violations = 0
    latency_ms: list[float] = []
    for row in rows:
        kind = str(row.get("kind") or row.get("category") or "unknown")
        legacy_bucket = legacy_per_kind.setdefault(
            kind, {"hit": 0, "total": 0}
        )
        legacy_bucket["total"] += 1
        expectation = str(row.get("answer_expectation") or "")
        if expectation and expectation not in {"must_answer", "must_abstain", "either"}:
            raise ValueError(f"invalid answer_expectation: {expectation!r}")
        bucket = None
        # Current kind recall is a positive-recall metric. Explicit v2
        # negative/optional cases are evaluated by abstention and precision
        # metrics instead of treating any retrieved candidate as a miss.
        # Legacy rows without the field retain the historical projection.
        if expectation not in {"must_abstain", "either"}:
            bucket = per_kind.setdefault(kind, {"hit": 0, "total": 0})
            bucket["total"] += 1
        expected = set(str(v) for v in (row.get("expected_evidence_message_ids") or ()))
        packed = set(str(v) for v in (row.get("packed_source_ids") or ()))
        answerable = (
            set(str(v) for v in (row.get("allowed_citation_ids") or ()))
            if "allowed_citation_ids" in row
            else packed
        )
        if expected:
            if bucket is not None:
                bucket["hit"] += int(bool(expected & answerable))
            legacy_bucket["hit"] += int(bool(expected & packed))
        else:
            no_retrieved_layer = not (
                bool(row.get("raw_hit"))
                or bool(row.get("fact_hit"))
                or bool(row.get("summary_hit"))
            )
            if bucket is not None:
                bucket["hit"] += int(no_retrieved_layer)
            legacy_bucket["hit"] += int(no_retrieved_layer)
        expected_layer = str(row.get("expected_layer") or "raw")
        layer_bucket = per_layer.setdefault(expected_layer, {"hit": 0, "total": 0})
        layer_bucket["total"] += 1
        if expected_layer == "none":
            layer_bucket["hit"] += int(
                not (
                    bool(row.get("raw_hit"))
                    or bool(row.get("fact_hit"))
                    or bool(row.get("summary_hit"))
                )
            )
        else:
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
        "legacy_kind_recall": {
            kind: (
                bucket["hit"] / bucket["total"]
                if bucket["total"]
                else 0.0
            )
            for kind, bucket in sorted(legacy_per_kind.items())
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
    optional answer_expectation, citation_precision, citation_recall,
    protocol_failure_codes,
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
            "abstention_evaluable_cases": 0,
            "abstention_either_cases": 0,
            "legacy_abstention_precision": 0.0,
            "legacy_abstention_recall": 0.0,
            "legacy_abstention_f1": 0.0,
            "legacy_abstention_evaluable_cases": 0,
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
    (
        abstention_precision,
        abstention_recall,
        abstention_f1,
        abstention_evaluable_cases,
    ) = _abstention_scores(rows, expectation_aware=True)
    (
        legacy_abstention_precision,
        legacy_abstention_recall,
        legacy_abstention_f1,
        legacy_abstention_evaluable_cases,
    ) = _abstention_scores(rows, expectation_aware=False)
    abstention_either_cases = sum(
        str(row.get("answer_expectation") or "") == "either" for row in rows
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
        "abstention_evaluable_cases": abstention_evaluable_cases,
        "abstention_either_cases": abstention_either_cases,
        "legacy_abstention_precision": legacy_abstention_precision,
        "legacy_abstention_recall": legacy_abstention_recall,
        "legacy_abstention_f1": legacy_abstention_f1,
        "legacy_abstention_evaluable_cases": legacy_abstention_evaluable_cases,
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


def _without_keys(
    metrics: Mapping[str, Any], excluded: set[str]
) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in excluded}


def _prefixed_metrics(
    metrics: Mapping[str, Any], prefix: str
) -> dict[str, Any]:
    return {
        key: value for key, value in metrics.items() if key.startswith(prefix)
    }


def _unavailable_comparison(
    current: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Make an incompatible baseline explicit instead of diffing it anyway."""

    return {
        "status": "not_comparable",
        "reason": reason,
        "before": None,
        "after": dict(current),
    }


def offline_baseline_diff(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Diff offline reports without mixing packed-ID and answerable recall.

    Older reports stored only ``kind_recall``. That field was calculated from
    packed source IDs, while current ``kind_recall`` is calculated from IDs
    that are legal citations. The legacy projection preserves the old
    packed-ID definition, so old baselines can only be compared there.
    """

    excluded = {"kind_recall", "legacy_kind_recall"}
    baseline_has_legacy = "legacy_kind_recall" in baseline
    current_kind = current.get("kind_recall") or {}
    current_legacy_kind = current.get("legacy_kind_recall") or {}

    if baseline_has_legacy and "kind_recall" in baseline:
        current_diff: Any = diff_metrics(
            baseline.get("kind_recall") or {}, current_kind
        )
    else:
        current_diff = _unavailable_comparison(
            current_kind,
            reason=(
                "baseline kind_recall uses the legacy packed-ID definition; "
                "it cannot be compared with current allowed-citation recall"
            ),
        )

    baseline_legacy_kind = (
        baseline.get("legacy_kind_recall")
        if baseline_has_legacy
        else baseline.get("kind_recall")
    )
    if isinstance(baseline_legacy_kind, Mapping):
        legacy_diff: Any = diff_metrics(baseline_legacy_kind, current_legacy_kind)
    else:
        legacy_diff = _unavailable_comparison(
            current_legacy_kind,
            reason="baseline has no packed-ID kind recall",
        )

    return {
        "shared": diff_metrics(
            _without_keys(baseline, excluded), _without_keys(current, excluded)
        ),
        "current": {"allowed_citation_kind_recall": current_diff},
        "legacy": {"packed_ids_kind_recall": legacy_diff},
    }


def fullchain_baseline_diff(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Diff fullchain reports without mixing abstention-label definitions.

    Old ``abstention_*`` fields use the empty-gold label. Current reports
    retain that definition under ``legacy_abstention_*`` and expose a separate
    expectation-aware family under ``abstention_*``. A dual-field baseline
    compares each family independently; an old baseline is explicitly marked
    unavailable for the current family.
    """

    abstention_prefix = "abstention_"
    legacy_prefix = "legacy_abstention_"
    excluded = {
        key
        for metrics in (baseline, current)
        for key in metrics
        if key.startswith(abstention_prefix) or key.startswith(legacy_prefix)
    }
    baseline_current = _prefixed_metrics(baseline, abstention_prefix)
    baseline_legacy = _prefixed_metrics(baseline, legacy_prefix)
    current_current = _prefixed_metrics(current, abstention_prefix)
    current_legacy = _prefixed_metrics(current, legacy_prefix)

    if baseline_legacy:
        current_diff: Any = diff_metrics(baseline_current, current_current)
        legacy_diff: Any = diff_metrics(baseline_legacy, current_legacy)
    else:
        current_diff = _unavailable_comparison(
            current_current,
            reason=(
                "baseline abstention_* uses the legacy empty-gold definition; "
                "it cannot be compared with current expectation-aware abstention"
            ),
        )
        projected_baseline_legacy = {
            f"legacy_{key}": value for key, value in baseline_current.items()
        }
        if projected_baseline_legacy:
            legacy_diff = diff_metrics(projected_baseline_legacy, current_legacy)
        else:
            legacy_diff = _unavailable_comparison(
                current_legacy,
                reason="baseline has no empty-gold abstention metrics",
            )

    return {
        "shared": diff_metrics(
            _without_keys(baseline, excluded), _without_keys(current, excluded)
        ),
        "current": {"expectation_aware_abstention": current_diff},
        "legacy": {"empty_gold_abstention": legacy_diff},
    }


def category_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category") or "unknown") for row in rows))


def dataset_coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize dataset stratification labels for a coverage report."""

    categories: Counter[str] = Counter(
        str(case.get("category") or "unknown") for case in cases
    )
    kinds: Counter[str] = Counter(
        str(case.get("kind") or "") for case in cases
    )
    layers: Counter[str] = Counter(
        str(case.get("expected_layer") or "unknown") for case in cases
    )
    expectations: Counter[str] = Counter(
        str(case.get("answer_expectation") or "unknown") for case in cases
    )
    subject_states: Counter[str] = Counter()
    time_intents: Counter[str] = Counter()
    for case in cases:
        for tag in (case.get("tags") or case.get("gate_tags") or ()):
            tag_text = str(tag)
            if tag_text.startswith("subject="):
                subject_states[tag_text] += 1
            elif tag_text.startswith("intent="):
                time_intents[tag_text] += 1
    return {
        "total": len(cases),
        "by_category": dict(sorted(categories.items())),
        "by_kind": dict(sorted(kinds.items())),
        "by_expected_layer": dict(sorted(layers.items())),
        "by_answer_expectation": dict(sorted(expectations.items())),
        "subject_states": dict(sorted(subject_states.items())),
        "time_intents": dict(sorted(time_intents.items())),
        "distractor_count": int(categories["distractor"]),
        "cross_group_count": int(categories["cross_group"]),
    }
