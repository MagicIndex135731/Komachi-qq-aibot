from scripts.memory_test_metrics import (
    classification_metrics,
    dataset_coverage,
    diff_metrics,
    fullchain_baseline_diff,
    fullchain_metrics,
    offline_baseline_diff,
    percentile,
)


def test_percentile_basic():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 100) == 4.0


def test_dataset_coverage_summarizes_stratification_labels():
    cases = [
        {
            "category": "profile",
            "kind": "profile",
            "expected_layer": "fact",
            "answer_expectation": "must_answer",
            "tags": ("subject=20001", "intent=profile", "layer=fact"),
        },
        {
            "category": "distractor",
            "kind": "distractor",
            "expected_layer": "raw",
            "answer_expectation": "must_abstain",
            "tags": ("subject=20002", "intent=distractor"),
        },
        {
            "category": "cross_group",
            "kind": "cross_group",
            "expected_layer": "raw",
            "answer_expectation": "either",
            "gate_tags": ("subject=20003", "intent=cross_group"),
        },
    ]
    report = dataset_coverage(cases)
    assert report["total"] == 3
    assert report["by_category"] == {
        "cross_group": 1,
        "distractor": 1,
        "profile": 1,
    }
    assert report["by_kind"] == {
        "cross_group": 1,
        "distractor": 1,
        "profile": 1,
    }
    assert report["by_expected_layer"] == {"fact": 1, "raw": 2}
    assert report["by_answer_expectation"] == {
        "either": 1,
        "must_abstain": 1,
        "must_answer": 1,
    }
    assert report["subject_states"] == {
        "subject=20001": 1,
        "subject=20002": 1,
        "subject=20003": 1,
    }
    assert report["distractor_count"] == 1
    assert report["cross_group_count"] == 1


def test_classification_metrics():
    rows = [
        {
            "category": "preference",
            "kind": "preference",
            "expected_layer": "fact",
            "expected_evidence_message_ids": ("a", "b"),
            "packed_source_ids": ("a", "c"),
            "fact_hit": True,
            "raw_hit": False,
            "summary_hit": False,
            "subject_expected": ("1",),
            "subject_actual": ("1",),
            "cross_group_violation": False,
            "latency_ms": 10.0,
        },
        {
            "category": "abstention",
            "kind": "abstention",
            "expected_layer": "none",
            "expected_evidence_message_ids": (),
            "packed_source_ids": (),
            "fact_hit": False,
            "raw_hit": False,
            "summary_hit": False,
            "subject_expected": None,
            "subject_actual": None,
            "cross_group_violation": False,
            "latency_ms": 30.0,
        },
    ]
    metrics = classification_metrics(rows)
    assert metrics["cases"] == 2
    assert metrics["kind_recall"]["preference"] == 1.0
    assert metrics["legacy_kind_recall"]["preference"] == 1.0
    assert metrics["kind_recall"]["abstention"] == 1.0
    assert metrics["subject_binding_accuracy"] == 1.0
    assert metrics["retrieval_latency_ms"]["p50"] == 20.0


def test_classification_metrics_prefer_answerable_sources_and_dual_report_legacy():
    rows = [
        {
            "category": "profile",
            "kind": "profile",
            "expected_layer": "fact",
            "expected_evidence_message_ids": ("gold",),
            # The legacy packed set includes recent fallback sources.
            "packed_source_ids": ("gold", "recent"),
            # Only this set is visible to the answer as legal evidence.
            "allowed_citation_ids": ("other",),
            "fact_hit": True,
        },
        {
            "category": "profile",
            "kind": "profile",
            "expected_layer": "fact",
            "expected_evidence_message_ids": ("gold",),
            "packed_source_ids": ("gold",),
            "allowed_citation_ids": ("gold",),
            "fact_hit": True,
        },
    ]

    metrics = classification_metrics(rows)

    assert metrics["kind_recall"]["profile"] == 0.5
    assert metrics["legacy_kind_recall"]["profile"] == 1.0


def test_classification_metrics_exclude_either_from_current_kind_recall():
    rows = [
        {
            "category": "mention",
            "kind": "mention",
            "answer_expectation": "either",
            "expected_layer": "raw",
            "expected_evidence_message_ids": (),
            "packed_source_ids": ("optional",),
            "allowed_citation_ids": ("optional",),
            "raw_hit": True,
            "fact_hit": False,
            "summary_hit": False,
        }
    ]

    metrics = classification_metrics(rows)

    assert "mention" not in metrics["kind_recall"]
    assert metrics["legacy_kind_recall"]["mention"] == 0.0


def test_classification_metrics_exclude_must_abstain_from_current_kind_recall():
    rows = [
        {
            "category": "distractor",
            "kind": "distractor",
            "answer_expectation": "must_abstain",
            "expected_layer": "none",
            "expected_evidence_message_ids": (),
            "packed_source_ids": ("irrelevant",),
            "allowed_citation_ids": ("irrelevant",),
            "raw_hit": True,
            "fact_hit": False,
            "summary_hit": False,
        }
    ]

    metrics = classification_metrics(rows)

    assert "distractor" not in metrics["kind_recall"]
    assert metrics["legacy_kind_recall"]["distractor"] == 0.0


def test_fullchain_metrics():
    rows = [
        {
            "answer_correct": True,
            "answer_grounded": True,
            "abstained": False,
            "expected_abstention": False,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "protocol_failure_codes": [],
            "input_tokens": 100,
            "output_tokens": 20,
            "ttft_ms": 50.0,
            "total_ms": 200.0,
            "cached": False,
        },
        {
            "answer_correct": False,
            "answer_grounded": False,
            "abstained": False,
            "expected_abstention": False,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "protocol_failure_codes": ["citation_outside_packet"],
            "input_tokens": 90,
            "output_tokens": 10,
            "ttft_ms": 80.0,
            "total_ms": 300.0,
            "cached": True,
        },
    ]
    metrics = fullchain_metrics(rows)
    assert metrics["cases"] == 2
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["protocol_failures"] == 1
    assert metrics["cache_hits"] == 1
    assert metrics["ttft_ms"]["p50"] == 65.0
    assert metrics["input_tokens"] == 190


def test_fullchain_abstention_metrics_exclude_either_and_dual_report_legacy():
    rows = [
        {
            "answer_expectation": "must_abstain",
            "expected_abstention": True,
            "abstained": True,
        },
        {
            "answer_expectation": "must_answer",
            "expected_abstention": False,
            "abstained": True,
        },
        {
            "answer_expectation": "either",
            # v2 rows no longer label either as must_abstain.  The legacy
            # projection must still reproduce v1's empty-gold label.
            "expected_abstention": False,
            "abstained": False,
        },
    ]

    metrics = fullchain_metrics(rows)

    assert metrics["abstention_evaluable_cases"] == 2
    assert metrics["abstention_either_cases"] == 1
    assert metrics["abstention_precision"] == 0.5
    assert metrics["abstention_recall"] == 1.0
    assert metrics["abstention_f1"] == 2 / 3
    assert metrics["legacy_abstention_precision"] == 0.5
    assert metrics["legacy_abstention_recall"] == 0.5
    assert metrics["legacy_abstention_f1"] == 0.5
    assert metrics["legacy_abstention_evaluable_cases"] == 3


def test_fullchain_legacy_abstention_uses_explicit_projection_when_present():
    metrics = fullchain_metrics(
        [
            {
                "answer_expectation": "either",
                "expected_abstention": False,
                "legacy_expected_abstention": False,
                "abstained": False,
            }
        ]
    )

    assert metrics["abstention_evaluable_cases"] == 0
    assert metrics["legacy_abstention_evaluable_cases"] == 1
    assert metrics["legacy_abstention_recall"] == 0.0


def test_fullchain_abstention_metrics_keep_legacy_rows_comparable():
    rows = [
        {"expected_abstention": True, "abstained": True},
        {"expected_abstention": False, "abstained": False},
    ]

    metrics = fullchain_metrics(rows)

    assert metrics["abstention_evaluable_cases"] == 2
    assert metrics["abstention_either_cases"] == 0
    assert metrics["abstention_precision"] == 1.0
    assert metrics["abstention_recall"] == 1.0
    assert metrics["abstention_f1"] == 1.0
    assert metrics["legacy_abstention_precision"] == 1.0
    assert metrics["legacy_abstention_recall"] == 1.0
    assert metrics["legacy_abstention_f1"] == 1.0


def test_diff_metrics():
    diff = diff_metrics({"recall": 0.5, "nested": {"a": 1}}, {"recall": 0.7, "nested": {"a": 2}})
    assert diff["recall"]["before"] == 0.5
    assert diff["recall"]["after"] == 0.7
    assert diff["nested"]["a"]["delta"] == 1


def test_legacy_baseline_diff_only_uses_legacy_metric_projections():
    offline = offline_baseline_diff(
        {"kind_recall": {"profile": 0.25}},
        {
            "kind_recall": {"profile": 0.5},
            "legacy_kind_recall": {"profile": 1.0},
        },
    )
    assert (
        offline["current"]["allowed_citation_kind_recall"]["status"]
        == "not_comparable"
    )
    assert offline["legacy"]["packed_ids_kind_recall"]["profile"] == {
        "before": 0.25,
        "after": 1.0,
        "delta": 0.75,
    }

    fullchain = fullchain_baseline_diff(
        {
            "abstention_precision": 0.25,
            "abstention_recall": 0.4,
            "abstention_f1": 0.3,
        },
        {
            "abstention_precision": 0.8,
            "abstention_recall": 0.9,
            "abstention_f1": 0.85,
            "legacy_abstention_precision": 0.5,
            "legacy_abstention_recall": 0.6,
            "legacy_abstention_f1": 0.55,
        },
    )
    assert (
        fullchain["current"]["expectation_aware_abstention"]["status"]
        == "not_comparable"
    )
    assert fullchain["legacy"]["empty_gold_abstention"][
        "legacy_abstention_precision"
    ] == {"before": 0.25, "after": 0.5, "delta": 0.25}


def test_dual_field_baseline_diff_keeps_current_and_legacy_separate():
    baseline = {
        "kind_recall": {"profile": 0.4},
        "legacy_kind_recall": {"profile": 0.7},
    }
    current = {
        "kind_recall": {"profile": 0.6},
        "legacy_kind_recall": {"profile": 0.9},
    }
    offline = offline_baseline_diff(baseline, current)
    assert offline["current"]["allowed_citation_kind_recall"]["profile"] == {
        "before": 0.4,
        "after": 0.6,
        "delta": 0.2,
    }
    assert offline["legacy"]["packed_ids_kind_recall"]["profile"] == {
        "before": 0.7,
        "after": 0.9,
        "delta": 0.2,
    }

    fullchain = fullchain_baseline_diff(
        {
            "abstention_precision": 0.4,
            "legacy_abstention_precision": 0.7,
        },
        {
            "abstention_precision": 0.6,
            "legacy_abstention_precision": 0.9,
        },
    )
    assert fullchain["current"]["expectation_aware_abstention"][
        "abstention_precision"
    ] == {"before": 0.4, "after": 0.6, "delta": 0.2}
    assert fullchain["legacy"]["empty_gold_abstention"][
        "legacy_abstention_precision"
    ] == {"before": 0.7, "after": 0.9, "delta": 0.2}
