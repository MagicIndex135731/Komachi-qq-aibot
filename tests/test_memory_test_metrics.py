from scripts.memory_test_metrics import (
    classification_metrics,
    diff_metrics,
    fullchain_metrics,
    percentile,
)


def test_percentile_basic():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 100) == 4.0


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
    assert metrics["kind_recall"]["abstention"] == 1.0
    assert metrics["subject_binding_accuracy"] == 1.0
    assert metrics["retrieval_latency_ms"]["p50"] == 20.0


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


def test_diff_metrics():
    diff = diff_metrics({"recall": 0.5, "nested": {"a": 1}}, {"recall": 0.7, "nested": {"a": 2}})
    assert diff["recall"]["before"] == 0.5
    assert diff["recall"]["after"] == 0.7
    assert diff["nested"]["a"]["delta"] == 1
