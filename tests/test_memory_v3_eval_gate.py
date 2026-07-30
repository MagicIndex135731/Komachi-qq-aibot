from __future__ import annotations

from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from scripts import run_memory_recall_eval as recall_runner
from scripts.evaluate_memory_recall import EvaluationCase, EvaluationResult, evaluate
from scripts.evaluate_memory_v3 import (
    MessageMetadata,
    audit_v3_quality_sources,
    build_v3_observation,
    evaluate_v3,
    load_v3_quality_sidecar,
    quality_sidecar_template,
    retrieval_fingerprint_sha256,
    validate_v3_dataset_contract,
    validate_v3_dataset_sources,
)


def _v3_case(**updates) -> EvaluationCase:
    values = {
        "group_id": 100,
        "query": "private query that must never be reported",
        "recent_context_message_ids": ("recent",),
        "expected_evidence_message_ids": (),
        "category": "abstention",
        "time_range": (
            "2026-07-28T00:00:00+08:00",
            "2026-07-29T00:00:00+08:00",
        ),
        "schema_version": 3,
        "requester_uin": "42",
        "allowed_subject_user_ids": ("42",),
        "allowed_evidence_user_ids": ("42",),
        "expected_answer_mode": "mention",
        "expected_coverage_strategy": "time_buckets",
        "minimum_time_bucket_count": 2,
        "forbidden_evidence_message_ids": (
            "cross-group",
            "blocked",
            "wrong-time",
        ),
        "gate_tags": (
            "abstention",
            "blocked_reserved",
            "cross_group",
            "first_person",
            "mention",
            "source_resolution",
            "subject",
            "time_bucket",
            "time_range",
        ),
        "contract_fields_complete": True,
    }
    values.update(updates)
    return EvaluationCase(**values)


def _meta(
    source_id: str,
    *,
    row_id: int,
    group_id: int = 100,
    user_id: str = "42",
    timestamp: datetime = datetime(2026, 7, 28, 8, tzinfo=UTC),
    delivery_state: str = "",
) -> MessageMetadata:
    return MessageMetadata(
        row_id=row_id,
        source_message_id=source_id,
        group_id=group_id,
        user_id=user_id,
        timestamp=timestamp,
        delivery_state=delivery_state,
    )


@pytest.mark.parametrize(
    ("target", "invalid_value", "expected_code"),
    [
        ("answer_accuracy", float("nan"), "AC_ANSWER_ACCURACY"),
        ("citation_precision", float("inf"), "AC_CITATION_PRECISION"),
        ("ttft_p95_ms", float("-inf"), "AC_TTFT_P95"),
        ("benchmark_p95", True, "AC_RETRIEVAL_P95"),
    ],
)
def test_v3_acceptance_rejects_non_finite_metrics(
    target,
    invalid_value,
    expected_code,
) -> None:
    zero_names = (
        "group_leak_count",
        "subject_leak_count",
        "time_leak_count",
        "ineligible_source_count",
        "unresolved_source_count",
        "outside_snapshot_source_count",
        "forbidden_source_count",
        "plan_mismatch_count",
        "derived_evidence_count",
        "retrieval_over_150_count",
        "packet_over_150_count",
        "packet_over_24k_count",
        "recent_over_60_count",
        "citation_not_in_packet_count",
        "citation_forbidden_source_count",
        "citation_unresolved_source_count",
        "citation_group_leak_count",
        "citation_subject_leak_count",
        "citation_time_leak_count",
        "citation_ineligible_source_count",
    )
    metrics = {
        **{name: 0 for name in zero_names},
        "recall_at_150": 1.0,
        "recall_within_24k": 1.0,
        "time_bucket_coverage_rate": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "grounded_answer_accuracy": 1.0,
        "answer_accuracy": 1.0,
        "abstention_f1": 1.0,
        "index_visibility_p95_ms": 100.0,
        "ttft_p95_ms": 100.0,
    }
    benchmark = {
        "p95_latency_ms": 100.0,
        "rerank_enabled": False,
        "network_enabled": False,
        "vector_success_verified": True,
    }
    if target == "benchmark_p95":
        benchmark["p95_latency_ms"] = invalid_value
    else:
        metrics[target] = invalid_value

    failures = recall_runner._v3_acceptance_failures(
        report={"metrics": metrics},
        benchmark=benchmark,
    )

    assert expected_code in failures


def test_v3_dataset_contract_requires_explicit_structural_gate_coverage() -> None:
    case = _v3_case()

    counts = validate_v3_dataset_contract((case,))

    assert counts["cross_group"] == 1
    with pytest.raises(ValueError, match="scope field"):
        validate_v3_dataset_contract(
            (EvaluationCase(100, "q", (), (), "abstention", schema_version=3),)
        )


def test_v3_dataset_sources_prove_real_cross_group_subject_time_and_safety_distractors() -> None:
    case = _v3_case()
    metadata = {
        "cross-group": _meta("cross-group", row_id=1, group_id=200),
        "blocked": _meta(
            "blocked",
            row_id=2,
            user_id="99",
            delivery_state="blocked",
        ),
        "wrong-time": _meta(
            "wrong-time",
            row_id=3,
            user_id="99",
            timestamp=datetime(2026, 7, 30, tzinfo=UTC),
        ),
    }

    validate_v3_dataset_sources(
        (case,),
        metadata=metadata,
        snapshot_watermarks={100: 10, 200: 10},
    )

    with pytest.raises(ValueError, match="blocked/reserved"):
        validate_v3_dataset_sources(
            (case,),
            metadata={
                **metadata,
                "blocked": _meta("blocked", row_id=2, user_id="99"),
            },
            snapshot_watermarks={100: 10, 200: 10},
        )


def test_v3_observation_uses_raw_source_ids_when_legacy_episode_units_are_empty() -> None:
    case = _v3_case(
        expected_evidence_message_ids=("gold",),
        category="exact",
        expected_answer_mode="exact",
        expected_coverage_strategy="relevance",
        minimum_time_bucket_count=0,
        gate_tags=("source_resolution",),
        forbidden_evidence_message_ids=(),
    )
    metadata = {
        "gold": _meta("gold", row_id=1),
        "recent": _meta("recent", row_id=2),
    }
    packed_message = SimpleNamespace(source_msg_id="gold")
    packed = SimpleNamespace(
        evidence_segments=(SimpleNamespace(messages=(packed_message,)),),
        recent_messages=(),
        facts=(),
        summaries=(),
    )
    trace = SimpleNamespace(
        retrieved_source_msg_ids=("gold",),
        # Current production trace exposes no episode units for raw_message_v3.
        retrieved_source_units=(),
        result=SimpleNamespace(packed_context=packed, estimated_tokens=12),
        resolved_query=SimpleNamespace(
            subject_ids=("42",),
            answer_mode="exact",
            coverage_strategy="relevance",
            time_range=SimpleNamespace(
                start_at=datetime(2026, 7, 27, 16, tzinfo=UTC),
                end_at=datetime(2026, 7, 28, 16, tzinfo=UTC),
            ),
        ),
    )

    observation = build_v3_observation(
        case_index=0,
        case=case,
        trace=trace,
        requester_uin="42",
        metadata=metadata,
        snapshot_watermark=10,
        history_packet_tokens=12,
        retrieval_latency_ms=4.0,
    )
    fingerprint = retrieval_fingerprint_sha256((observation,))
    report = evaluate_v3(
        cases=(case,),
        observations=(observation,),
        quality=None,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        gate_tag_counts={"source_resolution": 1},
    )

    assert report["metrics"]["recall_at_150"] == 1.0
    assert report["metrics"]["recall_within_24k"] == 1.0
    assert report["metrics"]["unresolved_source_count"] == 0
    assert "private query" not in json.dumps(report)


def test_generic_aggregator_treats_v3_raw_ids_as_single_source_units() -> None:
    case = EvaluationCase(100, "private", (), ("gold",), "exact")
    report = evaluate(
        cases=(case,),
        results=(
            EvaluationResult(
                case_index=0,
                variant="v3",
                retrieved_evidence_message_ids=("gold",),
                packed_evidence_message_ids=("gold",),
                context_tokens=1,
                latency_ms=1.0,
                rewrite_used=False,
                retrieved_evidence_units=(),
            ),
        ),
        dataset_sha256="a" * 64,
        recall_k=150,
        variants=("v3",),
    )

    assert report["variants"]["v3"]["recall_at_k"] == 1.0


def test_quality_sidecar_is_bound_to_retrieval_and_contains_no_answer_text(
    tmp_path,
) -> None:
    case = _v3_case(
        expected_evidence_message_ids=("gold",),
        category="exact",
        expected_answer_mode="exact",
        expected_coverage_strategy="relevance",
        minimum_time_bucket_count=0,
        gate_tags=("source_resolution",),
        forbidden_evidence_message_ids=(),
    )
    metadata = {"gold": _meta("gold", row_id=1)}
    trace = SimpleNamespace(
        retrieved_source_msg_ids=("gold",),
        result=SimpleNamespace(
            packed_context=SimpleNamespace(
                evidence_segments=(
                    SimpleNamespace(
                        messages=(SimpleNamespace(source_msg_id="gold"),)
                    ),
                ),
                recent_messages=(),
                facts=(),
                summaries=(),
            ),
            estimated_tokens=8,
        ),
        resolved_query=SimpleNamespace(
            subject_ids=("42",),
            answer_mode="exact",
            coverage_strategy="relevance",
            time_range=SimpleNamespace(
                start_at=datetime(2026, 7, 27, 16, tzinfo=UTC),
                end_at=datetime(2026, 7, 28, 16, tzinfo=UTC),
            ),
        ),
    )
    observation = build_v3_observation(
        case_index=0,
        case=case,
        trace=trace,
        requester_uin="42",
        metadata=metadata,
        snapshot_watermark=10,
        history_packet_tokens=8,
        retrieval_latency_ms=3.0,
    )
    fingerprint = retrieval_fingerprint_sha256((observation,))
    template = quality_sidecar_template(
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        case_count=1,
    )
    template.update(
        {
            "judge_provider": "controlled-replay",
            "judge_model": "gpt-5.6",
            "evaluated_at": "2026-07-29T12:00:00Z",
            "index_visibility_ms": [100.0] * 20,
        }
    )
    template["cases"][0].update(
        {
            "cited_source_message_ids": ["gold"],
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "total_prompt_tokens": 100,
            "ttft_ms": 200.0,
        }
    )
    sidecar_path = tmp_path / "quality.json"
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    quality = load_v3_quality_sidecar(
        sidecar_path,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        case_count=1,
    )
    audit = audit_v3_quality_sources(
        cases=(case,),
        observations=(observation,),
        quality=quality,
        metadata=metadata,
    )

    assert quality.cases[0].answer_correct is True
    assert all(value == 0 for value in audit.values())
    assert "private query" not in sidecar_path.read_text(encoding="utf-8")

    for invalid_case_index in (False, 0.0, 1):
        template["cases"][0]["case_index"] = invalid_case_index
        sidecar_path.write_text(json.dumps(template), encoding="utf-8")
        with pytest.raises(ValueError, match="wrong case_index"):
            load_v3_quality_sidecar(
                sidecar_path,
                dataset_sha256="a" * 64,
                snapshot_manifest_sha256="b" * 64,
                retrieval_fingerprint=fingerprint,
                case_count=1,
            )

    template["cases"][0]["case_index"] = 0
    for invalid_constant in ("NaN", "Infinity", "-Infinity"):
        sidecar_path.write_text(
            json.dumps(template).replace(
                '"ttft_ms": 200.0',
                f'"ttft_ms": {invalid_constant}',
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid V3 quality sidecar"):
            load_v3_quality_sidecar(
                sidecar_path,
                dataset_sha256="a" * 64,
                snapshot_manifest_sha256="b" * 64,
                retrieval_fingerprint=fingerprint,
                case_count=1,
            )
