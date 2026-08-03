from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_memory_recall_eval as recall_runner
from scripts import resume_memory_v3_quality_replay as resume_runner
from scripts import run_memory_v3_quality_replay as quality_runner
from scripts.memory_v3_quality_contract import prompt_contract_sha256
from tests.test_memory_v3_quality_resume import _build_artifacts
from scripts.evaluate_memory_recall import EvaluationCase, EvaluationResult, evaluate
from scripts.evaluate_memory_v3 import (
    MessageMetadata,
    _expected_fail_closed_judge_decision,
    _expected_abstention_for_quality,
    _citation_precision_score,
    audit_v3_quality_sources,
    build_v3_observation,
    evaluate_v3,
    load_v3_quality_sidecar,
    quality_sidecar_template,
    retrieval_fingerprint_sha256,
    validate_v3_dataset_contract,
    validate_v3_dataset_sources,
)


def test_citation_precision_accepts_judge_grounded_alternative_evidence() -> None:
    assert _citation_precision_score(
        gold={"expected"},
        citations={"alternative"},
        answer_grounded=True,
        citations_minimal=True,
        expected_evidence_available=True,
    ) == 1.0
    assert _citation_precision_score(
        gold={"expected"},
        citations={"unsupported"},
        answer_grounded=False,
        citations_minimal=True,
        expected_evidence_available=True,
    ) == 0.0
    assert _citation_precision_score(
        gold={"expected", "second"},
        citations={"expected", "extra"},
        answer_grounded=True,
        citations_minimal=True,
        expected_evidence_available=True,
    ) == 1.0
    assert _citation_precision_score(
        gold={"expected", "second"},
        citations={"expected", "extra"},
        answer_grounded=True,
        citations_minimal=False,
        expected_evidence_available=True,
    ) == 0.5
    assert _citation_precision_score(
        gold={"expected"},
        citations=set(),
        answer_grounded=False,
        citations_minimal=True,
        expected_evidence_available=False,
    ) == 1.0


@pytest.mark.parametrize("profile", ("legacy", "adaptive"))
def test_recall_and_quality_producers_accept_explicit_context_profile(profile: str) -> None:
    recall_args = recall_runner.build_argument_parser().parse_args(
        [
            "--database", "db.sqlite",
            "--manifest", "manifest.json",
            "--dataset", "cases.jsonl",
            "--results-output", "results.jsonl",
            "--report-output", "report.json",
            "--benchmark-output", "benchmark.json",
            "--review", "review.json",
            "--prepared-report", "prepared.json",
            "--context-profile", profile,
        ]
    )
    quality_args = quality_runner.build_argument_parser().parse_args(
        [
            "--database", "db.sqlite",
            "--manifest", "manifest.json",
            "--prepared-report", "prepared.json",
            "--dataset", "cases.jsonl",
            "--review", "review.json",
            "--quality-output", "quality.json",
            "--private-replay-output", "private.json",
            "--visibility-output", "visibility.json",
            "--context-profile", profile,
        ]
    )

    assert recall_args.context_profile == profile
    assert quality_args.context_profile == profile


def test_recall_gate_forwards_frozen_inputs_to_resume_contract(
    monkeypatch,
    tmp_path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "receipt",
            "dataset",
            "manifest",
            "prepared",
            "parent_sidecar",
            "parent_private",
            "visibility",
            "parent_gate",
            "parent_results",
            "parent_benchmark",
            "child_sidecar",
            "child_private",
        )
    }
    captured = {}

    def validate(receipt_path, **kwargs):
        captured["receipt_path"] = receipt_path
        captured.update(kwargs)
        raise ValueError("tampered resume contract")

    monkeypatch.setattr(resume_runner, "validate_quality_resume_receipt", validate)
    args = SimpleNamespace(
        quality_resume_receipt=paths["receipt"],
        dataset=paths["dataset"],
        manifest=paths["manifest"],
        prepared_report=paths["prepared"],
        quality_resume_parent_sidecar=paths["parent_sidecar"],
        quality_resume_parent_private_replay=paths["parent_private"],
        quality_visibility_artifact=paths["visibility"],
        quality_resume_parent_gate_report=paths["parent_gate"],
        quality_resume_parent_results=paths["parent_results"],
        quality_resume_parent_benchmark=paths["parent_benchmark"],
        quality_sidecar=paths["child_sidecar"],
        quality_private_replay=paths["child_private"],
    )
    with pytest.raises(ValueError, match="tampered resume contract"):
        recall_runner._validate_quality_resume_artifacts(args)
    assert captured["dataset_path"] == paths["dataset"]
    assert captured["manifest_path"] == paths["manifest"]
    assert captured["prepared_report_path"] == paths["prepared"]


def test_recall_gate_rejects_changed_resume_executable_contract(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _build_artifacts(tmp_path)
    monkeypatch.setattr(resume_runner, "resume_contract_sha256", lambda: "0" * 64)
    args = SimpleNamespace(
        quality_resume_receipt=paths["receipt"],
        dataset=paths["dataset"],
        manifest=paths["manifest"],
        prepared_report=paths["prepared"],
        quality_resume_parent_sidecar=paths["parent_public"],
        quality_resume_parent_private_replay=paths["parent_private"],
        quality_visibility_artifact=paths["visibility"],
        quality_resume_parent_gate_report=paths["gate"],
        quality_resume_parent_results=paths["results"],
        quality_resume_parent_benchmark=paths["benchmark"],
        quality_sidecar=paths["child_public"],
        quality_private_replay=paths["child_private"],
    )
    with pytest.raises(ValueError, match="executable contract"):
        recall_runner._validate_quality_resume_artifacts(args)


def test_abstention_expectation_accepts_audited_grounded_alternative() -> None:
    assert _expected_abstention_for_quality(
        expected_evidence_available=False,
        citations={"alternative"},
        answer_grounded=True,
        answer_correct=True,
        citations_minimal=True,
    ) is False
    assert _expected_abstention_for_quality(
        expected_evidence_available=False,
        citations={"unsupported"},
        answer_grounded=False,
        answer_correct=False,
        citations_minimal=True,
    ) is True
    assert _expected_abstention_for_quality(
        expected_evidence_available=True,
        citations=set(),
        answer_grounded=False,
        answer_correct=False,
        citations_minimal=True,
    ) is False
    assert _expected_abstention_for_quality(
        expected_evidence_available=False,
        citations={"alternative"},
        answer_grounded=True,
        answer_correct=False,
        citations_minimal=True,
    ) is True


def test_fail_closed_judge_reconciliation_rejects_abstention_on_supported_case() -> None:
    decision = _expected_fail_closed_judge_decision(
        case=_v3_case(expected_evidence_message_ids=("gold",)),
        raw_decision={
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": True,
            "reason_code": "EXPECTED_ABSTENTION",
        },
        generated_citations=(),
        generated_abstained=True,
        protocol_failure_codes=(),
        citation_failure_codes=(),
    )

    assert decision == {
        "answer_grounded": False,
        "answer_correct": False,
        "abstained": True,
        "reason_code": "EXPECTED_ABSTENTION",
    }


def test_fail_closed_judge_reconciliation_preserves_recorded_failure_order() -> None:
    decision = _expected_fail_closed_judge_decision(
        case=_v3_case(expected_evidence_message_ids=("gold",)),
        raw_decision={
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "reason_code": "SUPPORTED",
        },
        generated_citations=("outside",),
        generated_abstained=False,
        protocol_failure_codes=("citation_not_minimal",),
        citation_failure_codes=("citation_outside_packet", "citation_not_minimal"),
    )

    assert decision == {
        "answer_grounded": False,
        "answer_correct": False,
        "abstained": False,
        "reason_code": "citation_not_minimal+citation_outside_packet",
    }


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
    reply_to_message_id: str | None = None,
) -> MessageMetadata:
    return MessageMetadata(
        row_id=row_id,
        source_message_id=source_id,
        group_id=group_id,
        user_id=user_id,
        timestamp=timestamp,
        delivery_state=delivery_state,
        reply_to_message_id=reply_to_message_id,
    )


def test_snapshot_candidate_filter_excludes_only_live_same_group_rows() -> None:
    metadata = {
        "snapshot": _meta("snapshot", row_id=10),
        "live": _meta("live", row_id=11),
        "cross-group": _meta("cross-group", row_id=99, group_id=200),
    }
    filter_candidates = recall_runner._snapshot_candidate_filter(
        metadata=metadata,
        snapshot_watermarks={100: 10},
    )
    retained = filter_candidates(
        request=SimpleNamespace(group_id=100),
        resolved_query=SimpleNamespace(),
        candidates=(
            SimpleNamespace(source_msg_ids=("snapshot",)),
            SimpleNamespace(source_msg_ids=("live",)),
            SimpleNamespace(source_msg_ids=("missing",)),
            SimpleNamespace(source_msg_ids=("cross-group",)),
        ),
    )

    assert [row.source_msg_ids for row in retained] == [
        ("snapshot",),
        ("missing",),
        ("cross-group",),
    ]


@pytest.mark.parametrize(
    ("target", "invalid_value", "expected_code"),
    [
        ("answer_accuracy", float("nan"), "AC_ANSWER_ACCURACY"),
        ("citation_precision", float("inf"), "AC_CITATION_PRECISION"),
        ("ttft_p95_ms", float("-inf"), "AC_TTFT_P95"),
        ("benchmark_p95", True, "AC_RETRIEVAL_P95"),
        ("answer_protocol_failure_count", 1, "AC_ANSWER_PROTOCOL"),
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
        "answer_protocol_failure_count",
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


def test_v3_adaptive_acceptance_uses_wide_safety_caps() -> None:
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
        "retrieval_over_300_count",
        "packet_over_300_count",
        "packet_over_32k_count",
        "recent_over_120_count",
        "citation_not_in_packet_count",
        "citation_forbidden_source_count",
        "citation_unresolved_source_count",
        "citation_group_leak_count",
        "citation_subject_leak_count",
        "citation_time_leak_count",
        "citation_ineligible_source_count",
        "answer_protocol_failure_count",
    )
    metrics = {
        **{name: 0 for name in zero_names},
        "retrieval_over_150_count": 1,
        "packet_over_150_count": 1,
        "packet_over_24k_count": 1,
        "recent_over_60_count": 1,
        "recall_at_300": 1.0,
        "recall_within_32k": 1.0,
        "time_bucket_coverage_rate": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "grounded_answer_accuracy": 1.0,
        "answer_accuracy": 1.0,
        "abstention_f1": 1.0,
        "index_visibility_p95_ms": 100.0,
        "ttft_p95_ms": 100.0,
    }

    failures = recall_runner._v3_acceptance_failures(
        report={"metrics": metrics},
        benchmark={
            "p95_latency_ms": 100.0,
            "rerank_enabled": False,
            "network_enabled": False,
            "vector_success_verified": True,
        },
        adaptive_enabled=True,
    )

    assert failures == ()


def test_v3_acceptance_supports_an_explicit_latency_waiver() -> None:
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
        "retrieval_over_300_count",
        "packet_over_300_count",
        "packet_over_32k_count",
        "recent_over_120_count",
        "citation_not_in_packet_count",
        "citation_forbidden_source_count",
        "citation_unresolved_source_count",
        "citation_group_leak_count",
        "citation_subject_leak_count",
        "citation_time_leak_count",
        "citation_ineligible_source_count",
        "answer_protocol_failure_count",
    )
    report = {
        "metrics": {
            **{name: 0 for name in zero_names},
            "recall_at_300": 1.0,
            "recall_within_32k": 1.0,
            "time_bucket_coverage_rate": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "grounded_answer_accuracy": 1.0,
            "answer_accuracy": 1.0,
            "abstention_f1": 1.0,
            "index_visibility_p95_ms": 100.0,
            "ttft_p95_ms": 100.0,
        }
    }
    benchmark = {
        "p95_latency_ms": 546.0,
        "rerank_enabled": False,
        "network_enabled": False,
        "vector_success_verified": True,
    }

    assert "AC_RETRIEVAL_P95" in recall_runner._v3_acceptance_failures(
        report=report,
        benchmark=benchmark,
        adaptive_enabled=True,
    )
    assert "AC_RETRIEVAL_P95" not in recall_runner._v3_acceptance_failures(
        report=report,
        benchmark=benchmark,
        adaptive_enabled=True,
        max_retrieval_p95_ms=600.0,
    )


def test_adaptive_packet_32k_metric_uses_shared_recent_plus_history_total() -> None:
    case = EvaluationCase(100, "q", (), (), "abstention", schema_version=3)
    packed = SimpleNamespace(
        evidence_segments=(),
        recent_messages=(),
        facts=(),
        summaries=(),
    )
    trace = SimpleNamespace(
        retrieved_source_msg_ids=(),
        retrieved_source_units=(),
        result=SimpleNamespace(packed_context=packed, estimated_tokens=32_001),
        resolved_query=SimpleNamespace(
            subject_ids=None,
            answer_mode="abstention",
            coverage_strategy="relevance",
            time_range=None,
        ),
    )
    observation = build_v3_observation(
        case_index=0,
        case=case,
        trace=trace,
        requester_uin="42",
        metadata={},
        snapshot_watermark=10,
        history_packet_tokens=100,
        retrieval_latency_ms=1.0,
    )
    fingerprint = retrieval_fingerprint_sha256((observation,))

    report = evaluate_v3(
        cases=(case,),
        observations=(observation,),
        quality=None,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        gate_tag_counts={},
    )

    assert report["metrics"]["packet_over_32k_count"] == 1
    assert report["metrics"]["packet_over_24k_count"] == 0


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
        "recent": _meta("recent", row_id=4),
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


def test_v3_dataset_sources_reject_requester_mismatch_and_implicit_quote_plan_drift() -> None:
    case = _v3_case(
        expected_answer_mode="general_history",
        expected_coverage_strategy="relevance",
        minimum_time_bucket_count=0,
    )
    metadata = {
        "recent": _meta("recent", row_id=1, user_id="99"),
        "cross-group": _meta("cross-group", row_id=2, group_id=200),
        "blocked": _meta("blocked", row_id=3, delivery_state="blocked"),
        "wrong-time": _meta(
            "wrong-time",
            row_id=4,
            user_id="99",
            timestamp=datetime(2026, 7, 30, tzinfo=UTC),
        ),
    }

    with pytest.raises(ValueError, match="requester does not match"):
        validate_v3_dataset_sources(
            (case,),
            metadata=metadata,
            snapshot_watermarks={100: 10, 200: 10},
        )

    metadata["recent"] = _meta(
        "recent",
        row_id=1,
        reply_to_message_id="quoted",
    )
    metadata["quoted"] = _meta("quoted", row_id=5)
    with pytest.raises(ValueError, match="implicit quote"):
        validate_v3_dataset_sources(
            (case,),
            metadata=metadata,
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
    assert report["metrics"]["recall_at_300"] == 1.0
    assert report["metrics"]["recall_within_24k"] == 1.0
    assert report["metrics"]["recall_within_32k"] == 1.0
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
        context_profile="legacy",
    )
    private_path = tmp_path / "quality-private.json"
    answer_prompt = ["private prompt"]
    answer_raw = json.dumps(
        {
            "answer": "private answer",
            "cited_source_message_ids": ["gold"],
            "abstained": False,
        },
        separators=(",", ":"),
    )
    contract_raw = json.dumps(
        {"citations_minimal": True, "reason_code": "minimal"},
        separators=(",", ":"),
    )
    judge_raw = json.dumps(
        {
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "reason_code": "supported",
        },
        separators=(",", ":"),
    )
    private_payload = {
        "private_replay_version": 1,
        "dataset_sha256": "a" * 64,
        "snapshot_manifest_sha256": "b" * 64,
        "retrieval_fingerprint_sha256": fingerprint,
        "prompt_contract_sha256": prompt_contract_sha256(),
        "generator_model": "generator-model",
        "judge_model": "judge-model",
        "evaluated_at": "2026-07-29T12:00:00Z",
        "cases": [
            {
                "case_index": 0,
                "query": case.query,
                "answer_prompt": answer_prompt,
                "answer_prompt_sha256": hashlib.sha256(
                    json.dumps(
                        answer_prompt,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "answer": "private answer",
                "generated_citations": ["gold"],
                "generated_abstained": False,
                "answer_protocol_failure_codes": [],
                "answer_repair_count": 0,
                "answer_observation": {
                    "text": answer_raw,
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "ttft_ms": 200.0,
                    "model": "generator-model",
                    "endpoint": "responses",
                },
                "answer_attempts": [
                    {
                        "kind": "initial",
                        "prompt": answer_prompt,
                        "answer": {
                            "answer": "private answer",
                            "cited_source_message_ids": ["gold"],
                            "abstained": False,
                        },
                        "observation": {
                            "text": answer_raw,
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "ttft_ms": 200.0,
                            "model": "generator-model",
                            "endpoint": "responses",
                        },
                        "protocol_failure_codes": [],
                        "citation_contract_prompt": ["private contract prompt"],
                        "citation_contract_raw_output": contract_raw,
                        "citation_contract_observation": {
                            "text": contract_raw,
                            "input_tokens": 50,
                            "output_tokens": 5,
                            "ttft_ms": 100.0,
                            "model": "judge-model",
                            "endpoint": "responses",
                        },
                        "citation_contract_decision": {
                            "citations_minimal": True,
                            "reason_code": "minimal",
                        },
                    }
                ],
                "citation_contract_prompt": ["private contract prompt"],
                "citation_contract_raw_output": contract_raw,
                "citation_contract_observation": {
                    "text": contract_raw,
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "ttft_ms": 100.0,
                    "model": "judge-model",
                    "endpoint": "responses",
                },
                "citation_contract_decision": {
                    "citations_minimal": True,
                    "reason_code": "minimal",
                },
                "judge_prompt": ["private judge prompt"],
                "judge_raw_output": judge_raw,
                "judge_observation": {
                    "text": judge_raw,
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "ttft_ms": 100.0,
                    "model": "judge-model",
                    "endpoint": "responses",
                },
                "judge_decision": {
                    "answer_grounded": True,
                    "answer_correct": True,
                    "abstained": False,
                    "reason_code": "supported",
                },
                "citation_failure_codes": [],
            }
        ],
    }
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    private_sha256 = hashlib.sha256(private_path.read_bytes()).hexdigest()
    visibility_path = tmp_path / "quality-visibility.json"
    visibility_payload = {
        "visibility_version": 1,
        "measurement_mode": "disposable_sqlite_online_backup_clone",
        "source_snapshot_clone_sha256": "f" * 64,
        "vector_generation": 2,
        "sample_count": 20,
        "samples": [
            {
                "case_index": index,
                "nonce_sha256": f"{index:064x}",
                "fts_ms": 90.0,
                "vector_ms": 100.0,
                "overall_ms": 100.0,
            }
            for index in range(20)
        ],
        "dataset_sha256": "a" * 64,
        "snapshot_manifest_sha256": "b" * 64,
        "retrieval_fingerprint_sha256": fingerprint,
    }
    visibility_path.write_text(json.dumps(visibility_payload), encoding="utf-8")
    visibility_sha256 = hashlib.sha256(visibility_path.read_bytes()).hexdigest()
    template.update(
        {
            "private_replay_sha256": private_sha256,
            "visibility_artifact_sha256": visibility_sha256,
            "prompt_contract_sha256": prompt_contract_sha256(),
            "judge_provider": "responses-controlled-replay",
            "judge_model": "generator=generator-model;judge=judge-model",
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
            "answer_protocol_failure_codes": [],
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
        private_replay_path=private_path,
        visibility_artifact_path=visibility_path,
        expected_vector_generation=2,
        evaluation_cases=(case,),
        expected_answer_prompt_sha256_by_case={
            0: private_payload["cases"][0]["answer_prompt_sha256"]
        },
        expected_context_profile="legacy",
    )
    audit = audit_v3_quality_sources(
        cases=(case,),
        observations=(observation,),
        quality=quality,
        metadata=metadata,
    )

    assert quality.cases[0].answer_correct is True
    assert quality.cases[0].answer_protocol_failure_codes == ()
    assert all(value == 0 for value in audit.values())
    assert "private query" not in sidecar_path.read_text(encoding="utf-8")

    receipt_path = tmp_path / "quality-resume-receipt.json"
    receipt_path.write_text('{"resume_version":1}\n', encoding="utf-8")
    template["quality_version"] = 4
    template["resume_receipt_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    resumed = load_v3_quality_sidecar(
        sidecar_path,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        case_count=1,
        private_replay_path=private_path,
        visibility_artifact_path=visibility_path,
        expected_vector_generation=2,
        evaluation_cases=(case,),
        expected_answer_prompt_sha256_by_case={
            0: private_payload["cases"][0]["answer_prompt_sha256"]
        },
        expected_context_profile="legacy",
        resume_receipt_path=receipt_path,
    )
    assert resumed.resume_receipt_sha256 == template["resume_receipt_sha256"]
    with pytest.raises(ValueError, match="valid receipt"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
        )
    rebind_receipt_path = tmp_path / "quality-rebind-receipt.json"
    rebind_receipt_path.write_text('{"rebind_version":1}\n', encoding="utf-8")
    template["quality_version"] = 5
    template.pop("resume_receipt_sha256")
    template["rebind_receipt_sha256"] = hashlib.sha256(
        rebind_receipt_path.read_bytes()
    ).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    rebound = load_v3_quality_sidecar(
        sidecar_path,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        case_count=1,
        private_replay_path=private_path,
        visibility_artifact_path=visibility_path,
        expected_vector_generation=2,
        evaluation_cases=(case,),
        expected_answer_prompt_sha256_by_case={
            0: private_payload["cases"][0]["answer_prompt_sha256"]
        },
        expected_context_profile="legacy",
        rebind_receipt_path=rebind_receipt_path,
    )
    assert rebound.rebind_receipt_sha256 == template["rebind_receipt_sha256"]
    with pytest.raises(ValueError, match="valid receipt"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
        )
    template["quality_version"] = 3
    template.pop("rebind_receipt_sha256")
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    with pytest.raises(ValueError, match="context profile"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            expected_context_profile="adaptive",
        )

    with pytest.raises(ValueError, match="answer prompt binding"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            expected_answer_prompt_sha256_by_case={0: "0" * 64},
        )

    private_payload["cases"][0]["query"] = "wrong private query"
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = hashlib.sha256(private_path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="query does not match the dataset"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            evaluation_cases=(case,),
        )
    private_payload["cases"][0]["query"] = case.query

    for invalid_case_index in (False, 0.0, 1):
        private_payload["cases"][0]["case_index"] = invalid_case_index
        private_path.write_text(json.dumps(private_payload), encoding="utf-8")
        template["private_replay_sha256"] = hashlib.sha256(private_path.read_bytes()).hexdigest()
        sidecar_path.write_text(json.dumps(template), encoding="utf-8")
        with pytest.raises(ValueError, match="wrong case_index"):
            load_v3_quality_sidecar(
                sidecar_path,
                dataset_sha256="a" * 64,
                snapshot_manifest_sha256="b" * 64,
                retrieval_fingerprint=fingerprint,
                case_count=1,
                private_replay_path=private_path,
            )
    private_payload["cases"][0]["case_index"] = 0
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = private_sha256

    for invalid_case_index in (False, 0.0, 1):
        visibility_payload["samples"][0]["case_index"] = invalid_case_index
        visibility_path.write_text(json.dumps(visibility_payload), encoding="utf-8")
        template["visibility_artifact_sha256"] = hashlib.sha256(
            visibility_path.read_bytes()
        ).hexdigest()
        sidecar_path.write_text(json.dumps(template), encoding="utf-8")
        with pytest.raises(ValueError, match="visibility sample index"):
            load_v3_quality_sidecar(
                sidecar_path,
                dataset_sha256="a" * 64,
                snapshot_manifest_sha256="b" * 64,
                retrieval_fingerprint=fingerprint,
                case_count=1,
                visibility_artifact_path=visibility_path,
            )
    visibility_payload["samples"][0]["case_index"] = 0
    visibility_path.write_text(json.dumps(visibility_payload), encoding="utf-8")
    template["visibility_artifact_sha256"] = visibility_sha256
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    private_payload["cases"][0]["judge_raw_output"] = "hand-written decision"
    private_payload["cases"][0]["judge_observation"]["text"] = "hand-written decision"
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = hashlib.sha256(private_path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    private_payload["cases"][0]["judge_raw_output"] = judge_raw
    private_payload["cases"][0]["judge_observation"]["text"] = judge_raw
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = private_sha256
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    invalid_answer_raw = json.dumps(
        {
            "answer": 7,
            "cited_source_message_ids": ["gold"],
            "abstained": False,
        },
        separators=(",", ":"),
    )
    private_payload["cases"][0]["answer_attempts"][0]["answer"]["answer"] = 7
    private_payload["cases"][0]["answer_attempts"][0]["observation"]["text"] = invalid_answer_raw
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = hashlib.sha256(private_path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="generated answer"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    private_payload["cases"][0]["answer_attempts"][0]["answer"]["answer"] = "private answer"
    private_payload["cases"][0]["answer_attempts"][0]["observation"]["text"] = answer_raw
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = private_sha256
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    invalid_abstention_raw = json.dumps(
        {
            "answer": "not the fixed abstention text",
            "cited_source_message_ids": [],
            "abstained": True,
        },
        separators=(",", ":"),
    )
    attempt = private_payload["cases"][0]["answer_attempts"][0]
    attempt["answer"] = {
        "answer": "not the fixed abstention text",
        "cited_source_message_ids": [],
        "abstained": True,
    }
    attempt["observation"]["text"] = invalid_abstention_raw
    attempt["protocol_failure_codes"] = ["abstention_text_mismatch"]
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = hashlib.sha256(private_path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="omits final protocol failures"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    attempt["answer"] = {
        "answer": "private answer",
        "cited_source_message_ids": ["gold"],
        "abstained": False,
    }
    attempt["observation"]["text"] = answer_raw
    attempt["protocol_failure_codes"] = []
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    template["private_replay_sha256"] = private_sha256
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    private_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")

    template["cases"][0]["answer_protocol_failure_codes"] = [
        "citation_not_minimal"
    ]
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the sidecar"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    template["cases"][0]["answer_protocol_failure_codes"] = []
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")

    visibility_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_v3_quality_sidecar(
            sidecar_path,
            dataset_sha256="a" * 64,
            snapshot_manifest_sha256="b" * 64,
            retrieval_fingerprint=fingerprint,
            case_count=1,
            private_replay_path=private_path,
            visibility_artifact_path=visibility_path,
            expected_vector_generation=2,
        )
    visibility_path.write_text(json.dumps(visibility_payload), encoding="utf-8")

    template["cases"][0]["answer_protocol_failure_codes"] = [
        "citation_not_minimal"
    ]
    sidecar_path.write_text(json.dumps(template), encoding="utf-8")
    failed_quality = load_v3_quality_sidecar(
        sidecar_path,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        case_count=1,
    )
    failed_report = evaluate_v3(
        cases=(case,),
        observations=(observation,),
        quality=failed_quality,
        dataset_sha256="a" * 64,
        snapshot_manifest_sha256="b" * 64,
        retrieval_fingerprint=fingerprint,
        gate_tag_counts={},
    )
    assert failed_report["metrics"]["answer_protocol_failure_count"] == 1
    assert "AC_ANSWER_PROTOCOL" in recall_runner._v3_acceptance_failures(
        report=failed_report,
        benchmark={
            "p95_latency_ms": 100.0,
            "rerank_enabled": False,
            "network_enabled": False,
            "vector_success_verified": True,
        },
    )
    template["cases"][0]["answer_protocol_failure_codes"] = []

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
