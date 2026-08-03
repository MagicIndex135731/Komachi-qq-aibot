from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.rebind_memory_v3_quality as rebind_module
import scripts.resume_memory_v3_quality_replay as resume_module
from scripts.rebind_memory_v3_quality import (
    rebind_existing_quality_fingerprint,
    validate_quality_rebind_receipt,
)
from scripts.evaluate_memory_recall import EvaluationCase
from scripts.evaluate_memory_v3 import load_v3_quality_sidecar
from scripts.resume_memory_v3_quality_replay import (
    EXPECTED_CASE_COUNT,
    _canonical_sha256,
    _file_sha256,
    _retrieval_fingerprint,
    _retrieval_observation_payload,
    _text_sha256,
    reseal_existing_quality_resume,
    resume_case_rows,
    resume_contract_sha256,
    validate_quality_resume_receipt,
)
from scripts.run_memory_v3_quality_replay import ObservedGeneration, prompt_contract_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _observation(text: str, model: str) -> dict[str, object]:
    return {
        "text": text,
        "input_tokens": 10,
        "output_tokens": 3,
        "ttft_ms": 2.0,
        "model": model,
        "endpoint": "responses",
    }


def _answer(value: str, citations: list[str]) -> dict[str, object]:
    return {"answer": value, "cited_source_message_ids": citations, "abstained": False}


def _attempt(kind: str, *, answer: dict[str, object], minimal: bool) -> dict[str, object]:
    raw_answer = json.dumps(answer, separators=(",", ":"))
    raw_contract = json.dumps(
        {"citations_minimal": minimal, "reason_code": "minimal" if minimal else "extra"},
        separators=(",", ":"),
    )
    return {
        "kind": kind,
        "prompt": ["answer prompt"],
        "answer": answer,
        "observation": _observation(raw_answer, "generator"),
        "protocol_failure_codes": [],
        "citation_contract_prompt": ["Retrieved packet:\npacket"],
        "citation_contract_raw_output": raw_contract,
        "citation_contract_observation": _observation(raw_contract, "judge"),
        "citation_contract_decision": {
            "citations_minimal": minimal,
            "reason_code": "minimal" if minimal else "extra",
        },
    }


def _private_row(index: int, *, failed: bool = False, child: bool = False) -> dict[str, object]:
    initial = _attempt("initial", answer=_answer("fact", ["s1", "s2"]), minimal=False)
    repair = _attempt(
        "citation_repair",
        answer=_answer("fact", ["s1"] if child else ["s1", "s2"]),
        minimal=child,
    )
    answer_prompt = ["answer prompt"]
    prompt_sha = hashlib.sha256(
        json.dumps(answer_prompt, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    attempts = [initial, repair] if failed else [initial]
    final = attempts[-1]
    protocol = [] if child or not failed else ["citation_not_minimal"]
    contract = final["citation_contract_decision"]
    contract_raw = final["citation_contract_raw_output"]
    judge_raw = json.dumps(
        {
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "reason_code": "supported",
        },
        separators=(",", ":"),
    )
    return {
        "case_index": index,
        "query": f"query-{index}",
        "answer_prompt": answer_prompt,
        "answer_prompt_sha256": prompt_sha,
        "answer": "fact",
        "generated_citations": final["answer"]["cited_source_message_ids"],
        "generated_abstained": False,
        "answer_protocol_failure_codes": protocol,
        "answer_repair_count": 1 if failed else 0,
        "answer_observation": initial["observation"],
        "answer_attempts": attempts,
        "citation_contract_prompt": final["citation_contract_prompt"],
        "citation_contract_raw_output": contract_raw,
        "citation_contract_observation": final["citation_contract_observation"],
        "citation_contract_decision": contract,
        "judge_prompt": ["judge prompt"],
        "judge_raw_output": judge_raw,
        "judge_observation": _observation(judge_raw, "judge"),
        "judge_decision": {
            "answer_grounded": True,
            "answer_correct": True,
            "abstained": False,
            "reason_code": "supported",
        },
        "citation_failure_codes": [],
    }


def _public_row(index: int, *, failed: bool = False, child: bool = False) -> dict[str, object]:
    return {
        "case_index": index,
        "cited_source_message_ids": ["s1"] if child else ["s1", "s2"],
        "answer_grounded": True,
        "answer_correct": True,
        "abstained": False,
        "answer_protocol_failure_codes": (
            ["citation_not_minimal"] if failed and not child else []
        ),
        "total_prompt_tokens": 10,
        "ttft_ms": 2.0,
    }


def _build_artifacts(tmp_path: Path, *, case_index: int = 30) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "parent_public",
            "parent_private",
            "visibility",
            "gate",
            "benchmark",
            "child_public",
            "child_private",
            "receipt",
        )
    }
    paths["results"] = tmp_path / "results.jsonl"
    paths["dataset"] = tmp_path / "dataset.jsonl"
    paths["manifest"] = tmp_path / "manifest.json"
    paths["prepared"] = tmp_path / "prepared.json"
    paths["dataset"].write_text('{"case":true}\n', encoding="utf-8")
    _write_json(paths["manifest"], {"manifest": True})
    _write_json(paths["prepared"], {"phase": "prepared", "vector_generation": 3})
    parent_public_rows = [
        _public_row(index, failed=index == case_index) for index in range(EXPECTED_CASE_COUNT)
    ]
    child_public_rows = deepcopy(parent_public_rows)
    child_public_rows[case_index] = _public_row(case_index, failed=True, child=True)
    parent_private_rows = [
        _private_row(index, failed=index == case_index) for index in range(EXPECTED_CASE_COUNT)
    ]
    child_private_rows = deepcopy(parent_private_rows)
    child_private_rows[case_index] = _private_row(case_index, failed=True, child=True)
    results = [
        {
            "case_index": index,
            "retrieved_source_message_ids": ["s1", "s2"],
            "history_packet_source_message_ids": ["s1", "s2"],
            "history_packet_tokens": 10,
            "answer_prompt_sha256": parent_private_rows[index]["answer_prompt_sha256"],
        }
        for index in range(EXPECTED_CASE_COUNT)
    ]
    fingerprint = _retrieval_fingerprint(results)
    common = {
        "dataset_sha256": "a" * 64,
        "snapshot_manifest_sha256": "b" * 64,
        "retrieval_fingerprint_sha256": fingerprint,
        "context_profile": "adaptive",
        "prompt_contract_sha256": prompt_contract_sha256(),
        "visibility_artifact_sha256": "",
    }
    visibility = {
        "visibility_version": 1,
        "measurement_mode": "disposable_sqlite_online_backup_clone",
        "source_snapshot_clone_sha256": "d" * 64,
        "vector_generation": 3,
        "sample_count": 20,
        "samples": [
            {
                "case_index": index,
                "nonce_sha256": f"{index:064x}",
                "fts_ms": 1.0,
                "vector_ms": 1.0,
                "overall_ms": 1.0,
            }
            for index in range(20)
        ],
        "dataset_sha256": common["dataset_sha256"],
        "snapshot_manifest_sha256": common["snapshot_manifest_sha256"],
        "retrieval_fingerprint_sha256": fingerprint,
    }
    _write_json(paths["visibility"], visibility)
    common["visibility_artifact_sha256"] = _file_sha256(paths["visibility"])
    parent_private = {
        "private_replay_version": 1,
        "dataset_sha256": common["dataset_sha256"],
        "snapshot_manifest_sha256": common["snapshot_manifest_sha256"],
        "retrieval_fingerprint_sha256": fingerprint,
        "prompt_contract_sha256": common["prompt_contract_sha256"],
        "generator_model": "generator",
        "judge_model": "judge",
        "evaluated_at": "2026-08-03T00:00:00Z",
        "cases": parent_private_rows,
    }
    child_private = deepcopy(parent_private)
    child_private["evaluated_at"] = "2026-08-03T01:00:00Z"
    child_private["cases"] = child_private_rows
    _write_json(paths["parent_private"], parent_private)
    _write_json(paths["child_private"], child_private)
    parent_public = {
        "quality_version": 3,
        **common,
        "private_replay_sha256": _file_sha256(paths["parent_private"]),
        "judge_provider": "responses-controlled-replay",
        "judge_model": "generator=generator;judge=judge",
        "evaluated_at": parent_private["evaluated_at"],
        "index_visibility_ms": [1.0] * 20,
        "cases": parent_public_rows,
    }
    _write_json(paths["parent_public"], parent_public)
    paths["results"].write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in results),
        encoding="utf-8",
    )
    _write_json(paths["benchmark"], {"p95_latency_ms": 1.0})
    gate = {
        "dataset_sha256": common["dataset_sha256"],
        "snapshot_manifest_sha256": common["snapshot_manifest_sha256"],
        "retrieval_fingerprint_sha256": fingerprint,
        "context_profile": "adaptive",
        "vector_generation": 3,
        "quality_sidecar_sha256": _file_sha256(paths["parent_public"]),
        "results_sha256": _file_sha256(paths["results"]),
        "benchmark_sha256": _file_sha256(paths["benchmark"]),
        "metrics": {"answer_protocol_failure_count": 1},
        "acceptance": {"status": "failed", "error_codes": ["AC_ANSWER_PROTOCOL"]},
    }
    _write_json(paths["gate"], gate)
    parent_hashes = {
        "quality_sidecar": _file_sha256(paths["parent_public"]),
        "private_replay": _file_sha256(paths["parent_private"]),
        "visibility": _file_sha256(paths["visibility"]),
        "gate_report": _file_sha256(paths["gate"]),
        "results": _file_sha256(paths["results"]),
        "benchmark": _file_sha256(paths["benchmark"]),
        "dataset": _file_sha256(paths["dataset"]),
        "manifest": _file_sha256(paths["manifest"]),
        "prepared_report": _file_sha256(paths["prepared"]),
    }
    receipt = {
        "resume_receipt_version": 1,
        "resume_contract_sha256": resume_contract_sha256(),
        "parent_artifacts_sha256": parent_hashes,
        "child_private_replay_sha256": _file_sha256(paths["child_private"]),
        "bindings": {
            "dataset_sha256": common["dataset_sha256"],
            "snapshot_manifest_sha256": common["snapshot_manifest_sha256"],
            "retrieval_fingerprint_sha256": fingerprint,
            "prompt_contract_sha256": common["prompt_contract_sha256"],
            "vector_generation": 3,
            "context_profile": "adaptive",
        },
        "resumed_case_indexes": [case_index],
        "parent_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in parent_private_rows
        ],
        "child_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_private_rows
        ],
        "parent_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in parent_public_rows
        ],
        "child_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_public_rows
        ],
        "resumed_case_audit": {
            "case_index": case_index,
            "stable_retrieval_observation_sha256": _canonical_sha256(
                _retrieval_observation_payload(results[case_index])
            ),
            "answer_prompt_sha256": results[case_index]["answer_prompt_sha256"],
            "parent_failed_repair_raw_output_sha256": _text_sha256(
                parent_private_rows[case_index]["answer_attempts"][1]["observation"]["text"]
            ),
            "child_repair_raw_output_sha256": _text_sha256(
                child_private_rows[case_index]["answer_attempts"][1]["observation"]["text"]
            ),
            "child_citation_reviewer_raw_output_sha256": _text_sha256(
                child_private_rows[case_index]["citation_contract_raw_output"]
            ),
            "child_correctness_judge_raw_output_sha256": _text_sha256(
                child_private_rows[case_index]["judge_raw_output"]
            ),
        },
    }
    _write_json(paths["receipt"], receipt)
    child_public = {
        **parent_public,
        "quality_version": 4,
        "private_replay_sha256": _file_sha256(paths["child_private"]),
        "evaluated_at": child_private["evaluated_at"],
        "cases": child_public_rows,
        "resume_receipt_sha256": _file_sha256(paths["receipt"]),
    }
    _write_json(paths["child_public"], child_public)
    return paths


def _validate(paths: dict[str, Path]) -> dict[str, object]:
    return validate_quality_resume_receipt(
        paths["receipt"],
        dataset_path=paths["dataset"],
        manifest_path=paths["manifest"],
        prepared_report_path=paths["prepared"],
        parent_quality_sidecar_path=paths["parent_public"],
        parent_private_replay_path=paths["parent_private"],
        parent_visibility_path=paths["visibility"],
        parent_gate_report_path=paths["gate"],
        parent_results_path=paths["results"],
        parent_benchmark_path=paths["benchmark"],
        child_quality_sidecar_path=paths["child_public"],
        child_private_replay_path=paths["child_private"],
    )


def _build_rebind_artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = _build_artifacts(tmp_path / "resume")
    paths.update(
        {
            "old_passed_gate": tmp_path / "old-passed-gate.json",
            "old_passed_benchmark": tmp_path / "old-passed-benchmark.json",
            "new_failed_gate": tmp_path / "new-failed-gate.json",
            "new_results": tmp_path / "new-results.jsonl",
            "new_benchmark": tmp_path / "new-benchmark.json",
            "rebound_public": tmp_path / "rebound-quality.json",
            "rebound_private": tmp_path / "rebound-private.json",
            "rebound_visibility": tmp_path / "rebound-visibility.json",
            "rebind_receipt": tmp_path / "rebind-receipt.json",
        }
    )
    safety_metrics = {
        field: 0
        for field in (
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
        )
    }
    safety_metrics.update(
        recall_at_300=1.0,
        recall_within_32k=1.0,
        time_bucket_coverage_rate=1.0,
    )
    benchmark = {
        "network_enabled": False,
        "rerank_enabled": False,
        "vector_success_verified": True,
    }
    _write_json(paths["old_passed_benchmark"], benchmark)
    old_results = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
    ]
    new_results = deepcopy(old_results)
    new_results[0]["retrieved_source_message_ids"] = ["s1"]
    paths["new_results"].write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in new_results
        ),
        encoding="utf-8",
    )
    _write_json(paths["new_benchmark"], benchmark)
    old_public = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    gate_common = {
        "dataset_sha256": old_public["dataset_sha256"],
        "snapshot_manifest_sha256": old_public["snapshot_manifest_sha256"],
        "context_profile": old_public["context_profile"],
        "vector_generation": 3,
        "metrics": safety_metrics,
    }
    _write_json(
        paths["old_passed_gate"],
        {
            **gate_common,
            "retrieval_fingerprint_sha256": _retrieval_fingerprint(old_results),
            "results_sha256": _file_sha256(paths["results"]),
            "benchmark_sha256": _file_sha256(paths["old_passed_benchmark"]),
            "acceptance": {"status": "passed", "error_codes": []},
        },
    )
    _write_json(
        paths["new_failed_gate"],
        {
            **gate_common,
            "retrieval_fingerprint_sha256": _retrieval_fingerprint(new_results),
            "results_sha256": _file_sha256(paths["new_results"]),
            "benchmark_sha256": _file_sha256(paths["new_benchmark"]),
            "acceptance": {
                "status": "failed",
                "error_codes": ["AC_QUALITY_SIDECAR_INVALID"],
            },
        },
    )
    rebind_existing_quality_fingerprint(
        output_receipt_path=paths["rebind_receipt"],
        output_quality_sidecar_path=paths["rebound_public"],
        output_private_replay_path=paths["rebound_private"],
        output_visibility_path=paths["rebound_visibility"],
        dataset_path=paths["dataset"],
        manifest_path=paths["manifest"],
        prepared_report_path=paths["prepared"],
        old_quality_sidecar_path=paths["child_public"],
        old_private_replay_path=paths["child_private"],
        old_resume_receipt_path=paths["receipt"],
        old_visibility_path=paths["visibility"],
        old_gate_report_path=paths["old_passed_gate"],
        old_results_path=paths["results"],
        old_benchmark_path=paths["old_passed_benchmark"],
        old_resume_parent_quality_sidecar_path=paths["parent_public"],
        old_resume_parent_private_replay_path=paths["parent_private"],
        old_resume_parent_gate_report_path=paths["gate"],
        old_resume_parent_results_path=paths["results"],
        old_resume_parent_benchmark_path=paths["benchmark"],
        new_failed_gate_report_path=paths["new_failed_gate"],
        new_results_path=paths["new_results"],
        new_benchmark_path=paths["new_benchmark"],
    )
    return paths


def _validate_rebind(
    paths: dict[str, Path],
    *,
    swap_visibility: bool = False,
) -> dict[str, object]:
    return validate_quality_rebind_receipt(
        paths["rebind_receipt"],
        dataset_path=paths["dataset"],
        manifest_path=paths["manifest"],
        prepared_report_path=paths["prepared"],
        old_quality_sidecar_path=paths["child_public"],
        old_private_replay_path=paths["child_private"],
        old_resume_receipt_path=paths["receipt"],
        old_visibility_path=(
            paths["rebound_visibility"] if swap_visibility else paths["visibility"]
        ),
        old_gate_report_path=paths["old_passed_gate"],
        old_results_path=paths["results"],
        old_benchmark_path=paths["old_passed_benchmark"],
        old_resume_parent_quality_sidecar_path=paths["parent_public"],
        old_resume_parent_private_replay_path=paths["parent_private"],
        old_resume_parent_gate_report_path=paths["gate"],
        old_resume_parent_results_path=paths["results"],
        old_resume_parent_benchmark_path=paths["benchmark"],
        new_failed_gate_report_path=paths["new_failed_gate"],
        new_results_path=paths["new_results"],
        new_benchmark_path=paths["new_benchmark"],
        child_quality_sidecar_path=paths["rebound_public"],
        child_private_replay_path=paths["rebound_private"],
        child_visibility_path=(
            paths["visibility"] if swap_visibility else paths["rebound_visibility"]
        ),
    )


def test_valid_quality_rebind_changes_only_fingerprint_metadata(tmp_path: Path) -> None:
    paths = _build_rebind_artifacts(tmp_path)
    receipt = _validate_rebind(paths)
    old_public = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    new_public = json.loads(paths["rebound_public"].read_text(encoding="utf-8"))
    old_private = json.loads(paths["child_private"].read_text(encoding="utf-8"))
    new_private = json.loads(paths["rebound_private"].read_text(encoding="utf-8"))
    old_visibility = json.loads(paths["visibility"].read_text(encoding="utf-8"))
    new_visibility = json.loads(
        paths["rebound_visibility"].read_text(encoding="utf-8")
    )

    assert receipt["old_answer_prompt_sha256_by_case"] == receipt[
        "new_answer_prompt_sha256_by_case"
    ]
    assert old_public["cases"] == new_public["cases"]
    assert old_private["cases"] == new_private["cases"]
    assert new_public["visibility_artifact_sha256"] == _file_sha256(
        paths["rebound_visibility"]
    )
    assert {
        **new_visibility,
        "retrieval_fingerprint_sha256": old_visibility[
            "retrieval_fingerprint_sha256"
        ],
    } == old_visibility
    assert old_public["retrieval_fingerprint_sha256"] != new_public[
        "retrieval_fingerprint_sha256"
    ]

    loaded = load_v3_quality_sidecar(
        paths["rebound_public"],
        dataset_sha256=new_public["dataset_sha256"],
        snapshot_manifest_sha256=new_public["snapshot_manifest_sha256"],
        retrieval_fingerprint=new_public["retrieval_fingerprint_sha256"],
        case_count=EXPECTED_CASE_COUNT,
        visibility_artifact_path=paths["rebound_visibility"],
        expected_vector_generation=3,
        expected_context_profile="adaptive",
        rebind_receipt_path=paths["rebind_receipt"],
    )
    assert loaded.rebind_receipt_sha256 == new_public["rebind_receipt_sha256"]


def test_quality_rebind_rejects_swapped_parent_and_child_visibility(
    tmp_path: Path,
) -> None:
    paths = _build_rebind_artifacts(tmp_path)

    with pytest.raises(ValueError):
        _validate_rebind(paths, swap_visibility=True)


def test_quality_rebind_rejects_changed_answer_prompt(tmp_path: Path) -> None:
    paths = _build_rebind_artifacts(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["new_results"].read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["answer_prompt_sha256"] = "0" * 64
    paths["new_results"].write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    gate = json.loads(paths["new_failed_gate"].read_text(encoding="utf-8"))
    gate["results_sha256"] = _file_sha256(paths["new_results"])
    _write_json(paths["new_failed_gate"], gate)
    receipt = json.loads(paths["rebind_receipt"].read_text(encoding="utf-8"))
    receipt["artifacts_sha256"]["new_results"] = _file_sha256(paths["new_results"])
    receipt["artifacts_sha256"]["new_failed_gate_report"] = _file_sha256(
        paths["new_failed_gate"]
    )
    _write_json(paths["rebind_receipt"], receipt)
    public = json.loads(paths["rebound_public"].read_text(encoding="utf-8"))
    public["rebind_receipt_sha256"] = _file_sha256(paths["rebind_receipt"])
    _write_json(paths["rebound_public"], public)

    with pytest.raises(ValueError, match="answer prompts changed"):
        _validate_rebind(paths)


def test_quality_rebind_rejects_changed_executable_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _build_rebind_artifacts(tmp_path)
    monkeypatch.setattr(rebind_module, "rebind_contract_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="executable contract"):
        _validate_rebind(paths)


def test_valid_resume_receipt(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    assert _validate(paths)["resumed_case_indexes"] == [30]


def test_resume_receipt_rejects_tampered_parent(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    paths["parent_private"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="parent private replay"):
        _validate(paths)


def test_resume_receipt_rejects_changed_non_target_row(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    child = json.loads(paths["child_private"].read_text(encoding="utf-8"))
    child["cases"][1]["answer"] = "tampered"
    _write_json(paths["child_private"], child)
    public = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    public["private_replay_sha256"] = _file_sha256(paths["child_private"])
    _write_json(paths["child_public"], public)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["child_private_replay_sha256"] = _file_sha256(paths["child_private"])
    receipt["child_private_case_canonical_sha256"][1] = _canonical_sha256(child["cases"][1])
    _write_json(paths["receipt"], receipt)
    public["resume_receipt_sha256"] = _file_sha256(paths["receipt"])
    _write_json(paths["child_public"], public)
    with pytest.raises(ValueError, match="non-resumed case 1 changed"):
        _validate(paths)


def test_resume_receipt_rejects_wrong_case_and_hash(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["resumed_case_indexes"] = [29]
    _write_json(paths["receipt"], receipt)
    public = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    public["resume_receipt_sha256"] = _file_sha256(paths["receipt"])
    _write_json(paths["child_public"], public)
    with pytest.raises(ValueError, match="targeted failure"):
        _validate(paths)

    paths = _build_artifacts(tmp_path / "hash")
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["resumed_case_audit"]["answer_prompt_sha256"] = "0" * 64
    _write_json(paths["receipt"], receipt)
    public = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    public["resume_receipt_sha256"] = _file_sha256(paths["receipt"])
    _write_json(paths["child_public"], public)
    with pytest.raises(ValueError, match="audit does not match"):
        _validate(paths)


def test_resume_receipt_rejects_non_standard_json(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path)
    paths["receipt"].write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON artifact"):
        _validate(paths)


def test_resume_receipt_rejects_changed_executable_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _build_artifacts(tmp_path)
    monkeypatch.setattr(resume_module, "resume_contract_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="executable contract"):
        _validate(paths)


@pytest.mark.parametrize("name", ("dataset", "manifest", "prepared"))
def test_resume_receipt_rejects_changed_parent_input_digest(
    tmp_path: Path,
    name: str,
) -> None:
    paths = _build_artifacts(tmp_path)
    paths[name].write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="parent artifact hash"):
        _validate(paths)


def test_reseal_existing_adds_contract_without_changing_model_artifacts(
    tmp_path: Path,
) -> None:
    paths = _build_artifacts(tmp_path / "source")
    legacy_receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    legacy_receipt.pop("resume_contract_sha256")
    _write_json(paths["receipt"], legacy_receipt)
    old_child = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    old_child["resume_receipt_sha256"] = _file_sha256(paths["receipt"])
    _write_json(paths["child_public"], old_child)
    old_private_bytes = paths["child_private"].read_bytes()
    output_receipt = tmp_path / "r27-receipt.json"
    output_quality = tmp_path / "r27-quality.json"
    output_private = tmp_path / "r27-private.json"

    result = reseal_existing_quality_resume(
        existing_receipt_path=paths["receipt"],
        existing_child_quality_sidecar_path=paths["child_public"],
        existing_child_private_replay_path=paths["child_private"],
        output_receipt_path=output_receipt,
        output_child_quality_sidecar_path=output_quality,
        output_child_private_replay_path=output_private,
        dataset_path=paths["dataset"],
        manifest_path=paths["manifest"],
        prepared_report_path=paths["prepared"],
        parent_quality_sidecar_path=paths["parent_public"],
        parent_private_replay_path=paths["parent_private"],
        parent_visibility_path=paths["visibility"],
        parent_gate_report_path=paths["gate"],
        parent_results_path=paths["results"],
        parent_benchmark_path=paths["benchmark"],
    )

    assert output_private.read_bytes() == old_private_bytes
    assert result["private_replay_sha256"] == hashlib.sha256(old_private_bytes).hexdigest()
    assert result["resume_contract_sha256"] == resume_contract_sha256()
    validate_quality_resume_receipt(
        output_receipt,
        dataset_path=paths["dataset"],
        manifest_path=paths["manifest"],
        prepared_report_path=paths["prepared"],
        parent_quality_sidecar_path=paths["parent_public"],
        parent_private_replay_path=paths["parent_private"],
        parent_visibility_path=paths["visibility"],
        parent_gate_report_path=paths["gate"],
        parent_results_path=paths["results"],
        parent_benchmark_path=paths["benchmark"],
        child_quality_sidecar_path=output_quality,
        child_private_replay_path=output_private,
    )


def test_reseal_existing_rejects_changed_legacy_binding(tmp_path: Path) -> None:
    paths = _build_artifacts(tmp_path / "source")
    legacy_receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    legacy_receipt.pop("resume_contract_sha256")
    legacy_receipt["parent_artifacts_sha256"]["dataset"] = "0" * 64
    _write_json(paths["receipt"], legacy_receipt)
    old_child = json.loads(paths["child_public"].read_text(encoding="utf-8"))
    old_child["resume_receipt_sha256"] = _file_sha256(paths["receipt"])
    _write_json(paths["child_public"], old_child)

    with pytest.raises(ValueError, match="parent artifact hash"):
        reseal_existing_quality_resume(
            existing_receipt_path=paths["receipt"],
            existing_child_quality_sidecar_path=paths["child_public"],
            existing_child_private_replay_path=paths["child_private"],
            output_receipt_path=tmp_path / "r27-receipt.json",
            output_child_quality_sidecar_path=tmp_path / "r27-quality.json",
            output_child_private_replay_path=None,
            dataset_path=paths["dataset"],
            manifest_path=paths["manifest"],
            prepared_report_path=paths["prepared"],
            parent_quality_sidecar_path=paths["parent_public"],
            parent_private_replay_path=paths["parent_private"],
            parent_visibility_path=paths["visibility"],
            parent_gate_report_path=paths["gate"],
            parent_results_path=paths["results"],
            parent_benchmark_path=paths["benchmark"],
        )


class _FakeTransport:
    def __init__(self, values: list[ObservedGeneration]) -> None:
        self.values = values
        self.calls = 0

    def generate(self, prompt_lines: list[str], *, model: str) -> ObservedGeneration:
        del prompt_lines, model
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_resume_case_uses_three_calls_and_preserves_initial_attempt() -> None:
    initial_answer = _answer("fact", ["s1", "s2"])
    answer_prompt = [
        "Allowed citation IDs JSON list: [\"s1\",\"s2\"]. IDs shown elsewhere are not citable."
    ]
    parent = _private_row(30, failed=True)
    parent["answer_prompt"] = answer_prompt
    parent["answer_prompt_sha256"] = hashlib.sha256(
        json.dumps(answer_prompt, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    parent["answer_attempts"][0]["answer"] = initial_answer
    parent["citation_contract_prompt"] = ["Retrieved packet:\nsource=s1: fact\nsource=s2: reaction"]
    repair_raw = json.dumps(_answer("fact", ["s1"]), separators=(",", ":"))
    citation_raw = json.dumps(
        {"citations_minimal": True, "reason_code": "minimal"}, separators=(",", ":")
    )
    judge_raw = json.dumps(
        {
            "answer_grounded": True,
            "answer_correct": False,
            "abstained": False,
            "reason_code": "reference_mismatch",
        },
        separators=(",", ":"),
    )
    transport = _FakeTransport(
        [
            ObservedGeneration(repair_raw, 10, 3, 1.0, "generator"),
            ObservedGeneration(citation_raw, 10, 3, 1.0, "judge"),
            ObservedGeneration(judge_raw, 10, 3, 1.0, "judge"),
        ]
    )
    case = EvaluationCase(
        group_id=1,
        query="question",
        recent_context_message_ids=("recent",),
        expected_evidence_message_ids=("s1",),
        category="exact",
        forbidden_evidence_message_ids=(),
    )
    child_public, child_private, _ = resume_case_rows(
        case=case,
        case_index=30,
        parent_public_case=_public_row(30, failed=True),
        parent_private_case=parent,
        result_row={"history_packet_source_message_ids": ["s1", "s2"]},
        transport=transport,
        generator_model="generator",
        judge_model="judge",
        generation_attempts=1,
        gold_text="source=s1: fact",
        known_source_ids=("s1", "s2"),
        ineligible_source_ids=(),
    )
    assert transport.calls == 3
    assert child_public["answer_protocol_failure_codes"] == []
    assert child_public["answer_correct"] is False
    assert child_private["answer_attempts"][0] == parent["answer_attempts"][0]
    assert child_private["generated_citations"] == ["s1"]
