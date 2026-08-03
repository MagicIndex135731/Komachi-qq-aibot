"""Rebind validated Memory V3 quality artifacts after retrieval-only changes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scripts.resume_memory_v3_quality_replay import (
    EXPECTED_CASE_COUNT,
    RESUMED_QUALITY_VERSION,
    _CHILD_PUBLIC_FIELDS,
    _PARENT_PUBLIC_FIELDS,
    _PRIVATE_CASE_FIELDS,
    _PRIVATE_REPLAY_FIELDS,
    _PUBLIC_CASE_FIELDS,
    _canonical_bytes,
    _canonical_sha256,
    _exact_fields,
    _file_sha256,
    _is_sha256,
    _load_strict_json,
    _load_strict_jsonl,
    _native_index,
    _retrieval_fingerprint,
    _validate_rows,
    _write_new_bytes,
    validate_quality_resume_receipt,
)


REBOUND_QUALITY_VERSION = 5
QUALITY_REBIND_RECEIPT_VERSION = 1
_REBOUND_PUBLIC_FIELDS = {*_PARENT_PUBLIC_FIELDS, "rebind_receipt_sha256"}
_REBIND_ARTIFACT_FIELDS = {
    "dataset",
    "manifest",
    "prepared_report",
    "old_quality_sidecar",
    "old_private_replay",
    "old_resume_receipt",
    "old_visibility",
    "old_gate_report",
    "old_results",
    "old_benchmark",
    "new_failed_gate_report",
    "new_results",
    "new_benchmark",
    "child_private_replay",
    "child_visibility",
}
_REBIND_RECEIPT_FIELDS = {
    "quality_rebind_receipt_version",
    "rebind_contract_sha256",
    "artifacts_sha256",
    "bindings",
    "old_answer_prompt_sha256_by_case",
    "new_answer_prompt_sha256_by_case",
    "old_private_case_canonical_sha256",
    "child_private_case_canonical_sha256",
    "old_public_case_canonical_sha256",
    "child_public_case_canonical_sha256",
}


def rebind_contract_sha256() -> str:
    """Bind receipts to the exact standalone rebind executable source."""

    return _file_sha256(Path(__file__).resolve())


def _answer_prompt_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError("quality rebind results must contain exactly 64 rows")
    values: list[str] = []
    for index, row in enumerate(rows):
        _native_index(row.get("case_index"), expected=index)
        value = row.get("answer_prompt_sha256")
        if not _is_sha256(value):
            raise ValueError(f"quality rebind result {index} has an invalid prompt hash")
        values.append(value)
    return values


def _validate_rebind_gate(
    gate: Mapping[str, Any],
    *,
    results_path: Path,
    benchmark_path: Path,
    expected_status: str,
    required_error: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = _load_strict_jsonl(results_path)
    benchmark = _load_strict_json(benchmark_path)
    acceptance = gate.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("status") != expected_status:
        raise ValueError("quality rebind gate has an invalid acceptance status")
    errors = acceptance.get("error_codes")
    if not isinstance(errors, list) or any(not isinstance(value, str) for value in errors):
        raise ValueError("quality rebind gate has invalid error codes")
    if required_error is None and errors:
        raise ValueError("old quality rebind gate did not pass")
    if required_error is not None and required_error not in errors:
        raise ValueError("new quality rebind gate did not record invalid old quality")
    if gate.get("results_sha256") != _file_sha256(results_path):
        raise ValueError("quality rebind gate results hash does not match")
    if gate.get("benchmark_sha256") != _file_sha256(benchmark_path):
        raise ValueError("quality rebind gate benchmark hash does not match")
    if gate.get("retrieval_fingerprint_sha256") != _retrieval_fingerprint(results):
        raise ValueError("quality rebind gate retrieval fingerprint does not match")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("quality rebind gate metrics are missing")
    safety_metrics = (
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
    if any(metrics.get(field) != 0 for field in safety_metrics):
        raise ValueError("quality rebind gate retrieval safety audit failed")
    for field, minimum in (
        ("recall_at_300", 0.80),
        ("recall_within_32k", 0.80),
        ("time_bucket_coverage_rate", 1.0),
    ):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
            raise ValueError(f"quality rebind gate metric {field} failed")
    if (
        benchmark.get("network_enabled") is not False
        or benchmark.get("rerank_enabled") is not False
        or benchmark.get("vector_success_verified") is not True
    ):
        raise ValueError("quality rebind benchmark safety contract failed")
    return results, benchmark


def validate_quality_rebind_receipt(
    receipt_path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    prepared_report_path: Path,
    old_quality_sidecar_path: Path,
    old_private_replay_path: Path,
    old_resume_receipt_path: Path,
    old_visibility_path: Path,
    old_gate_report_path: Path,
    old_results_path: Path,
    old_benchmark_path: Path,
    old_resume_parent_quality_sidecar_path: Path,
    old_resume_parent_private_replay_path: Path,
    old_resume_parent_gate_report_path: Path,
    old_resume_parent_results_path: Path,
    old_resume_parent_benchmark_path: Path,
    new_failed_gate_report_path: Path,
    new_results_path: Path,
    new_benchmark_path: Path,
    child_quality_sidecar_path: Path,
    child_private_replay_path: Path,
    child_visibility_path: Path,
) -> dict[str, Any]:
    receipt = _exact_fields(
        _load_strict_json(receipt_path),
        _REBIND_RECEIPT_FIELDS,
        label="quality rebind receipt",
    )
    if receipt.get("quality_rebind_receipt_version") != QUALITY_REBIND_RECEIPT_VERSION:
        raise ValueError("unsupported quality rebind receipt version")
    if receipt.get("rebind_contract_sha256") != rebind_contract_sha256():
        raise ValueError("quality rebind executable contract does not match")

    validate_quality_resume_receipt(
        old_resume_receipt_path,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        prepared_report_path=prepared_report_path,
        parent_quality_sidecar_path=old_resume_parent_quality_sidecar_path,
        parent_private_replay_path=old_resume_parent_private_replay_path,
        parent_visibility_path=old_visibility_path,
        parent_gate_report_path=old_resume_parent_gate_report_path,
        parent_results_path=old_resume_parent_results_path,
        parent_benchmark_path=old_resume_parent_benchmark_path,
        child_quality_sidecar_path=old_quality_sidecar_path,
        child_private_replay_path=old_private_replay_path,
    )

    old_public = _exact_fields(
        _load_strict_json(old_quality_sidecar_path),
        _CHILD_PUBLIC_FIELDS,
        label="old resumed quality sidecar",
    )
    old_private = _exact_fields(
        _load_strict_json(old_private_replay_path),
        _PRIVATE_REPLAY_FIELDS,
        label="old resumed private replay",
    )
    child_public = _exact_fields(
        _load_strict_json(child_quality_sidecar_path),
        _REBOUND_PUBLIC_FIELDS,
        label="rebound quality sidecar",
    )
    child_private = _exact_fields(
        _load_strict_json(child_private_replay_path),
        _PRIVATE_REPLAY_FIELDS,
        label="rebound private replay",
    )
    old_visibility = _load_strict_json(old_visibility_path)
    child_visibility = _load_strict_json(child_visibility_path)
    if old_public.get("quality_version") != RESUMED_QUALITY_VERSION:
        raise ValueError("quality rebind parent is not v4")
    if child_public.get("quality_version") != REBOUND_QUALITY_VERSION:
        raise ValueError("quality rebind child is not v5")
    if child_public.get("rebind_receipt_sha256") != _file_sha256(receipt_path):
        raise ValueError("rebound quality sidecar does not bind its receipt")
    if child_public.get("private_replay_sha256") != _file_sha256(child_private_replay_path):
        raise ValueError("rebound quality sidecar does not bind its private replay")
    if child_public.get("visibility_artifact_sha256") != _file_sha256(
        child_visibility_path
    ):
        raise ValueError("rebound quality sidecar does not bind its visibility artifact")

    old_gate = _load_strict_json(old_gate_report_path)
    new_gate = _load_strict_json(new_failed_gate_report_path)
    old_results, _ = _validate_rebind_gate(
        old_gate,
        results_path=old_results_path,
        benchmark_path=old_benchmark_path,
        expected_status="passed",
        required_error=None,
    )
    new_results, _ = _validate_rebind_gate(
        new_gate,
        results_path=new_results_path,
        benchmark_path=new_benchmark_path,
        expected_status="failed",
        required_error="AC_QUALITY_SIDECAR_INVALID",
    )
    old_prompts = _answer_prompt_hashes(old_results)
    new_prompts = _answer_prompt_hashes(new_results)
    if old_prompts != new_prompts:
        raise ValueError("quality rebind answer prompts changed")

    artifacts = _exact_fields(
        receipt.get("artifacts_sha256"),
        _REBIND_ARTIFACT_FIELDS,
        label="quality rebind artifact hashes",
    )
    actual_artifacts = {
        "dataset": _file_sha256(dataset_path),
        "manifest": _file_sha256(manifest_path),
        "prepared_report": _file_sha256(prepared_report_path),
        "old_quality_sidecar": _file_sha256(old_quality_sidecar_path),
        "old_private_replay": _file_sha256(old_private_replay_path),
        "old_resume_receipt": _file_sha256(old_resume_receipt_path),
        "old_visibility": _file_sha256(old_visibility_path),
        "old_gate_report": _file_sha256(old_gate_report_path),
        "old_results": _file_sha256(old_results_path),
        "old_benchmark": _file_sha256(old_benchmark_path),
        "new_failed_gate_report": _file_sha256(new_failed_gate_report_path),
        "new_results": _file_sha256(new_results_path),
        "new_benchmark": _file_sha256(new_benchmark_path),
        "child_private_replay": _file_sha256(child_private_replay_path),
        "child_visibility": _file_sha256(child_visibility_path),
    }
    if artifacts != actual_artifacts:
        raise ValueError("quality rebind artifact hash does not match")

    prepared = _load_strict_json(prepared_report_path)
    generation = prepared.get("vector_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("quality rebind prepared generation is invalid")
    bindings = _exact_fields(
        receipt.get("bindings"),
        {
            "dataset_sha256",
            "snapshot_manifest_sha256",
            "vector_generation",
            "context_profile",
            "prompt_contract_sha256",
            "old_retrieval_fingerprint_sha256",
            "new_retrieval_fingerprint_sha256",
        },
        label="quality rebind bindings",
    )
    old_fingerprint = _retrieval_fingerprint(old_results)
    new_fingerprint = _retrieval_fingerprint(new_results)
    expected_bindings = {
        "dataset_sha256": old_public.get("dataset_sha256"),
        "snapshot_manifest_sha256": old_public.get("snapshot_manifest_sha256"),
        "vector_generation": generation,
        "context_profile": old_public.get("context_profile"),
        "prompt_contract_sha256": old_public.get("prompt_contract_sha256"),
        "old_retrieval_fingerprint_sha256": old_fingerprint,
        "new_retrieval_fingerprint_sha256": new_fingerprint,
    }
    if bindings != expected_bindings:
        raise ValueError("quality rebind semantic bindings do not match")
    for gate in (old_gate, new_gate):
        for field in ("dataset_sha256", "snapshot_manifest_sha256", "context_profile"):
            if gate.get(field) != expected_bindings[field]:
                raise ValueError(f"quality rebind gate {field} changed")
        if gate.get("vector_generation") != generation:
            raise ValueError("quality rebind gate vector generation changed")
    if old_public.get("retrieval_fingerprint_sha256") != old_fingerprint:
        raise ValueError("old quality retrieval fingerprint does not match old results")
    if child_public.get("retrieval_fingerprint_sha256") != new_fingerprint:
        raise ValueError("rebound quality retrieval fingerprint does not match new results")
    if old_private.get("retrieval_fingerprint_sha256") != old_fingerprint:
        raise ValueError("old private retrieval fingerprint does not match")
    if child_private.get("retrieval_fingerprint_sha256") != new_fingerprint:
        raise ValueError("rebound private retrieval fingerprint does not match")
    expected_child_visibility = deepcopy(old_visibility)
    expected_child_visibility["retrieval_fingerprint_sha256"] = new_fingerprint
    if child_visibility != expected_child_visibility:
        raise ValueError("quality rebind changed visibility metadata")
    if receipt.get("old_answer_prompt_sha256_by_case") != old_prompts:
        raise ValueError("quality rebind old prompt hashes do not match")
    if receipt.get("new_answer_prompt_sha256_by_case") != new_prompts:
        raise ValueError("quality rebind new prompt hashes do not match")

    old_public_rows = _validate_rows(
        old_public.get("cases"), fields=_PUBLIC_CASE_FIELDS, label="old public cases"
    )
    child_public_rows = _validate_rows(
        child_public.get("cases"), fields=_PUBLIC_CASE_FIELDS, label="rebound public cases"
    )
    old_private_rows = _validate_rows(
        old_private.get("cases"), fields=_PRIVATE_CASE_FIELDS, label="old private cases"
    )
    child_private_rows = _validate_rows(
        child_private.get("cases"), fields=_PRIVATE_CASE_FIELDS, label="rebound private cases"
    )
    row_sets = (
        ("old_public_case_canonical_sha256", old_public_rows),
        ("child_public_case_canonical_sha256", child_public_rows),
        ("old_private_case_canonical_sha256", old_private_rows),
        ("child_private_case_canonical_sha256", child_private_rows),
    )
    hashes: dict[str, list[str]] = {}
    for field, rows in row_sets:
        actual = [_canonical_sha256(row) for row in rows]
        if receipt.get(field) != actual:
            raise ValueError(f"quality rebind {field} does not match")
        hashes[field] = actual
    if (
        hashes["old_public_case_canonical_sha256"]
        != hashes["child_public_case_canonical_sha256"]
        or hashes["old_private_case_canonical_sha256"]
        != hashes["child_private_case_canonical_sha256"]
    ):
        raise ValueError("quality rebind changed a case row")

    preserved_public = _PARENT_PUBLIC_FIELDS - {
        "quality_version",
        "retrieval_fingerprint_sha256",
        "private_replay_sha256",
        "visibility_artifact_sha256",
    }
    if any(old_public.get(field) != child_public.get(field) for field in preserved_public):
        raise ValueError("quality rebind changed public metadata")
    preserved_private = _PRIVATE_REPLAY_FIELDS - {"retrieval_fingerprint_sha256"}
    if any(old_private.get(field) != child_private.get(field) for field in preserved_private):
        raise ValueError("quality rebind changed private metadata")
    return receipt


def rebind_existing_quality_fingerprint(
    *,
    output_receipt_path: Path,
    output_quality_sidecar_path: Path,
    output_private_replay_path: Path,
    output_visibility_path: Path,
    **paths: Path,
) -> dict[str, str]:
    old_public_path = paths["old_quality_sidecar_path"]
    old_private_path = paths["old_private_replay_path"]
    old_public = _load_strict_json(old_public_path)
    old_private = _load_strict_json(old_private_path)
    new_results = _load_strict_jsonl(paths["new_results_path"])
    new_fingerprint = _retrieval_fingerprint(new_results)
    child_private = deepcopy(old_private)
    child_private["retrieval_fingerprint_sha256"] = new_fingerprint
    private_bytes = _canonical_bytes(child_private) + b"\n"
    private_sha = hashlib.sha256(private_bytes).hexdigest()
    child_visibility = deepcopy(_load_strict_json(paths["old_visibility_path"]))
    child_visibility["retrieval_fingerprint_sha256"] = new_fingerprint
    visibility_bytes = _canonical_bytes(child_visibility) + b"\n"
    visibility_sha = hashlib.sha256(visibility_bytes).hexdigest()
    child_public = deepcopy(old_public)
    child_public.pop("resume_receipt_sha256", None)
    child_public["quality_version"] = REBOUND_QUALITY_VERSION
    child_public["retrieval_fingerprint_sha256"] = new_fingerprint
    child_public["private_replay_sha256"] = private_sha
    child_public["visibility_artifact_sha256"] = visibility_sha

    old_results = _load_strict_jsonl(paths["old_results_path"])
    old_prompts = _answer_prompt_hashes(old_results)
    new_prompts = _answer_prompt_hashes(new_results)
    artifacts = {
        "dataset": _file_sha256(paths["dataset_path"]),
        "manifest": _file_sha256(paths["manifest_path"]),
        "prepared_report": _file_sha256(paths["prepared_report_path"]),
        "old_quality_sidecar": _file_sha256(old_public_path),
        "old_private_replay": _file_sha256(old_private_path),
        "old_resume_receipt": _file_sha256(paths["old_resume_receipt_path"]),
        "old_visibility": _file_sha256(paths["old_visibility_path"]),
        "old_gate_report": _file_sha256(paths["old_gate_report_path"]),
        "old_results": _file_sha256(paths["old_results_path"]),
        "old_benchmark": _file_sha256(paths["old_benchmark_path"]),
        "new_failed_gate_report": _file_sha256(paths["new_failed_gate_report_path"]),
        "new_results": _file_sha256(paths["new_results_path"]),
        "new_benchmark": _file_sha256(paths["new_benchmark_path"]),
        "child_private_replay": private_sha,
        "child_visibility": visibility_sha,
    }
    prepared = _load_strict_json(paths["prepared_report_path"])
    receipt = {
        "quality_rebind_receipt_version": QUALITY_REBIND_RECEIPT_VERSION,
        "rebind_contract_sha256": rebind_contract_sha256(),
        "artifacts_sha256": artifacts,
        "bindings": {
            "dataset_sha256": old_public["dataset_sha256"],
            "snapshot_manifest_sha256": old_public["snapshot_manifest_sha256"],
            "vector_generation": prepared["vector_generation"],
            "context_profile": old_public["context_profile"],
            "prompt_contract_sha256": old_public["prompt_contract_sha256"],
            "old_retrieval_fingerprint_sha256": old_public[
                "retrieval_fingerprint_sha256"
            ],
            "new_retrieval_fingerprint_sha256": new_fingerprint,
        },
        "old_answer_prompt_sha256_by_case": old_prompts,
        "new_answer_prompt_sha256_by_case": new_prompts,
        "old_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in old_private["cases"]
        ],
        "child_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_private["cases"]
        ],
        "old_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in old_public["cases"]
        ],
        "child_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_public["cases"]
        ],
    }
    receipt_bytes = _canonical_bytes(receipt) + b"\n"
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    child_public["rebind_receipt_sha256"] = receipt_sha
    public_bytes = _canonical_bytes(child_public) + b"\n"

    output_paths = {
        output_receipt_path.resolve(),
        output_quality_sidecar_path.resolve(),
        output_private_replay_path.resolve(),
        output_visibility_path.resolve(),
    }
    if len(output_paths) != 4 or any(path.exists() for path in output_paths):
        raise ValueError("quality rebind outputs must be distinct new paths")
    with tempfile.TemporaryDirectory(prefix="memory-v3-quality-rebind-") as directory:
        temporary = Path(directory)
        candidate_receipt = temporary / "receipt.json"
        candidate_public = temporary / "quality.json"
        candidate_private = temporary / "private.json"
        candidate_visibility = temporary / "visibility.json"
        candidate_receipt.write_bytes(receipt_bytes)
        candidate_public.write_bytes(public_bytes)
        candidate_private.write_bytes(private_bytes)
        candidate_visibility.write_bytes(visibility_bytes)
        validate_quality_rebind_receipt(
            candidate_receipt,
            child_quality_sidecar_path=candidate_public,
            child_private_replay_path=candidate_private,
            child_visibility_path=candidate_visibility,
            **paths,
        )
    created: list[Path] = []
    try:
        for path, value, private in (
            (output_private_replay_path, private_bytes, True),
            (output_visibility_path, visibility_bytes, False),
            (output_receipt_path, receipt_bytes, True),
            (output_quality_sidecar_path, public_bytes, False),
        ):
            _write_new_bytes(path, value, private=private)
            created.append(path)
        validate_quality_rebind_receipt(
            output_receipt_path,
            child_quality_sidecar_path=output_quality_sidecar_path,
            child_private_replay_path=output_private_replay_path,
            child_visibility_path=output_visibility_path,
            **paths,
        )
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "quality_sidecar_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "private_replay_sha256": private_sha,
        "visibility_artifact_sha256": visibility_sha,
        "rebind_receipt_sha256": receipt_sha,
        "rebind_contract_sha256": receipt["rebind_contract_sha256"],
        "retrieval_fingerprint_sha256": new_fingerprint,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Memory V3 quality rebind")
    for name in (
        "dataset",
        "manifest",
        "prepared-report",
        "old-quality-sidecar",
        "old-private-replay",
        "old-resume-receipt",
        "old-visibility",
        "old-gate-report",
        "old-results",
        "old-benchmark",
        "old-resume-parent-quality-sidecar",
        "old-resume-parent-private-replay",
        "old-resume-parent-gate-report",
        "old-resume-parent-results",
        "old-resume-parent-benchmark",
        "new-failed-gate-report",
        "new-results",
        "new-benchmark",
        "quality-output",
        "private-replay-output",
        "visibility-output",
        "rebind-receipt-output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = rebind_existing_quality_fingerprint(
        output_receipt_path=args.rebind_receipt_output,
        output_quality_sidecar_path=args.quality_output,
        output_private_replay_path=args.private_replay_output,
        output_visibility_path=args.visibility_output,
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        prepared_report_path=args.prepared_report,
        old_quality_sidecar_path=args.old_quality_sidecar,
        old_private_replay_path=args.old_private_replay,
        old_resume_receipt_path=args.old_resume_receipt,
        old_visibility_path=args.old_visibility,
        old_gate_report_path=args.old_gate_report,
        old_results_path=args.old_results,
        old_benchmark_path=args.old_benchmark,
        old_resume_parent_quality_sidecar_path=args.old_resume_parent_quality_sidecar,
        old_resume_parent_private_replay_path=args.old_resume_parent_private_replay,
        old_resume_parent_gate_report_path=args.old_resume_parent_gate_report,
        old_resume_parent_results_path=args.old_resume_parent_results,
        old_resume_parent_benchmark_path=args.old_resume_parent_benchmark,
        new_failed_gate_report_path=args.new_failed_gate_report,
        new_results_path=args.new_results,
        new_benchmark_path=args.new_benchmark,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
