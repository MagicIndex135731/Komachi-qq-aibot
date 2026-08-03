from __future__ import annotations

"""Resume one failed Memory V3 quality case without replaying successful cases."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from app.config import AppSettings
from app.core.memory_backfill import (
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)

try:
    from scripts.evaluate_memory_recall import load_evaluation_cases
    from scripts.evaluate_memory_v3 import load_message_metadata, load_v3_quality_sidecar
    from scripts.run_memory_v3_quality_replay import (
        AnswerGenerationOutcome,
        GeneratedAnswer,
        ObservedResponsesTransport,
        QualityReplayError,
        _generate_citation_repair_with_retry,
        _generate_valid_json,
        _isolated_llm_client,
        _load_gold_text,
        build_answer_repair_prompt,
        build_citation_contract_prompt,
        build_judge_prompt,
        finalize_replay_case_judgment,
        parse_citation_contract_decision,
        parse_judge_decision,
    )
except ImportError:  # Direct script execution.
    from evaluate_memory_recall import load_evaluation_cases
    from evaluate_memory_v3 import load_message_metadata, load_v3_quality_sidecar
    from run_memory_v3_quality_replay import (
        AnswerGenerationOutcome,
        GeneratedAnswer,
        ObservedResponsesTransport,
        QualityReplayError,
        _generate_citation_repair_with_retry,
        _generate_valid_json,
        _isolated_llm_client,
        _load_gold_text,
        build_answer_repair_prompt,
        build_citation_contract_prompt,
        build_judge_prompt,
        finalize_replay_case_judgment,
        parse_citation_contract_decision,
        parse_judge_decision,
    )


QUALITY_RESUME_RECEIPT_VERSION = 1
RESUMED_QUALITY_VERSION = 4
EXPECTED_CASE_COUNT = 64
EXPECTED_FAILURE_CODE = "citation_not_minimal"
EXPECTED_GATE_CODE = "AC_ANSWER_PROTOCOL"

_PUBLIC_CASE_FIELDS = {
    "case_index",
    "cited_source_message_ids",
    "answer_grounded",
    "answer_correct",
    "abstained",
    "answer_protocol_failure_codes",
    "total_prompt_tokens",
    "ttft_ms",
}
_PRIVATE_CASE_FIELDS = {
    "case_index",
    "query",
    "answer_prompt",
    "answer_prompt_sha256",
    "answer",
    "generated_citations",
    "generated_abstained",
    "answer_protocol_failure_codes",
    "answer_repair_count",
    "answer_observation",
    "answer_attempts",
    "citation_contract_prompt",
    "citation_contract_raw_output",
    "citation_contract_observation",
    "citation_contract_decision",
    "judge_prompt",
    "judge_raw_output",
    "judge_observation",
    "judge_decision",
    "citation_failure_codes",
}
_PARENT_PUBLIC_FIELDS = {
    "quality_version",
    "dataset_sha256",
    "snapshot_manifest_sha256",
    "retrieval_fingerprint_sha256",
    "context_profile",
    "private_replay_sha256",
    "visibility_artifact_sha256",
    "prompt_contract_sha256",
    "judge_provider",
    "judge_model",
    "evaluated_at",
    "index_visibility_ms",
    "cases",
}
_CHILD_PUBLIC_FIELDS = {*_PARENT_PUBLIC_FIELDS, "resume_receipt_sha256"}
_PRIVATE_REPLAY_FIELDS = {
    "private_replay_version",
    "dataset_sha256",
    "snapshot_manifest_sha256",
    "retrieval_fingerprint_sha256",
    "prompt_contract_sha256",
    "generator_model",
    "judge_model",
    "evaluated_at",
    "cases",
}
_PARENT_ARTIFACT_FIELDS = {
    "quality_sidecar",
    "private_replay",
    "visibility",
    "gate_report",
    "results",
    "benchmark",
    "dataset",
    "manifest",
    "prepared_report",
}
_RECEIPT_FIELDS = {
    "resume_receipt_version",
    "resume_contract_sha256",
    "parent_artifacts_sha256",
    "child_private_replay_sha256",
    "bindings",
    "resumed_case_indexes",
    "parent_private_case_canonical_sha256",
    "child_private_case_canonical_sha256",
    "parent_public_case_canonical_sha256",
    "child_public_case_canonical_sha256",
    "resumed_case_audit",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_strict_json(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {Path(path).name}")
    return value


def _load_strict_jsonl(path: Path | str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line, parse_constant=_reject_constant) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSONL artifact: {Path(path).name}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("results JSONL contains a non-object row")
    return rows


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_contract_sha256() -> str:
    """Bind receipts to the exact targeted-resume executable source."""

    return _file_sha256(Path(__file__).resolve())


def _text_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("raw model output is not text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _native_index(value: object, *, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("case_index must be a native non-negative integer")
    if expected is not None and value != expected:
        raise ValueError(f"case_index {value} does not match expected index {expected}")
    return value


def _validate_rows(
    rows: object,
    *,
    fields: set[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(f"{label} must contain exactly {EXPECTED_CASE_COUNT} rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"{label} row {index} fields do not match the contract")
        _native_index(row.get("case_index"), expected=index)
    return rows


def _retrieval_observation_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "case_index",
        "retrieved_source_message_ids",
        "history_packet_source_message_ids",
        "history_packet_tokens",
    )
    if any(field not in row for field in required):
        raise ValueError("results row lacks a retrieval fingerprint field")
    _native_index(row["case_index"])
    for field in required[1:3]:
        value = row[field]
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"results row has invalid {field}")
    tokens = row["history_packet_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("results row has invalid history_packet_tokens")
    return {field: row[field] for field in required}


def _retrieval_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256([_retrieval_observation_payload(row) for row in rows])


def _validate_parent_failure(
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    *,
    case_index: int,
) -> None:
    failed = [
        index
        for index, row in enumerate(public_rows)
        if row.get("answer_protocol_failure_codes")
    ]
    if failed != [case_index]:
        raise ValueError("parent quality sidecar does not have one targeted failure")
    if public_rows[case_index].get("answer_protocol_failure_codes") != [EXPECTED_FAILURE_CODE]:
        raise ValueError("target parent failure is not uniquely citation_not_minimal")
    private = private_rows[case_index]
    if private.get("answer_protocol_failure_codes") != [EXPECTED_FAILURE_CODE]:
        raise ValueError("parent private failure does not match the public sidecar")
    attempts = private.get("answer_attempts")
    if (
        private.get("answer_repair_count") != 1
        or not isinstance(attempts, list)
        or len(attempts) != 2
        or not isinstance(attempts[0], dict)
        or not isinstance(attempts[1], dict)
        or attempts[0].get("kind") != "initial"
        or attempts[1].get("kind") != "citation_repair"
    ):
        raise ValueError("parent target has no unique failed repair attempt")
    final_contract = attempts[1].get("citation_contract_decision")
    if not isinstance(final_contract, dict) or final_contract.get("citations_minimal") is not False:
        raise ValueError("parent repair did not fail citation minimality")


def _validate_public_private_reconciliation(
    public_rows: Sequence[Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    *,
    results: Sequence[Mapping[str, Any]],
) -> None:
    for index, (public, private, result) in enumerate(
        zip(public_rows, private_rows, results, strict=True)
    ):
        prompt = private.get("answer_prompt")
        if not isinstance(prompt, list) or any(not isinstance(line, str) for line in prompt):
            raise ValueError(f"private case {index} has an invalid answer prompt")
        prompt_sha = hashlib.sha256(_canonical_bytes(prompt)).hexdigest()
        if (
            private.get("answer_prompt_sha256") != prompt_sha
            or result.get("answer_prompt_sha256") != prompt_sha
        ):
            raise ValueError(f"case {index} answer prompt binding does not match")
        observation = private.get("answer_observation")
        decision = private.get("judge_decision")
        if not isinstance(observation, dict) or not isinstance(decision, dict):
            raise ValueError(f"private case {index} has invalid replay observations")
        if (
            public.get("cited_source_message_ids") != private.get("generated_citations")
            or public.get("answer_protocol_failure_codes")
            != private.get("answer_protocol_failure_codes")
            or public.get("answer_grounded") != decision.get("answer_grounded")
            or public.get("answer_correct") != decision.get("answer_correct")
            or public.get("abstained") != decision.get("abstained")
            or public.get("total_prompt_tokens") != observation.get("input_tokens")
            or public.get("ttft_ms") != observation.get("ttft_ms")
        ):
            raise ValueError(f"public/private case {index} does not reconcile")


def _validate_gate_and_results(
    *,
    parent_public: Mapping[str, Any],
    gate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    parent_quality_sidecar_path: Path | str,
    parent_results_path: Path | str,
    parent_benchmark_path: Path | str,
) -> None:
    acceptance = gate.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance != {
        "status": "failed",
        "error_codes": [EXPECTED_GATE_CODE],
    }:
        raise ValueError("parent gate did not fail uniquely with AC_ANSWER_PROTOCOL")
    if gate.get("quality_sidecar_sha256") != _file_sha256(parent_quality_sidecar_path):
        raise ValueError("parent gate quality sidecar binding does not match")
    if gate.get("results_sha256") != _file_sha256(parent_results_path):
        raise ValueError("parent gate results binding does not match")
    if gate.get("benchmark_sha256") != _file_sha256(parent_benchmark_path):
        raise ValueError("parent gate benchmark binding does not match")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("answer_protocol_failure_count") != 1:
        raise ValueError("parent gate answer protocol failure count is not one")
    for field in (
        "dataset_sha256",
        "snapshot_manifest_sha256",
        "retrieval_fingerprint_sha256",
        "context_profile",
    ):
        if gate.get(field) != parent_public.get(field):
            raise ValueError(f"parent gate {field} binding does not match")
    if len(results) != EXPECTED_CASE_COUNT:
        raise ValueError("parent results must contain exactly 64 rows")
    for index, row in enumerate(results):
        _native_index(row.get("case_index"), expected=index)
        prompt_sha = row.get("answer_prompt_sha256")
        if not _is_sha256(prompt_sha):
            raise ValueError(f"results row {index} has an invalid answer prompt hash")
    if _retrieval_fingerprint(results) != parent_public.get("retrieval_fingerprint_sha256"):
        raise ValueError("parent results retrieval fingerprint does not match")


def _exact_fields(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields do not match the contract")
    return value


def validate_quality_resume_receipt(
    receipt_path: Path,
    *,
    dataset_path: Path,
    manifest_path: Path,
    prepared_report_path: Path,
    parent_quality_sidecar_path: Path,
    parent_private_replay_path: Path,
    parent_visibility_path: Path,
    parent_gate_report_path: Path,
    parent_results_path: Path,
    parent_benchmark_path: Path,
    child_quality_sidecar_path: Path,
    child_private_replay_path: Path,
) -> dict[str, Any]:
    """Strictly validate a v4 targeted-resume receipt and all bound artifacts."""

    receipt = _exact_fields(_load_strict_json(receipt_path), _RECEIPT_FIELDS, label="resume receipt")
    if receipt.get("resume_receipt_version") != QUALITY_RESUME_RECEIPT_VERSION:
        raise ValueError("unsupported quality resume receipt version")
    if receipt.get("resume_contract_sha256") != resume_contract_sha256():
        raise ValueError("quality resume executable contract does not match")
    resumed = receipt.get("resumed_case_indexes")
    if not isinstance(resumed, list) or len(resumed) != 1:
        raise ValueError("resume receipt must bind exactly one case")
    case_index = _native_index(resumed[0])

    parent_public = _exact_fields(
        _load_strict_json(parent_quality_sidecar_path),
        _PARENT_PUBLIC_FIELDS,
        label="parent quality sidecar",
    )
    parent_private = _exact_fields(
        _load_strict_json(parent_private_replay_path),
        _PRIVATE_REPLAY_FIELDS,
        label="parent private replay",
    )
    child_public = _exact_fields(
        _load_strict_json(child_quality_sidecar_path),
        _CHILD_PUBLIC_FIELDS,
        label="child quality sidecar",
    )
    child_private = _exact_fields(
        _load_strict_json(child_private_replay_path),
        _PRIVATE_REPLAY_FIELDS,
        label="child private replay",
    )
    gate = _load_strict_json(parent_gate_report_path)
    _load_strict_json(parent_benchmark_path)
    results = _load_strict_jsonl(parent_results_path)

    parent_hashes = _exact_fields(
        receipt.get("parent_artifacts_sha256"),
        _PARENT_ARTIFACT_FIELDS,
        label="parent artifact hashes",
    )
    actual_parent_hashes = {
        "quality_sidecar": _file_sha256(parent_quality_sidecar_path),
        "private_replay": _file_sha256(parent_private_replay_path),
        "visibility": _file_sha256(parent_visibility_path),
        "gate_report": _file_sha256(parent_gate_report_path),
        "results": _file_sha256(parent_results_path),
        "benchmark": _file_sha256(parent_benchmark_path),
        "dataset": _file_sha256(dataset_path),
        "manifest": _file_sha256(manifest_path),
        "prepared_report": _file_sha256(prepared_report_path),
    }
    if any(parent_hashes.get(field) != value for field, value in actual_parent_hashes.items()):
        raise ValueError("parent artifact hash does not match the resume receipt")
    if receipt.get("child_private_replay_sha256") != _file_sha256(child_private_replay_path):
        raise ValueError("child private replay hash does not match the resume receipt")
    if child_public.get("quality_version") != RESUMED_QUALITY_VERSION:
        raise ValueError("resumed quality sidecar is not v4")
    if parent_public.get("quality_version") != 3:
        raise ValueError("targeted resume parent quality sidecar is not v3")
    if child_public.get("resume_receipt_sha256") != _file_sha256(receipt_path):
        raise ValueError("child quality sidecar does not bind the resume receipt")
    if child_public.get("private_replay_sha256") != _file_sha256(child_private_replay_path):
        raise ValueError("child quality sidecar does not bind its private replay")
    if parent_public.get("visibility_artifact_sha256") != _file_sha256(parent_visibility_path):
        raise ValueError("parent visibility hash does not match the sidecar")
    if parent_public.get("private_replay_sha256") != _file_sha256(parent_private_replay_path):
        raise ValueError("parent private replay hash does not match the sidecar")
    if child_public.get("visibility_artifact_sha256") != parent_public.get("visibility_artifact_sha256"):
        raise ValueError("resumed sidecar did not preserve visibility bytes")

    parent_public_rows = _validate_rows(
        parent_public.get("cases"), fields=_PUBLIC_CASE_FIELDS, label="parent public cases"
    )
    child_public_rows = _validate_rows(
        child_public.get("cases"), fields=_PUBLIC_CASE_FIELDS, label="child public cases"
    )
    parent_private_rows = _validate_rows(
        parent_private.get("cases"), fields=_PRIVATE_CASE_FIELDS, label="parent private cases"
    )
    child_private_rows = _validate_rows(
        child_private.get("cases"), fields=_PRIVATE_CASE_FIELDS, label="child private cases"
    )
    _validate_parent_failure(
        parent_public_rows,
        parent_private_rows,
        case_index=case_index,
    )
    _validate_gate_and_results(
        parent_public=parent_public,
        gate=gate,
        results=results,
        parent_quality_sidecar_path=parent_quality_sidecar_path,
        parent_results_path=parent_results_path,
        parent_benchmark_path=parent_benchmark_path,
    )
    _validate_public_private_reconciliation(
        parent_public_rows,
        parent_private_rows,
        results=results,
    )
    _validate_public_private_reconciliation(
        child_public_rows,
        child_private_rows,
        results=results,
    )

    bindings = _exact_fields(
        receipt.get("bindings"),
        {
            "dataset_sha256",
            "snapshot_manifest_sha256",
            "retrieval_fingerprint_sha256",
            "prompt_contract_sha256",
            "vector_generation",
            "context_profile",
        },
        label="resume bindings",
    )
    for field in (
        "dataset_sha256",
        "snapshot_manifest_sha256",
        "retrieval_fingerprint_sha256",
        "prompt_contract_sha256",
        "context_profile",
    ):
        if parent_public.get(field) != bindings.get(field) or child_public.get(field) != bindings.get(field):
            raise ValueError(f"resume {field} binding does not match parent and child")
    if gate.get("vector_generation") != bindings.get("vector_generation"):
        raise ValueError("resume vector generation does not match the parent gate")
    parent_private_bindings = {
        field: parent_private.get(field)
        for field in (
            "dataset_sha256",
            "snapshot_manifest_sha256",
            "retrieval_fingerprint_sha256",
            "prompt_contract_sha256",
        )
    }
    child_private_bindings = {
        field: child_private.get(field)
        for field in parent_private_bindings
    }
    if parent_private_bindings != child_private_bindings or any(
        bindings.get(field) != value for field, value in parent_private_bindings.items()
    ):
        raise ValueError("private replay bindings do not match the receipt")
    if (
        parent_private.get("private_replay_version") != 1
        or child_private.get("private_replay_version") != 1
        or parent_public.get("judge_model")
        != f"generator={parent_private.get('generator_model')};judge={parent_private.get('judge_model')}"
        or child_public.get("judge_model")
        != f"generator={child_private.get('generator_model')};judge={child_private.get('judge_model')}"
        or parent_public.get("evaluated_at") != parent_private.get("evaluated_at")
        or child_public.get("evaluated_at") != child_private.get("evaluated_at")
    ):
        raise ValueError("quality replay metadata does not reconcile")
    preserved_public_fields = _PARENT_PUBLIC_FIELDS - {
        "quality_version",
        "private_replay_sha256",
        "evaluated_at",
        "cases",
    }
    if any(parent_public.get(field) != child_public.get(field) for field in preserved_public_fields):
        raise ValueError("resumed public metadata changed outside the allowed fields")
    preserved_private_fields = _PRIVATE_REPLAY_FIELDS - {"evaluated_at", "cases"}
    if any(parent_private.get(field) != child_private.get(field) for field in preserved_private_fields):
        raise ValueError("resumed private metadata changed outside the allowed fields")

    row_hash_fields = (
        ("parent_private_case_canonical_sha256", parent_private_rows),
        ("child_private_case_canonical_sha256", child_private_rows),
        ("parent_public_case_canonical_sha256", parent_public_rows),
        ("child_public_case_canonical_sha256", child_public_rows),
    )
    computed: dict[str, list[str]] = {}
    for field, rows in row_hash_fields:
        values = receipt.get(field)
        if (
            not isinstance(values, list)
            or len(values) != EXPECTED_CASE_COUNT
            or any(not _is_sha256(value) for value in values)
        ):
            raise ValueError(f"resume receipt has invalid {field}")
        actual = [_canonical_sha256(row) for row in rows]
        if values != actual:
            raise ValueError(f"resume receipt {field} does not match artifact rows")
        computed[field] = actual
    for index in range(EXPECTED_CASE_COUNT):
        if index == case_index:
            continue
        if (
            computed["parent_private_case_canonical_sha256"][index]
            != computed["child_private_case_canonical_sha256"][index]
            or computed["parent_public_case_canonical_sha256"][index]
            != computed["child_public_case_canonical_sha256"][index]
        ):
            raise ValueError(f"non-resumed case {index} changed")
    if _canonical_sha256(parent_private_rows[case_index]["answer_attempts"][0]) != _canonical_sha256(
        child_private_rows[case_index]["answer_attempts"][0]
    ):
        raise ValueError("resumed case did not preserve its initial attempt")
    if child_public_rows[case_index].get("answer_protocol_failure_codes") != []:
        raise ValueError("resumed public case still has an answer protocol failure")
    if child_private_rows[case_index].get("answer_protocol_failure_codes") != []:
        raise ValueError("resumed private case still has an answer protocol failure")

    audit = _exact_fields(
        receipt.get("resumed_case_audit"),
        {
            "case_index",
            "stable_retrieval_observation_sha256",
            "answer_prompt_sha256",
            "parent_failed_repair_raw_output_sha256",
            "child_repair_raw_output_sha256",
            "child_citation_reviewer_raw_output_sha256",
            "child_correctness_judge_raw_output_sha256",
        },
        label="resumed case audit",
    )
    _native_index(audit.get("case_index"), expected=case_index)
    if any(not _is_sha256(audit.get(field)) for field in set(audit) - {"case_index"}):
        raise ValueError("resumed case audit contains an invalid SHA-256")
    parent_case = parent_private_rows[case_index]
    child_case = child_private_rows[case_index]
    parent_attempts = parent_case["answer_attempts"]
    child_attempts = child_case["answer_attempts"]
    if not isinstance(child_attempts, list) or len(child_attempts) != 2:
        raise ValueError("resumed child case has an invalid repair audit")
    expected_audit = {
        "case_index": case_index,
        "stable_retrieval_observation_sha256": _canonical_sha256(
            _retrieval_observation_payload(results[case_index])
        ),
        "answer_prompt_sha256": results[case_index]["answer_prompt_sha256"],
        "parent_failed_repair_raw_output_sha256": _text_sha256(
            parent_attempts[1]["observation"]["text"]
        ),
        "child_repair_raw_output_sha256": _text_sha256(
            child_attempts[1]["observation"]["text"]
        ),
        "child_citation_reviewer_raw_output_sha256": _text_sha256(
            child_case["citation_contract_raw_output"]
        ),
        "child_correctness_judge_raw_output_sha256": _text_sha256(
            child_case["judge_raw_output"]
        ),
    }
    if audit != expected_audit:
        raise ValueError("resumed case audit does not match the bound artifacts")
    if parent_case.get("answer_prompt_sha256") != audit["answer_prompt_sha256"] or child_case.get(
        "answer_prompt_sha256"
    ) != audit["answer_prompt_sha256"]:
        raise ValueError("resumed answer prompt binding changed")
    return receipt


def _extract_packet_text(citation_prompt: object) -> str:
    if not isinstance(citation_prompt, list) or len(citation_prompt) != 1 or not isinstance(
        citation_prompt[0], str
    ):
        raise ValueError("parent citation contract prompt is invalid")
    marker = "Retrieved packet:\n"
    before, separator, packet = citation_prompt[0].rpartition(marker)
    del before
    if not separator or not packet:
        raise ValueError("parent citation contract prompt has no packet")
    return packet


def _extract_allowed_citation_ids(answer_prompt: object) -> tuple[str, ...]:
    if not isinstance(answer_prompt, list) or any(not isinstance(line, str) for line in answer_prompt):
        raise ValueError("parent answer prompt is invalid")
    marker = "Allowed citation IDs JSON list: "
    for line in answer_prompt:
        position = line.find(marker)
        if position < 0:
            continue
        value, _ = json.JSONDecoder(parse_constant=_reject_constant).raw_decode(
            line[position + len(marker) :]
        )
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            break
        return tuple(value)
    raise ValueError("parent answer prompt has no allowed citation list")


def resume_case_rows(
    *,
    case: object,
    case_index: int,
    parent_public_case: Mapping[str, Any],
    parent_private_case: Mapping[str, Any],
    result_row: Mapping[str, Any],
    transport: object,
    generator_model: str,
    judge_model: str,
    generation_attempts: int,
    gold_text: str,
    known_source_ids: Sequence[str],
    ineligible_source_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Perform the only three normal resume calls: repair, citation review, judge."""

    _native_index(case_index)
    answer_prompt = parent_private_case["answer_prompt"]
    allowed_ids = _extract_allowed_citation_ids(answer_prompt)
    packet_text = _extract_packet_text(parent_private_case["citation_contract_prompt"])
    attempts = parent_private_case["answer_attempts"]
    initial_attempt = deepcopy(attempts[0])
    initial_answer_value = initial_attempt["answer"]
    initial_answer = GeneratedAnswer(
        answer=str(initial_answer_value["answer"]),
        cited_source_message_ids=tuple(initial_answer_value["cited_source_message_ids"]),
        abstained=bool(initial_answer_value["abstained"]),
    )
    repair_prompt = build_answer_repair_prompt(
        original_prompt=answer_prompt,
        answer=initial_answer,
        protocol_failure_codes=(EXPECTED_FAILURE_CODE,),
    )
    citation_cap = max(0, len(initial_answer.cited_source_message_ids) - 1)
    repair_prompt.append(
        "The previous citation reviewer rejected the current citation set as "
        "non-minimal. Keep the answer text and abstained value byte-for-byte "
        f"unchanged, but return at most {citation_cap} cited source message ID(s). "
        "Choose only the single most direct supporting source when one source is "
        "sufficient; do not retain a second citation merely for context."
    )
    repaired = _generate_citation_repair_with_retry(
        transport,
        repair_prompt,
        model=generator_model,
        attempts=generation_attempts,
        original_answer=initial_answer,
        allowed_citation_ids=allowed_ids,
    )
    if repaired.protocol_failure_codes:
        raise QualityReplayError("QUALITY_REPLAY_RESUME_REPAIR_INVALID")
    citation_prompt = build_citation_contract_prompt(
        case=case,
        answer=repaired.answer,
        packet_text=packet_text,
    )
    citation_observed, citation_decision = _generate_valid_json(
        transport,
        citation_prompt,
        model=judge_model,
        attempts=generation_attempts,
        parser=parse_citation_contract_decision,
    )
    if not citation_decision.citations_minimal:
        raise QualityReplayError("QUALITY_REPLAY_RESUME_CITATION_NOT_MINIMAL")
    judge_prompt = build_judge_prompt(
        case=case,
        answer=repaired.answer,
        packet_text=packet_text,
        gold_text=gold_text,
    )
    judge_observed, raw_decision = _generate_valid_json(
        transport,
        judge_prompt,
        model=judge_model,
        attempts=generation_attempts,
        parser=parse_judge_decision,
    )
    outcome = AnswerGenerationOutcome(repaired.observation, repaired.answer)
    decision, citation_failures = finalize_replay_case_judgment(
        case=case,
        answer_outcome=outcome,
        raw_decision=raw_decision,
        packet_source_ids=result_row["history_packet_source_message_ids"],
        forbidden_source_ids=getattr(case, "forbidden_evidence_message_ids"),
        known_source_ids=known_source_ids,
        ineligible_source_ids=ineligible_source_ids,
    )
    if citation_failures:
        raise QualityReplayError("QUALITY_REPLAY_RESUME_JUDGMENT_FAILED")
    repair_attempt = {
        "kind": "citation_repair",
        "prompt": repair_prompt,
        "answer": asdict(repaired.answer),
        "observation": asdict(repaired.observation),
        "protocol_failure_codes": [],
        "citation_contract_prompt": citation_prompt,
        "citation_contract_raw_output": citation_observed.text,
        "citation_contract_observation": asdict(citation_observed),
        "citation_contract_decision": asdict(citation_decision),
    }
    child_private = deepcopy(dict(parent_private_case))
    child_private.update(
        {
            "answer": repaired.answer.answer,
            "generated_citations": list(repaired.answer.cited_source_message_ids),
            "generated_abstained": repaired.answer.abstained,
            "answer_protocol_failure_codes": [],
            "answer_repair_count": 1,
            "answer_attempts": [initial_attempt, repair_attempt],
            "citation_contract_prompt": citation_prompt,
            "citation_contract_raw_output": citation_observed.text,
            "citation_contract_observation": asdict(citation_observed),
            "citation_contract_decision": asdict(citation_decision),
            "judge_prompt": judge_prompt,
            "judge_raw_output": judge_observed.text,
            "judge_observation": asdict(judge_observed),
            "judge_decision": asdict(decision),
            "citation_failure_codes": list(citation_failures),
        }
    )
    child_public = deepcopy(dict(parent_public_case))
    child_public.update(
        {
            "cited_source_message_ids": list(repaired.answer.cited_source_message_ids),
            "answer_grounded": decision.answer_grounded,
            "answer_correct": decision.answer_correct,
            "abstained": decision.abstained,
            "answer_protocol_failure_codes": [],
        }
    )
    raw_hashes = {
        "parent_failed_repair_raw_output_sha256": _text_sha256(
            attempts[1]["observation"]["text"]
        ),
        "child_repair_raw_output_sha256": _text_sha256(repaired.observation.text),
        "child_citation_reviewer_raw_output_sha256": _text_sha256(
            citation_observed.text
        ),
        "child_correctness_judge_raw_output_sha256": _text_sha256(
            judge_observed.text
        ),
    }
    return child_public, child_private, raw_hashes


def _write_new_json(path: Path, value: object, *, private: bool) -> str:
    rendered = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600 if private else 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(rendered).hexdigest()


def _write_new_bytes(path: Path, value: bytes, *, private: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600 if private else 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(value).hexdigest()


def reseal_existing_quality_resume(
    *,
    existing_receipt_path: Path,
    existing_child_quality_sidecar_path: Path,
    existing_child_private_replay_path: Path,
    output_receipt_path: Path,
    output_child_quality_sidecar_path: Path,
    output_child_private_replay_path: Path | None,
    dataset_path: Path,
    manifest_path: Path,
    prepared_report_path: Path,
    parent_quality_sidecar_path: Path,
    parent_private_replay_path: Path,
    parent_visibility_path: Path,
    parent_gate_report_path: Path,
    parent_results_path: Path,
    parent_benchmark_path: Path,
) -> dict[str, str]:
    """Offline-only re-seal of a pre-contract v4 targeted replay."""

    legacy_fields = _RECEIPT_FIELDS - {"resume_contract_sha256"}
    old_receipt = _exact_fields(
        _load_strict_json(existing_receipt_path),
        legacy_fields,
        label="legacy resume receipt",
    )
    old_child = _exact_fields(
        _load_strict_json(existing_child_quality_sidecar_path),
        _CHILD_PUBLIC_FIELDS,
        label="existing resumed quality sidecar",
    )
    if old_child.get("quality_version") != RESUMED_QUALITY_VERSION:
        raise ValueError("existing resumed quality sidecar is not v4")
    if old_child.get("resume_receipt_sha256") != _file_sha256(existing_receipt_path):
        raise ValueError("existing resumed sidecar does not bind the legacy receipt")
    private_bytes = existing_child_private_replay_path.read_bytes()
    if old_child.get("private_replay_sha256") != hashlib.sha256(private_bytes).hexdigest():
        raise ValueError("existing resumed sidecar does not bind its private replay")

    output_paths = {
        output_receipt_path.resolve(),
        output_child_quality_sidecar_path.resolve(),
    }
    if output_child_private_replay_path is not None:
        output_paths.add(output_child_private_replay_path.resolve())
    input_paths = {
        existing_receipt_path.resolve(),
        existing_child_quality_sidecar_path.resolve(),
        existing_child_private_replay_path.resolve(),
        dataset_path.resolve(),
        manifest_path.resolve(),
        prepared_report_path.resolve(),
        parent_quality_sidecar_path.resolve(),
        parent_private_replay_path.resolve(),
        parent_visibility_path.resolve(),
        parent_gate_report_path.resolve(),
        parent_results_path.resolve(),
        parent_benchmark_path.resolve(),
    }
    if len(output_paths) != (3 if output_child_private_replay_path is not None else 2):
        raise ValueError("reseal outputs must be distinct")
    if output_paths & input_paths or any(path.exists() for path in output_paths):
        raise ValueError("reseal outputs must be new paths")

    new_receipt = dict(old_receipt)
    new_receipt["resume_contract_sha256"] = resume_contract_sha256()
    receipt_bytes = _canonical_bytes(new_receipt) + b"\n"
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    new_child = deepcopy(old_child)
    new_child["resume_receipt_sha256"] = receipt_sha
    child_bytes = _canonical_bytes(new_child) + b"\n"

    with tempfile.TemporaryDirectory(prefix="memory-v3-quality-reseal-") as directory:
        temporary = Path(directory)
        candidate_receipt = temporary / "receipt.json"
        candidate_child = temporary / "quality.json"
        candidate_private = temporary / "private.json"
        candidate_receipt.write_bytes(receipt_bytes)
        candidate_child.write_bytes(child_bytes)
        candidate_private.write_bytes(private_bytes)
        validate_quality_resume_receipt(
            candidate_receipt,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            prepared_report_path=prepared_report_path,
            parent_quality_sidecar_path=parent_quality_sidecar_path,
            parent_private_replay_path=parent_private_replay_path,
            parent_visibility_path=parent_visibility_path,
            parent_gate_report_path=parent_gate_report_path,
            parent_results_path=parent_results_path,
            parent_benchmark_path=parent_benchmark_path,
            child_quality_sidecar_path=candidate_child,
            child_private_replay_path=candidate_private,
        )

    created: list[Path] = []
    try:
        private_output = existing_child_private_replay_path
        if output_child_private_replay_path is not None:
            _write_new_bytes(output_child_private_replay_path, private_bytes, private=True)
            created.append(output_child_private_replay_path)
            private_output = output_child_private_replay_path
        _write_new_bytes(output_receipt_path, receipt_bytes, private=True)
        created.append(output_receipt_path)
        _write_new_bytes(output_child_quality_sidecar_path, child_bytes, private=False)
        created.append(output_child_quality_sidecar_path)
        validate_quality_resume_receipt(
            output_receipt_path,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            prepared_report_path=prepared_report_path,
            parent_quality_sidecar_path=parent_quality_sidecar_path,
            parent_private_replay_path=parent_private_replay_path,
            parent_visibility_path=parent_visibility_path,
            parent_gate_report_path=parent_gate_report_path,
            parent_results_path=parent_results_path,
            parent_benchmark_path=parent_benchmark_path,
            child_quality_sidecar_path=output_child_quality_sidecar_path,
            child_private_replay_path=private_output,
        )
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "quality_sidecar_sha256": hashlib.sha256(child_bytes).hexdigest(),
        "private_replay_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "resume_receipt_sha256": receipt_sha,
        "resume_contract_sha256": new_receipt["resume_contract_sha256"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume one Memory V3 quality replay case")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prepared-report", required=True, type=Path)
    parser.add_argument("--parent-quality-sidecar", required=True, type=Path)
    parser.add_argument("--parent-private-replay", required=True, type=Path)
    parser.add_argument("--parent-visibility", required=True, type=Path)
    parser.add_argument("--parent-gate-report", required=True, type=Path)
    parser.add_argument("--parent-results", required=True, type=Path)
    parser.add_argument("--parent-benchmark", required=True, type=Path)
    parser.add_argument("--quality-output", required=True, type=Path)
    parser.add_argument("--private-replay-output", type=Path)
    parser.add_argument("--resume-receipt-output", required=True, type=Path)
    parser.add_argument("--reseal-existing", action="store_true")
    parser.add_argument("--existing-quality-sidecar", type=Path)
    parser.add_argument("--existing-private-replay", type=Path)
    parser.add_argument("--existing-resume-receipt", type=Path)
    parser.add_argument("--case-index", type=int, default=30)
    parser.add_argument("--generation-attempts", type=int, default=1)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.reseal_existing:
        existing_paths = (
            args.existing_quality_sidecar,
            args.existing_private_replay,
            args.existing_resume_receipt,
        )
        if not all(path is not None for path in existing_paths):
            raise ValueError("reseal-existing requires all existing child artifacts")
        result = reseal_existing_quality_resume(
            existing_receipt_path=args.existing_resume_receipt,
            existing_child_quality_sidecar_path=args.existing_quality_sidecar,
            existing_child_private_replay_path=args.existing_private_replay,
            output_receipt_path=args.resume_receipt_output,
            output_child_quality_sidecar_path=args.quality_output,
            output_child_private_replay_path=args.private_replay_output,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            prepared_report_path=args.prepared_report,
            parent_quality_sidecar_path=args.parent_quality_sidecar,
            parent_private_replay_path=args.parent_private_replay,
            parent_visibility_path=args.parent_visibility,
            parent_gate_report_path=args.parent_gate_report,
            parent_results_path=args.parent_results,
            parent_benchmark_path=args.parent_benchmark,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.private_replay_output is None:
        raise ValueError("targeted replay requires private-replay-output")
    case_index = _native_index(args.case_index)
    output_paths = {
        args.quality_output.resolve(),
        args.private_replay_output.resolve(),
        args.resume_receipt_output.resolve(),
    }
    parent_paths = {
        args.parent_quality_sidecar.resolve(),
        args.parent_private_replay.resolve(),
        args.parent_visibility.resolve(),
        args.parent_gate_report.resolve(),
        args.parent_results.resolve(),
        args.parent_benchmark.resolve(),
    }
    if len(output_paths) != 3 or output_paths & parent_paths:
        raise ValueError("resume outputs must be distinct new paths")
    if any(path.exists() for path in output_paths):
        raise ValueError("resume output path already exists")

    cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    if len(cases) != EXPECTED_CASE_COUNT or case_index >= len(cases):
        raise ValueError("resume dataset or case index is invalid")
    parent_public = _load_strict_json(args.parent_quality_sidecar)
    parent_private = _load_strict_json(args.parent_private_replay)
    gate = _load_strict_json(args.parent_gate_report)
    results = _load_strict_jsonl(args.parent_results)
    _load_strict_json(args.parent_benchmark)
    manifest = _load_strict_json(args.manifest)
    prepared = _load_strict_json(args.prepared_report)
    public_rows = _validate_rows(
        parent_public.get("cases"), fields=_PUBLIC_CASE_FIELDS, label="parent public cases"
    )
    private_rows = _validate_rows(
        parent_private.get("cases"), fields=_PRIVATE_CASE_FIELDS, label="parent private cases"
    )
    _validate_parent_failure(public_rows, private_rows, case_index=case_index)
    _validate_gate_and_results(
        parent_public=parent_public,
        gate=gate,
        results=results,
        parent_quality_sidecar_path=args.parent_quality_sidecar,
        parent_results_path=args.parent_results,
        parent_benchmark_path=args.parent_benchmark,
    )
    manifest_sha = message_ledger_manifest_sha256(manifest)
    if dataset_sha256 != parent_public.get("dataset_sha256"):
        raise ValueError("dataset hash does not match the parent replay")
    if manifest_sha != parent_public.get("snapshot_manifest_sha256"):
        raise ValueError("manifest hash does not match the parent replay")
    generation = prepared.get("vector_generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
        or prepared.get("phase") != "prepared"
        or prepared.get("database_path") != str(args.database.resolve())
        or prepared.get("vector_status") != "ready"
        or not isinstance(prepared.get("vector_identity"), dict)
        or prepared.get("manifest_sha256") != manifest_sha
        or gate.get("vector_generation") != generation
    ):
        raise ValueError("prepared generation does not match the parent gate")
    if not verify_message_ledger_manifest(args.database, manifest).matches:
        raise ValueError("database no longer matches the parent manifest")
    if parent_public.get("private_replay_sha256") != _file_sha256(args.parent_private_replay):
        raise ValueError("parent private replay hash does not match")
    if parent_public.get("visibility_artifact_sha256") != _file_sha256(args.parent_visibility):
        raise ValueError("parent visibility artifact hash does not match")
    if results[case_index]["answer_prompt_sha256"] != private_rows[case_index][
        "answer_prompt_sha256"
    ]:
        raise ValueError("target answer prompt hash does not match the gate result")
    load_v3_quality_sidecar(
        args.parent_quality_sidecar,
        dataset_sha256=dataset_sha256,
        snapshot_manifest_sha256=manifest_sha,
        retrieval_fingerprint=parent_public["retrieval_fingerprint_sha256"],
        case_count=len(cases),
        private_replay_path=args.parent_private_replay,
        visibility_artifact_path=args.parent_visibility,
        expected_vector_generation=generation,
        evaluation_cases=cases,
        expected_answer_prompt_sha256_by_case={
            index: row["answer_prompt_sha256"] for index, row in enumerate(results)
        },
        expected_context_profile=parent_public["context_profile"],
    )

    settings = AppSettings()
    client = _isolated_llm_client(settings)
    try:
        generator_model = str(parent_private.get("generator_model") or "")
        judge_model = str(parent_private.get("judge_model") or "")
        if not generator_model or not judge_model or client.responses_model != generator_model:
            raise ValueError("configured model does not match the parent replay")
        metadata = load_message_metadata(args.database)
        child_public_case, child_private_case, raw_hashes = resume_case_rows(
            case=cases[case_index],
            case_index=case_index,
            parent_public_case=public_rows[case_index],
            parent_private_case=private_rows[case_index],
            result_row=results[case_index],
            transport=ObservedResponsesTransport(client),
            generator_model=generator_model,
            judge_model=judge_model,
            generation_attempts=max(1, args.generation_attempts),
            gold_text=_load_gold_text(args.database, cases[case_index]),
            known_source_ids=tuple(metadata),
            ineligible_source_ids=tuple(
                source_id for source_id, row in metadata.items() if not row.eligible
            ),
        )
    finally:
        client.http_client.close()

    child_private = deepcopy(parent_private)
    child_private["evaluated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    child_private["cases"] = deepcopy(private_rows)
    child_private["cases"][case_index] = child_private_case
    child_private_sha = _write_new_json(args.private_replay_output, child_private, private=True)

    child_public = deepcopy(parent_public)
    child_public["quality_version"] = RESUMED_QUALITY_VERSION
    child_public["evaluated_at"] = child_private["evaluated_at"]
    child_public["private_replay_sha256"] = child_private_sha
    child_public["cases"] = deepcopy(public_rows)
    child_public["cases"][case_index] = child_public_case
    parent_artifacts = {
        "quality_sidecar": _file_sha256(args.parent_quality_sidecar),
        "private_replay": _file_sha256(args.parent_private_replay),
        "visibility": _file_sha256(args.parent_visibility),
        "gate_report": _file_sha256(args.parent_gate_report),
        "results": _file_sha256(args.parent_results),
        "benchmark": _file_sha256(args.parent_benchmark),
        "dataset": _file_sha256(args.dataset),
        "manifest": _file_sha256(args.manifest),
        "prepared_report": _file_sha256(args.prepared_report),
    }
    receipt = {
        "resume_receipt_version": QUALITY_RESUME_RECEIPT_VERSION,
        "resume_contract_sha256": resume_contract_sha256(),
        "parent_artifacts_sha256": parent_artifacts,
        "child_private_replay_sha256": child_private_sha,
        "bindings": {
            "dataset_sha256": dataset_sha256,
            "snapshot_manifest_sha256": manifest_sha,
            "retrieval_fingerprint_sha256": parent_public["retrieval_fingerprint_sha256"],
            "prompt_contract_sha256": parent_public["prompt_contract_sha256"],
            "vector_generation": generation,
            "context_profile": parent_public["context_profile"],
        },
        "resumed_case_indexes": [case_index],
        "parent_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in private_rows
        ],
        "child_private_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_private["cases"]
        ],
        "parent_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in public_rows
        ],
        "child_public_case_canonical_sha256": [
            _canonical_sha256(row) for row in child_public["cases"]
        ],
        "resumed_case_audit": {
            "case_index": case_index,
            "stable_retrieval_observation_sha256": _canonical_sha256(
                _retrieval_observation_payload(results[case_index])
            ),
            "answer_prompt_sha256": results[case_index]["answer_prompt_sha256"],
            **raw_hashes,
        },
    }
    try:
        receipt_sha = _write_new_json(args.resume_receipt_output, receipt, private=True)
        child_public["resume_receipt_sha256"] = receipt_sha
        _write_new_json(args.quality_output, child_public, private=False)
        validate_quality_resume_receipt(
            args.resume_receipt_output,
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            prepared_report_path=args.prepared_report,
            parent_quality_sidecar_path=args.parent_quality_sidecar,
            parent_private_replay_path=args.parent_private_replay,
            parent_visibility_path=args.parent_visibility,
            parent_gate_report_path=args.parent_gate_report,
            parent_results_path=args.parent_results,
            parent_benchmark_path=args.parent_benchmark,
            child_quality_sidecar_path=args.quality_output,
            child_private_replay_path=args.private_replay_output,
        )
    except Exception:
        args.quality_output.unlink(missing_ok=True)
        args.resume_receipt_output.unlink(missing_ok=True)
        args.private_replay_output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "case_index": case_index,
                "quality_sidecar_sha256": _file_sha256(args.quality_output),
                "private_replay_sha256": child_private_sha,
                "resume_receipt_sha256": receipt_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except (OSError, ValueError, QualityReplayError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
