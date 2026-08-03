from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

try:
    from .evaluate_memory_recall import EvaluationCase
except ImportError:  # Direct script execution.
    from evaluate_memory_recall import EvaluationCase

try:
    from .memory_v3_quality_contract import (
        QUALITY_REPLAY_PROVIDER,
        answer_contract_failure_codes,
        prompt_contract_sha256,
    )
except ImportError:  # Direct script execution.
    from memory_v3_quality_contract import (
        QUALITY_REPLAY_PROVIDER,
        answer_contract_failure_codes,
        prompt_contract_sha256,
    )


V3_DATASET_SCHEMA_VERSION = 3
V3_QUALITY_SIDECAR_VERSION = 2
V3_ANSWER_MODES = frozenset(
    {
        "exact",
        "mention",
        "dated_history",
        "summary",
        "assessment",
        "current_fact",
        "general_history",
    }
)
V3_COVERAGE_STRATEGIES = frozenset(
    {"relevance", "chronological", "time_buckets"}
)


def _strict_replay_json_object(value: object, *, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("V3 private replay raw output is not text")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant: {constant}")

    parsed = json.loads(value, parse_constant=reject_constant)
    if not isinstance(parsed, dict) or set(parsed) != fields:
        raise ValueError("V3 private replay raw output fields are invalid")
    return parsed


def _validate_replay_observation(
    value: object,
    *,
    expected_model: str,
) -> dict[str, Any]:
    fields = {"text", "input_tokens", "output_tokens", "ttft_ms", "model", "endpoint"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("V3 private replay observation schema is invalid")
    if (
        not isinstance(value["text"], str)
        or isinstance(value["input_tokens"], bool)
        or not isinstance(value["input_tokens"], int)
        or value["input_tokens"] <= 0
        or isinstance(value["output_tokens"], bool)
        or not isinstance(value["output_tokens"], int)
        or value["output_tokens"] < 0
        or isinstance(value["ttft_ms"], bool)
        or not isinstance(value["ttft_ms"], (int, float))
        or not math.isfinite(value["ttft_ms"])
        or value["ttft_ms"] < 0
        or value["model"] != expected_model
        or value["endpoint"] != "responses"
    ):
        raise ValueError("V3 private replay observation values are invalid")
    return value


def _validate_reason_code(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 96
        and not any(character.isspace() for character in value)
    )


def _validate_replay_answer(value: object) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != {
        "answer",
        "cited_source_message_ids",
        "abstained",
    }:
        raise ValueError("V3 private replay answer schema is invalid")
    answer = value["answer"]
    citations = value["cited_source_message_ids"]
    failures = answer_contract_failure_codes(
        answer=answer,
        citations=citations,
        abstained=value["abstained"],
    )
    return value, failures


V3_REQUIRED_GATE_TAGS = frozenset(
    {
        "abstention",
        "blocked_reserved",
        "cross_group",
        "first_person",
        "mention",
        "source_resolution",
        "subject",
        "time_bucket",
        "time_range",
    }
)
_INELIGIBLE_DELIVERY_STATES = frozenset(
    {"reserved", "blocked", "uncertain", "deleted"}
)


@dataclass(frozen=True, slots=True)
class MessageMetadata:
    row_id: int
    source_message_id: str
    group_id: int
    user_id: str
    timestamp: datetime
    delivery_state: str
    reply_to_message_id: str | None = None

    @property
    def eligible(self) -> bool:
        return self.delivery_state not in _INELIGIBLE_DELIVERY_STATES


@dataclass(frozen=True, slots=True)
class V3Observation:
    case_index: int
    retrieved_source_message_ids: tuple[str, ...]
    history_packet_source_message_ids: tuple[str, ...]
    history_packet_tokens: int
    memory_context_tokens: int
    recent_message_count: int
    eligible_history_count: int
    covered_time_bucket_count: int
    required_time_bucket_count: int
    retrieval_latency_ms: float
    group_leak_count: int
    subject_leak_count: int
    time_leak_count: int
    ineligible_source_count: int
    unresolved_source_count: int
    outside_snapshot_source_count: int
    forbidden_source_count: int
    plan_mismatch_count: int
    derived_evidence_count: int


@dataclass(frozen=True, slots=True)
class V3QualityCase:
    case_index: int
    cited_source_message_ids: tuple[str, ...]
    answer_grounded: bool
    answer_correct: bool
    abstained: bool
    answer_protocol_failure_codes: tuple[str, ...]
    total_prompt_tokens: int
    ttft_ms: float


@dataclass(frozen=True, slots=True)
class V3QualitySidecar:
    dataset_sha256: str
    snapshot_manifest_sha256: str
    retrieval_fingerprint_sha256: str
    private_replay_sha256: str
    visibility_artifact_sha256: str
    prompt_contract_sha256: str
    judge_provider: str
    judge_model: str
    evaluated_at: str
    index_visibility_ms: tuple[float, ...]
    cases: tuple[V3QualityCase, ...]


def validate_v3_dataset_contract(cases: Sequence[EvaluationCase]) -> dict[str, int]:
    """Validate frozen V3 scopes before any private query is executed."""

    if not cases:
        raise ValueError("V3 evaluation dataset is empty")
    tag_counts: Counter[str] = Counter()
    for index, case in enumerate(cases):
        if case.schema_version != V3_DATASET_SCHEMA_VERSION:
            raise ValueError(f"case {index} is not a V3 evaluation contract")
        if not case.contract_fields_complete:
            raise ValueError(f"case {index} does not freeze every V3 scope field")
        if case.requester_uin is None:
            raise ValueError(f"case {index} does not freeze requester_uin")
        if case.expected_answer_mode not in V3_ANSWER_MODES:
            raise ValueError(f"case {index} has an invalid expected_answer_mode")
        if case.expected_coverage_strategy not in V3_COVERAGE_STRATEGIES:
            raise ValueError(
                f"case {index} has an invalid expected_coverage_strategy"
            )
        if case.time_range is not None:
            start_at, end_at = (_parse_datetime(value) for value in case.time_range)
            if start_at >= end_at:
                raise ValueError(f"case {index} time_range is not a half-open range")
        if case.expected_coverage_strategy == "time_buckets":
            if case.minimum_time_bucket_count < 2:
                raise ValueError(
                    f"case {index} time-bucket coverage has no multi-bucket minimum"
                )
        elif case.minimum_time_bucket_count:
            raise ValueError(
                f"case {index} sets time buckets for a non-time-bucket strategy"
            )
        tags = set(case.gate_tags)
        if len(tags) != len(case.gate_tags):
            raise ValueError(f"case {index} has duplicate gate tags")
        if "abstention" in tags and case.expected_evidence_message_ids:
            raise ValueError(f"case {index} abstention contract has gold evidence")
        if case.category == "abstention" and "abstention" not in tags:
            raise ValueError(f"case {index} abstention is not gate tagged")
        if "subject" in tags and (
            case.allowed_subject_user_ids is None
            or case.allowed_evidence_user_ids is None
        ):
            raise ValueError(f"case {index} subject scope is not frozen")
        if "first_person" in tags and (
            case.allowed_subject_user_ids != (case.requester_uin,)
        ):
            raise ValueError(f"case {index} first-person subject is not requester")
        if "time_range" in tags and case.time_range is None:
            raise ValueError(f"case {index} time scope is not frozen")
        if "time_bucket" in tags and (
            case.expected_coverage_strategy != "time_buckets"
        ):
            raise ValueError(f"case {index} time-bucket strategy is not frozen")
        if "mention" in tags and case.expected_answer_mode != "mention":
            raise ValueError(f"case {index} mention mode is not frozen")
        if (
            {"blocked_reserved", "cross_group"} & tags
            and not case.forbidden_evidence_message_ids
        ):
            raise ValueError(f"case {index} has no frozen forbidden sources")
        tag_counts.update(tags)
    missing_tags = sorted(V3_REQUIRED_GATE_TAGS - set(tag_counts))
    if missing_tags:
        raise ValueError(
            "V3 evaluation dataset is missing gate coverage: "
            + ",".join(missing_tags)
        )
    return dict(sorted(tag_counts.items()))


def load_message_metadata(
    database: Path | str,
) -> dict[str, MessageMetadata]:
    """Load provenance-only ledger metadata; never selects chat正文."""

    connection = sqlite3.connect(str(database))
    try:
        rows = connection.execute(
            "SELECT id, platform_msg_id, group_id, user_id, timestamp, "
            "reply_to_msg_id, "
            "CASE WHEN json_valid(raw_json) "
            "THEN coalesce(json_extract(raw_json, '$.delivery_state'), '') "
            "ELSE '' END AS delivery_state "
            "FROM messages WHERE group_id IS NOT NULL"
        )
        metadata: dict[str, MessageMetadata] = {}
        for row in rows:
            source_id = str(row[1])
            if source_id in metadata:
                raise ValueError("message ledger contains duplicate source IDs")
            metadata[source_id] = MessageMetadata(
                row_id=int(row[0]),
                source_message_id=source_id,
                group_id=int(row[2]),
                user_id=str(row[3]),
                timestamp=_parse_ledger_datetime(row[4]),
                delivery_state=str(row[6] or "").strip().casefold(),
                reply_to_message_id=(str(row[5]) if row[5] else None),
            )
        return metadata
    finally:
        connection.close()


def validate_v3_dataset_sources(
    cases: Sequence[EvaluationCase],
    *,
    metadata: Mapping[str, MessageMetadata],
    snapshot_watermarks: Mapping[int, int],
) -> None:
    """Prove that each tagged leak case has a real frozen distractor."""

    for index, case in enumerate(cases):
        recent_rows = tuple(
            metadata.get(source_id)
            for source_id in case.recent_context_message_ids
        )
        if not recent_rows or any(item is None for item in recent_rows):
            raise ValueError(f"case {index} has unresolved recent context")
        resolved_recent = tuple(item for item in recent_rows if item is not None)
        for row in resolved_recent:
            watermark = snapshot_watermarks.get(row.group_id)
            if (
                row.group_id != case.group_id
                or watermark is None
                or row.row_id > int(watermark)
                or not row.eligible
            ):
                raise ValueError(f"case {index} recent context violates snapshot scope")
        target = resolved_recent[-1]
        if target.user_id != str(case.requester_uin):
            raise ValueError(f"case {index} requester does not match recent target")
        effective_quote_id = case.quoted_context_message_id or target.reply_to_message_id
        effective_quote = metadata.get(effective_quote_id) if effective_quote_id else None
        quote_is_in_scope = (
            effective_quote is not None
            and effective_quote.group_id == case.group_id
            and effective_quote.row_id
            <= int(snapshot_watermarks.get(case.group_id, 0))
            and effective_quote.eligible
        )
        if quote_is_in_scope and case.expected_answer_mode != "exact":
            raise ValueError(f"case {index} implicit quote changes answer mode")
        if case.expected_answer_mode == "exact" and not quote_is_in_scope:
            raise ValueError(f"case {index} exact mode has no scoped quote")

        forbidden = [
            metadata.get(source_id)
            for source_id in case.forbidden_evidence_message_ids
        ]
        if any(item is None for item in forbidden):
            raise ValueError(f"case {index} has an unresolved forbidden source")
        forbidden_rows = tuple(item for item in forbidden if item is not None)
        for row in forbidden_rows:
            watermark = snapshot_watermarks.get(row.group_id)
            if watermark is None or row.row_id > int(watermark):
                raise ValueError(f"case {index} forbidden source is outside snapshot")
        tags = set(case.gate_tags)
        if "cross_group" in tags and not any(
            row.group_id != case.group_id for row in forbidden_rows
        ):
            raise ValueError(f"case {index} has no cross-group distractor")
        if "blocked_reserved" in tags and not any(
            not row.eligible for row in forbidden_rows
        ):
            raise ValueError(f"case {index} has no blocked/reserved distractor")
        if "subject" in tags:
            allowed = set(case.allowed_evidence_user_ids or ())
            if not any(
                row.group_id == case.group_id and row.user_id not in allowed
                for row in forbidden_rows
            ):
                raise ValueError(f"case {index} has no wrong-subject distractor")
        if "time_range" in tags:
            if not any(
                row.group_id == case.group_id
                and not _within_time_range(row.timestamp, case.time_range)
                for row in forbidden_rows
            ):
                raise ValueError(f"case {index} has no out-of-range distractor")


def build_v3_observation(
    *,
    case_index: int,
    case: EvaluationCase,
    trace: object,
    requester_uin: str,
    metadata: Mapping[str, MessageMetadata],
    snapshot_watermark: int,
    history_packet_tokens: int,
    retrieval_latency_ms: float,
) -> V3Observation:
    """Build a content-free structural observation from one V3 trace."""

    retrieved_ids = tuple(
        dict.fromkeys(
            str(source_id)
            for source_id in getattr(trace, "retrieved_source_msg_ids", ())
            if str(source_id)
        )
    )
    packed = getattr(getattr(trace, "result"), "packed_context")
    evidence_segments = tuple(getattr(packed, "evidence_segments", ()))
    history_ids = tuple(
        dict.fromkeys(
            str(getattr(message, "source_msg_id"))
            for segment in evidence_segments
            for message in getattr(segment, "messages", ())
            if str(getattr(message, "source_msg_id", ""))
        )
    )
    derived_evidence_count = len(tuple(getattr(packed, "facts", ()))) + len(
        tuple(getattr(packed, "summaries", ()))
    )
    eligible_rows = _eligible_history_rows(
        case,
        metadata=metadata,
        snapshot_watermark=snapshot_watermark,
    )
    covered_buckets, required_buckets = _time_bucket_coverage(
        case,
        eligible_rows=eligible_rows,
        selected_rows=tuple(
            metadata[source_id]
            for source_id in history_ids
            if source_id in metadata
        ),
    )
    audit_ids = tuple(dict.fromkeys((*retrieved_ids, *history_ids)))
    audit_rows = [metadata.get(source_id) for source_id in audit_ids]
    unresolved_count = sum(row is None for row in audit_rows)
    resolved_rows = tuple(row for row in audit_rows if row is not None)
    allowed_evidence_users = case.allowed_evidence_user_ids
    subject_leak_count = (
        0
        if allowed_evidence_users is None
        else sum(
            row.user_id not in set(allowed_evidence_users)
            for row in resolved_rows
        )
    )
    resolved_query = getattr(trace, "resolved_query")
    plan_mismatches = _plan_mismatch_count(
        case,
        resolved_query=resolved_query,
        requester_uin=requester_uin,
    )
    forbidden = set(case.forbidden_evidence_message_ids)
    return V3Observation(
        case_index=int(case_index),
        retrieved_source_message_ids=retrieved_ids,
        history_packet_source_message_ids=history_ids,
        history_packet_tokens=max(0, int(history_packet_tokens)),
        memory_context_tokens=max(
            0, int(getattr(getattr(trace, "result"), "estimated_tokens"))
        ),
        recent_message_count=len(tuple(getattr(packed, "recent_messages", ()))),
        eligible_history_count=len(eligible_rows),
        covered_time_bucket_count=covered_buckets,
        required_time_bucket_count=required_buckets,
        retrieval_latency_ms=max(0.0, float(retrieval_latency_ms)),
        group_leak_count=sum(row.group_id != case.group_id for row in resolved_rows),
        subject_leak_count=subject_leak_count,
        time_leak_count=sum(
            not _within_time_range(row.timestamp, case.time_range)
            for row in resolved_rows
        ),
        ineligible_source_count=sum(not row.eligible for row in resolved_rows),
        unresolved_source_count=unresolved_count,
        outside_snapshot_source_count=sum(
            row.row_id > int(snapshot_watermark)
            for row in resolved_rows
            if row.group_id == case.group_id
        ),
        forbidden_source_count=len(forbidden & set(audit_ids)),
        plan_mismatch_count=plan_mismatches,
        derived_evidence_count=derived_evidence_count,
    )


def retrieval_fingerprint_sha256(
    observations: Sequence[V3Observation],
) -> str:
    payload = [
        {
            "case_index": observation.case_index,
            "retrieved_source_message_ids": list(
                observation.retrieved_source_message_ids
            ),
            "history_packet_source_message_ids": list(
                observation.history_packet_source_message_ids
            ),
            "history_packet_tokens": observation.history_packet_tokens,
        }
        for observation in observations
    ]
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def quality_sidecar_template(
    *,
    dataset_sha256: str,
    snapshot_manifest_sha256: str,
    retrieval_fingerprint: str,
    case_count: int,
) -> dict[str, Any]:
    """Return a content-free template for the controlled answer replay."""

    return {
        "quality_version": V3_QUALITY_SIDECAR_VERSION,
        "dataset_sha256": dataset_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "retrieval_fingerprint_sha256": retrieval_fingerprint,
        "private_replay_sha256": "",
        "visibility_artifact_sha256": "",
        "prompt_contract_sha256": "",
        "judge_provider": "",
        "judge_model": "",
        "evaluated_at": "",
        "index_visibility_ms": [],
        "cases": [
            {
                "case_index": index,
                "cited_source_message_ids": [],
                "answer_grounded": None,
                "answer_correct": None,
                "abstained": None,
                "answer_protocol_failure_codes": [],
                "total_prompt_tokens": None,
                "ttft_ms": None,
            }
            for index in range(case_count)
        ],
    }


def load_v3_quality_sidecar(
    path: Path | str,
    *,
    dataset_sha256: str,
    snapshot_manifest_sha256: str,
    retrieval_fingerprint: str,
    case_count: int,
    minimum_visibility_samples: int = 20,
    private_replay_path: Path | str | None = None,
    visibility_artifact_path: Path | str | None = None,
    expected_vector_generation: int | None = None,
    evaluation_cases: Sequence[EvaluationCase] | None = None,
    expected_answer_prompt_sha256_by_case: Mapping[int, str] | None = None,
) -> V3QualitySidecar:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid V3 quality sidecar") from exc
    if not isinstance(payload, dict):
        raise ValueError("V3 quality sidecar must be an object")
    expected_fields = {
        "quality_version",
        "dataset_sha256",
        "snapshot_manifest_sha256",
        "retrieval_fingerprint_sha256",
        "private_replay_sha256",
        "visibility_artifact_sha256",
        "prompt_contract_sha256",
        "judge_provider",
        "judge_model",
        "evaluated_at",
        "index_visibility_ms",
        "cases",
    }
    if set(payload) != expected_fields:
        raise ValueError("V3 quality sidecar fields do not match the contract")
    if payload["quality_version"] != V3_QUALITY_SIDECAR_VERSION:
        raise ValueError("V3 quality sidecar version is unsupported")
    bindings = (
        ("dataset_sha256", dataset_sha256),
        ("snapshot_manifest_sha256", snapshot_manifest_sha256),
        ("retrieval_fingerprint_sha256", retrieval_fingerprint),
    )
    if any(payload[field] != expected for field, expected in bindings):
        raise ValueError("V3 quality sidecar binding does not match this run")
    for field in (
        "private_replay_sha256",
        "visibility_artifact_sha256",
        "prompt_contract_sha256",
    ):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"V3 quality sidecar has invalid {field}")
    if payload["prompt_contract_sha256"] != prompt_contract_sha256():
        raise ValueError("V3 quality sidecar prompt contract is unsupported")
    private_payload: dict[str, Any] | None = None
    if expected_answer_prompt_sha256_by_case is not None:
        expected_case_indexes = set(range(int(case_count)))
        if (
            private_replay_path is None
            or len(expected_answer_prompt_sha256_by_case) != int(case_count)
            or set(expected_answer_prompt_sha256_by_case) != expected_case_indexes
            or any(
                isinstance(case_index, bool) or not isinstance(case_index, int)
                for case_index in expected_answer_prompt_sha256_by_case
            )
            or any(
                not isinstance(prompt_sha256, str)
                or len(prompt_sha256) != 64
                or any(character not in "0123456789abcdef" for character in prompt_sha256)
                for prompt_sha256 in expected_answer_prompt_sha256_by_case.values()
            )
        ):
            raise ValueError("V3 expected answer prompt bindings are invalid")
    if private_replay_path is not None:
        try:
            private_bytes = Path(private_replay_path).read_bytes()
            private_payload = json.loads(
                private_bytes.decode("utf-8"),
                parse_constant=reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid V3 private replay artifact") from exc
        if hashlib.sha256(private_bytes).hexdigest() != payload["private_replay_sha256"]:
            raise ValueError("V3 private replay hash does not match the sidecar")
        if not isinstance(private_payload, dict):
            raise ValueError("V3 private replay artifact must be an object")
        private_bindings = {
            "dataset_sha256": dataset_sha256,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "retrieval_fingerprint_sha256": retrieval_fingerprint,
            "prompt_contract_sha256": payload["prompt_contract_sha256"],
        }
        if any(
            private_payload.get(field) != expected
            for field, expected in private_bindings.items()
        ):
            raise ValueError("V3 private replay binding does not match the sidecar")
        private_cases = private_payload.get("cases")
        if not isinstance(private_cases, list) or len(private_cases) != int(case_count):
            raise ValueError("V3 private replay case count does not match the sidecar")
    for field in ("judge_provider", "judge_model", "evaluated_at"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"V3 quality sidecar has no {field}")
    if payload["judge_provider"] != QUALITY_REPLAY_PROVIDER:
        raise ValueError("V3 quality sidecar uses an unsupported replay provider")
    _parse_datetime(payload["evaluated_at"])
    visibility = _finite_number_list(
        payload["index_visibility_ms"],
        field="index_visibility_ms",
    )
    if len(visibility) < int(minimum_visibility_samples):
        raise ValueError("V3 quality sidecar has insufficient visibility samples")
    if visibility_artifact_path is not None:
        try:
            visibility_bytes = Path(visibility_artifact_path).read_bytes()
            visibility_payload = json.loads(
                visibility_bytes.decode("utf-8"),
                parse_constant=reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid V3 visibility artifact") from exc
        if (
            hashlib.sha256(visibility_bytes).hexdigest()
            != payload["visibility_artifact_sha256"]
        ):
            raise ValueError("V3 visibility artifact hash does not match the sidecar")
        expected_visibility_fields = {
            "visibility_version",
            "measurement_mode",
            "source_snapshot_clone_sha256",
            "vector_generation",
            "sample_count",
            "samples",
            "dataset_sha256",
            "snapshot_manifest_sha256",
            "retrieval_fingerprint_sha256",
        }
        if (
            not isinstance(visibility_payload, dict)
            or set(visibility_payload) != expected_visibility_fields
        ):
            raise ValueError("V3 visibility artifact fields do not match the contract")
        visibility_bindings = {
            "dataset_sha256": dataset_sha256,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "retrieval_fingerprint_sha256": retrieval_fingerprint,
        }
        if any(
            visibility_payload.get(field) != expected
            for field, expected in visibility_bindings.items()
        ):
            raise ValueError("V3 visibility artifact binding does not match the sidecar")
        if (
            expected_vector_generation is not None
            and visibility_payload.get("vector_generation")
            != int(expected_vector_generation)
        ):
            raise ValueError("V3 visibility artifact uses the wrong vector generation")
        samples = visibility_payload.get("samples")
        if (
            visibility_payload.get("visibility_version") != 1
            or visibility_payload.get("measurement_mode")
            != "disposable_sqlite_online_backup_clone"
            or not isinstance(samples, list)
            or visibility_payload.get("sample_count") != len(samples)
            or len(samples) != len(visibility)
        ):
            raise ValueError("V3 visibility artifact sample contract is invalid")
        sample_values: list[float] = []
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or set(sample) != {
                "case_index",
                "nonce_sha256",
                "fts_ms",
                "vector_ms",
                "overall_ms",
            }:
                raise ValueError("V3 visibility sample fields are invalid")
            case_index = sample.get("case_index")
            if (
                isinstance(case_index, bool)
                or not isinstance(case_index, int)
                or case_index != index
            ):
                raise ValueError("V3 visibility sample index is invalid")
            nonce_sha = sample.get("nonce_sha256")
            if (
                not isinstance(nonce_sha, str)
                or len(nonce_sha) != 64
                or any(character not in "0123456789abcdef" for character in nonce_sha)
            ):
                raise ValueError("V3 visibility sample nonce hash is invalid")
            timings = _finite_number_list(
                [sample.get("fts_ms"), sample.get("vector_ms"), sample.get("overall_ms")],
                field="visibility sample timings",
            )
            if timings[2] != max(timings[0], timings[1]):
                raise ValueError("V3 visibility sample overall timing is invalid")
            sample_values.append(timings[2])
        if tuple(sample_values) != visibility:
            raise ValueError("V3 visibility samples do not match the sidecar")
    rows = payload["cases"]
    if not isinstance(rows, list) or len(rows) != int(case_count):
        raise ValueError("V3 quality sidecar must contain one row per case")
    parsed_rows: list[V3QualityCase] = []
    required_case_fields = {
        "case_index",
        "cited_source_message_ids",
        "answer_grounded",
        "answer_correct",
        "abstained",
        "answer_protocol_failure_codes",
        "total_prompt_tokens",
        "ttft_ms",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required_case_fields:
            raise ValueError(f"V3 quality case {index} has invalid fields")
        case_index = row["case_index"]
        if (
            isinstance(case_index, bool)
            or not isinstance(case_index, int)
            or case_index != index
        ):
            raise ValueError(f"V3 quality case {index} has a wrong case_index")
        cited = row["cited_source_message_ids"]
        if (
            not isinstance(cited, list)
            or any(not isinstance(item, str) or not item for item in cited)
            or len(set(cited)) != len(cited)
        ):
            raise ValueError(f"V3 quality case {index} has invalid citations")
        for field in ("answer_grounded", "answer_correct", "abstained"):
            if not isinstance(row[field], bool):
                raise ValueError(f"V3 quality case {index} has invalid {field}")
        protocol_failures = row["answer_protocol_failure_codes"]
        if (
            not isinstance(protocol_failures, list)
            or any(
                not isinstance(item, str)
                or not item
                or len(item) > 96
                or any(character.isspace() for character in item)
                for item in protocol_failures
            )
            or len(set(protocol_failures)) != len(protocol_failures)
        ):
            raise ValueError(
                f"V3 quality case {index} has invalid answer protocol failures"
            )
        total_prompt_tokens = row["total_prompt_tokens"]
        if (
            isinstance(total_prompt_tokens, bool)
            or not isinstance(total_prompt_tokens, int)
            or total_prompt_tokens < 0
        ):
            raise ValueError(
                f"V3 quality case {index} has invalid total_prompt_tokens"
            )
        ttft_ms = row["ttft_ms"]
        if (
            isinstance(ttft_ms, bool)
            or not isinstance(ttft_ms, (int, float))
            or not math.isfinite(ttft_ms)
            or ttft_ms < 0
        ):
            raise ValueError(f"V3 quality case {index} has invalid ttft_ms")
        parsed_rows.append(
            V3QualityCase(
                case_index=index,
                cited_source_message_ids=tuple(cited),
                answer_grounded=row["answer_grounded"],
                answer_correct=row["answer_correct"],
                abstained=row["abstained"],
                answer_protocol_failure_codes=tuple(protocol_failures),
                total_prompt_tokens=total_prompt_tokens,
                ttft_ms=float(ttft_ms),
            )
        )
    if private_payload is not None:
        if evaluation_cases is not None and len(evaluation_cases) != int(case_count):
            raise ValueError("V3 quality evaluation case count does not match the sidecar")
        expected_private_fields = {
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
        if set(private_payload) != expected_private_fields:
            raise ValueError("V3 private replay fields do not match the contract")
        if (
            private_payload.get("private_replay_version") != 1
            or private_payload.get("evaluated_at") != payload["evaluated_at"]
            or payload["judge_model"]
            != f"generator={private_payload.get('generator_model')};judge={private_payload.get('judge_model')}"
        ):
            raise ValueError("V3 private replay metadata does not match the sidecar")
        expected_private_case_fields = {
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
        for index, (private_case, public_case) in enumerate(
            zip(private_payload["cases"], rows, strict=True)
        ):
            if (
                not isinstance(private_case, dict)
                or set(private_case) != expected_private_case_fields
            ):
                raise ValueError(f"V3 private replay case {index} fields are invalid")
            case_index = private_case.get("case_index")
            if (
                isinstance(case_index, bool)
                or not isinstance(case_index, int)
                or case_index != index
            ):
                raise ValueError(f"V3 private replay case {index} has a wrong case_index")
            query = private_case.get("query")
            if not isinstance(query, str):
                raise ValueError(f"V3 private replay case {index} query is invalid")
            if evaluation_cases is not None and query != evaluation_cases[index].query:
                raise ValueError(f"V3 private replay case {index} query does not match the dataset")
            if (
                expected_answer_prompt_sha256_by_case is not None
                and private_case.get("answer_prompt_sha256")
                != expected_answer_prompt_sha256_by_case[index]
            ):
                raise ValueError(f"V3 private replay case {index} answer prompt binding does not match")
            answer_prompt = private_case.get("answer_prompt")
            if not isinstance(answer_prompt, list) or any(
                not isinstance(item, str) for item in answer_prompt
            ):
                raise ValueError(f"V3 private replay case {index} prompt is invalid")
            rendered_prompt = json.dumps(
                answer_prompt,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            if private_case.get("answer_prompt_sha256") != hashlib.sha256(
                rendered_prompt
            ).hexdigest():
                raise ValueError(f"V3 private replay case {index} prompt hash is invalid")
            generator_model = str(private_payload["generator_model"])
            judge_model = str(private_payload["judge_model"])
            answer_observation = _validate_replay_observation(
                private_case.get("answer_observation"),
                expected_model=generator_model,
            )
            decision = private_case.get("judge_decision")
            if (
                not isinstance(decision, dict)
                or set(decision)
                != {
                    "answer_grounded",
                    "answer_correct",
                    "abstained",
                    "reason_code",
                }
                or any(
                    not isinstance(decision[field], bool)
                    for field in ("answer_grounded", "answer_correct", "abstained")
                )
                or not _validate_reason_code(decision["reason_code"])
            ):
                raise ValueError(f"V3 private replay case {index} result schema is invalid")
            raw_judge = _strict_replay_json_object(
                private_case["judge_raw_output"],
                fields={"answer_grounded", "answer_correct", "abstained", "reason_code"},
            )
            protocol_codes = private_case["answer_protocol_failure_codes"]
            citation_failure_codes = private_case["citation_failure_codes"]
            generated_citations = private_case["generated_citations"]
            generated_abstained = private_case["generated_abstained"]
            repair_count = private_case["answer_repair_count"]
            if (
                not isinstance(generated_citations, list)
                or any(not isinstance(item, str) or not item for item in generated_citations)
                or not isinstance(generated_abstained, bool)
                or not isinstance(protocol_codes, list)
                or any(not isinstance(item, str) or not item for item in protocol_codes)
                or not isinstance(citation_failure_codes, list)
                or any(not isinstance(item, str) or not item for item in citation_failure_codes)
                or isinstance(repair_count, bool)
                or not isinstance(repair_count, int)
                or repair_count not in (0, 1)
            ):
                raise ValueError(f"V3 private replay case {index} answer schema is invalid")
            expected_decision = raw_judge
            if evaluation_cases is not None:
                expected_decision = _expected_fail_closed_judge_decision(
                    case=evaluation_cases[index],
                    raw_decision=raw_judge,
                    generated_citations=generated_citations,
                    generated_abstained=generated_abstained,
                    protocol_failure_codes=protocol_codes,
                    citation_failure_codes=citation_failure_codes,
                )
            if expected_decision != decision:
                raise ValueError(f"V3 private replay case {index} raw judge does not match")
            judge_observation = _validate_replay_observation(
                private_case["judge_observation"],
                expected_model=judge_model,
            )
            if judge_observation["text"] != private_case["judge_raw_output"]:
                raise ValueError(f"V3 private replay case {index} judge observation does not match")
            reconciled = (
                private_case.get("generated_citations")
                == public_case["cited_source_message_ids"]
                and private_case.get("answer_protocol_failure_codes")
                == public_case["answer_protocol_failure_codes"]
                and decision.get("answer_grounded") == public_case["answer_grounded"]
                and decision.get("answer_correct") == public_case["answer_correct"]
                and decision.get("abstained") == public_case["abstained"]
                and answer_observation.get("input_tokens")
                == public_case["total_prompt_tokens"]
                and answer_observation.get("ttft_ms") == public_case["ttft_ms"]
            )
            if not reconciled:
                raise ValueError(f"V3 private replay case {index} does not match the sidecar")
            if (
                not generated_abstained
                and not generated_citations
                and "citation_missing" not in protocol_codes
            ):
                raise ValueError(f"V3 private replay case {index} omits missing-citation failure")
            contract_decision = private_case["citation_contract_decision"]
            if (
                not isinstance(contract_decision, dict)
                or set(contract_decision) != {"citations_minimal", "reason_code"}
                or not isinstance(contract_decision["citations_minimal"], bool)
                or not _validate_reason_code(contract_decision["reason_code"])
            ):
                raise ValueError(f"V3 private replay case {index} citation contract is invalid")
            raw_contract = _strict_replay_json_object(
                private_case["citation_contract_raw_output"],
                fields={"citations_minimal", "reason_code"},
            )
            if raw_contract != contract_decision:
                raise ValueError(f"V3 private replay case {index} raw citation contract does not match")
            citation_observation = _validate_replay_observation(
                private_case["citation_contract_observation"],
                expected_model=judge_model,
            )
            if citation_observation["text"] != private_case["citation_contract_raw_output"]:
                raise ValueError(f"V3 private replay case {index} citation observation does not match")
            if (
                not contract_decision["citations_minimal"]
                and "citation_not_minimal" not in protocol_codes
            ):
                raise ValueError(f"V3 private replay case {index} omits minimal-citation failure")
            attempts = private_case["answer_attempts"]
            if (
                not isinstance(attempts, list)
                or len(attempts) != repair_count + 1
                or len(attempts) not in (1, 2)
            ):
                raise ValueError(f"V3 private replay case {index} repair audit is invalid")
            expected_attempt_fields = {
                "kind",
                "prompt",
                "answer",
                "observation",
                "protocol_failure_codes",
                "citation_contract_prompt",
                "citation_contract_raw_output",
                "citation_contract_observation",
                "citation_contract_decision",
            }
            for attempt_index, attempt in enumerate(attempts):
                if (
                    not isinstance(attempt, dict)
                    or set(attempt) != expected_attempt_fields
                    or attempt.get("kind")
                    != ("initial" if attempt_index == 0 else "citation_repair")
                    or not isinstance(attempt.get("answer"), dict)
                ):
                    raise ValueError(f"V3 private replay case {index} attempt is invalid")
                _, recomputed_answer_failures = _validate_replay_answer(attempt["answer"])
                attempt_prompt = attempt["prompt"]
                contract_prompt = attempt["citation_contract_prompt"]
                attempt_protocol_codes = attempt["protocol_failure_codes"]
                if (
                    not isinstance(attempt_prompt, list)
                    or any(not isinstance(item, str) for item in attempt_prompt)
                    or not isinstance(contract_prompt, list)
                    or any(not isinstance(item, str) for item in contract_prompt)
                    or not isinstance(attempt_protocol_codes, list)
                    or any(not isinstance(item, str) or not item for item in attempt_protocol_codes)
                ):
                    raise ValueError(f"V3 private replay case {index} attempt audit is invalid")
                attempt_observation = _validate_replay_observation(
                    attempt["observation"],
                    expected_model=generator_model,
                )
                raw_answer = _strict_replay_json_object(
                    attempt_observation["text"],
                    fields={"answer", "cited_source_message_ids", "abstained"},
                )
                _validate_replay_answer(raw_answer)
                if raw_answer != attempt["answer"]:
                    raise ValueError(f"V3 private replay case {index} raw answer does not match")
                if not set(recomputed_answer_failures) <= set(attempt_protocol_codes):
                    raise ValueError(f"V3 private replay case {index} omits answer contract failures")
                attempt_contract = attempt["citation_contract_decision"]
                if (
                    not isinstance(attempt_contract, dict)
                    or set(attempt_contract) != {"citations_minimal", "reason_code"}
                    or not isinstance(attempt_contract["citations_minimal"], bool)
                    or not _validate_reason_code(attempt_contract["reason_code"])
                ):
                    raise ValueError(f"V3 private replay case {index} attempt contract is invalid")
                raw_attempt_contract = _strict_replay_json_object(
                    attempt["citation_contract_raw_output"],
                    fields={"citations_minimal", "reason_code"},
                )
                attempt_contract_observation = _validate_replay_observation(
                    attempt["citation_contract_observation"],
                    expected_model=judge_model,
                )
                if (
                    raw_attempt_contract != attempt_contract
                    or attempt_contract_observation["text"]
                    != attempt["citation_contract_raw_output"]
                ):
                    raise ValueError(f"V3 private replay case {index} attempt contract does not match")
            initial_attempt = attempts[0]
            final_attempt = attempts[-1]
            _, final_answer_failures = _validate_replay_answer(final_attempt["answer"])
            required_final_failures = {
                *final_answer_failures,
                *final_attempt["protocol_failure_codes"],
            }
            if not final_attempt["citation_contract_decision"]["citations_minimal"]:
                required_final_failures.add("citation_not_minimal")
            if not required_final_failures <= set(protocol_codes):
                raise ValueError(
                    f"V3 private replay case {index} omits final protocol failures"
                )
            if (
                initial_attempt["answer"].get("answer")
                != final_attempt["answer"].get("answer")
                or initial_attempt["answer"].get("abstained")
                != final_attempt["answer"].get("abstained")
                or private_case["answer"] != final_attempt["answer"].get("answer")
                or private_case["generated_citations"]
                != final_attempt["answer"].get("cited_source_message_ids")
                or private_case["generated_abstained"]
                != final_attempt["answer"].get("abstained")
                or private_case["answer_observation"]
                != initial_attempt["observation"]
                or private_case["citation_contract_decision"]
                != final_attempt["citation_contract_decision"]
                or private_case["citation_contract_prompt"]
                != final_attempt["citation_contract_prompt"]
                or private_case["citation_contract_raw_output"]
                != final_attempt["citation_contract_raw_output"]
                or private_case["citation_contract_observation"]
                != final_attempt["citation_contract_observation"]
            ):
                raise ValueError(f"V3 private replay case {index} repair changed substantive output")
    return V3QualitySidecar(
        dataset_sha256=dataset_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        retrieval_fingerprint_sha256=retrieval_fingerprint,
        private_replay_sha256=payload["private_replay_sha256"],
        visibility_artifact_sha256=payload["visibility_artifact_sha256"],
        prompt_contract_sha256=payload["prompt_contract_sha256"],
        judge_provider=payload["judge_provider"],
        judge_model=payload["judge_model"],
        evaluated_at=payload["evaluated_at"],
        index_visibility_ms=visibility,
        cases=tuple(parsed_rows),
    )


def _expected_fail_closed_judge_decision(
    *,
    case: EvaluationCase,
    raw_decision: Mapping[str, Any],
    generated_citations: Sequence[str],
    generated_abstained: bool,
    protocol_failure_codes: Sequence[str],
    citation_failure_codes: Sequence[str],
) -> dict[str, Any]:
    failures = tuple(dict.fromkeys((*protocol_failure_codes, *citation_failure_codes)))
    abstained = bool(raw_decision["abstained"] or generated_abstained)
    if failures:
        return {
            "answer_grounded": False,
            "answer_correct": False,
            "abstained": abstained,
            "reason_code": "+".join(failures),
        }

    citations_present = bool(generated_citations)
    grounded = bool(raw_decision["answer_grounded"])
    correct = bool(raw_decision["answer_correct"])
    if not case.expected_evidence_message_ids:
        grounded = grounded and not citations_present
        correct = correct and abstained and not citations_present
    else:
        grounded = grounded and citations_present and not abstained
        correct = correct and grounded and not abstained
    return {
        "answer_grounded": grounded,
        "answer_correct": correct,
        "abstained": abstained,
        "reason_code": raw_decision["reason_code"],
    }


def evaluate_v3(
    *,
    cases: Sequence[EvaluationCase],
    observations: Sequence[V3Observation],
    quality: V3QualitySidecar | None,
    dataset_sha256: str,
    snapshot_manifest_sha256: str,
    retrieval_fingerprint: str,
    gate_tag_counts: Mapping[str, int],
) -> dict[str, Any]:
    if len(cases) != len(observations):
        raise ValueError("V3 observations must contain exactly one row per case")
    ordered = tuple(sorted(observations, key=lambda item: item.case_index))
    if [item.case_index for item in ordered] != list(range(len(cases))):
        raise ValueError("V3 observation case indexes are incomplete")

    recalls_150: list[float] = []
    recalls_24k: list[float] = []
    for case, observation in zip(cases, ordered, strict=True):
        gold = set(case.expected_evidence_message_ids)
        retrieved = set(observation.retrieved_source_message_ids[:150])
        packed = set(observation.history_packet_source_message_ids)
        expected_abstention = not gold
        recalls_150.append(
            float(not retrieved)
            if expected_abstention
            else len(gold & retrieved) / len(gold)
        )
        recalls_24k.append(
            float(not packed)
            if expected_abstention
            else len(gold & packed) / len(gold)
        )

    packet_counts = [
        len(item.history_packet_source_message_ids) for item in ordered
    ]
    eligible_total = sum(item.eligible_history_count for item in ordered)
    packet_total = sum(packet_counts)
    metrics: dict[str, Any] = {
        "recall_at_150": _mean(recalls_150),
        "recall_within_24k": _mean(recalls_24k),
        "group_leak_count": sum(item.group_leak_count for item in ordered),
        "subject_leak_count": sum(item.subject_leak_count for item in ordered),
        "time_leak_count": sum(item.time_leak_count for item in ordered),
        "ineligible_source_count": sum(
            item.ineligible_source_count for item in ordered
        ),
        "unresolved_source_count": sum(
            item.unresolved_source_count for item in ordered
        ),
        "outside_snapshot_source_count": sum(
            item.outside_snapshot_source_count for item in ordered
        ),
        "forbidden_source_count": sum(
            item.forbidden_source_count for item in ordered
        ),
        "plan_mismatch_count": sum(item.plan_mismatch_count for item in ordered),
        "derived_evidence_count": sum(
            item.derived_evidence_count for item in ordered
        ),
        "retrieval_over_150_count": sum(
            len(item.retrieved_source_message_ids) > 150 for item in ordered
        ),
        "packet_over_150_count": sum(count > 150 for count in packet_counts),
        "packet_over_24k_count": sum(
            item.history_packet_tokens > 24_000 for item in ordered
        ),
        "recent_over_60_count": sum(
            item.recent_message_count > 60 for item in ordered
        ),
        "time_bucket_coverage_rate": _mean(
            [
                float(
                    item.covered_time_bucket_count
                    >= item.required_time_bucket_count
                )
                for item in ordered
                if item.required_time_bucket_count > 0
            ],
            empty=1.0,
        ),
        "eligible_history_count": eligible_total,
        "history_packet_message_count": packet_total,
        "packet_to_eligible_ratio": (
            packet_total / eligible_total if eligible_total else 0.0
        ),
        "eligible_to_packet_compression_ratio": (
            eligible_total / packet_total if packet_total else 0.0
        ),
        "mean_history_packet_tokens": _mean(
            [float(item.history_packet_tokens) for item in ordered]
        ),
        "p95_history_packet_tokens": _percentile(
            [float(item.history_packet_tokens) for item in ordered],
            0.95,
        ),
        "mean_memory_context_tokens": _mean(
            [float(item.memory_context_tokens) for item in ordered]
        ),
        "retrieval_p50_ms": _percentile(
            [item.retrieval_latency_ms for item in ordered],
            0.50,
        ),
        "retrieval_p95_ms": _percentile(
            [item.retrieval_latency_ms for item in ordered],
            0.95,
        ),
    }
    if quality is None:
        metrics.update(
            {
                "citation_precision": None,
                "citation_recall": None,
                "grounded_answer_accuracy": None,
                "answer_accuracy": None,
                "answer_protocol_failure_count": None,
                "abstention_precision": None,
                "abstention_recall": None,
                "abstention_f1": None,
                "mean_total_prompt_tokens": None,
                "p95_total_prompt_tokens": None,
                "ttft_p50_ms": None,
                "ttft_p95_ms": None,
                "index_visibility_p50_ms": None,
                "index_visibility_p95_ms": None,
            }
        )
    else:
        _add_quality_metrics(
            metrics,
            cases=cases,
            observations=ordered,
            quality=quality,
        )

    return {
        "evaluation_schema_version": V3_DATASET_SCHEMA_VERSION,
        "memory_path": "raw_message_v3",
        "dataset_sha256": dataset_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "retrieval_fingerprint_sha256": retrieval_fingerprint,
        "case_count": len(cases),
        "gate_tag_counts": dict(sorted(gate_tag_counts.items())),
        "quality_sidecar_present": quality is not None,
        "metrics": metrics,
    }


def audit_v3_quality_sources(
    *,
    cases: Sequence[EvaluationCase],
    observations: Sequence[V3Observation],
    quality: V3QualitySidecar,
    metadata: Mapping[str, MessageMetadata],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case, observation, judgment in zip(
        cases,
        observations,
        quality.cases,
        strict=True,
    ):
        packet = set(observation.history_packet_source_message_ids)
        allowed_users = case.allowed_evidence_user_ids
        forbidden = set(case.forbidden_evidence_message_ids)
        for source_id in judgment.cited_source_message_ids:
            if source_id not in packet:
                counts["citation_not_in_packet_count"] += 1
            if source_id in forbidden:
                counts["citation_forbidden_source_count"] += 1
            row = metadata.get(source_id)
            if row is None:
                counts["citation_unresolved_source_count"] += 1
                continue
            if row.group_id != case.group_id:
                counts["citation_group_leak_count"] += 1
            if (
                allowed_users is not None
                and row.user_id not in set(allowed_users)
            ):
                counts["citation_subject_leak_count"] += 1
            if not _within_time_range(row.timestamp, case.time_range):
                counts["citation_time_leak_count"] += 1
            if not row.eligible:
                counts["citation_ineligible_source_count"] += 1
    keys = (
        "citation_not_in_packet_count",
        "citation_forbidden_source_count",
        "citation_unresolved_source_count",
        "citation_group_leak_count",
        "citation_subject_leak_count",
        "citation_time_leak_count",
        "citation_ineligible_source_count",
    )
    return {key: int(counts[key]) for key in keys}


def observation_as_safe_dict(observation: V3Observation) -> dict[str, Any]:
    value = asdict(observation)
    value["variant"] = "v3"
    value["retrieved_source_message_ids"] = list(
        observation.retrieved_source_message_ids
    )
    value["history_packet_source_message_ids"] = list(
        observation.history_packet_source_message_ids
    )
    return value


def _eligible_history_rows(
    case: EvaluationCase,
    *,
    metadata: Mapping[str, MessageMetadata],
    snapshot_watermark: int,
) -> tuple[MessageMetadata, ...]:
    allowed = case.allowed_evidence_user_ids
    if allowed == ():
        return ()
    rows = (
        row
        for row in metadata.values()
        if row.group_id == case.group_id
        and row.row_id <= int(snapshot_watermark)
        and row.eligible
        and (allowed is None or row.user_id in set(allowed))
        and _within_time_range(row.timestamp, case.time_range)
    )
    return tuple(sorted(rows, key=lambda row: (row.timestamp, row.row_id)))


def _time_bucket_coverage(
    case: EvaluationCase,
    *,
    eligible_rows: Sequence[MessageMetadata],
    selected_rows: Sequence[MessageMetadata],
) -> tuple[int, int]:
    requested = int(case.minimum_time_bucket_count)
    if requested <= 0 or not eligible_rows:
        return 0, 0
    if case.time_range is None:
        start_at = min(row.timestamp for row in eligible_rows)
        end_at = max(row.timestamp for row in eligible_rows) + timedelta(
            microseconds=1
        )
    else:
        start_at, end_at = (_parse_datetime(value) for value in case.time_range)
    width = (end_at - start_at).total_seconds()
    if width <= 0:
        return 0, requested

    def bucket(row: MessageMetadata) -> int | None:
        if not start_at <= row.timestamp < end_at:
            return None
        offset = (row.timestamp - start_at).total_seconds()
        return min(requested - 1, int(offset / width * requested))

    nonempty = {value for row in eligible_rows if (value := bucket(row)) is not None}
    selected = {
        value for row in selected_rows if (value := bucket(row)) is not None
    }
    required = min(requested, len(nonempty))
    return len(selected & nonempty), required


def _plan_mismatch_count(
    case: EvaluationCase,
    *,
    resolved_query: object,
    requester_uin: str,
) -> int:
    mismatches = int(str(requester_uin) != str(case.requester_uin))
    if str(getattr(resolved_query, "requester_id", "")) != str(
        case.requester_uin
    ):
        mismatches += 1
    if int(getattr(resolved_query, "group_id", 0) or 0) != int(case.group_id):
        mismatches += 1
    actual_subject = getattr(resolved_query, "subject_ids", None)
    if actual_subject is not None:
        actual_subject = tuple(str(item) for item in actual_subject)
    if actual_subject != case.allowed_subject_user_ids:
        mismatches += 1
    if str(getattr(resolved_query, "answer_mode", "")) != str(
        case.expected_answer_mode
    ):
        mismatches += 1
    actual_coverage = getattr(
        resolved_query,
        "coverage_strategy",
        getattr(resolved_query, "coverage_mode", ""),
    )
    if str(actual_coverage) != str(case.expected_coverage_strategy):
        mismatches += 1
    actual_range = getattr(resolved_query, "time_range", None)
    if not _same_time_range(actual_range, case.time_range):
        mismatches += 1
    return mismatches


def _same_time_range(
    actual: object | None,
    expected: tuple[str, str] | None,
) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    actual_start = getattr(actual, "start", getattr(actual, "start_at", None))
    actual_end = getattr(actual, "end", getattr(actual, "end_at", None))
    if actual_start is None or actual_end is None:
        return False
    expected_start, expected_end = (_parse_datetime(value) for value in expected)
    return (
        _parse_datetime(actual_start) == expected_start
        and _parse_datetime(actual_end) == expected_end
    )


def _within_time_range(
    timestamp: datetime,
    time_range: tuple[str, str] | None,
) -> bool:
    if time_range is None:
        return True
    start_at, end_at = (_parse_datetime(value) for value in time_range)
    return start_at <= timestamp < end_at


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        if not normalized:
            raise ValueError("timestamp must not be empty")
        parsed = datetime.fromisoformat(normalized)
    else:
        raise ValueError("timestamp must be an ISO-8601 value")
    if parsed.tzinfo is None:
        raise ValueError("V3 timestamps must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def _parse_ledger_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        if not normalized:
            raise ValueError("ledger timestamp must not be empty")
        parsed = datetime.fromisoformat(normalized)
    else:
        raise ValueError("ledger timestamp must be an ISO-8601 value")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finite_number_list(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    parsed: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"{field} must contain finite non-negative numbers")
        parsed.append(float(item))
    return tuple(parsed)


def _add_quality_metrics(
    metrics: dict[str, Any],
    *,
    cases: Sequence[EvaluationCase],
    observations: Sequence[V3Observation],
    quality: V3QualitySidecar,
) -> None:
    citation_precision: list[float] = []
    citation_recall: list[float] = []
    true_positive = false_positive = false_negative = 0
    for case, observation, judgment in zip(
        cases,
        observations,
        quality.cases,
        strict=True,
    ):
        gold = set(case.expected_evidence_message_ids)
        expected_evidence_available = bool(
            gold & set(observation.history_packet_source_message_ids)
        )
        citations = set(judgment.cited_source_message_ids)
        citations_minimal = (
            "citation_not_minimal"
            not in judgment.answer_protocol_failure_codes
        )
        citation_precision.append(
            _citation_precision_score(
                gold=gold,
                citations=citations,
                answer_grounded=judgment.answer_grounded,
                citations_minimal=citations_minimal,
                expected_evidence_available=expected_evidence_available,
            )
        )
        citation_recall.append(
            len(gold & citations) / len(gold) if gold else float(not citations)
        )
        expected_abstention = _expected_abstention_for_quality(
            expected_evidence_available=expected_evidence_available,
            citations=citations,
            answer_grounded=judgment.answer_grounded,
            answer_correct=judgment.answer_correct,
            citations_minimal=citations_minimal,
        )
        true_positive += int(expected_abstention and judgment.abstained)
        false_positive += int(not expected_abstention and judgment.abstained)
        false_negative += int(expected_abstention and not judgment.abstained)
    abstention_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    abstention_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    metrics.update(
        {
            "citation_precision": _mean(citation_precision),
            "citation_recall": _mean(citation_recall),
            "grounded_answer_accuracy": _mean(
                [
                    float(row.answer_grounded and row.answer_correct)
                    for row in quality.cases
                ]
            ),
            "answer_accuracy": _mean(
                [float(row.answer_correct) for row in quality.cases]
            ),
            "answer_protocol_failure_count": sum(
                bool(row.answer_protocol_failure_codes) for row in quality.cases
            ),
            "abstention_precision": abstention_precision,
            "abstention_recall": abstention_recall,
            "abstention_f1": (
                2
                * abstention_precision
                * abstention_recall
                / (abstention_precision + abstention_recall)
                if abstention_precision + abstention_recall
                else 0.0
            ),
            "mean_total_prompt_tokens": _mean(
                [float(row.total_prompt_tokens) for row in quality.cases]
            ),
            "p95_total_prompt_tokens": _percentile(
                [float(row.total_prompt_tokens) for row in quality.cases],
                0.95,
            ),
            "ttft_p50_ms": _percentile(
                [row.ttft_ms for row in quality.cases],
                0.50,
            ),
            "ttft_p95_ms": _percentile(
                [row.ttft_ms for row in quality.cases],
                0.95,
            ),
            "index_visibility_p50_ms": _percentile(
                quality.index_visibility_ms,
                0.50,
            ),
            "index_visibility_p95_ms": _percentile(
                quality.index_visibility_ms,
                0.95,
            ),
        }
    )


def _citation_precision_score(
    *,
    gold: set[str],
    citations: set[str],
    answer_grounded: bool,
    citations_minimal: bool,
    expected_evidence_available: bool,
) -> float:
    if not citations:
        return float(not expected_evidence_available)
    if answer_grounded and citations_minimal:
        return 1.0
    overlap = gold & citations
    if overlap:
        return len(overlap) / len(citations)
    return 0.0


def _expected_abstention_for_quality(
    *,
    expected_evidence_available: bool,
    citations: set[str],
    answer_grounded: bool,
    answer_correct: bool,
    citations_minimal: bool,
) -> bool:
    alternative_evidence_supported = bool(
        citations and answer_grounded and answer_correct and citations_minimal
    )
    return not (expected_evidence_available or alternative_evidence_supported)


def _mean(values: Sequence[float], *, empty: float = 0.0) -> float:
    return sum(values) / len(values) if values else float(empty)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    index = max(0, math.ceil(float(percentile) * len(ordered)) - 1)
    return ordered[index]
