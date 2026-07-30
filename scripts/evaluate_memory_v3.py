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


V3_DATASET_SCHEMA_VERSION = 3
V3_QUALITY_SIDECAR_VERSION = 1
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
    total_prompt_tokens: int
    ttft_ms: float


@dataclass(frozen=True, slots=True)
class V3QualitySidecar:
    dataset_sha256: str
    snapshot_manifest_sha256: str
    retrieval_fingerprint_sha256: str
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
                delivery_state=str(row[5] or "").strip().casefold(),
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
    for field in ("judge_provider", "judge_model", "evaluated_at"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"V3 quality sidecar has no {field}")
    _parse_datetime(payload["evaluated_at"])
    visibility = _finite_number_list(
        payload["index_visibility_ms"],
        field="index_visibility_ms",
    )
    if len(visibility) < int(minimum_visibility_samples):
        raise ValueError("V3 quality sidecar has insufficient visibility samples")
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
                total_prompt_tokens=total_prompt_tokens,
                ttft_ms=float(ttft_ms),
            )
        )
    return V3QualitySidecar(
        dataset_sha256=dataset_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        retrieval_fingerprint_sha256=retrieval_fingerprint,
        judge_provider=payload["judge_provider"],
        judge_model=payload["judge_model"],
        evaluated_at=payload["evaluated_at"],
        index_visibility_ms=visibility,
        cases=tuple(parsed_rows),
    )


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
        _add_quality_metrics(metrics, cases=cases, quality=quality)

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
    quality: V3QualitySidecar,
) -> None:
    citation_precision: list[float] = []
    citation_recall: list[float] = []
    true_positive = false_positive = false_negative = 0
    for case, judgment in zip(cases, quality.cases, strict=True):
        gold = set(case.expected_evidence_message_ids)
        citations = set(judgment.cited_source_message_ids)
        citation_precision.append(
            len(gold & citations) / len(citations)
            if citations
            else float(not gold)
        )
        citation_recall.append(
            len(gold & citations) / len(gold) if gold else float(not citations)
        )
        expected_abstention = not gold
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


def _mean(values: Sequence[float], *, empty: float = 0.0) -> float:
    return sum(values) / len(values) if values else float(empty)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    index = max(0, math.ceil(float(percentile) * len(ordered)) - 1)
    return ordered[index]
