"""Measure answer/judge variance from a frozen private full-chain detail file.

The input and private output contain prompts and model text and must remain in
gitignored local storage.  The public output is an aggregate-only projection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from app.config import AppSettings
from scripts.memory_test_fullchain import (
    DEFAULT_ANSWER_EFFORT,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_AUX_EFFORT,
    DEFAULT_AUX_MODEL,
    PROVIDER_ATTEMPTS,
    PROVIDER_BACKOFF_SECONDS,
    _build_eval_clients,
    _cache_load,
    _cache_save,
    _generate_with_retries,
    _parse_answer_loose,
    _sanitize_citation_ids,
    _sha256,
)
from scripts.run_memory_v3_quality_replay import (
    GeneratedAnswer,
    ObservedResponsesTransport,
    QualityReplayError,
    build_answer_repair_prompt,
    parse_judge_decision,
)


CONTRACT_VERSION = "memory-test-confidence-v7"
SCHEMA_VERSION = 5
LEGACY_JUDGE_PACKET_MODE = "legacy-stored"
RESAMPLE_JUDGE_PACKET_MODE = "full"
UNSAMPLED_JUDGE_PACKET_MODE = "not-sampled"
_ANSWER_MARKER = "Generated answer:\n"
_CITATION_MARKER = "\nGenerated citation IDs:\n"
_ABSTAINED_MARKER = "\nGenerated abstained flag:\n"
_PACKET_MARKER = "\nRetrieved packet:\n"
_SIGNATURE_LENGTH = 64


def _valid_signature(value: Any) -> bool:
    text = str(value or "")
    return len(text) == _SIGNATURE_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def _load_latest_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("case_id"):
            raise ValueError("confidence input rows must have case_id")
        latest[str(value["case_id"])] = value
    rows = list(latest.values())
    base_signatures = {str(row.get("resume_base_signature") or "") for row in rows}
    if any(not _valid_signature(row.get("case_input_signature")) for row in rows):
        raise ValueError("confidence input requires signed fullchain rows")
    if len(base_signatures) != 1 or not _valid_signature(
        next(iter(base_signatures), "")
    ):
        raise ValueError("confidence input rows must share one signed fullchain base")
    return rows


def _replace_judge_answer(
    prompt_lines: Sequence[str],
    *,
    answer: str,
    cited_ids: Sequence[str],
    abstained: bool,
) -> list[str]:
    """Replace only generated-answer fields in a frozen judge prompt."""

    replaced: list[str] = []
    found = False
    for line in prompt_lines:
        if _ANSWER_MARKER not in line:
            replaced.append(str(line))
            continue
        prefix, rest = str(line).split(_ANSWER_MARKER, 1)
        _, rest = rest.split(_CITATION_MARKER, 1)
        _, rest = rest.split(_ABSTAINED_MARKER, 1)
        _, suffix = rest.split(_PACKET_MARKER, 1)
        replaced.append(
            prefix
            + _ANSWER_MARKER
            + answer
            + _CITATION_MARKER
            + json.dumps(list(cited_ids), ensure_ascii=False)
            + _ABSTAINED_MARKER
            + json.dumps(bool(abstained))
            + _PACKET_MARKER
            + suffix
        )
        found = True
    if not found:
        raise ValueError("judge prompt does not contain replaceable answer fields")
    return replaced


def _extract_judge_answer(
    prompt_lines: Sequence[str],
) -> tuple[str, tuple[str, ...], bool]:
    """Recover generated-answer fields from a frozen private judge prompt."""

    for line in prompt_lines:
        if _ANSWER_MARKER not in line:
            continue
        _, rest = str(line).split(_ANSWER_MARKER, 1)
        answer, rest = rest.split(_CITATION_MARKER, 1)
        citations_json, rest = rest.split(_ABSTAINED_MARKER, 1)
        abstained_json, _ = rest.split(_PACKET_MARKER, 1)
        citations = json.loads(citations_json)
        abstained = json.loads(abstained_json)
        if (
            not isinstance(citations, list)
            or any(not isinstance(value, str) for value in citations)
            or not isinstance(abstained, bool)
        ):
            raise ValueError("judge prompt generated-answer fields are invalid")
        return answer, tuple(citations), abstained
    raise ValueError("judge prompt does not contain generated-answer fields")


def _string_sequence_field(
    row: Mapping[str, Any], field: str, *, allow_empty: bool
) -> tuple[str, ...]:
    """Read a private string-array field without accepting scalar coercion."""

    value = row.get(field)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"confidence replay requires private {field}")
    resolved = tuple(value)
    if not allow_empty and not resolved:
        raise ValueError(f"confidence replay requires private {field}")
    return resolved


def _validate_confidence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    requires_full_judge_packet: bool,
    requires_answer_allowlist: bool,
) -> None:
    """Validate every private input before checkpoint mutation or transport.

    A confidence checkpoint is a valid completed-case prefix only after the
    complete input batch has passed its private-data contract.  In particular,
    do not let a malformed later row leave an apparently resumable prefix.
    """

    base_signatures = {str(row.get("resume_base_signature") or "") for row in rows}
    if rows and (
        len(base_signatures) != 1
        or not _valid_signature(next(iter(base_signatures), ""))
    ):
        raise ValueError("confidence input rows must share one signed fullchain base")
    if any(not _valid_signature(row.get("case_input_signature")) for row in rows):
        raise ValueError("confidence input requires signed fullchain rows")
    if any(row.get("protocol_failure_codes") for row in rows):
        raise ValueError("confidence input contains protocol failures")

    for row in rows:
        _string_sequence_field(row, "answer_prompt", allow_empty=False)
        judge_prompt = _string_sequence_field(row, "judge_prompt", allow_empty=False)
        # Both fields are retained in the private checkpoint.  Parsing them here
        # also proves they can be safely frozen/replaced before any judge call.
        _extract_judge_answer(judge_prompt)
        if requires_full_judge_packet:
            full_prompt = _string_sequence_field(
                row, "judge_prompt_full", allow_empty=False
            )
            _extract_judge_answer(full_prompt)
        if requires_answer_allowlist:
            _string_sequence_field(row, "allowed_citation_ids", allow_empty=True)


def _observation_payload(observation: Any) -> dict[str, Any]:
    return {
        "text": str(observation.text),
        "input_tokens": int(getattr(observation, "input_tokens", 0)),
        "output_tokens": int(getattr(observation, "output_tokens", 0)),
        "ttft_ms": float(getattr(observation, "ttft_ms", 0.0)),
        "model": str(getattr(observation, "model", "")),
        "usage_estimated": bool(getattr(observation, "usage_estimated", False)),
        "attempt_count": int(getattr(observation, "attempt_count", 1)),
        "no_event_attempts": int(getattr(observation, "no_event_attempts", 0)),
    }


def _generate_cached(
    *,
    transport: Any,
    prompt: Sequence[str],
    model: str,
    effort: str,
    axis: str,
    sample_index: int,
    cache_dir: Path,
    provider_attempts: int,
    provider_backoff: float,
) -> tuple[Any, bool]:
    key = _sha256(
        CONTRACT_VERSION
        + "|"
        + axis
        + "|"
        + str(sample_index)
        + "|"
        + model
        + "|"
        + effort
        + "|"
        + json.dumps(list(prompt), ensure_ascii=False)
    )
    cached = _cache_load(cache_dir, key)
    if cached is not None:
        return SimpleNamespace(**cached), True
    observation = _generate_with_retries(
        transport,
        prompt,
        model=model,
        attempts=provider_attempts,
        backoff=provider_backoff,
    )
    _cache_save(cache_dir, key, _observation_payload(observation))
    return observation, False


def _decision_payload(decision: Any) -> dict[str, Any]:
    return {
        "answer_grounded": bool(decision.answer_grounded),
        "answer_correct": bool(decision.answer_correct),
        "abstained": bool(decision.abstained),
        "reason_code": str(decision.reason_code),
    }


def _decision_key(value: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    return (
        bool(value.get("answer_grounded")),
        bool(value.get("answer_correct")),
        bool(value.get("abstained")),
    )


def _majority_decision(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [
        sample for sample in samples if isinstance(sample.get("decision"), Mapping)
    ]
    if not valid:
        return {
            "answer_grounded": False,
            "answer_correct": False,
            "abstained": False,
            "grounded_correct": False,
            "valid_samples": 0,
        }
    keys = Counter(_decision_key(sample["decision"]) for sample in valid)
    winner, _ = keys.most_common(1)[0]
    return {
        "answer_grounded": winner[0],
        "answer_correct": winner[1],
        "abstained": winner[2],
        "grounded_correct": winner[0] and winner[1],
        "valid_samples": len(valid),
    }


def _judge_samples(
    *,
    prompt: Sequence[str],
    baseline_decision: Mapping[str, Any] | None,
    repeats: int,
    axis: str,
    transport: Any,
    model: str,
    effort: str,
    cache_dir: Path,
    provider_attempts: int,
    provider_backoff: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if baseline_decision is not None:
        samples.append(
            {"sample_index": 0, "cached": True, "decision": dict(baseline_decision)}
        )
    start = 1 if baseline_decision is not None else 0
    for sample_index in range(start, repeats):
        item: dict[str, Any] = {"sample_index": sample_index}
        try:
            observation, cached = _generate_cached(
                transport=transport,
                prompt=prompt,
                model=model,
                effort=effort,
                axis=axis,
                sample_index=sample_index,
                cache_dir=cache_dir,
                provider_attempts=provider_attempts,
                provider_backoff=provider_backoff,
            )
            item.update(_observation_payload(observation))
            item["cached"] = cached
            item["decision"] = _decision_payload(parse_judge_decision(observation.text))
        except QualityReplayError as exc:
            item["provider_error"] = str(exc)
            item["provider_failure_kind"] = str(
                getattr(exc, "failure_kind", None) or "provider_failed"
            )
        except (ValueError, json.JSONDecodeError) as exc:
            item["protocol_error"] = type(exc).__name__
        samples.append(item)
    return samples


def _baseline_decision(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("protocol_failure_codes"):
        return None
    return {
        "answer_grounded": bool(row.get("answer_grounded")),
        "answer_correct": bool(row.get("answer_correct")),
        "abstained": bool(row.get("abstained")),
        "reason_code": str(row.get("judge_reason_code") or "baseline"),
    }


def _repair_answer(
    *,
    answer_prompt: Sequence[str],
    parsed: Any,
    allowed_ids: Sequence[str],
    transport: Any,
    judge_model: str,
    judge_effort: str,
    cache_dir: Path,
    axis: str,
    provider_attempts: int,
    provider_backoff: float,
) -> tuple[Any, dict[str, Any] | None]:
    cited = tuple(str(value) for value in parsed.cited_source_message_ids)
    allowed = tuple(str(value) for value in allowed_ids)
    if parsed.abstained or (cited and set(cited) <= set(allowed)):
        return parsed, None
    sanitized, citation_sanitized = _sanitize_citation_ids(cited, allowed)
    if citation_sanitized:
        return (
            GeneratedAnswer(
                answer=parsed.answer,
                cited_source_message_ids=sanitized,
                abstained=parsed.abstained,
            ),
            {
                "protocol_failure_codes": [],
                "observations": [],
                "citation_sanitized": True,
                "method": "deterministic_allowlist_filter",
            },
        )
    original = GeneratedAnswer(
        answer=parsed.answer,
        cited_source_message_ids=cited,
        abstained=parsed.abstained,
    )
    prompt = build_answer_repair_prompt(
        original_prompt=answer_prompt,
        answer=original,
        protocol_failure_codes=(
            ("citation_missing",) if not cited else ("citation_outside_allowlist",)
        ),
    )
    observations: list[dict[str, Any]] = []
    allowed_set = set(allowed)
    for sample_index in range(provider_attempts):
        observation, cached = _generate_cached(
            transport=transport,
            prompt=prompt,
            model=judge_model,
            effort=judge_effort,
            axis=axis,
            sample_index=sample_index,
            cache_dir=cache_dir,
            provider_attempts=provider_attempts,
            provider_backoff=provider_backoff,
        )
        payload = _observation_payload(observation)
        payload["cached"] = cached
        observations.append(payload)
        try:
            repaired = _parse_answer_loose(observation.text)
        except (ValueError, json.JSONDecodeError):
            continue
        if (
            repaired.answer == original.answer
            and repaired.abstained == original.abstained
            and set(repaired.cited_source_message_ids) <= allowed_set
        ):
            return repaired, {
                "protocol_failure_codes": [],
                "observations": observations,
            }
    return parsed, {
        "protocol_failure_codes": ["citation_repair_invalid"],
        "observations": observations,
    }


def run_confidence_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    answer_transport: Any,
    judge_transport: Any,
    cache_dir: Path,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    answer_effort: str = DEFAULT_ANSWER_EFFORT,
    judge_model: str = DEFAULT_AUX_MODEL,
    judge_effort: str = DEFAULT_AUX_EFFORT,
    answer_repeats: int = 1,
    judge_repeats: int = 3,
    provider_attempts: int = PROVIDER_ATTEMPTS,
    provider_backoff: float = PROVIDER_BACKOFF_SECONDS,
    checkpoint_path: Path | None = None,
    compare_full_judge_packet: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if answer_repeats < 1 or answer_repeats % 2 == 0:
        raise ValueError("answer_repeats must be a positive odd integer")
    if judge_repeats < 1 or judge_repeats % 2 == 0:
        raise ValueError("judge_repeats must be a positive odd integer")
    requires_full_judge_packet = answer_repeats > 1 or compare_full_judge_packet
    _validate_confidence_rows(
        rows,
        requires_full_judge_packet=requires_full_judge_packet,
        requires_answer_allowlist=answer_repeats > 1,
    )
    private_rows: list[dict[str, Any]] = []
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("", encoding="utf-8")
    for row_index, row in enumerate(rows, start=1):
        answer_prompt = _string_sequence_field(row, "answer_prompt", allow_empty=False)
        judge_prompt = _string_sequence_field(row, "judge_prompt", allow_empty=False)
        baseline_judge_packet_mode = str(
            row.get("judge_packet_mode") or LEGACY_JUDGE_PACKET_MODE
        )
        resample_judge_packet_mode = (
            RESAMPLE_JUDGE_PACKET_MODE
            if requires_full_judge_packet
            else UNSAMPLED_JUDGE_PACKET_MODE
        )
        full_judge_prompt = (
            _string_sequence_field(row, "judge_prompt_full", allow_empty=False)
            if requires_full_judge_packet
            else ()
        )
        repair_allowed_ids = (
            _string_sequence_field(row, "allowed_citation_ids", allow_empty=True)
            if answer_repeats > 1
            else tuple(str(value) for value in row.get("packed_source_ids") or ())
        )
        _, _, frozen_generated_abstained = _extract_judge_answer(judge_prompt)
        generated_abstained = bool(
            row.get("generated_abstained", frozen_generated_abstained)
        )
        canonical_judge_prompt = tuple(
            _replace_judge_answer(
                judge_prompt,
                answer=str(row.get("answer") or ""),
                cited_ids=tuple(row.get("cited_source_message_ids") or ()),
                abstained=generated_abstained,
            )
        )
        baseline = _baseline_decision(row)
        baseline_judges = _judge_samples(
            prompt=canonical_judge_prompt,
            baseline_decision=baseline,
            repeats=judge_repeats,
            axis="baseline-judge|" + str(row["case_id"]),
            transport=judge_transport,
            model=judge_model,
            effort=judge_effort,
            cache_dir=cache_dir,
            provider_attempts=provider_attempts,
            provider_backoff=provider_backoff,
        )
        answer_samples: list[dict[str, Any]] = []
        baseline_answer = {
            "sample_index": 0,
            "sample_role": "baseline",
            "answer": str(row.get("answer") or ""),
            "cited_source_message_ids": list(row.get("cited_source_message_ids") or ()),
            "abstained": generated_abstained,
            "decision_envelope": row.get("decision_envelope_shadow"),
            "decision_envelope_validation": row.get("decision_envelope_validation"),
            "decision_envelope_reanswered": bool(row.get("decision_envelope_reanswered")),
            "decision_envelope_expanded": bool(row.get("decision_envelope_expanded")),
            "judge_packet_mode": baseline_judge_packet_mode,
            "judge_samples": baseline_judges,
            "judge_majority": _majority_decision(baseline_judges),
        }
        answer_samples.append(baseline_answer)
        if (
            requires_full_judge_packet
            and baseline_judge_packet_mode != RESAMPLE_JUDGE_PACKET_MODE
        ):
            frozen_full_prompt = _replace_judge_answer(
                full_judge_prompt,
                answer=str(row.get("answer") or ""),
                cited_ids=tuple(row.get("cited_source_message_ids") or ()),
                abstained=generated_abstained,
            )
            frozen_full_judges = _judge_samples(
                prompt=frozen_full_prompt,
                baseline_decision=None,
                repeats=judge_repeats,
                axis="baseline-full-judge|" + str(row["case_id"]),
                transport=judge_transport,
                model=judge_model,
                effort=judge_effort,
                cache_dir=cache_dir,
                provider_attempts=provider_attempts,
                provider_backoff=provider_backoff,
            )
            answer_samples.append(
                {
                    "sample_index": 0,
                    "sample_role": "baseline-full-rejudge",
                    "answer": str(row.get("answer") or ""),
                    "cited_source_message_ids": list(
                        row.get("cited_source_message_ids") or ()
                    ),
                    "abstained": generated_abstained,
                    "judge_packet_mode": RESAMPLE_JUDGE_PACKET_MODE,
                    "judge_samples": frozen_full_judges,
                    "judge_majority": _majority_decision(frozen_full_judges),
                }
            )
        for sample_index in range(1, answer_repeats):
            sample: dict[str, Any] = {
                "sample_index": sample_index,
                "sample_role": "resample",
            }
            try:
                observation, cached = _generate_cached(
                    transport=answer_transport,
                    prompt=answer_prompt,
                    model=answer_model,
                    effort=answer_effort,
                    axis="answer|" + str(row["case_id"]),
                    sample_index=sample_index,
                    cache_dir=cache_dir,
                    provider_attempts=provider_attempts,
                    provider_backoff=provider_backoff,
                )
                sample.update(_observation_payload(observation))
                sample["cached"] = cached
                parsed = _parse_answer_loose(observation.text)
                parsed, repair = _repair_answer(
                    answer_prompt=answer_prompt,
                    parsed=parsed,
                    allowed_ids=repair_allowed_ids,
                    transport=judge_transport,
                    judge_model=judge_model,
                    judge_effort=judge_effort,
                    cache_dir=cache_dir,
                    axis=f"answer-{sample_index}-repair|{row['case_id']}",
                    provider_attempts=provider_attempts,
                    provider_backoff=provider_backoff,
                )
                sample["repair"] = repair
                sample["citation_sanitized"] = bool(
                    repair and repair.get("citation_sanitized")
                )
                repair_protocol_failure_codes = tuple(
                    str(value)
                    for value in (
                        repair.get("protocol_failure_codes")
                        if isinstance(repair, Mapping)
                        else ()
                    )
                    or ()
                )
                sample["answer"] = parsed.answer
                sample["cited_source_message_ids"] = list(
                    parsed.cited_source_message_ids
                )
                sample["abstained"] = bool(parsed.abstained)
                sample_prompt = _replace_judge_answer(
                    full_judge_prompt,
                    answer=parsed.answer,
                    cited_ids=parsed.cited_source_message_ids,
                    abstained=parsed.abstained,
                )
                sample["judge_packet_mode"] = RESAMPLE_JUDGE_PACKET_MODE
                if repair_protocol_failure_codes:
                    sample["repair_protocol_failure_codes"] = list(
                        repair_protocol_failure_codes
                    )
                else:
                    judges = _judge_samples(
                        prompt=sample_prompt,
                        baseline_decision=None,
                        repeats=judge_repeats,
                        axis=f"answer-{sample_index}-judge|{row['case_id']}",
                        transport=judge_transport,
                        model=judge_model,
                        effort=judge_effort,
                        cache_dir=cache_dir,
                        provider_attempts=provider_attempts,
                        provider_backoff=provider_backoff,
                    )
                    sample["judge_samples"] = judges
                    sample["judge_majority"] = _majority_decision(judges)
            except QualityReplayError as exc:
                sample["provider_error"] = str(exc)
                sample["provider_failure_kind"] = str(
                    getattr(exc, "failure_kind", None) or "provider_failed"
                )
            except (ValueError, json.JSONDecodeError) as exc:
                sample["protocol_error"] = type(exc).__name__
            answer_samples.append(sample)
        private_row = {
            "case_id": str(row["case_id"]),
            "category": str(row.get("category") or "unknown"),
            "case_input_signature": str(row["case_input_signature"]),
            "resume_base_signature": str(row["resume_base_signature"]),
            "packet_source_hash": _sha256(
                json.dumps(row.get("packed_source_ids") or (), ensure_ascii=False)
            ),
            "answer_prompt": list(answer_prompt),
            "judge_prompt": list(canonical_judge_prompt),
            "judge_prompt_full": list(full_judge_prompt),
            "baseline_judge_packet_mode": baseline_judge_packet_mode,
            "resample_judge_packet_mode": resample_judge_packet_mode,
            "compare_full_judge_packet": bool(compare_full_judge_packet),
            "answer_samples": answer_samples,
        }
        private_rows.append(private_row)
        if checkpoint_path is not None:
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(private_row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        print(
            "confidence_progress " f"completed={row_index} total={len(rows)}",
            file=sys.stderr,
            flush=True,
        )
    return private_rows, build_public_summary(private_rows)


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def build_public_summary(private_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in private_rows:
        buckets[str(row.get("category") or "unknown")].append(row)

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        judge_flips = 0
        answer_flips = 0
        answer_sampled = 0
        cross_mode_disagreements = 0
        cross_mode_sampled = 0
        full_packet_comparison_requested = 0
        full_packet_comparison_performed = 0
        majority_gc = 0
        baseline_gc = 0
        provider_failures = 0
        protocol_failures = 0
        baseline_packet_modes: set[str] = set()
        resample_packet_modes: set[str] = set()
        for row in rows:
            full_packet_comparison_requested += int(
                bool(row.get("compare_full_judge_packet"))
            )
            baseline_packet_modes.add(
                str(
                    row.get("baseline_judge_packet_mode")
                    or LEGACY_JUDGE_PACKET_MODE
                )
            )
            answers = list(row.get("answer_samples") or ())
            resample_packet_modes.add(
                str(
                    row.get("resample_judge_packet_mode")
                    or (
                        LEGACY_JUDGE_PACKET_MODE
                        if len(answers) > 1
                        else UNSAMPLED_JUDGE_PACKET_MODE
                    )
                )
            )
            if not answers:
                protocol_failures += 1
                continue
            baseline = answers[0]
            judges = list(baseline.get("judge_samples") or ())
            decisions = {
                _decision_key(sample["decision"])
                for sample in judges
                if isinstance(sample.get("decision"), Mapping)
            }
            judge_flips += int(len(decisions) > 1)
            baseline_majority = baseline.get("judge_majority") or {}
            baseline_gc += int(
                bool(baseline.get("judge_samples"))
                and bool(
                    (baseline.get("judge_samples") or [{}])[0]
                    .get("decision", {})
                    .get("answer_grounded")
                )
                and bool(
                    (baseline.get("judge_samples") or [{}])[0]
                    .get("decision", {})
                    .get("answer_correct")
                )
            )
            majority_gc += int(bool(baseline_majority.get("grounded_correct")))
            outcomes_by_packet_mode: dict[str, list[bool]] = defaultdict(list)
            frozen_baseline_by_role: dict[
                str, tuple[str, bool, tuple[str, tuple[str, ...], bool]]
            ] = {}
            for answer_index, answer in enumerate(answers):
                repair = answer.get("repair")
                if isinstance(repair, Mapping) and repair.get(
                    "protocol_failure_codes"
                ):
                    continue
                majority = answer.get("judge_majority") or {}
                if int(majority.get("valid_samples") or 0) <= 0:
                    continue
                packet_mode = str(
                    answer.get("judge_packet_mode")
                    or (
                        row.get("baseline_judge_packet_mode")
                        if answer_index == 0
                        else row.get("resample_judge_packet_mode")
                    )
                    or LEGACY_JUDGE_PACKET_MODE
                )
                outcomes_by_packet_mode[packet_mode].append(
                    bool(majority.get("grounded_correct"))
                )
                sample_role = str(answer.get("sample_role") or "")
                if answer_index == 0 and not sample_role:
                    sample_role = "baseline"
                if sample_role in {"baseline", "baseline-full-rejudge"}:
                    frozen_baseline_by_role[sample_role] = (
                        packet_mode,
                        bool(majority.get("grounded_correct")),
                        (
                            str(answer.get("answer") or ""),
                            tuple(
                                str(value)
                                for value in answer.get(
                                    "cited_source_message_ids"
                                )
                                or ()
                            ),
                            bool(answer.get("abstained")),
                        ),
                    )
            comparable_outcomes = [
                outcomes
                for outcomes in outcomes_by_packet_mode.values()
                if len(outcomes) > 1
            ]
            if comparable_outcomes:
                answer_sampled += 1
                answer_flips += int(
                    any(len(set(outcomes)) > 1 for outcomes in comparable_outcomes)
                )
            frozen_baseline = frozen_baseline_by_role.get("baseline")
            frozen_full_rejudge = frozen_baseline_by_role.get(
                "baseline-full-rejudge"
            )
            if (
                frozen_baseline is not None
                and frozen_full_rejudge is not None
                and frozen_baseline[0] != frozen_full_rejudge[0]
                and frozen_baseline[2] == frozen_full_rejudge[2]
            ):
                cross_mode_sampled += 1
                full_packet_comparison_performed += int(
                    bool(row.get("compare_full_judge_packet"))
                )
                cross_mode_disagreements += int(
                    frozen_baseline[1] != frozen_full_rejudge[1]
                )
            for answer in answers:
                provider_failures += int(bool(answer.get("provider_error")))
                protocol_failures += int(bool(answer.get("protocol_error")))
                repair = answer.get("repair")
                protocol_failures += int(
                    isinstance(repair, Mapping)
                    and bool(repair.get("protocol_failure_codes"))
                )
                for sample in answer.get("judge_samples") or ():
                    provider_failures += int(bool(sample.get("provider_error")))
                    protocol_failures += int(bool(sample.get("protocol_error")))
        total = len(rows)
        return {
            "cases": total,
            "baseline_grounded_correct": baseline_gc,
            "judge_majority_grounded_correct": majority_gc,
            "judge_majority_rate": round(majority_gc / total, 6) if total else 0.0,
            "judge_majority_wilson95": _wilson_interval(majority_gc, total),
            "judge_flip_cases": judge_flips,
            "judge_flip_rate": round(judge_flips / total, 6) if total else 0.0,
            "answer_sampled_cases": answer_sampled,
            "answer_outcome_flip_cases": answer_flips,
            "answer_outcome_flip_rate": (
                round(answer_flips / answer_sampled, 6) if answer_sampled else 0.0
            ),
            "cross_mode_sampled_cases": cross_mode_sampled,
            "cross_mode_outcome_disagreement_cases": cross_mode_disagreements,
            "cross_mode_outcome_disagreement_rate": (
                round(cross_mode_disagreements / cross_mode_sampled, 6)
                if cross_mode_sampled
                else 0.0
            ),
            "full_judge_packet_comparison_requested_cases": (
                full_packet_comparison_requested
            ),
            "full_judge_packet_comparison_performed_cases": (
                full_packet_comparison_performed
            ),
            "provider_failures": provider_failures,
            "protocol_failures": protocol_failures,
            "baseline_judge_packet_modes": sorted(baseline_packet_modes),
            "resample_judge_packet_modes": sorted(resample_packet_modes),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "overall": aggregate(private_rows),
        "by_category": {
            category: aggregate(rows) for category, rows in sorted(buckets.items())
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _result_exit_code(public: Mapping[str, Any]) -> int:
    overall = public.get("overall") or {}
    return (
        0
        if not overall.get("provider_failures") and not overall.get("protocol_failures")
        else 2
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure full-chain answer/judge variance."
    )
    parser.add_argument("--input-detail", required=True, type=Path)
    parser.add_argument("--output-private", required=True, type=Path)
    parser.add_argument("--output-public", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--categories", default="")
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case IDs for targeted private replay.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--answer-repeats", type=int, default=1)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument(
        "--compare-full-judge-packet",
        action="store_true",
        help=(
            "Rejudge the frozen baseline answer against judge_prompt_full; "
            "does not generate another answer."
        ),
    )
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-effort", default=DEFAULT_ANSWER_EFFORT)
    parser.add_argument("--judge-model", default=DEFAULT_AUX_MODEL)
    parser.add_argument("--judge-effort", default=DEFAULT_AUX_EFFORT)
    parser.add_argument("--provider-attempts", type=int, default=PROVIDER_ATTEMPTS)
    parser.add_argument(
        "--provider-backoff", type=float, default=PROVIDER_BACKOFF_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    rows = _load_latest_rows(args.input_detail)
    categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    }
    if categories:
        rows = [row for row in rows if str(row.get("category") or "") in categories]
    case_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
    if case_ids:
        rows = [row for row in rows if str(row.get("case_id") or "") in case_ids]
    if args.limit > 0:
        rows = rows[: args.limit]
    settings = AppSettings()
    answer_client, judge_client = _build_eval_clients(
        settings,
        answer_model=args.answer_model,
        answer_effort=args.answer_effort,
        aux_model=args.judge_model,
        aux_effort=args.judge_effort,
    )
    private_rows, public = run_confidence_replay(
        rows,
        answer_transport=ObservedResponsesTransport(
            answer_client, max_attempts=args.provider_attempts
        ),
        judge_transport=ObservedResponsesTransport(
            judge_client, max_attempts=args.provider_attempts
        ),
        cache_dir=args.cache_dir,
        answer_model=args.answer_model,
        answer_effort=args.answer_effort,
        judge_model=args.judge_model,
        judge_effort=args.judge_effort,
        answer_repeats=args.answer_repeats,
        judge_repeats=args.judge_repeats,
        provider_attempts=args.provider_attempts,
        provider_backoff=args.provider_backoff,
        checkpoint_path=args.output_private,
        compare_full_judge_packet=args.compare_full_judge_packet,
    )
    _write_jsonl(args.output_private, private_rows)
    args.output_public.parent.mkdir(parents=True, exist_ok=True)
    args.output_public.write_text(
        json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return _result_exit_code(public)


if __name__ == "__main__":
    raise SystemExit(main())
