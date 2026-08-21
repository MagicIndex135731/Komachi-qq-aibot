from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

from sqlalchemy import select, text

from app.config import AppSettings, load_runtime_config
from app.core.memory_backfill import (
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)
from app.core.memory_backfill_runner import group_watermarks_from_manifest
from app.core.legacy_memory_context import (
    GroupMemoryContextRequest,
    member_label_for_user,
)
from app.core.memory_context_packer import EvidenceMessage
from app.core.memory_context_packer import MemoryContextPacker
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine, session_scope
from app.storage.models import MemoryBackfillRun, RetrievalIndexState
from app.storage.repositories import MessageRepository, UserRepository
try:
    from .evaluate_memory_recall import (
        EvaluationCase,
        load_evaluation_cases,
        validate_real_dataset_review,
    )
    from .evaluate_memory_v3 import (
        V3Observation,
        audit_v3_quality_sources,
        build_v3_observation,
        evaluate_v3,
        load_message_metadata,
        load_v3_quality_sidecar,
        observation_as_safe_dict,
        quality_sidecar_template,
        retrieval_fingerprint_sha256,
        validate_v3_dataset_contract,
        validate_v3_dataset_sources,
    )
except ImportError:  # Direct script execution.
    from evaluate_memory_recall import (
        EvaluationCase,
        load_evaluation_cases,
        validate_real_dataset_review,
    )
    from evaluate_memory_v3 import (
        V3Observation,
        audit_v3_quality_sources,
        build_v3_observation,
        evaluate_v3,
        load_message_metadata,
        load_v3_quality_sidecar,
        observation_as_safe_dict,
        quality_sidecar_template,
        retrieval_fingerprint_sha256,
        validate_v3_dataset_contract,
        validate_v3_dataset_sources,
    )


class AcceptanceGateError(RuntimeError):
    def __init__(self, codes: Sequence[str]) -> None:
        super().__init__("production memory evaluation acceptance gate failed")
        self.codes = tuple(codes)


def _validate_quality_resume_artifacts(args: argparse.Namespace) -> dict:
    from scripts.resume_memory_v3_quality_replay import (
        validate_quality_resume_receipt,
    )

    return validate_quality_resume_receipt(
        args.quality_resume_receipt,
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        prepared_report_path=args.prepared_report,
        parent_quality_sidecar_path=args.quality_resume_parent_sidecar,
        parent_private_replay_path=args.quality_resume_parent_private_replay,
        parent_visibility_path=args.quality_visibility_artifact,
        parent_gate_report_path=args.quality_resume_parent_gate_report,
        parent_results_path=args.quality_resume_parent_results,
        parent_benchmark_path=args.quality_resume_parent_benchmark,
        child_quality_sidecar_path=args.quality_sidecar,
        child_private_replay_path=args.quality_private_replay,
    )


def _validate_quality_rebind_artifacts(args: argparse.Namespace) -> dict:
    from scripts.rebind_memory_v3_quality import validate_quality_rebind_receipt

    return validate_quality_rebind_receipt(
        args.quality_rebind_receipt,
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        prepared_report_path=args.prepared_report,
        old_quality_sidecar_path=args.quality_rebind_parent_sidecar,
        old_private_replay_path=args.quality_rebind_parent_private_replay,
        old_resume_receipt_path=args.quality_resume_receipt,
        old_visibility_path=args.quality_rebind_parent_visibility,
        old_gate_report_path=args.quality_rebind_parent_gate_report,
        old_results_path=args.quality_rebind_parent_results,
        old_benchmark_path=args.quality_rebind_parent_benchmark,
        old_resume_parent_quality_sidecar_path=args.quality_resume_parent_sidecar,
        old_resume_parent_private_replay_path=args.quality_resume_parent_private_replay,
        old_resume_parent_gate_report_path=args.quality_resume_parent_gate_report,
        old_resume_parent_results_path=args.quality_resume_parent_results,
        old_resume_parent_benchmark_path=args.quality_resume_parent_benchmark,
        new_failed_gate_report_path=args.quality_rebind_source_gate_report,
        new_results_path=args.quality_rebind_source_results,
        new_benchmark_path=args.quality_rebind_source_benchmark,
        child_quality_sidecar_path=args.quality_sidecar,
        child_private_replay_path=args.quality_private_replay,
        child_visibility_path=args.quality_visibility_artifact,
    )

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the fail-closed raw-message V3 production evaluation."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--benchmark-output", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument(
        "--prepared-report",
        required=True,
        type=Path,
        help="Prepare-phase V3 backfill report whose ready generation is evaluated.",
    )
    parser.add_argument(
        "--quality-sidecar",
        type=Path,
        help="Content-free controlled-answer/index-visibility judgment JSON",
    )
    parser.add_argument(
        "--quality-private-replay",
        type=Path,
        help="Private controlled replay artifact bound by the quality sidecar",
    )
    parser.add_argument(
        "--quality-visibility-artifact",
        type=Path,
        help="Disposable-clone visibility artifact bound by the quality sidecar",
    )
    parser.add_argument("--quality-resume-receipt", type=Path)
    parser.add_argument("--quality-resume-parent-sidecar", type=Path)
    parser.add_argument("--quality-resume-parent-private-replay", type=Path)
    parser.add_argument("--quality-resume-parent-gate-report", type=Path)
    parser.add_argument("--quality-resume-parent-results", type=Path)
    parser.add_argument("--quality-resume-parent-benchmark", type=Path)
    parser.add_argument("--quality-rebind-receipt", type=Path)
    parser.add_argument("--quality-rebind-parent-sidecar", type=Path)
    parser.add_argument("--quality-rebind-parent-private-replay", type=Path)
    parser.add_argument("--quality-rebind-parent-visibility", type=Path)
    parser.add_argument("--quality-rebind-parent-gate-report", type=Path)
    parser.add_argument("--quality-rebind-parent-results", type=Path)
    parser.add_argument("--quality-rebind-parent-benchmark", type=Path)
    parser.add_argument("--quality-rebind-source-gate-report", type=Path)
    parser.add_argument("--quality-rebind-source-results", type=Path)
    parser.add_argument("--quality-rebind-source-benchmark", type=Path)
    parser.add_argument(
        "--quality-template-output",
        type=Path,
        help="Write a retrieval-bound judgment template; missing judgments still fail",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=250)
    parser.add_argument(
        "--max-retrieval-p95-ms",
        type=float,
        default=500.0,
        help="Maximum accepted local retrieval P95 in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--context-profile",
        choices=("legacy", "adaptive"),
        default="adaptive",
        help="Context budget contract to evaluate and bind into the gate report.",
    )
    parser.add_argument("--enforce-real-dataset", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except AcceptanceGateError as exc:
        _print_safe_failure(exc.codes)
        return 2
    except (OSError, ValueError, RuntimeError):
        _print_safe_failure(("EVAL_INPUT_OR_RUNTIME_INVALID",))
        return 2


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.warmup < 20:
        raise ValueError("warmup must be at least 20")
    if not math.isfinite(args.max_retrieval_p95_ms) or args.max_retrieval_p95_ms <= 0:
        raise ValueError("max-retrieval-p95-ms must be a positive finite number")
    resume_paths = (
        args.quality_resume_receipt,
        args.quality_resume_parent_sidecar,
        args.quality_resume_parent_private_replay,
        args.quality_resume_parent_gate_report,
        args.quality_resume_parent_results,
        args.quality_resume_parent_benchmark,
    )
    if any(path is not None for path in resume_paths) and not all(
        path is not None for path in resume_paths
    ):
        raise ValueError("quality resume artifacts must be provided together")
    rebind_paths = (
        args.quality_rebind_receipt,
        args.quality_rebind_parent_sidecar,
        args.quality_rebind_parent_private_replay,
        args.quality_rebind_parent_visibility,
        args.quality_rebind_parent_gate_report,
        args.quality_rebind_parent_results,
        args.quality_rebind_parent_benchmark,
        args.quality_rebind_source_gate_report,
        args.quality_rebind_source_results,
        args.quality_rebind_source_benchmark,
    )
    if any(path is not None for path in rebind_paths) and not all(
        path is not None for path in rebind_paths
    ):
        raise ValueError("quality rebind artifacts must be provided together")
    if all(path is not None for path in rebind_paths) and not all(
        path is not None for path in resume_paths
    ):
        raise ValueError("quality rebind requires the complete parent resume chain")
    cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    try:
        gate_tag_counts = validate_v3_dataset_contract(cases)
    except ValueError as exc:
        raise AcceptanceGateError(("AC_DATASET_V3_CONTRACT",)) from exc
    measured_runs = max(250, len(cases) * 5, args.benchmark_runs)

    functional_settings = AppSettings().model_copy(
        update={
            "memory_orchestration_v2_enabled": True,
            "memory_orchestration_shadow_mode": False,
            "memory_raw_v3_enabled": True,
            "memory_adaptive_context_enabled": args.context_profile == "adaptive",
            "memory_query_rewrite_enabled": True,
            "memory_llm_rerank_enabled": False,
            "memory_max_evidence_messages": 150,
            "memory_history_context_budget_tokens": 24_000,
            "context_recent_limit": 60,
            "llm_builtin_web_search": False,
        }
    )
    _validate_v3_runtime_settings(functional_settings)
    runtime_config = load_runtime_config(functional_settings)
    prepared = _load_prepared_report(
        args.prepared_report,
        database=args.database,
    )
    prepared_generation = int(prepared["vector_generation"])
    filter_manifest = _load_strict_json_object(args.manifest)
    filter_watermarks = group_watermarks_from_manifest(filter_manifest)
    metadata = load_message_metadata(args.database)
    candidate_filter = _snapshot_candidate_filter(
        metadata=metadata,
        snapshot_watermarks=filter_watermarks,
    )
    engine = build_engine(args.database)
    try:
        llm_client = build_llm_client(settings=functional_settings, engine=engine)
        runtime = build_memory_runtime(
            settings=functional_settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(functional_settings.bot_qq),
            raw_message_embedding_generation_override=prepared_generation,
            evaluation_candidate_filter=candidate_filter,
        )
        _validate_local_vector_runtime(runtime, warm=True)
        manifest = _validate_v3_rollout_state(
            engine=engine,
            runtime=runtime,
            database=args.database,
            manifest_path=args.manifest,
            prepared_report=prepared,
        )
        snapshot_manifest_sha256 = message_ledger_manifest_sha256(manifest)
        group_watermarks = group_watermarks_from_manifest(manifest)
        validate_real_dataset_review(
            cases,
            dataset_sha256=dataset_sha256,
            review_path=args.review,
            database=args.database,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            snapshot_watermarks=group_watermarks,
        )
        try:
            validate_v3_dataset_sources(
                cases,
                metadata=metadata,
                snapshot_watermarks=group_watermarks,
            )
        except ValueError as exc:
            raise AcceptanceGateError(("AC_DATASET_V3_SOURCE_COVERAGE",)) from exc
        requests = tuple(
            _build_request(
                engine=engine,
                settings=functional_settings,
                case=case,
                snapshot_watermark=group_watermarks[case.group_id],
            )
            for case in cases
        )
        observations: list[V3Observation] = []
        answer_prompt_sha256_by_case: list[str] = []
        # Delayed import avoids the quality replay module's intentional import of
        # this evaluator while still using the exact production replay prompt.
        from scripts.run_memory_v3_quality_replay import build_answer_prompt

        vector_succeeded = False
        for case_index, request in enumerate(requests):
            started = perf_counter()
            trace = runtime.v2_provider.evaluate(request)
            latency_ms = (perf_counter() - started) * 1000
            vector_succeeded = _require_successful_vector_trace(
                trace,
                previously_succeeded=vector_succeeded,
            )
            observation = build_v3_observation(
                case_index=case_index,
                case=cases[case_index],
                trace=trace,
                requester_uin=str(request.current_user_id),
                metadata=metadata,
                snapshot_watermark=group_watermarks[cases[case_index].group_id],
                history_packet_tokens=_history_packet_tokens(
                    trace.result.packed_context
                ),
                retrieval_latency_ms=latency_ms,
            )
            observations.append(observation)
            answer_prompt = build_answer_prompt(
                case=cases[case_index],
                trace=trace,
                runtime_config=runtime_config,
            )
            answer_prompt_sha256_by_case.append(
                hashlib.sha256(
                    json.dumps(
                        answer_prompt,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )

        _validate_local_vector_runtime(runtime, warm=False)
        if not vector_succeeded:
            raise AcceptanceGateError(("AC_VECTOR_NOT_EXERCISED",))

        benchmark_settings = functional_settings.model_copy(
            update={"memory_query_rewrite_enabled": False}
        )
        benchmark_runtime = build_memory_runtime(
            settings=benchmark_settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(benchmark_settings.bot_qq),
            raw_message_embedding_generation_override=prepared_generation,
            evaluation_candidate_filter=candidate_filter,
        )
        _validate_local_vector_runtime(benchmark_runtime, warm=True)
        benchmark = _benchmark(
            requests=requests,
            provider=benchmark_runtime.v2_provider,
            warmup=args.warmup,
            runs=measured_runs,
            enforce_vector=True,
        )
        _validate_local_vector_runtime(benchmark_runtime, warm=False)
        retrieval_fingerprint = retrieval_fingerprint_sha256(observations)
        quality_template = quality_sidecar_template(
            dataset_sha256=dataset_sha256,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            retrieval_fingerprint=retrieval_fingerprint,
            case_count=len(cases),
            context_profile=args.context_profile,
        )
        if args.quality_template_output is not None:
            _write_json(args.quality_template_output, quality_template)

        quality = None
        input_failures: list[str] = []
        if args.quality_sidecar is None:
            input_failures.append("AC_QUALITY_SIDECAR_REQUIRED")
        elif args.quality_private_replay is None:
            input_failures.append("AC_QUALITY_PRIVATE_REPLAY_REQUIRED")
        elif args.quality_visibility_artifact is None:
            input_failures.append("AC_QUALITY_VISIBILITY_ARTIFACT_REQUIRED")
        else:
            try:
                quality_version = _load_strict_json_object(args.quality_sidecar).get(
                    "quality_version"
                )
                quality = load_v3_quality_sidecar(
                    args.quality_sidecar,
                    dataset_sha256=dataset_sha256,
                    snapshot_manifest_sha256=snapshot_manifest_sha256,
                    retrieval_fingerprint=retrieval_fingerprint,
                    case_count=len(cases),
                    private_replay_path=args.quality_private_replay,
                    visibility_artifact_path=args.quality_visibility_artifact,
                    expected_vector_generation=prepared_generation,
                    expected_context_profile=args.context_profile,
                    evaluation_cases=cases,
                    expected_answer_prompt_sha256_by_case={
                        index: value
                        for index, value in enumerate(answer_prompt_sha256_by_case)
                    },
                    resume_receipt_path=(
                        args.quality_resume_receipt if quality_version == 4 else None
                    ),
                    rebind_receipt_path=(
                        args.quality_rebind_receipt if quality_version == 5 else None
                    ),
                )
            except ValueError:
                input_failures.append("AC_QUALITY_SIDECAR_INVALID")

        resume_receipt_sha256: str | None = None
        if quality is not None and quality.resume_receipt_sha256 is not None:
            try:
                _validate_quality_resume_artifacts(args)
                resume_receipt_sha256 = quality.resume_receipt_sha256
            except (OSError, ValueError):
                input_failures.append("AC_QUALITY_RESUME_RECEIPT_INVALID")
        elif all(path is not None for path in resume_paths):
            if not all(path is not None for path in rebind_paths):
                input_failures.append("AC_QUALITY_RESUME_RECEIPT_INVALID")

        rebind_receipt_sha256: str | None = None
        if quality is not None and quality.rebind_receipt_sha256 is not None:
            try:
                _validate_quality_rebind_artifacts(args)
                rebind_receipt_sha256 = quality.rebind_receipt_sha256
            except (OSError, ValueError):
                input_failures.append("AC_QUALITY_REBIND_RECEIPT_INVALID")
        elif all(path is not None for path in rebind_paths):
            input_failures.append("AC_QUALITY_REBIND_RECEIPT_INVALID")

        report = evaluate_v3(
            cases=cases,
            observations=observations,
            quality=quality,
            dataset_sha256=dataset_sha256,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            retrieval_fingerprint=retrieval_fingerprint,
            gate_tag_counts=gate_tag_counts,
        )
        if resume_receipt_sha256 is not None:
            report["quality_resume_receipt_sha256"] = resume_receipt_sha256
        if rebind_receipt_sha256 is not None:
            report["quality_rebind_receipt_sha256"] = rebind_receipt_sha256
        report["vector_generation"] = prepared_generation
        report["context_profile"] = (
            "adaptive"
            if functional_settings.memory_adaptive_context_enabled
            else "legacy"
        )
        report["acceptance_limits"] = {
            "retrieval_p95_ms": float(args.max_retrieval_p95_ms),
        }
        report["quality_sidecar_sha256"] = (
            _file_sha256(args.quality_sidecar)
            if quality is not None and args.quality_sidecar is not None
            else None
        )
        if quality is not None:
            report["metrics"].update(
                audit_v3_quality_sources(
                    cases=cases,
                    observations=observations,
                    quality=quality,
                    metadata=metadata,
                )
            )
        failures = tuple(
            dict.fromkeys(
                (
                    *input_failures,
                    *_v3_acceptance_failures(
                        report=report,
                        benchmark=benchmark,
                        adaptive_enabled=functional_settings.memory_adaptive_context_enabled,
                        max_retrieval_p95_ms=args.max_retrieval_p95_ms,
                    ),
                )
            )
        )
        report["acceptance"] = {
            "status": "failed" if failures else "passed",
            "error_codes": list(failures),
        }
        result_rows = []
        for observation, answer_prompt_sha256 in zip(
            observations,
            answer_prompt_sha256_by_case,
            strict=True,
        ):
            row = observation_as_safe_dict(observation)
            row["answer_prompt_sha256"] = answer_prompt_sha256
            result_rows.append(row)
        _write_jsonl(args.results_output, result_rows)
        _write_json(args.benchmark_output, benchmark)
        report["results_sha256"] = _file_sha256(args.results_output)
        report["benchmark_sha256"] = _file_sha256(args.benchmark_output)
        _write_json(args.report_output, report)
        if failures:
            raise AcceptanceGateError(failures)
        print(
            json.dumps(
                {
                    "dataset_sha256": dataset_sha256,
                    "case_count": len(cases),
                    "result_count": len(observations),
                    "memory_path": "raw_message_v3",
                    "acceptance": "passed",
                    "benchmark": benchmark,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    finally:
        engine.dispose()


def _build_request(
    *,
    engine: object,
    settings: AppSettings,
    case: EvaluationCase,
    snapshot_watermark: int,
) -> GroupMemoryContextRequest:
    with session_scope(engine) as session:
        messages = MessageRepository(session)
        rows = []
        for source_id in case.recent_context_message_ids:
            row = messages.get_by_platform_msg_id(source_id)
            if row is None or int(row.group_id or 0) != int(case.group_id):
                raise ValueError("evaluation recent context violates group scope")
            if int(row.id) > int(snapshot_watermark):
                raise ValueError("evaluation recent context is outside snapshot")
            rows.append(row)
        if not rows:
            raise ValueError("evaluation case has no snapshot-bound recent context")
        target = rows[-1]
        if str(target.user_id) != str(case.requester_uin):
            raise ValueError("evaluation requester does not match the frozen dataset")
        if any(
            messages.is_reserved_outbound(row)
            or messages.is_qq_blocked_outbound(row)
            or messages.is_delivery_uncertain_outbound(row)
            for row in rows
        ):
            raise ValueError("evaluation recent context contains ineligible content")
        users_by_id = UserRepository(session).get_users_by_ids(
            [int(row.user_id) for row in rows]
        )
        recent = tuple(
            EvidenceMessage(
                source_msg_id=str(row.platform_msg_id),
                speaker=member_label_for_user(
                    user_id=int(row.user_id),
                    users_by_id=users_by_id,
                    bot_user_id=settings.bot_qq,
                    bot_display_name=str(settings.bot_qq),
                ),
                content=str(row.plain_text or ""),
                sent_at=row.timestamp,
                blocked=messages.is_qq_blocked_outbound(row),
                group_id=int(row.group_id or 0),
                reply_to_msg_id=row.reply_to_msg_id,
                is_bot=int(row.user_id) == int(settings.bot_qq),
                user_id=int(row.user_id),
            )
            for row in rows
        )
        quoted = None
        quoted_source_id = case.quoted_context_message_id or target.reply_to_msg_id
        if quoted_source_id:
            quoted_row = messages.get_by_platform_msg_id(quoted_source_id)
            if quoted_row is not None and int(quoted_row.group_id or 0) == int(
                case.group_id
            ) and int(quoted_row.id) <= int(snapshot_watermark):
                quoted_users = UserRepository(session).get_users_by_ids(
                    [int(quoted_row.user_id)]
                )
                quoted = EvidenceMessage(
                    source_msg_id=str(quoted_row.platform_msg_id),
                    speaker=member_label_for_user(
                        user_id=int(quoted_row.user_id),
                        users_by_id=quoted_users,
                        bot_user_id=settings.bot_qq,
                        bot_display_name=str(settings.bot_qq),
                    ),
                    content=str(quoted_row.plain_text or ""),
                    sent_at=quoted_row.timestamp,
                    blocked=messages.is_qq_blocked_outbound(quoted_row),
                    group_id=int(quoted_row.group_id or 0),
                    reply_to_msg_id=quoted_row.reply_to_msg_id,
                    is_bot=int(quoted_row.user_id) == int(settings.bot_qq),
                    user_id=int(quoted_row.user_id),
                )

    available_input = max(
        1,
        settings.llm_context_window_tokens
        - settings.llm_max_output_tokens
        - settings.llm_context_safety_margin_tokens
        - (
            settings.llm_tool_context_reserve_tokens
            if settings.llm_builtin_web_search
            else 0
        ),
    )
    return GroupMemoryContextRequest(
        group_id=case.group_id,
        query=case.query,
        recent_messages=recent,
        quoted_message=quoted,
        target_message_id=None,
        available_input=available_input,
        now=_utc(target.timestamp),
        current_user_id=int(target.user_id),
    )


def _snapshot_candidate_filter(*, metadata, snapshot_watermarks):
    """Exclude only live rows above the frozen evaluation watermark.

    Missing, cross-group, and ineligible provenance deliberately remains in
    the trace so the corresponding safety gates can still fail closed.
    """

    frozen_watermarks = {
        int(group_id): int(watermark)
        for group_id, watermark in snapshot_watermarks.items()
    }

    def filter_candidates(*, request, resolved_query, candidates):
        del resolved_query
        group_id = int(request.group_id)
        watermark = frozen_watermarks[group_id]
        retained = []
        for candidate in candidates:
            rows = tuple(
                metadata.get(str(source_id))
                for source_id in getattr(candidate, "source_msg_ids", ())
            )
            if any(
                row is not None
                and int(row.group_id) == group_id
                and int(row.row_id) > watermark
                for row in rows
            ):
                continue
            retained.append(candidate)
        return tuple(retained)

    return filter_candidates


def _benchmark(
    *,
    requests,
    provider,
    warmup: int,
    runs: int,
    enforce_vector: bool = False,
) -> dict:
    vector_succeeded = False
    for index in range(warmup):
        trace = provider.evaluate(requests[index % len(requests)])
        if enforce_vector:
            vector_succeeded = _require_successful_vector_trace(
                trace,
                previously_succeeded=vector_succeeded,
            )
    latencies: list[float] = []
    for index in range(runs):
        started = perf_counter()
        trace = provider.evaluate(requests[index % len(requests)])
        latencies.append((perf_counter() - started) * 1000)
        if enforce_vector:
            vector_succeeded = _require_successful_vector_trace(
                trace,
                previously_succeeded=vector_succeeded,
            )
    if enforce_vector and not vector_succeeded:
        raise AcceptanceGateError(("AC_VECTOR_NOT_EXERCISED",))
    ordered = sorted(latencies)
    p50 = ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)]
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "warmup_runs": int(warmup),
        "measured_runs": int(runs),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "rewrite_enabled": False,
        "rerank_enabled": False,
        "network_enabled": False,
        "vector_success_verified": bool(vector_succeeded) if enforce_vector else False,
    }


def _require_successful_vector_trace(
    trace: object,
    *,
    previously_succeeded: bool,
) -> bool:
    attempted = set(getattr(trace, "attempted_channels", ()))
    failed = set(getattr(trace, "failed_channels", ()))
    if "vector" in failed:
        raise AcceptanceGateError(("AC_VECTOR_QUERY_FAILED",))
    candidate_counts = dict(getattr(trace, "channel_candidate_counts", ()))
    return previously_succeeded or (
        "vector" in attempted
        and "vector" not in failed
        and int(candidate_counts.get("vector", 0)) > 0
    )


def _history_packet_tokens(packed: object) -> int:
    recent = tuple(getattr(packed, "recent_messages", ()))
    recent_text = "\n\n".join(
        MemoryContextPacker._render_recent(message) for message in recent
    )
    recent_tokens = MemoryContextPacker._fallback_token_count(recent_text)
    return max(0, int(getattr(packed, "estimated_tokens", 0)) - recent_tokens)


def _validate_v3_runtime_settings(settings: AppSettings) -> None:
    if not settings.memory_raw_v3_enabled:
        raise AcceptanceGateError(("AC_V3_PATH_DISABLED",))
    if settings.memory_llm_rerank_enabled:
        raise AcceptanceGateError(("AC_RERANK_FORBIDDEN",))
    if settings.memory_adaptive_context_enabled:
        if settings.memory_adaptive_max_history_messages != 300:
            raise AcceptanceGateError(("AC_PACKET_MESSAGE_LIMIT_CONFIG",))
        if settings.memory_adaptive_max_recent_messages != 60:
            raise AcceptanceGateError(("AC_RECENT_LIMIT_CONFIG",))
        if settings.memory_normal_context_budget_tokens != 32_000:
            raise AcceptanceGateError(("AC_PACKET_TOKEN_LIMIT_CONFIG",))
        if settings.memory_effective_context_budget_chars != 48_000:
            raise AcceptanceGateError(("AC_PACKET_CHAR_LIMIT_CONFIG",))
    else:
        if settings.memory_max_evidence_messages != 150:
            raise AcceptanceGateError(("AC_PACKET_MESSAGE_LIMIT_CONFIG",))
        if settings.memory_history_context_budget_tokens != 24_000:
            raise AcceptanceGateError(("AC_PACKET_TOKEN_LIMIT_CONFIG",))
        if settings.context_recent_limit != 60:
            raise AcceptanceGateError(("AC_RECENT_LIMIT_CONFIG",))


def _load_prepared_report(
    path: Path | str,
    *,
    database: Path | str,
) -> dict:
    try:
        prepared = _load_strict_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",)) from exc
    if (
        not isinstance(prepared, dict)
        or prepared.get("phase") != "prepared"
        or prepared.get("database_path") != str(Path(database).resolve())
        or prepared.get("vector_status") != "ready"
        or not isinstance(prepared.get("vector_identity"), dict)
    ):
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",))
    try:
        generation = int(prepared["vector_generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",)) from exc
    if generation <= 0:
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",))
    return prepared


def _v3_acceptance_failures(
    *,
    report: dict,
    benchmark: dict,
    adaptive_enabled: bool = False,
    max_retrieval_p95_ms: float = 500.0,
) -> tuple[str, ...]:
    metrics = report["metrics"]
    failures: list[str] = []
    zero_gates = (
        ("group_leak_count", "AC_GROUP_LEAK"),
        ("subject_leak_count", "AC_SUBJECT_LEAK"),
        ("time_leak_count", "AC_TIME_LEAK"),
        ("ineligible_source_count", "AC_INELIGIBLE_SOURCE"),
        ("unresolved_source_count", "AC_SOURCE_RESOLUTION"),
        ("outside_snapshot_source_count", "AC_SOURCE_OUTSIDE_SNAPSHOT"),
        ("forbidden_source_count", "AC_FORBIDDEN_SOURCE"),
        ("plan_mismatch_count", "AC_QUERY_PLAN_CONTRACT"),
        ("derived_evidence_count", "AC_DERIVED_EVIDENCE_FORBIDDEN"),
        ("retrieval_over_150_count", "AC_RECALL_CANDIDATE_LIMIT"),
        ("packet_over_150_count", "AC_PACKET_MESSAGE_LIMIT"),
        ("packet_over_24k_count", "AC_PACKET_TOKEN_LIMIT"),
        ("recent_over_60_count", "AC_RECENT_MESSAGE_LIMIT"),
        ("citation_not_in_packet_count", "AC_CITATION_NOT_IN_PACKET"),
        ("citation_forbidden_source_count", "AC_CITATION_FORBIDDEN_SOURCE"),
        ("citation_unresolved_source_count", "AC_CITATION_SOURCE_RESOLUTION"),
        ("citation_group_leak_count", "AC_CITATION_GROUP_LEAK"),
        ("citation_subject_leak_count", "AC_CITATION_SUBJECT_LEAK"),
        ("citation_time_leak_count", "AC_CITATION_TIME_LEAK"),
        ("citation_ineligible_source_count", "AC_CITATION_INELIGIBLE_SOURCE"),
        ("answer_protocol_failure_count", "AC_ANSWER_PROTOCOL"),
    )
    if adaptive_enabled:
        legacy_limit_metrics = {
            "retrieval_over_150_count",
            "packet_over_150_count",
            "packet_over_24k_count",
            "recent_over_60_count",
        }
        zero_gates = tuple(
            item for item in zero_gates if item[0] not in legacy_limit_metrics
        ) + (
            ("retrieval_over_300_count", "AC_RECALL_CANDIDATE_LIMIT"),
            ("packet_over_300_count", "AC_PACKET_MESSAGE_LIMIT"),
            ("packet_over_32k_count", "AC_PACKET_TOKEN_LIMIT"),
            ("recent_over_120_count", "AC_RECENT_MESSAGE_LIMIT"),
        )
    for metric_name, error_code in zero_gates:
        value = _finite_number(metrics.get(metric_name))
        if value is None or value != 0.0:
            failures.append(error_code)
    threshold_gates = (
        ("recall_at_150", 0.80, "AC_RECALL_AT_150"),
        ("recall_within_24k", 0.80, "AC_RECALL_WITHIN_24K"),
        ("time_bucket_coverage_rate", 1.0, "AC_TIME_BUCKET_COVERAGE"),
        ("citation_precision", 0.95, "AC_CITATION_PRECISION"),
        ("citation_recall", 0.80, "AC_CITATION_RECALL"),
        ("grounded_answer_accuracy", 0.80, "AC_GROUNDED_ANSWER"),
        ("answer_accuracy", 0.80, "AC_ANSWER_ACCURACY"),
        ("abstention_f1", 0.90, "AC_ABSTENTION_F1"),
    )
    if adaptive_enabled:
        threshold_gates = tuple(
            item
            for item in threshold_gates
            if item[0] not in {"recall_at_150", "recall_within_24k"}
        ) + (
            ("recall_at_300", 0.80, "AC_RECALL_AT_300"),
            ("recall_within_32k", 0.80, "AC_RECALL_WITHIN_32K"),
        )
    for metric_name, threshold, error_code in threshold_gates:
        value = _finite_number(metrics.get(metric_name))
        if value is None or value < threshold:
            failures.append(error_code)
    visibility_p95 = _finite_number(metrics.get("index_visibility_p95_ms"))
    if visibility_p95 is None or visibility_p95 > 5_000.0:
        failures.append("AC_INDEX_VISIBILITY_P95")
    ttft_p95 = _finite_number(metrics.get("ttft_p95_ms"))
    if ttft_p95 is None or ttft_p95 > 15_000.0:
        failures.append("AC_TTFT_P95")
    retrieval_p95 = _finite_number(benchmark.get("p95_latency_ms"))
    if retrieval_p95 is None or retrieval_p95 >= float(max_retrieval_p95_ms):
        failures.append("AC_RETRIEVAL_P95")
    if benchmark.get("rerank_enabled") is not False:
        failures.append("AC_RERANK_FORBIDDEN")
    if benchmark.get("network_enabled") is not False:
        failures.append("AC_BENCHMARK_NETWORK_FORBIDDEN")
    if benchmark.get("vector_success_verified") is not True:
        failures.append("AC_VECTOR_NOT_EXERCISED")
    return tuple(failures)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _load_strict_json_object(path: Path | str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must be an object")
    return payload


def _acceptance_failures(*, report: dict, benchmark: dict) -> tuple[str, ...]:
    v1_categories = report["variants"]["v1"]["categories"]
    v2_categories = report["variants"]["v2"]["categories"]
    failures: list[str] = []
    if float(v2_categories["exact"]["recall_at_k"]) < 0.95:
        failures.append("AC4_KEYWORD_RECALL")
    vague_categories = ("paraphrase", "vague_reference")
    v1_count = sum(int(v1_categories[name]["case_count"]) for name in vague_categories)
    v2_count = sum(int(v2_categories[name]["case_count"]) for name in vague_categories)
    v1_recall = sum(
        float(v1_categories[name]["recall_at_k"]) * int(v1_categories[name]["case_count"])
        for name in vague_categories
    ) / v1_count
    v2_recall = sum(
        float(v2_categories[name]["recall_at_k"]) * int(v2_categories[name]["case_count"])
        for name in vague_categories
    ) / v2_count
    if v2_recall < 0.80 and v2_recall - v1_recall < 0.25:
        failures.append("AC5_VAGUE_REWRITE_RECALL")
    if float(benchmark["p95_latency_ms"]) >= 500.0:
        failures.append("AC7_LOCAL_P95")
    return tuple(failures)


def _validate_local_vector_runtime(runtime: object, *, warm: bool) -> None:
    provider = getattr(runtime, "embedding_provider", None)
    if (
        provider is None
        or not bool(getattr(provider, "available", False))
        or getattr(runtime, "embedding_generation", None) is None
    ):
        raise AcceptanceGateError(("AC_VECTOR_UNAVAILABLE",))
    identity = getattr(provider, "identity", None)
    if getattr(identity, "provider", None) != "local":
        raise AcceptanceGateError(("AC7_NETWORK_PROVIDER_FORBIDDEN",))
    if warm:
        vector = provider.embed_query("memory-v3-vector-readiness-probe")
        if vector is None or not bool(getattr(provider, "available", False)):
            raise AcceptanceGateError(("AC_VECTOR_UNAVAILABLE",))


def _validate_v3_rollout_state(
    *,
    engine: object,
    runtime: object,
    database: Path | str,
    manifest_path: Path | str,
    prepared_report: dict,
) -> dict:
    try:
        manifest = _load_strict_json_object(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise AcceptanceGateError(("AC_SNAPSHOT_MANIFEST_INVALID",)) from exc
    if (
        prepared_report.get("manifest_sha256")
        != message_ledger_manifest_sha256(manifest)
    ):
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",))
    if not verify_message_ledger_manifest(database, manifest).matches:
        raise AcceptanceGateError(("AC_FINAL_LEDGER_MISMATCH",))
    with engine.connect() as connection:
        integrity = connection.execute(text("PRAGMA integrity_check")).scalar_one()
    if str(integrity).strip().casefold() != "ok":
        raise AcceptanceGateError(("AC_DATABASE_INTEGRITY",))

    generation = int(getattr(runtime, "embedding_generation"))
    if generation != int(prepared_report["vector_generation"]):
        raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",))
    prepared_identity = prepared_report["vector_identity"]
    runtime_identity = runtime.embedding_provider.identity
    with engine.connect() as connection:
        index_state = connection.execute(
            text(
                "SELECT status, is_active, document_family, total_documents, "
                "indexed_documents, provider, model, dimensions, version "
                "FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": generation},
        ).one_or_none()
        if (
            index_state is None
            or str(index_state.status) != "ready"
            or bool(index_state.is_active)
            or str(index_state.document_family) != "raw_message_v3"
            or int(index_state.total_documents or 0) <= 0
            or int(index_state.indexed_documents or 0)
            != int(index_state.total_documents or 0)
        ):
            raise AcceptanceGateError(("AC_RAW_V3_VECTOR_NOT_READY",))
        actual_identity = {
            "provider": str(index_state.provider),
            "model": str(index_state.model),
            "dimensions": int(index_state.dimensions),
            "version": str(index_state.version),
            "document_family": str(index_state.document_family),
        }
        expected_identity = {
            "provider": str(prepared_identity.get("provider", "")),
            "model": str(prepared_identity.get("model", "")),
            "dimensions": int(prepared_identity.get("dimensions", 0)),
            "version": str(prepared_identity.get("version", "")),
            "document_family": "raw_message_v3",
        }
        provider_identity = {
            "provider": str(runtime_identity.provider),
            "model": str(runtime_identity.model),
            "dimensions": int(runtime_identity.dimensions),
            "version": str(runtime_identity.version),
            "document_family": "raw_message_v3",
        }
        if (
            actual_identity != expected_identity
            or actual_identity != provider_identity
        ):
            raise AcceptanceGateError(("AC_PREPARED_REPORT_INVALID",))

    watermarks = group_watermarks_from_manifest(manifest)
    with engine.connect() as connection:
        for group_id, watermark in watermarks.items():
            values = {
                "group_id": int(group_id),
                "watermark": int(watermark),
                "generation": generation,
            }
            eligible = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM messages m "
                        "WHERE m.group_id = :group_id AND m.id <= :watermark "
                        "AND trim(coalesce(m.plain_text, '')) <> '' "
                        "AND (CASE WHEN json_valid(m.raw_json) "
                        "THEN coalesce(json_extract(m.raw_json, '$.delivery_state'), '') "
                        "ELSE '' END) NOT IN ('reserved','blocked','uncertain','deleted')"
                    ),
                    values,
                ).scalar_one()
                or 0
            )
            projected = int(
                connection.execute(
                    text(
                        "SELECT count(DISTINCT m.id) "
                        "FROM messages m "
                        "JOIN retrieval_document_messages rdm "
                        "ON rdm.message_id = m.id AND rdm.group_id = m.group_id "
                        "JOIN retrieval_documents rd "
                        "ON rd.id = rdm.document_id AND rd.group_id = rdm.group_id "
                        "WHERE m.group_id = :group_id AND m.id <= :watermark "
                        "AND trim(coalesce(m.plain_text, '')) <> '' "
                        "AND (CASE WHEN json_valid(m.raw_json) "
                        "THEN coalesce(json_extract(m.raw_json, '$.delivery_state'), '') "
                        "ELSE '' END) NOT IN ('reserved','blocked','uncertain','deleted') "
                        "AND rd.document_kind = 'raw_message_v3' "
                        "AND rd.source_table = 'messages' AND rd.status = 'active' "
                        "AND rd.embedding_eligible = 1 "
                        "AND rd.embedding_generation = :generation "
                        "AND rd.embedding_status = 'ready'"
                    ),
                    values,
                ).scalar_one()
                or 0
            )
            if projected != eligible:
                raise AcceptanceGateError(("AC_RAW_V3_PROJECTION_COVERAGE",))
            unsafe_projected = int(
                connection.execute(
                    text(
                        "SELECT count(DISTINCT rd.id) "
                        "FROM messages m "
                        "JOIN retrieval_document_messages rdm "
                        "ON rdm.message_id = m.id AND rdm.group_id = m.group_id "
                        "JOIN retrieval_documents rd "
                        "ON rd.id = rdm.document_id AND rd.group_id = rdm.group_id "
                        "WHERE m.group_id = :group_id AND m.id <= :watermark "
                        "AND (CASE WHEN json_valid(m.raw_json) "
                        "THEN coalesce(json_extract(m.raw_json, '$.delivery_state'), '') "
                        "ELSE '' END) IN ('reserved','blocked','uncertain','deleted') "
                        "AND rd.document_kind = 'raw_message_v3' "
                        "AND rd.status = 'active'"
                    ),
                    values,
                ).scalar_one()
                or 0
            )
            if unsafe_projected:
                raise AcceptanceGateError(("AC_RAW_V3_UNSAFE_PROJECTION",))
        provenance_errors = int(
            connection.execute(
                text(
                    "SELECT count(*) FROM retrieval_documents rd "
                    "WHERE rd.document_kind = 'raw_message_v3' "
                    "AND rd.source_table = 'messages' AND rd.status = 'active' "
                    "AND rd.embedding_generation = :generation "
                    "AND ("
                    "NOT EXISTS (SELECT 1 FROM retrieval_document_messages rdm "
                    "WHERE rdm.document_id = rd.id AND rdm.group_id = rd.group_id) "
                    "OR (SELECT count(*) FROM retrieval_document_messages rdm "
                    "WHERE rdm.document_id = rd.id AND rdm.group_id = rd.group_id) <> 1"
                    ")"
                ),
                {"generation": generation},
            ).scalar_one()
            or 0
        )
        if provenance_errors:
            raise AcceptanceGateError(("AC_RAW_V3_PROVENANCE",))
    return manifest


def _validate_rollout_state(
    *,
    engine: object,
    runtime: object,
    database: Path | str,
    run_key: str,
) -> dict:
    generation = int(getattr(runtime, "embedding_generation"))
    with session_scope(engine) as session:
        index_state = session.execute(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == "vector",
                RetrievalIndexState.generation == generation,
            )
        ).scalar_one_or_none()
        if (
            index_state is None
            or str(index_state.status) != "ready"
            or not bool(index_state.is_active)
        ):
            raise AcceptanceGateError(("AC_VECTOR_NOT_ACTIVE",))
        run = session.execute(
            select(MemoryBackfillRun).where(
                MemoryBackfillRun.run_key == str(run_key),
            )
        ).scalar_one_or_none()
        if (
            run is None
            or str(run.status) != "completed"
            or str(run.index_generation) != f"vector:{generation}"
        ):
            raise AcceptanceGateError(("AC_BACKFILL_CONTRACT",))
        manifest = dict(run.manifest_json or {})
    if not verify_message_ledger_manifest(database, manifest).matches:
        raise AcceptanceGateError(("AC_FINAL_LEDGER_MISMATCH",))
    return manifest


def _print_safe_failure(codes: Sequence[str]) -> None:
    print(
        json.dumps(
            {"status": "failed", "error_codes": list(codes)},
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
