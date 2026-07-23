from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

from sqlalchemy import select

from app.config import AppSettings
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
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine, create_all, session_scope
from app.storage.models import MemoryBackfillRun, RetrievalIndexState
from app.storage.repositories import MessageRepository, UserRepository
try:
    from .evaluate_memory_recall import (
        EvaluationCase,
        EvaluationResult,
        evaluate,
        load_evaluation_cases,
        validate_real_dataset_review,
    )
except ImportError:  # Direct script execution.
    from evaluate_memory_recall import (
        EvaluationCase,
        EvaluationResult,
        evaluate,
        load_evaluation_cases,
        validate_real_dataset_review,
    )


class AcceptanceGateError(RuntimeError):
    def __init__(self, codes: Sequence[str]) -> None:
        super().__init__("production memory evaluation acceptance gate failed")
        self.codes = tuple(codes)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute real database-backed V1/V2 memory recall evaluation."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--benchmark-output", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--backfill-run-key", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--benchmark-runs", type=int, default=250)
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
    cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    measured_runs = max(250, len(cases) * 5, args.benchmark_runs)

    functional_settings = AppSettings().model_copy(
        update={
            "memory_orchestration_v2_enabled": True,
            "memory_orchestration_shadow_mode": False,
            "memory_query_rewrite_enabled": True,
            "memory_llm_rerank_enabled": False,
            "llm_builtin_web_search": False,
        }
    )
    engine = build_engine(args.database)
    try:
        create_all(engine)
        llm_client = build_llm_client(settings=functional_settings, engine=engine)
        runtime = build_memory_runtime(
            settings=functional_settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(functional_settings.bot_qq),
        )
        _validate_local_vector_runtime(runtime, warm=True)
        manifest = _validate_rollout_state(
            engine=engine,
            runtime=runtime,
            database=args.database,
            run_key=args.backfill_run_key,
        )
        validate_real_dataset_review(
            cases,
            dataset_sha256=dataset_sha256,
            review_path=args.review,
            database=args.database,
            snapshot_manifest_sha256=message_ledger_manifest_sha256(manifest),
            snapshot_watermarks=group_watermarks_from_manifest(manifest),
        )
        group_watermarks = group_watermarks_from_manifest(manifest)
        requests = tuple(
            _build_request(
                engine=engine,
                settings=functional_settings,
                case=case,
                snapshot_watermark=group_watermarks[case.group_id],
            )
            for case in cases
        )
        results: list[EvaluationResult] = []
        vector_succeeded = False
        for case_index, request in enumerate(requests):
            started = perf_counter()
            v1 = runtime.memory_orchestrator.legacy_provider(request)
            recent_source_ids = {
                message.source_msg_id for message in request.recent_messages
            }
            v1_retrieved = tuple(
                source_id
                for source_id in v1.selected_source_msg_ids
                if source_id not in recent_source_ids
            )
            results.append(
                EvaluationResult(
                    case_index=case_index,
                    variant="v1",
                    retrieved_evidence_message_ids=v1_retrieved,
                    packed_evidence_message_ids=tuple(v1.selected_source_msg_ids),
                    context_tokens=max(0, int(v1.estimated_tokens)),
                    latency_ms=(perf_counter() - started) * 1000,
                    rewrite_used=False,
                    retrieved_evidence_units=tuple((source_id,) for source_id in v1_retrieved),
                )
            )

            started = perf_counter()
            trace = runtime.v2_provider.evaluate(request)
            vector_succeeded = _require_successful_vector_trace(
                trace,
                previously_succeeded=vector_succeeded,
            )
            results.append(
                EvaluationResult(
                    case_index=case_index,
                    variant="v2",
                    retrieved_evidence_message_ids=(
                        trace.retrieved_source_msg_ids
                    ),
                    packed_evidence_message_ids=tuple(
                        trace.result.selected_source_msg_ids
                    ),
                    context_tokens=max(0, int(trace.result.estimated_tokens)),
                    latency_ms=(perf_counter() - started) * 1000,
                    rewrite_used=bool(trace.resolved_query.rewrite_used),
                    retrieved_evidence_units=trace.retrieved_source_units,
                )
            )

        _validate_local_vector_runtime(runtime, warm=False)
        if not vector_succeeded:
            raise AcceptanceGateError(("AC_VECTOR_NOT_EXERCISED",))

        report = evaluate(
            cases=cases,
            results=results,
            dataset_sha256=dataset_sha256,
            recall_k=10,
        )
        benchmark_settings = functional_settings.model_copy(
            update={"memory_query_rewrite_enabled": False}
        )
        benchmark_runtime = build_memory_runtime(
            settings=benchmark_settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(benchmark_settings.bot_qq),
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
        failures = _acceptance_failures(report=report, benchmark=benchmark)
        report["acceptance"] = {
            "status": "failed" if failures else "passed",
            "error_codes": list(failures),
        }
        _write_jsonl(
            args.results_output,
            (
                {
                    **asdict(result),
                    "retrieved_evidence_message_ids": list(result.retrieved_evidence_message_ids),
                    "packed_evidence_message_ids": list(result.packed_evidence_message_ids),
                    "retrieved_evidence_units": [
                        list(unit) for unit in result.retrieved_evidence_units
                    ],
                }
                for result in results
            ),
        )
        _write_json(args.report_output, report)
        _write_json(args.benchmark_output, benchmark)
        if failures:
            raise AcceptanceGateError(failures)
        print(
            json.dumps(
                {
                    "dataset_sha256": dataset_sha256,
                    "case_count": len(cases),
                    "result_count": len(results),
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
        target = rows[-1]
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
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "warmup_runs": int(warmup),
        "measured_runs": int(runs),
        "mean_latency_ms": sum(latencies) / len(latencies),
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
        vector = provider.embed_query("memory-v2-vector-readiness-probe")
        if vector is None or not bool(getattr(provider, "available", False)):
            raise AcceptanceGateError(("AC_VECTOR_UNAVAILABLE",))


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
        )
        + "\n",
        encoding="utf-8",
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
