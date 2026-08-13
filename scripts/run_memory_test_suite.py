"""Unified driver for the memory test platform.

Stages:
  prepare   copy snapshot to a read-only work copy + integrity/FTS check
  dataset   build the stratified case corpus (>=3000 by default)
  offline   run resolver->retriever->packer over every case (no model calls)
  fullchain run a stratified subset through real upstream model calls
  stress    (optional) run the existing 300-case memory stress eval
  report    aggregate metrics, optional baseline diff, optional gate

Example:
  python -m scripts.run_memory_test_suite --database /tmp/bot.db --all
  python -m scripts.run_memory_test_suite --database /tmp/bot.db \
      --stage fullchain --fullchain-limit 300 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.config import AppSettings
from app.core.legacy_memory_context import GroupMemoryContextRequest
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine as _build_engine
from scripts import build_memory_test_dataset as dataset_builder
from scripts import memory_test_fullchain as fullchain
from scripts.memory_test_metrics import (
    category_counts,
    classification_metrics,
    diff_metrics,
    fullchain_metrics,
)


DEFAULT_WORKDIR = Path("data/test-platform")


def _engine(database: Path):
    return _build_engine(database)


def _iter_rows(engine, statement: str, parameters: dict[str, Any] | None = None):
    with engine.connect() as connection:
        yield from connection.execute(text(statement), parameters or {})


def stage_prepare(database: Path, workdir: Path) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    snapshot = workdir / "snapshot.db"
    source = sqlite3.connect(str(database))
    target = sqlite3.connect(str(snapshot))
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    migration_error = ""
    try:
        from sqlalchemy import create_engine as _migration_engine_factory
        from app.storage.db import _apply_schema_migrations

        migration_engine = _migration_engine_factory(f"sqlite:///{snapshot}")
        try:
            with migration_engine.begin() as connection:
                _apply_schema_migrations(connection)
        finally:
            migration_engine.dispose()
    except Exception as exc:  # pragma: no cover - defensive for odd snapshots
        migration_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    check = sqlite3.connect(str(snapshot))
    try:
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        fts_tables = [
            str(row[0])
            for row in check.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE '%fts%'"
            )
        ]
        message_count = check.execute("SELECT count(*) FROM messages").fetchone()[0]
        message_columns = {
            str(row[1]) for row in check.execute("PRAGMA table_info(messages)")
        }
        document_columns = {
            str(row[1])
            for row in check.execute("PRAGMA table_info(retrieval_documents)")
        }
    finally:
        check.close()
    schema_issues: list[str] = []
    if "raw_json" not in message_columns:
        schema_issues.append("messages.raw_json missing")
    for column in ("embedding_eligible", "embedding_status", "embedding_generation"):
        if column not in document_columns:
            schema_issues.append(f"retrieval_documents.{column} missing")
    meta = {
        "source": str(database),
        "snapshot": str(snapshot),
        "integrity_check": integrity,
        "message_count": int(message_count),
        "fts_tables": fts_tables,
        "schema_ready": not schema_issues,
        "schema_issues": schema_issues,
        "migration_error": migration_error or None,
    }
    (workdir / "snapshot-meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def stage_dataset(
    database: Path,
    workdir: Path,
    *,
    count: int,
    seed: int,
    group_ids: Sequence[int],
) -> dict[str, Any]:
    engine = _engine(database)
    cases = dataset_builder.build_cases(
        engine,
        count=count,
        seed=seed,
        group_ids=group_ids or None,
    )
    output = workdir / "cases.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "cases": len(cases),
        "output": str(output),
        "by_category": category_counts(cases),
    }


def stage_offline(
    database: Path,
    workdir: Path,
    *,
    rewrite_enabled: bool,
    channel_timeout: float,
) -> dict[str, Any]:
    cases = fullchain._load_cases(workdir / "cases.jsonl")
    settings = AppSettings().model_copy(
        update={
            "memory_query_rewrite_enabled": bool(rewrite_enabled),
            "memory_retrieval_channel_timeout_seconds": float(channel_timeout),
        }
    )
    engine = _engine(database)
    llm_client = build_llm_client(settings=settings, engine=engine)
    runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name="小町",
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            _offline_case(
                engine=engine,
                runtime=runtime,
                case=case,
                settings=settings,
            )
        )
    output = workdir / "offline-results.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = classification_metrics(rows)
    return {"cases": len(rows), "metrics": metrics, "output": str(output)}


def _offline_case(*, engine, runtime, case: Mapping[str, Any], settings: AppSettings) -> dict[str, Any]:
    group_id = int(case["group_id"])
    recent_ids = tuple(
        int(value)
        for value in (case.get("recent_context_message_ids") or ())
        if str(value).strip()
    )
    recent = []
    for message_id in recent_ids:
        row = fullchain._load_message(engine, message_id)
        if row is not None:
            recent.append(fullchain._evidence(row, int(settings.bot_qq)))
    quoted = None
    quoted_id = case.get("quoted_context_message_id")
    if quoted_id:
        quoted_row = fullchain._load_message_by_platform(engine, str(quoted_id))
        if quoted_row is not None:
            quoted = fullchain._evidence(quoted_row, int(settings.bot_qq))
    request = GroupMemoryContextRequest(
        group_id=group_id,
        query=str(case["query"]),
        recent_messages=tuple(recent),
        quoted_message=quoted,
        target_message_id=str(case.get("target_message_id") or ""),
        available_input=34000,
        now=fullchain._parse_dt(case.get("now_iso")) or None,
        current_user_id=int(case.get("requester_uin") or 0),
        use_full_history=True,
    )
    started = perf_counter()
    trace = runtime.v2_provider.evaluate(request)
    latency_ms = (perf_counter() - started) * 1000
    packed = trace.result.packed_context
    expected = tuple(
        str(value) for value in (case.get("expected_evidence_message_ids") or ())
    )
    packed_ids = tuple(str(value) for value in (packed.source_msg_ids or ()))
    expected_layer = str(case.get("expected_layer") or "raw")
    tags = " ".join(str(value) for value in (case.get("tags") or ()))
    expected_subject = (
        tuple(str(value) for value in (case.get("allowed_subject_user_ids") or ()))
        if case.get("allowed_subject_user_ids") is not None
        else None
    )
    actual_subject = getattr(trace.resolved_query, "subject_ids", None)
    actual_subject_tuple = (
        tuple(str(value) for value in actual_subject)
        if actual_subject is not None
        else None
    )
    cross_group_violation = False
    if "cross_group" in tags and packed_ids:
        placeholders = ",".join(f":pid_{index}" for index in range(len(packed_ids)))
        parameters = {f"pid_{index}": str(value) for index, value in enumerate(packed_ids)}
        if placeholders:
            rows = list(
                _iter_rows(
                    engine,
                    "SELECT platform_msg_id, group_id FROM messages "
                    f"WHERE platform_msg_id IN ({placeholders})",
                    parameters,
                )
            )
            foreign_sources = [
                str(row[0])
                for row in rows
                if row[1] is not None and int(row[1]) != group_id
            ]
            cross_group_violation = bool(foreign_sources)
    return {
        "case_id": str(case.get("case_id") or ""),
        "category": str(case.get("category") or "unknown"),
        "kind": str(case.get("kind") or ""),
        "expected_layer": expected_layer,
        "expected_evidence_message_ids": expected,
        "packed_source_ids": packed_ids,
        "subject_expected": expected_subject,
        "subject_actual": actual_subject_tuple,
        "subject_match": actual_subject_tuple == expected_subject,
        "raw_hit": bool(packed.evidence_segments),
        "fact_hit": bool(packed.facts),
        "summary_hit": bool(packed.summaries),
        "latency_ms": latency_ms,
        "cross_group_violation": cross_group_violation,
        "attempted_channels": list(
            str(value) for value in getattr(trace, "attempted_channels", ())
        ),
        "failed_channels": list(
            str(value) for value in getattr(trace, "failed_channels", ())
        ),
        "all_channels_failed": bool(
            getattr(trace, "failed_channels", ())
            and set(getattr(trace, "failed_channels", ()))
            == set(getattr(trace, "attempted_channels", ()))
        ),
        "query": str(case.get("query", ""))[:120],
    }


def stage_fullchain(
    database: Path,
    workdir: Path,
    *,
    limit: int,
    seed: int,
    model: str,
    judge_model: str,
    dry_run: bool,
    resume: bool,
    rewrite_enabled: bool,
    channel_timeout: float,
    input_price_mtok: float,
    output_price_mtok: float,
    provider_attempts: int,
    provider_backoff: float,
    answer_model: str,
    answer_effort: str,
    aux_model: str,
    aux_effort: str,
) -> dict[str, Any]:
    cases = fullchain._load_cases(workdir / "cases.jsonl")
    engine = _engine(database)
    output = workdir / "fullchain-results.jsonl"
    detail_path = workdir / "fullchain-results.detail.jsonl"
    # Recover rows from an interrupted run before resuming so checkpointed
    # rows are never lost when --resume skips already-completed cases.
    if detail_path.exists():
        recovered: list[dict[str, Any]] = []
        for line in detail_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("case_id"):
                recovered.append(value)
        if recovered:
            fullchain._merge_results(output, recovered)
    rows, summary = fullchain.run_cases(
        engine,
        cases,
        limit=limit,
        seed=seed,
        cache_dir=workdir / "cache",
        model=model,
        judge_model=judge_model,
        dry_run=dry_run,
        resume=resume,
        rewrite_enabled=rewrite_enabled,
        channel_timeout=channel_timeout,
        input_price_mtok=input_price_mtok,
        output_price_mtok=output_price_mtok,
        provider_attempts=provider_attempts,
        provider_backoff=provider_backoff,
        answer_model=answer_model,
        answer_effort=answer_effort,
        aux_model=aux_model,
        aux_effort=aux_effort,
        progress_path=workdir / "progress-fullchain.jsonl",
        detail_path=detail_path,
    )
    if rows:
        fullchain._merge_results(output, rows)
    metrics = fullchain_metrics(rows) if rows else {}
    return {"summary": summary, "metrics": metrics, "output": str(output)}


def stage_stress(
    database: Path,
    workdir: Path,
    *,
    groups: str,
    limit_cases: int,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "stress.json"
    command = [
        sys.executable,
        "-m",
        "scripts.memory_stress_eval",
        "run",
        "--database",
        str(database),
        "--groups",
        groups,
        "--limit-cases",
        str(limit_cases),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return {"status": "failed", "returncode": result.returncode}
    return {"status": "ok", "output": str(output)}


def stage_report(
    workdir: Path,
    *,
    baseline_dir: Path | None,
    gate_grounded: float | None,
    gate_recall: float | None,
    gate_protocol_failures: int | None,
    gate_p95_ms: float | None,
) -> dict[str, Any]:
    offline_metrics: dict[str, Any] = {}
    fullchain_rows: list[dict[str, Any]] = []
    offline_path = workdir / "offline-results.jsonl"
    if offline_path.exists():
        rows = [
            json.loads(line)
            for line in offline_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        offline_metrics = classification_metrics(rows)
    fullchain_path = workdir / "fullchain-results.jsonl"
    if fullchain_path.exists():
        fullchain_rows = [
            json.loads(line)
            for line in fullchain_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    fc_metrics = fullchain_metrics(fullchain_rows)
    stress: dict[str, Any] = {}
    stress_path = workdir / "stress.json"
    if stress_path.exists():
        try:
            stress = json.loads(stress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stress = {}
    report = {
        "offline": offline_metrics,
        "fullchain": fc_metrics,
        "stress": stress,
        "baseline": None,
    }
    if baseline_dir is not None:
        baseline_report_path = Path(baseline_dir) / "report.json"
        if baseline_report_path.exists():
            baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
            report["baseline"] = {
                "path": str(baseline_report_path),
                "offline_diff": diff_metrics(
                    baseline.get("offline") or {}, offline_metrics
                ),
                "fullchain_diff": diff_metrics(
                    baseline.get("fullchain") or {}, fc_metrics
                ),
            }
    gate_results: dict[str, Any] = {}
    if gate_grounded is not None:
        gate_results["grounded_answer_accuracy"] = {
            "threshold": gate_grounded,
            "actual": fc_metrics.get("grounded_answer_accuracy"),
            "passed": bool(
                fc_metrics.get("grounded_answer_accuracy", 0.0) >= gate_grounded
            ),
        }
    if gate_recall is not None:
        kind_recall = offline_metrics.get("kind_recall") or {}
        values = [value for value in kind_recall.values() if isinstance(value, (int, float))]
        average_recall = sum(values) / len(values) if values else 0.0
        gate_results["average_kind_recall"] = {
            "threshold": gate_recall,
            "actual": average_recall,
            "passed": bool(average_recall >= gate_recall),
        }
    if gate_protocol_failures is not None:
        actual = int(fc_metrics.get("protocol_failures", 0))
        gate_results["protocol_failures"] = {
            "threshold": gate_protocol_failures,
            "actual": actual,
            "passed": bool(actual <= gate_protocol_failures),
        }
    if gate_p95_ms is not None:
        actual = fc_metrics.get("ttft_ms", {}).get("p95", 0.0)
        gate_results["ttft_p95_ms"] = {
            "threshold": gate_p95_ms,
            "actual": actual,
            "passed": bool(actual <= gate_p95_ms),
        }
    report["gate"] = gate_results
    (workdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (workdir / "report.md").write_text(
        _render_report_markdown(report),
        encoding="utf-8",
    )
    return report


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Memory Test Platform Report", ""]
    offline = report.get("offline") or {}
    if offline:
        lines.append("## Offline retrieval")
        lines.append(f"- cases: {offline.get('cases')}")
        lines.append(f"- kind recall: {json.dumps(offline.get('kind_recall'), ensure_ascii=False)}")
        lines.append(f"- layer hit rate: {json.dumps(offline.get('layer_hit_rate'), ensure_ascii=False)}")
        lines.append(f"- subject binding accuracy: {offline.get('subject_binding_accuracy')}")
        lines.append(f"- cross-group violations: {offline.get('cross_group_violations')}")
        lines.append(f"- retrieval latency p50/p95: {offline.get('retrieval_latency_ms')}")
        lines.append("")
    fullchain_metrics_value = report.get("fullchain") or {}
    if fullchain_metrics_value:
        lines.append("## Full-chain (real model)")
        for key, value in fullchain_metrics_value.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    gate = report.get("gate") or {}
    if gate:
        lines.append("## Gate")
        for key, value in gate.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified memory test platform driver.")
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--stage",
        choices=("prepare", "dataset", "offline", "fullchain", "stress", "report", "all"),
        default="all",
    )
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--group-ids", type=str, default="")
    parser.add_argument("--fullchain-limit", type=int, default=300)
    parser.add_argument("--model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rewrite", dest="rewrite_enabled", action="store_false")
    parser.add_argument("--rewrite-enabled", dest="rewrite_enabled", action="store_true")
    parser.set_defaults(rewrite_enabled=False)
    parser.add_argument("--channel-timeout", type=float, default=0.5)
    parser.add_argument("--input-price-mtok", type=float, default=fullchain.DEFAULT_INPUT_PRICE_MT)
    parser.add_argument("--output-price-mtok", type=float, default=fullchain.DEFAULT_OUTPUT_PRICE_MT)
    parser.add_argument("--provider-attempts", type=int, default=fullchain.PROVIDER_ATTEMPTS)
    parser.add_argument("--provider-backoff", type=float, default=fullchain.PROVIDER_BACKOFF_SECONDS)
    parser.add_argument("--answer-model", default=fullchain.DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-effort", default=fullchain.DEFAULT_ANSWER_EFFORT)
    parser.add_argument("--aux-model", default=fullchain.DEFAULT_AUX_MODEL)
    parser.add_argument("--aux-effort", default=fullchain.DEFAULT_AUX_EFFORT)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-groups", type=str, default="")
    parser.add_argument("--stress-limit", type=int, default=300)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--gate-grounded-accuracy", type=float, default=None)
    parser.add_argument("--gate-recall", type=float, default=None)
    parser.add_argument("--gate-protocol-failures", type=int, default=None)
    parser.add_argument("--gate-ttft-p95-ms", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.database is None:
        raise SystemExit("--database is required")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    database = args.database
    prepared_snapshot = workdir / "snapshot.db"
    if prepared_snapshot.exists() and args.stage != "prepare":
        # Every stage after prepare must run against the read-only snapshot copy.
        database = prepared_snapshot
    group_ids = (
        [int(value) for value in args.group_ids.split(",") if value.strip()]
        if args.group_ids
        else []
    )
    stages = ["prepare", "dataset", "offline", "fullchain", "report"]
    if args.stage != "all":
        stages = [args.stage]
    result: dict[str, Any] = {}
    for stage in stages:
        if stage == "prepare":
            result["prepare"] = stage_prepare(args.database, workdir)
        elif stage == "dataset":
            result["dataset"] = stage_dataset(
                database,
                workdir,
                count=args.count,
                seed=args.seed,
                group_ids=group_ids,
            )
        elif stage == "offline":
            result["offline"] = stage_offline(
                database,
                workdir,
                rewrite_enabled=args.rewrite_enabled,
                channel_timeout=args.channel_timeout,
            )
        elif stage == "fullchain":
            result["fullchain"] = stage_fullchain(
                database,
                workdir,
                limit=args.fullchain_limit,
                seed=args.seed,
                model=args.model,
                judge_model=args.judge_model,
                dry_run=args.dry_run,
                resume=args.resume,
                rewrite_enabled=args.rewrite_enabled,
                channel_timeout=args.channel_timeout,
                input_price_mtok=args.input_price_mtok,
                output_price_mtok=args.output_price_mtok,
                provider_attempts=args.provider_attempts,
                provider_backoff=args.provider_backoff,
                answer_model=args.answer_model,
                answer_effort=args.answer_effort,
                aux_model=args.aux_model,
                aux_effort=args.aux_effort,
            )
        elif stage == "stress":
            if not args.stress_groups:
                raise SystemExit("--stress-groups is required for the stress stage")
            result["stress"] = stage_stress(
                database,
                workdir,
                groups=args.stress_groups,
                limit_cases=args.stress_limit,
            )
        elif stage == "report":
            result["report"] = stage_report(
                workdir,
                baseline_dir=args.baseline_dir,
                gate_grounded=args.gate_grounded_accuracy,
                gate_recall=args.gate_recall,
                gate_protocol_failures=args.gate_protocol_failures,
                gate_p95_ms=args.gate_ttft_p95_ms,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    gate = (result.get("report") or {}).get("gate") or {}
    failed = [key for key, value in gate.items() if not bool(value.get("passed"))]
    if failed:
        raise SystemExit(f"gate failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
