from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select, text

from app.config import AppSettings
from app.core.memory_backfill import (
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)
from app.core.memory_backfill_runner import group_watermarks_from_manifest
from app.providers.semantic_embeddings import build_embedding_provider
from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    create_retrieval_vector_generation,
    create_all,
    refresh_retrieval_vector_generation,
    rollback_retrieval_vector_generation,
    resume_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import Message, RetrievalDocument, RetrievalDocumentMessage
from app.storage.repositories import RetrievalDocumentRepository
try:
    from .evaluate_memory_v3 import (
        V3Observation,
        audit_v3_quality_sources,
        evaluate_v3,
        load_message_metadata,
        load_v3_quality_sidecar,
        retrieval_fingerprint_sha256,
        validate_v3_dataset_contract,
    )
    from .evaluate_memory_recall import load_evaluation_cases
    from .run_memory_recall_eval import _v3_acceptance_failures
except ImportError:  # Direct script execution.
    from evaluate_memory_v3 import (
        V3Observation,
        audit_v3_quality_sources,
        evaluate_v3,
        load_message_metadata,
        load_v3_quality_sidecar,
        retrieval_fingerprint_sha256,
        validate_v3_dataset_contract,
    )
    from evaluate_memory_recall import load_evaluation_cases
    from run_memory_recall_eval import _v3_acceptance_failures


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Idempotently backfill raw_message_v3 FTS and local vectors."
    )
    parser.add_argument(
        "--phase",
        choices=("prepare", "activate", "rollback"),
        required=True,
        help="Prepare, explicitly activate, or roll back a reviewed generation.",
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prepared-report", type=Path)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--quality-sidecar", type=Path)
    parser.add_argument("--quality-private-replay", type=Path)
    parser.add_argument("--quality-visibility-artifact", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fts-only", action="store_true")
    parser.add_argument("--resume-generation", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    _validate_arguments(args)
    manifest = _load_strict_json(args.manifest)
    if args.phase == "activate":
        return _activate_prepared_generation(args, manifest)
    if args.phase == "rollback":
        return _rollback_prepared_generation(args, manifest)
    return _prepare_generation(args, manifest)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.resume_generation is not None and args.resume_generation < 1:
        raise ValueError("resume-generation must be positive")
    if args.fts_only and args.resume_generation is not None:
        raise ValueError("resume-generation cannot be used with fts-only")
    if args.phase == "prepare" and args.prepared_report is not None:
        raise ValueError("prepared-report is only valid for activate/rollback")
    if args.phase != "activate" and args.gate_report is not None:
        raise ValueError("gate-report is only valid for activate")
    if args.phase != "activate" and (
        args.dataset is not None
        or args.quality_sidecar is not None
        or args.quality_private_replay is not None
        or args.quality_visibility_artifact is not None
        or args.results is not None
        or args.benchmark_report is not None
    ):
        raise ValueError(
            "quality artifacts are only valid for activate"
        )
    if args.phase == "activate":
        if args.prepared_report is None:
            raise ValueError("activate requires prepared-report")
        if args.gate_report is None:
            raise ValueError("activate requires gate-report")
        if args.dataset is None:
            raise ValueError("activate requires dataset")
        if args.quality_sidecar is None:
            raise ValueError("activate requires quality-sidecar")
        if args.quality_private_replay is None:
            raise ValueError("activate requires quality-private-replay")
        if args.quality_visibility_artifact is None:
            raise ValueError("activate requires quality-visibility-artifact")
        if args.results is None:
            raise ValueError("activate requires results")
        if args.benchmark_report is None:
            raise ValueError("activate requires benchmark-report")
        if args.fts_only:
            raise ValueError("fts-only generations cannot be activated")
        if args.resume_generation is not None:
            raise ValueError("resume-generation is only valid for prepare")
    if args.phase == "rollback":
        if args.prepared_report is None:
            raise ValueError("rollback requires prepared-report")
        if args.fts_only:
            raise ValueError("fts-only generations cannot be rolled back")
        if args.resume_generation is not None:
            raise ValueError("resume-generation is only valid for prepare")


def _prepare_generation(
    args: argparse.Namespace,
    manifest: dict,
) -> int:
    ledger_before = verify_message_ledger_manifest(args.database, manifest)
    if not ledger_before.matches:
        raise RuntimeError("live message ledger differs from the verified backup")
    watermarks = group_watermarks_from_manifest(manifest)
    engine = build_engine(args.database)
    try:
        create_all(engine)
        provider = None
        vector_generation = None
        vector_identity = None
        resumed_generation = args.resume_generation is not None
        active_before = _active_vector_generation(engine)
        if not args.fts_only:
            settings = AppSettings()
            provider = build_embedding_provider(
                provider=settings.memory_embedding_provider,
                device=settings.memory_embedding_device,
                model=settings.memory_embedding_model,
                dimensions=settings.memory_embedding_dimensions,
                cache_dir=settings.memory_embedding_cache_dir,
                local_files_only=settings.memory_embedding_local_files_only,
                version=settings.memory_embedding_version,
                base_url=settings.memory_embedding_base_url,
                api_key=settings.memory_embedding_api_key,
                timeout_seconds=settings.memory_embedding_timeout_seconds,
            )
            if not provider.available:
                raise RuntimeError("embedding provider is unavailable")
        if provider is not None and provider.available:
            identity = provider.identity
            vector_identity = {
                "provider": str(identity.provider),
                "model": str(identity.model),
                "dimensions": int(identity.dimensions),
                "version": str(identity.version),
                "document_family": "raw_message_v3",
            }
            if args.resume_generation is None:
                vector_generation = create_retrieval_vector_generation(
                    engine,
                    provider=identity.provider,
                    model=identity.model,
                    dimensions=identity.dimensions,
                    version=identity.version,
                    document_family="raw_message_v3",
                )
            else:
                vector_generation = resume_retrieval_vector_generation(
                    engine,
                    generation=args.resume_generation,
                    provider=identity.provider,
                    model=identity.model,
                    dimensions=identity.dimensions,
                    version=identity.version,
                    document_family="raw_message_v3",
                )
            if vector_generation is None:
                raise RuntimeError("vector generation is unavailable or not resumable")

        manifest_bounds = dict(watermarks)
        projected = _project_raw_ranges(
            engine,
            generation=vector_generation,
            lower_bounds={group_id: 0 for group_id in manifest_bounds},
            upper_bounds=manifest_bounds,
            batch_size=args.batch_size,
        )
        pre_activation_live_bounds = _snapshot_live_high_watermarks(
            engine,
            manifest_watermarks=watermarks,
        )
        projected += _project_raw_ranges(
            engine,
            generation=vector_generation,
            lower_bounds=watermarks,
            upper_bounds=pre_activation_live_bounds,
            batch_size=args.batch_size,
        )
        pre_activation_target_bounds = _merge_high_watermarks(
            manifest_bounds,
            pre_activation_live_bounds,
        )
        embedded = 0
        if vector_generation is not None and provider is not None:
            embedded += _embed_raw_generation(
                engine,
                provider=provider,
                generation=vector_generation,
                upper_bounds=pre_activation_target_bounds,
                batch_size=args.batch_size,
            )
            coverage = refresh_retrieval_vector_generation(
                engine,
                generation=vector_generation,
                mark_ready=True,
            )
        else:
            coverage = None

        ledger_after = verify_message_ledger_manifest(args.database, manifest)
        if not ledger_after.matches:
            raise RuntimeError("raw message ledger changed inside the backup watermark")
        if vector_generation is not None:
            if coverage is None or coverage.status != "ready":
                raise RuntimeError(
                    "raw_message_v3 vector generation coverage is incomplete"
                )
        live_above_watermark = _count_live_messages(
            engine,
            manifest_watermarks=watermarks,
            live_high_watermarks=pre_activation_live_bounds,
        )
        report = {
            "phase": "prepared",
            "completed_at": datetime.now(UTC).isoformat(),
            "database_path": str(args.database.resolve()),
            "manifest_sha256": message_ledger_manifest_sha256(manifest),
            "groups": len(watermarks),
            "projected_eligible_messages": projected,
            "embedded_documents": embedded,
            "resumed_generation": resumed_generation,
            "expected_active_generation": active_before,
            "live_above_watermark": live_above_watermark,
            "live_catchup_high_watermarks": {
                str(group_id): watermark
                for group_id, watermark in sorted(pre_activation_live_bounds.items())
            },
            "vector_generation": vector_generation,
            "vector_identity": vector_identity,
            "vector_status": getattr(coverage, "status", "fts-only"),
            "vector_total_documents": getattr(coverage, "total_documents", 0),
            "vector_indexed_documents": getattr(coverage, "indexed_documents", 0),
            "ledger_matches": True,
        }
        _emit_report(report, output=args.output)
        return 0
    finally:
        engine.dispose()


def _activate_prepared_generation(
    args: argparse.Namespace,
    manifest: dict,
) -> int:
    prepared = _load_strict_json(args.prepared_report)
    if prepared.get("phase") != "prepared":
        raise ValueError("prepared-report does not describe a prepared generation")
    if prepared.get("manifest_sha256") != message_ledger_manifest_sha256(manifest):
        raise ValueError("prepared-report does not match the manifest")
    if prepared.get("database_path") != str(args.database.resolve()):
        raise ValueError("prepared-report does not match the database")
    generation_value = prepared.get("vector_generation")
    if generation_value is None:
        raise ValueError("prepared-report contains no vector generation")
    generation = int(generation_value)
    expected_value = prepared.get("expected_active_generation")
    expected_active_generation = (
        int(expected_value) if expected_value is not None else None
    )
    prepared_identity = prepared.get("vector_identity")
    if not isinstance(prepared_identity, dict):
        raise ValueError("prepared-report contains no vector identity")
    gate = _load_strict_json(args.gate_report)
    quality_sidecar = _load_strict_json(args.quality_sidecar)
    _validate_activation_gate(
        gate,
        manifest_sha256=message_ledger_manifest_sha256(manifest),
        generation=generation,
        dataset_sha256=_file_sha256(args.dataset),
        quality_sidecar_sha256=_file_sha256(args.quality_sidecar),
        quality_private_replay_sha256=_file_sha256(args.quality_private_replay),
        quality_visibility_artifact_sha256=_file_sha256(
            args.quality_visibility_artifact
        ),
        results_sha256=_file_sha256(args.results),
        benchmark_sha256=_file_sha256(args.benchmark_report),
        quality_sidecar=quality_sidecar,
    )
    _validate_activation_quality_artifacts(
        args,
        gate=gate,
        manifest_sha256=message_ledger_manifest_sha256(manifest),
        generation=generation,
    )

    watermarks = group_watermarks_from_manifest(manifest)
    engine = build_engine(args.database)
    try:
        create_all(engine)
        state = _vector_generation_state(engine, generation=generation)
        expected_identity = {
            "provider": str(prepared_identity.get("provider", "")),
            "model": str(prepared_identity.get("model", "")),
            "dimensions": int(prepared_identity.get("dimensions", 0)),
            "version": str(prepared_identity.get("version", "")),
            "document_family": "raw_message_v3",
        }
        if (
            state is None
            or state["identity"] != expected_identity
            or state["status"] != "ready"
            or state["is_active"]
        ):
            raise RuntimeError("prepared vector generation state changed")

        settings = AppSettings()
        provider = build_embedding_provider(
            provider=settings.memory_embedding_provider,
            device=settings.memory_embedding_device,
            model=settings.memory_embedding_model,
            dimensions=settings.memory_embedding_dimensions,
            cache_dir=settings.memory_embedding_cache_dir,
            local_files_only=settings.memory_embedding_local_files_only,
            version=settings.memory_embedding_version,
            base_url=settings.memory_embedding_base_url,
            api_key=settings.memory_embedding_api_key,
            timeout_seconds=settings.memory_embedding_timeout_seconds,
        )
        if not provider.available:
            raise RuntimeError("embedding provider is unavailable")
        provider_identity = {
            "provider": str(provider.identity.provider),
            "model": str(provider.identity.model),
            "dimensions": int(provider.identity.dimensions),
            "version": str(provider.identity.version),
            "document_family": "raw_message_v3",
        }
        if provider_identity != expected_identity:
            raise RuntimeError("embedding identity differs from prepared generation")

        pre_activation_live_bounds = _snapshot_live_high_watermarks(
            engine,
            manifest_watermarks=watermarks,
        )
        projected = _project_raw_ranges(
            engine,
            generation=generation,
            lower_bounds=watermarks,
            upper_bounds=pre_activation_live_bounds,
            batch_size=args.batch_size,
        )
        pre_activation_target_bounds = _merge_high_watermarks(
            watermarks,
            pre_activation_live_bounds,
        )
        embedded = _embed_raw_generation(
            engine,
            provider=provider,
            generation=generation,
            upper_bounds=pre_activation_target_bounds,
            batch_size=args.batch_size,
        )
        coverage = refresh_retrieval_vector_generation(
            engine,
            generation=generation,
            mark_ready=True,
        )
        if coverage.status != "ready":
            raise RuntimeError("pre-activation raw_message_v3 catch-up is incomplete")

        def ledger_guard() -> bool:
            return verify_message_ledger_manifest(
                args.database,
                manifest,
            ).matches

        if not activate_retrieval_vector_generation(
            engine,
            generation=generation,
            expected_active_generation=expected_active_generation,
            pre_activation_check=ledger_guard,
        ):
            raise RuntimeError(
                "vector generation activation or locked ledger check was rejected"
            )

        try:
            post_activation_live_bounds = _snapshot_live_high_watermarks(
                engine,
                manifest_watermarks=watermarks,
            )
            projected += _project_raw_ranges(
                engine,
                generation=generation,
                lower_bounds=pre_activation_live_bounds,
                upper_bounds=post_activation_live_bounds,
                batch_size=args.batch_size,
            )
            final_target_bounds = _merge_high_watermarks(
                watermarks,
                post_activation_live_bounds,
            )
            embedded += _embed_raw_generation(
                engine,
                provider=provider,
                generation=generation,
                upper_bounds=final_target_bounds,
                batch_size=args.batch_size,
            )
            coverage = refresh_retrieval_vector_generation(
                engine,
                generation=generation,
                mark_ready=True,
            )
            if coverage.status != "ready":
                raise RuntimeError(
                    "post-activation raw_message_v3 catch-up is incomplete"
                )
            report = {
                **prepared,
                "phase": "activated",
                "completed_at": datetime.now(UTC).isoformat(),
                "projected_eligible_messages": projected,
                "embedded_documents": embedded,
                "live_above_watermark": _count_live_messages(
                    engine,
                    manifest_watermarks=watermarks,
                    live_high_watermarks=post_activation_live_bounds,
                ),
                "live_catchup_high_watermarks": {
                    str(group_id): watermark
                    for group_id, watermark in sorted(
                        post_activation_live_bounds.items()
                    )
                },
                "vector_status": coverage.status,
                "vector_total_documents": coverage.total_documents,
                "vector_indexed_documents": coverage.indexed_documents,
                "ledger_matches": True,
                "gate_report_sha256": _file_sha256(args.gate_report),
            }
            _emit_report(report, output=args.output)
            return 0
        except Exception as activation_error:
            if expected_active_generation is None:
                raise RuntimeError(
                    "activation failed after CAS and no rollback generation exists"
                ) from activation_error
            try:
                rolled_back = rollback_retrieval_vector_generation(
                    engine,
                    generation=expected_active_generation,
                    expected_active_generation=generation,
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "activation and automatic rollback both failed"
                ) from rollback_error
            if not rolled_back:
                raise RuntimeError(
                    "activation failed and automatic rollback was rejected"
                ) from activation_error
            raise RuntimeError(
                "activation failed after CAS; legacy generation was restored"
            ) from activation_error
    finally:
        engine.dispose()


def _rollback_prepared_generation(
    args: argparse.Namespace,
    manifest: dict,
) -> int:
    prepared = _load_strict_json(args.prepared_report)
    if prepared.get("phase") != "prepared":
        raise ValueError("prepared-report does not describe a prepared generation")
    if prepared.get("manifest_sha256") != message_ledger_manifest_sha256(manifest):
        raise ValueError("prepared-report does not match the manifest")
    if prepared.get("database_path") != str(args.database.resolve()):
        raise ValueError("prepared-report does not match the database")
    try:
        raw_generation = int(prepared["vector_generation"])
        legacy_generation = int(prepared["expected_active_generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "prepared-report does not contain a rollback generation pair"
        ) from exc
    engine = build_engine(args.database)
    try:
        create_all(engine)
        with engine.connect() as connection:
            integrity = connection.execute(
                text("PRAGMA integrity_check")
            ).scalar_one()
        if str(integrity).strip().casefold() != "ok":
            raise RuntimeError("database integrity check failed")
        if not rollback_retrieval_vector_generation(
            engine,
            generation=legacy_generation,
            expected_active_generation=raw_generation,
        ):
            raise RuntimeError("vector generation rollback CAS was rejected")
        report = {
            "phase": "rolled_back",
            "completed_at": datetime.now(UTC).isoformat(),
            "database_path": str(args.database.resolve()),
            "manifest_sha256": message_ledger_manifest_sha256(manifest),
            "deactivated_generation": raw_generation,
            "active_generation": legacy_generation,
            "database_integrity": "ok",
            "ledger_check": "not_required_for_rollback",
        }
        _emit_report(report, output=args.output)
        return 0
    finally:
        engine.dispose()


def _vector_generation_state(engine, *, generation: int) -> dict | None:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT provider, model, dimensions, version, document_family, "
                "status, is_active FROM retrieval_index_state "
                "WHERE channel = 'vector' AND generation = :generation"
            ),
            {"generation": int(generation)},
        ).one_or_none()
    if row is None:
        return None
    return {
        "identity": {
            "provider": str(row.provider),
            "model": str(row.model),
            "dimensions": int(row.dimensions),
            "version": str(row.version),
            "document_family": str(row.document_family or ""),
        },
        "status": str(row.status),
        "is_active": bool(row.is_active),
    }


def _validate_activation_gate(
    gate: dict,
    *,
    manifest_sha256: str,
    generation: int,
    dataset_sha256: str,
    quality_sidecar_sha256: str,
    quality_private_replay_sha256: str,
    quality_visibility_artifact_sha256: str,
    results_sha256: str,
    benchmark_sha256: str,
    quality_sidecar: dict,
) -> None:
    required_fields = {
        "evaluation_schema_version",
        "memory_path",
        "dataset_sha256",
        "snapshot_manifest_sha256",
        "retrieval_fingerprint_sha256",
        "case_count",
        "quality_sidecar_present",
        "quality_sidecar_sha256",
        "results_sha256",
        "benchmark_sha256",
        "metrics",
        "vector_generation",
        "acceptance",
    }
    if not isinstance(gate, dict) or not required_fields <= set(gate):
        raise ValueError("activation gate fields do not match the contract")
    acceptance = gate.get("acceptance")
    if (
        not isinstance(acceptance, dict)
        or set(acceptance) != {"status", "error_codes"}
        or acceptance.get("status") != "passed"
        or acceptance.get("error_codes") != []
    ):
        raise RuntimeError("activation gate did not pass")
    if gate.get("evaluation_schema_version") != 3:
        raise ValueError("activation gate schema is unsupported")
    if gate.get("memory_path") != "raw_message_v3":
        raise ValueError("activation gate is not for raw_message_v3")
    if gate.get("snapshot_manifest_sha256") != manifest_sha256:
        raise ValueError("activation gate does not match the manifest")
    if int(gate.get("vector_generation", 0)) != int(generation):
        raise ValueError("activation gate does not match the vector generation")
    if gate.get("dataset_sha256") != dataset_sha256:
        raise ValueError("activation gate does not match the dataset")
    if gate.get("quality_sidecar_sha256") != quality_sidecar_sha256:
        raise ValueError("activation gate does not match the quality sidecar")
    if gate.get("results_sha256") != results_sha256:
        raise ValueError("activation gate does not match the evaluation results")
    if gate.get("benchmark_sha256") != benchmark_sha256:
        raise ValueError("activation gate does not match the benchmark")
    if gate.get("quality_sidecar_present") is not True:
        raise RuntimeError("activation gate has no reviewed quality sidecar")
    if not isinstance(quality_sidecar, dict):
        raise ValueError("quality sidecar is not an object")
    if quality_sidecar.get("private_replay_sha256") != quality_private_replay_sha256:
        raise ValueError("private replay artifact does not match the quality sidecar")
    if (
        quality_sidecar.get("visibility_artifact_sha256")
        != quality_visibility_artifact_sha256
    ):
        raise ValueError("visibility artifact does not match the quality sidecar")
    quality_bindings = {
        "dataset_sha256": dataset_sha256,
        "snapshot_manifest_sha256": manifest_sha256,
        "retrieval_fingerprint_sha256": gate.get(
            "retrieval_fingerprint_sha256"
        ),
    }
    if any(
        quality_sidecar.get(field) != expected
        for field, expected in quality_bindings.items()
    ):
        raise ValueError("quality sidecar binding does not match the gate")
    if int(gate.get("case_count", 0)) < 64:
        raise ValueError("activation gate has insufficient cases")
    metrics = gate.get("metrics")
    zero_metrics = {
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
    }
    minimum_metrics = {
        "recall_at_150": 0.80,
        "recall_within_24k": 0.80,
        "time_bucket_coverage_rate": 1.0,
        "citation_precision": 0.95,
        "citation_recall": 0.80,
        "grounded_answer_accuracy": 0.80,
        "answer_accuracy": 0.80,
        "abstention_f1": 0.90,
    }
    if not isinstance(metrics, dict):
        raise ValueError("activation gate has no metrics")
    if any(_finite_number(metrics.get(name)) != 0.0 for name in zero_metrics):
        raise RuntimeError("activation gate contains a hard safety failure")
    minimum_values = {
        name: _finite_number(metrics.get(name)) for name in minimum_metrics
    }
    latency_values = {
        name: _finite_number(metrics.get(name))
        for name in (
            "index_visibility_p95_ms",
            "ttft_p95_ms",
            "retrieval_p95_ms",
        )
    }
    if any(
        value is None
        for value in (*minimum_values.values(), *latency_values.values())
    ):
        raise ValueError("activation gate metrics are incomplete")
    below_minimum = any(
        minimum_values[name] < minimum
        for name, minimum in minimum_metrics.items()
    )
    latency_failure = (
        latency_values["index_visibility_p95_ms"] > 5_000.0
        or latency_values["ttft_p95_ms"] > 15_000.0
        or latency_values["retrieval_p95_ms"] >= 500.0
    )
    if below_minimum or latency_failure:
        raise RuntimeError("activation gate metrics do not pass")


def _validate_activation_quality_artifacts(
    args: argparse.Namespace,
    *,
    gate: dict,
    manifest_sha256: str,
    generation: int,
) -> None:
    case_count = gate.get("case_count")
    retrieval_fingerprint = gate.get("retrieval_fingerprint_sha256")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < 64
        or not isinstance(retrieval_fingerprint, str)
        or len(retrieval_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in retrieval_fingerprint)
    ):
        raise ValueError("activation gate quality bindings are invalid")
    evaluation_cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    if dataset_sha256 != _file_sha256(args.dataset):
        raise ValueError("activation dataset hash is invalid")
    gate_tag_counts = validate_v3_dataset_contract(evaluation_cases)
    observations, prompt_hashes = _load_activation_results(
        args.results,
        case_count=case_count,
    )
    computed_fingerprint = retrieval_fingerprint_sha256(observations)
    if computed_fingerprint != retrieval_fingerprint:
        raise ValueError("activation results do not match the retrieval fingerprint")
    quality = load_v3_quality_sidecar(
        args.quality_sidecar,
        dataset_sha256=_file_sha256(args.dataset),
        snapshot_manifest_sha256=manifest_sha256,
        retrieval_fingerprint=retrieval_fingerprint,
        case_count=case_count,
        private_replay_path=args.quality_private_replay,
        visibility_artifact_path=args.quality_visibility_artifact,
        expected_vector_generation=generation,
        evaluation_cases=evaluation_cases,
        expected_answer_prompt_sha256_by_case={
            index: value for index, value in enumerate(prompt_hashes)
        },
    )
    metadata = load_message_metadata(args.database)
    recomputed = evaluate_v3(
        cases=evaluation_cases,
        observations=observations,
        quality=quality,
        dataset_sha256=dataset_sha256,
        snapshot_manifest_sha256=manifest_sha256,
        retrieval_fingerprint=computed_fingerprint,
        gate_tag_counts=gate_tag_counts,
    )
    recomputed["vector_generation"] = generation
    recomputed["quality_sidecar_sha256"] = _file_sha256(args.quality_sidecar)
    recomputed["metrics"].update(
        audit_v3_quality_sources(
            cases=evaluation_cases,
            observations=observations,
            quality=quality,
            metadata=metadata,
        )
    )
    benchmark = _load_activation_benchmark(
        args.benchmark_report,
        case_count=case_count,
    )
    failures = _v3_acceptance_failures(report=recomputed, benchmark=benchmark)
    recomputed["acceptance"] = {
        "status": "failed" if failures else "passed",
        "error_codes": list(failures),
    }
    recomputed["results_sha256"] = _file_sha256(args.results)
    recomputed["benchmark_sha256"] = _file_sha256(args.benchmark_report)
    for field, value in recomputed.items():
        if gate.get(field) != value:
            raise ValueError(f"activation gate field was not reproduced: {field}")
    if failures:
        raise RuntimeError("recomputed activation gate did not pass")


_OBSERVATION_SEQUENCE_FIELDS = {
    "retrieved_source_message_ids",
    "history_packet_source_message_ids",
}
_OBSERVATION_FLOAT_FIELDS = {"retrieval_latency_ms"}


def _load_activation_results(
    path: Path,
    *,
    case_count: int,
) -> tuple[tuple[V3Observation, ...], tuple[str, ...]]:
    observation_fields = tuple(V3Observation.__dataclass_fields__)
    expected_fields = {
        *observation_fields,
        "variant",
        "answer_prompt_sha256",
    }
    observations: list[V3Observation] = []
    prompt_hashes: list[str] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line, parse_constant=_reject_json_constant)
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("activation result fields do not match the contract")
        if row.get("variant") != "v3":
            raise ValueError("activation result is not for V3")
        case_index = row.get("case_index")
        if (
            isinstance(case_index, bool)
            or not isinstance(case_index, int)
            or case_index != index
        ):
            raise ValueError("activation result case indexes are invalid")
        prompt_hash = row.get("answer_prompt_sha256")
        if not _is_sha256(prompt_hash):
            raise ValueError("activation result prompt hash is invalid")
        values: dict[str, Any] = {}
        for field in observation_fields:
            value = row[field]
            if field in _OBSERVATION_SEQUENCE_FIELDS:
                if (
                    not isinstance(value, list)
                    or any(not isinstance(item, str) or not item for item in value)
                    or len(value) != len(set(value))
                ):
                    raise ValueError("activation result source IDs are invalid")
                values[field] = tuple(value)
            elif field in _OBSERVATION_FLOAT_FIELDS:
                resolved = _finite_number(value)
                if resolved is None or resolved < 0:
                    raise ValueError("activation result latency is invalid")
                values[field] = resolved
            else:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("activation result counters are invalid")
                values[field] = value
        observations.append(V3Observation(**values))
        prompt_hashes.append(prompt_hash)
    if len(observations) != case_count:
        raise ValueError("activation results do not contain exactly one row per case")
    return tuple(observations), tuple(prompt_hashes)


def _load_activation_benchmark(path: Path, *, case_count: int) -> dict:
    benchmark = _load_strict_json(path)
    expected_fields = {
        "warmup_runs",
        "measured_runs",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "rewrite_enabled",
        "rerank_enabled",
        "network_enabled",
        "vector_success_verified",
    }
    if set(benchmark) != expected_fields:
        raise ValueError("activation benchmark fields do not match the contract")
    for field in ("warmup_runs", "measured_runs"):
        value = benchmark[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("activation benchmark run counts are invalid")
    if benchmark["warmup_runs"] < 20:
        raise ValueError("activation benchmark warmup is insufficient")
    if benchmark["measured_runs"] < max(250, int(case_count) * 5):
        raise ValueError("activation benchmark sample count is insufficient")
    for field in ("mean_latency_ms", "p50_latency_ms", "p95_latency_ms"):
        value = _finite_number(benchmark[field])
        if value is None or value < 0:
            raise ValueError("activation benchmark latency is invalid")
    if (
        benchmark["rewrite_enabled"] is not False
        or benchmark["rerank_enabled"] is not False
        or benchmark["network_enabled"] is not False
        or benchmark["vector_success_verified"] is not True
    ):
        raise ValueError("activation benchmark runtime contract is invalid")
    return benchmark


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _load_strict_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit_report(report: dict, *, output: Path | None) -> None:
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _merge_high_watermarks(
    *watermark_sets: dict[int, int],
) -> dict[int, int]:
    merged: dict[int, int] = {}
    for watermarks in watermark_sets:
        for group_id, watermark in watermarks.items():
            resolved_group_id = int(group_id)
            merged[resolved_group_id] = max(
                merged.get(resolved_group_id, 0),
                int(watermark),
            )
    return dict(sorted(merged.items()))


def _snapshot_live_high_watermarks(
    engine,
    *,
    manifest_watermarks: dict[int, int],
) -> dict[int, int]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT group_id, max(id) AS watermark "
                "FROM messages WHERE group_id IS NOT NULL "
                "GROUP BY group_id ORDER BY group_id"
            )
        )
        live = {
            int(group_id): int(watermark)
            for group_id, watermark in rows
            if int(watermark) > int(manifest_watermarks.get(int(group_id), 0))
        }
    return live


def _project_raw_ranges(
    engine,
    *,
    generation: int | None,
    lower_bounds: dict[int, int],
    upper_bounds: dict[int, int],
    batch_size: int,
) -> int:
    projected = 0
    for group_id, upper_bound in sorted(upper_bounds.items()):
        last_id = int(lower_bounds.get(int(group_id), 0))
        resolved_upper = int(upper_bound)
        while last_id < resolved_upper:
            with session_scope(engine) as session:
                rows = list(
                    session.scalars(
                        select(Message)
                        .where(
                            Message.group_id == int(group_id),
                            Message.id > last_id,
                            Message.id <= resolved_upper,
                        )
                        .order_by(Message.id.asc())
                        .limit(int(batch_size))
                    )
                )
                if not rows:
                    break
                documents = RetrievalDocumentRepository(session)
                for row in rows:
                    document = documents.project_raw_message_v3(
                        group_id=int(group_id),
                        message_id=int(row.id),
                        embedding_generation=generation,
                    )
                    projected += int(document is not None)
                last_id = int(rows[-1].id)
    return projected


def _embed_raw_generation(
    engine,
    *,
    provider,
    generation: int,
    upper_bounds: dict[int, int],
    batch_size: int,
) -> int:
    embedded = 0
    for group_id, upper_bound in sorted(upper_bounds.items()):
        while True:
            with session_scope(engine) as session:
                documents = list(
                    session.scalars(
                        select(RetrievalDocument)
                        .join(
                            RetrievalDocumentMessage,
                            (
                                RetrievalDocumentMessage.document_id
                                == RetrievalDocument.id
                            )
                            & (
                                RetrievalDocumentMessage.group_id
                                == RetrievalDocument.group_id
                            ),
                        )
                        .join(
                            Message,
                            (Message.id == RetrievalDocumentMessage.message_id)
                            & (Message.group_id == RetrievalDocumentMessage.group_id),
                        )
                        .where(
                            RetrievalDocument.group_id == int(group_id),
                            RetrievalDocument.document_kind == "raw_message_v3",
                            RetrievalDocument.source_table == "messages",
                            RetrievalDocument.status == "active",
                            RetrievalDocument.embedding_eligible.is_(True),
                            RetrievalDocument.embedding_generation == int(generation),
                            RetrievalDocument.embedding_status != "ready",
                            RetrievalDocumentMessage.group_id == int(group_id),
                            Message.group_id == int(group_id),
                            Message.id <= int(upper_bound),
                        )
                        .distinct()
                        .order_by(RetrievalDocument.id.asc())
                        .limit(int(batch_size))
                    )
                )
            if not documents:
                break
            vectors = provider.embed_documents(
                [str(document.content) for document in documents]
            )
            if vectors is None or len(vectors) != len(documents):
                raise RuntimeError("embedding provider returned an incomplete batch")
            embedded += write_retrieval_vector_embeddings(
                engine,
                generation=int(generation),
                rows=[
                    (
                        int(document.id),
                        int(document.group_id),
                        vector,
                    )
                    for document, vector in zip(
                        documents,
                        vectors,
                        strict=True,
                    )
                ],
            )
    return embedded


def _count_live_messages(
    engine,
    *,
    manifest_watermarks: dict[int, int],
    live_high_watermarks: dict[int, int],
) -> int:
    total = 0
    with engine.connect() as connection:
        for group_id, upper_bound in sorted(live_high_watermarks.items()):
            total += int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM messages "
                        "WHERE group_id = :group_id AND id > :lower_bound "
                        "AND id <= :upper_bound"
                    ),
                    {
                        "group_id": int(group_id),
                        "lower_bound": int(manifest_watermarks.get(int(group_id), 0)),
                        "upper_bound": int(upper_bound),
                    },
                ).scalar_one()
                or 0
            )
    return total


def _active_vector_generation(engine) -> int | None:
    with engine.connect() as connection:
        value = connection.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).scalar_one_or_none()
    return int(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
