from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

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
        args.dataset is not None or args.quality_sidecar is not None
    ):
        raise ValueError("dataset and quality-sidecar are only valid for activate")
    if args.phase == "activate":
        if args.prepared_report is None:
            raise ValueError("activate requires prepared-report")
        if args.gate_report is None:
            raise ValueError("activate requires gate-report")
        if args.dataset is None:
            raise ValueError("activate requires dataset")
        if args.quality_sidecar is None:
            raise ValueError("activate requires quality-sidecar")
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
        quality_sidecar=quality_sidecar,
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
            raise RuntimeError("post-activation raw_message_v3 catch-up is incomplete")

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
                for group_id, watermark in sorted(post_activation_live_bounds.items())
            },
            "vector_status": coverage.status,
            "vector_total_documents": coverage.total_documents,
            "vector_indexed_documents": coverage.indexed_documents,
            "ledger_matches": True,
            "gate_report_sha256": _file_sha256(args.gate_report),
        }
        _emit_report(report, output=args.output)
        return 0
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
    if gate.get("quality_sidecar_present") is not True:
        raise RuntimeError("activation gate has no reviewed quality sidecar")
    if not isinstance(quality_sidecar, dict):
        raise ValueError("quality sidecar is not an object")
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
