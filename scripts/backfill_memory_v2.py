from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sqlalchemy import text

from app.config import AppSettings
from app.core.memory_backfill import verify_message_ledger_manifest
from app.core.memory_backfill_runner import (
    collect_backfill_coverage,
    finalize_backfill_run,
    group_watermarks_from_manifest,
    run_memory_backfill,
)
from app.core.memory_background_service import (
    CompactionEpisodeDeriver,
    MemoryBackgroundService,
    SqlAlchemyMemoryBackgroundStore,
)
from app.main import (
    MEMORY_COMPACTION_GENERATION,
    MEMORY_SEGMENTATION_GENERATION,
    build_llm_client,
)
from app.providers.semantic_embeddings import build_embedding_provider
from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    create_all,
    ensure_retrieval_vector_generation,
    session_scope,
)
from app.storage.repositories import MemoryBackfillRunRepository


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume manifest-bounded Memory V2 backfill."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("backup manifest root must be an object")
    ledger = verify_message_ledger_manifest(args.database, manifest)
    if not ledger.matches:
        raise RuntimeError("live message ledger differs from the verified backup")

    settings = AppSettings()
    engine = build_engine(args.database)
    try:
        create_all(engine)
        llm_client = build_llm_client(settings=settings, engine=engine)
        embedding_provider = build_embedding_provider(
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
        identity = embedding_provider.identity
        if not embedding_provider.available:
            raise RuntimeError(
                "semantic embedding provider is unavailable; rollout remains shadow-only"
            )
        vector_generation = (
            ensure_retrieval_vector_generation(
                engine,
                provider=identity.provider,
                model=identity.model,
                dimensions=identity.dimensions,
                version=identity.version,
            )
            if embedding_provider.available
            else None
        )
        if vector_generation is None:
            raise RuntimeError(
                "semantic vector generation is unavailable; rollout remains shadow-only"
            )
        generation_label = (
            f"vector:{vector_generation}"
            if vector_generation is not None
            else "fts-only"
        )
        store = SqlAlchemyMemoryBackgroundStore(
            engine,
            batch_size=max(500, settings.memory_episode_max_messages * 10),
            max_attempts=settings.memory_compaction_retry_limit,
            embedding_provider=identity.provider,
            embedding_model=identity.model,
            embedding_version=identity.version,
            embedding_dimensions=identity.dimensions,
            embedding_generation=vector_generation,
        )
        background = MemoryBackgroundService(
            store=store,
            deriver=CompactionEpisodeDeriver(
                llm_client=llm_client,
                max_facts=settings.memory_compaction_max_facts,
            ),
            worker_id=f"backfill-{args.run_key}",
            segmentation_generation=MEMORY_SEGMENTATION_GENERATION,
            compaction_generation=MEMORY_COMPACTION_GENERATION,
            idle_minutes=settings.memory_episode_idle_minutes,
            max_messages=settings.memory_episode_max_messages,
            max_tokens=settings.memory_episode_max_tokens,
            chunk_max_tokens=settings.memory_chunk_max_tokens,
            chunk_overlap_messages=settings.memory_chunk_overlap_messages,
            bot_user_id=settings.bot_qq,
            embedder=embedding_provider,
            lease_seconds=60,
        )
        report = run_memory_backfill(
            engine=engine,
            background_service=background,
            manifest=manifest,
            run_key=args.run_key,
            segmentation_generation=MEMORY_SEGMENTATION_GENERATION,
            compaction_generation=MEMORY_COMPACTION_GENERATION,
            index_generation=generation_label,
            max_steps=args.max_steps,
            finalize=False,
        )
        if vector_generation is not None:
            try:
                active_generation = _active_vector_generation(engine)
                activated = activate_retrieval_vector_generation(
                    engine,
                    generation=vector_generation,
                    expected_active_generation=active_generation,
                )
            except Exception:
                _mark_backfill_failed(
                    engine,
                    run_id=report.run_id,
                    error_code="VectorActivationFailed",
                )
                raise
            if not activated:
                _mark_backfill_failed(
                    engine,
                    run_id=report.run_id,
                    error_code="VectorActivationFailed",
                )
                raise RuntimeError("vector generation failed coverage-checked activation")
        ledger_after = _verify_final_ledger(
            database=args.database,
            manifest=manifest,
            engine=engine,
            run_id=report.run_id,
        )
        finalize_backfill_run(engine=engine, run_id=report.run_id)
        report = collect_backfill_coverage(
            engine=engine,
            run_id=report.run_id,
            run_key=args.run_key,
            watermarks=group_watermarks_from_manifest(manifest),
        )
        safe_report = report.as_safe_dict()
        safe_report["ledger_after"] = ledger_after
        rendered = json.dumps(
            safe_report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    finally:
        engine.dispose()


def _active_vector_generation(engine) -> int | None:
    with engine.connect() as connection:
        value = connection.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel = 'vector' AND is_active = 1"
            )
        ).scalar_one_or_none()
    return int(value) if value is not None else None


def _verify_final_ledger(*, database: Path, manifest: dict, engine, run_id: int) -> dict:
    try:
        final_ledger = verify_message_ledger_manifest(database, manifest)
    except Exception:
        _mark_backfill_failed(
            engine,
            run_id=run_id,
            error_code="LedgerVerificationFailed",
        )
        raise
    if not final_ledger.matches:
        _mark_backfill_failed(
            engine,
            run_id=run_id,
            error_code="LedgerMismatch",
        )
        raise RuntimeError("live message ledger changed within the snapshot watermark")
    outside = {
        key: comparison.rows_above_watermark
        for key, comparison in sorted(final_ledger.buckets.items())
        if comparison.rows_above_watermark
    }
    return {
        "matches": True,
        "rows_above_watermark": outside,
        "rows_above_watermark_total": sum(outside.values()),
    }


def _mark_backfill_failed(engine, *, run_id: int, error_code: str) -> None:
    with session_scope(engine) as session:
        MemoryBackfillRunRepository(session).update_status(
            run_id=int(run_id),
            status="failed",
            completed_at=None,
            last_error_code=str(error_code),
        )


if __name__ == "__main__":
    raise SystemExit(main())
