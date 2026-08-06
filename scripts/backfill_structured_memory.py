"""Full-history structured memory backfill with resume and coverage reports.

Reuses the manifest-bounded episode backfill machinery but derives a live
manifest from the current database instead of a frozen backup snapshot. The
backfill fills summaries (episode level) and structured ``memory_items`` for
every group's messages below each group's current watermark, using the same
idempotent episode/compaction-generation keys as the runtime worker.

Commands:
  plan       read-only inventory and LLM-call estimate
  run        enqueue and drain the backfill for missing coverage
  status     coverage report for an existing run
  finalize   mark a drained run completed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.engine import Engine

from app.config import AppSettings
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
from app.storage.db import create_all, session_scope


INDEX_GENERATION = "fts:layered-backfill"


def _build_engine(database: Path, *, create: bool) -> Engine:
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        future=True,
    )

    @sqlalchemy_event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    if create:
        create_all(engine)
    return engine


def _live_watermarks(engine, *, limit_groups: int | None = None) -> dict[int, int]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT group_id, MAX(id) FROM messages "
                "WHERE group_id IS NOT NULL GROUP BY group_id ORDER BY group_id"
            )
        )
        watermarks = {
            int(group_id): int(max_id)
            for group_id, max_id in rows
            if max_id is not None
        }
    if limit_groups is not None and limit_groups > 0:
        watermarks = dict(
            list(sorted(watermarks.items()))[: int(limit_groups)]
        )
    return watermarks


def _live_manifest(watermarks: dict[int, int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "layered-structured-backfill",
        "buckets": {
            f"group:{group_id}": {
                "group_id": int(group_id),
                "watermark": int(watermark),
            }
            for group_id, watermark in sorted(watermarks.items())
        },
    }


def _plan_inventory(engine, watermarks: dict[int, int]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    with engine.connect() as connection:
        for group_id, watermark in sorted(watermarks.items()):
            parameters = {
                "group_id": int(group_id),
                "watermark": int(watermark),
                "segmentation_prefix": f"{MEMORY_SEGMENTATION_GENERATION}%",
            }
            messages = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM messages "
                        "WHERE group_id = :group_id AND id <= :watermark"
                    ),
                    parameters,
                ).scalar_one()
            )
            eligible = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM messages "
                        "WHERE group_id = :group_id AND id <= :watermark "
                        "AND (json_extract(raw_json, '$.delivery_state') IS NULL "
                        "OR json_extract(raw_json, '$.delivery_state') "
                        "NOT IN ('reserved', 'uncertain'))"
                    ),
                    parameters,
                ).scalar_one()
            )
            assigned = int(
                connection.execute(
                    text(
                        "SELECT COUNT(DISTINCT em.message_id) FROM episode_messages em "
                        "JOIN conversation_episodes e ON e.id = em.episode_id "
                        "WHERE em.group_id = :group_id AND em.message_id <= :watermark "
                        "AND e.is_current = 1 "
                        "AND e.segmentation_version LIKE :segmentation_prefix"
                    ),
                    parameters,
                ).scalar_one()
            )
            summaries = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM summaries "
                        "WHERE scope_type = 'group' AND scope_id = :group_id_text"
                    ),
                    {"group_id_text": str(group_id)},
                ).scalar_one()
            )
            memory_items = int(
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM memory_items "
                        "WHERE scope_type = 'group' AND scope_id = :group_id_text"
                    ),
                    {"group_id_text": str(group_id)},
                ).scalar_one()
            )
            groups.append(
                {
                    "group_id": int(group_id),
                    "watermark": int(watermark),
                    "messages": messages,
                    "eligible_messages": eligible,
                    "assigned_messages": assigned,
                    "gap_messages": max(0, eligible - assigned),
                    "summaries": summaries,
                    "memory_items": memory_items,
                }
            )
    return groups


def _estimate_llm_calls(groups: list[dict[str, Any]], *, batch_size: int) -> int:
    return sum(
        -(-group["gap_messages"] // max(1, int(batch_size)))
        for group in groups
    )


def _build_background(settings: AppSettings, engine, llm_client, run_key: str):
    store = SqlAlchemyMemoryBackgroundStore(
        engine,
        batch_size=max(500, settings.memory_episode_max_messages * 10),
        max_attempts=settings.memory_compaction_retry_limit,
        embedding_provider="",
        embedding_model="",
        embedding_version="",
        embedding_dimensions=None,
        embedding_generation=None,
    )
    return MemoryBackgroundService(
        store=store,
        deriver=CompactionEpisodeDeriver(
            llm_client=llm_client,
            max_facts=settings.memory_compaction_max_facts,
        ),
        worker_id=f"structured-backfill-{run_key}",
        segmentation_generation=MEMORY_SEGMENTATION_GENERATION,
        compaction_generation=MEMORY_COMPACTION_GENERATION,
        idle_minutes=settings.memory_episode_idle_minutes,
        max_messages=settings.memory_episode_max_messages,
        max_tokens=settings.memory_episode_max_tokens,
        chunk_max_tokens=settings.memory_chunk_max_tokens,
        chunk_overlap_messages=settings.memory_chunk_overlap_messages,
        bot_user_id=settings.bot_qq,
        embedder=None,
        lease_seconds=60,
    )


def _load_run(engine, run_key: str) -> tuple[int, dict[int, int]]:
    with session_scope(engine) as session:
        row = session.execute(
            text(
                "SELECT id, snapshot_watermarks_json FROM memory_backfill_runs "
                "WHERE run_key = :run_key ORDER BY id DESC LIMIT 1"
            ),
            {"run_key": str(run_key)},
        ).one_or_none()
    if row is None:
        raise ValueError(f"no backfill run found for run_key={run_key}")
    raw_watermarks = row.snapshot_watermarks_json
    if isinstance(raw_watermarks, str):
        raw_watermarks = json.loads(raw_watermarks or "{}")
    watermarks = {
        int(group_id): int(watermark)
        for group_id, watermark in (raw_watermarks or {}).items()
    }
    return int(row.id), watermarks


def _print_report(report) -> None:
    rendered = json.dumps(
        report.as_safe_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    print(rendered)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-history structured memory backfill (summaries + facts)."
    )
    parser.add_argument("command", choices=("plan", "run", "status", "finalize"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--run-key", default="")
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--create-all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    settings = AppSettings()
    engine = _build_engine(
        args.database,
        create=bool(args.create_all),
    )
    try:
        if args.command == "plan":
            watermarks = _live_watermarks(
                engine,
                limit_groups=args.limit_groups,
            )
            groups = _plan_inventory(engine, watermarks)
            print(
                json.dumps(
                    {
                        "command": "plan",
                        "groups": groups,
                        "estimated_llm_calls": _estimate_llm_calls(
                            groups,
                            batch_size=settings.memory_episode_max_messages,
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0

        run_key = str(args.run_key or "").strip()
        if not run_key:
            raise ValueError("--run-key is required for run/status/finalize")
        if args.command == "status":
            run_id, watermarks = _load_run(engine, run_key)
            _print_report(
                collect_backfill_coverage(
                    engine=engine,
                    run_id=run_id,
                    run_key=run_key,
                    watermarks=watermarks,
                )
            )
            return 0
        if args.command == "finalize":
            run_id, watermarks = _load_run(engine, run_key)
            report = collect_backfill_coverage(
                engine=engine,
                run_id=run_id,
                run_key=run_key,
                watermarks=watermarks,
            )
            if report.pending_jobs or report.running_jobs or report.failed_jobs:
                raise RuntimeError(
                    "backfill run still has unfinished jobs; run status first"
                )
            finalize_backfill_run(engine=engine, run_id=run_id)
            _print_report(
                collect_backfill_coverage(
                    engine=engine,
                    run_id=run_id,
                    run_key=run_key,
                    watermarks=watermarks,
                )
            )
            return 0

        watermarks = _live_watermarks(
            engine,
            limit_groups=args.limit_groups,
        )
        groups = _plan_inventory(engine, watermarks)
        print(
            json.dumps(
                {
                    "command": "run",
                    "run_key": run_key,
                    "estimated_llm_calls": _estimate_llm_calls(
                        groups,
                        batch_size=settings.memory_episode_max_messages,
                    ),
                    "groups": groups,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if args.limit_groups and not groups:
            return 0
        llm_client = build_llm_client(settings=settings, engine=engine)
        background = _build_background(settings, engine, llm_client, run_key)
        manifest = _live_manifest(watermarks)
        report = run_memory_backfill(
            engine=engine,
            background_service=background,
            manifest=manifest,
            run_key=run_key,
            segmentation_generation=MEMORY_SEGMENTATION_GENERATION,
            compaction_generation=MEMORY_COMPACTION_GENERATION,
            index_generation=INDEX_GENERATION,
            max_steps=args.max_steps,
            finalize=False,
        )
        if args.finalize:
            finalize_backfill_run(engine=engine, run_id=report.run_id)
            report = collect_backfill_coverage(
                engine=engine,
                run_id=report.run_id,
                run_key=run_key,
                watermarks=watermarks,
            )
        _print_report(report)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
