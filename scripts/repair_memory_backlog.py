"""Repair known memory-system backlog issues.

Commands:
  cleanup_jobs      mark legacy memory_compaction queued/failed jobs completed
  requeue_episodes  requeue episode processing for failed episodes
  backfill_canonical fill canonical_key for active facts that lack one

Usage (inside the production container):
    python scripts/repair_memory_backlog.py --db /workspace/data/bot.db plan
    python scripts/repair_memory_backlog.py --db /workspace/data/bot.db run
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/workspace")

from sqlalchemy import text
from sqlalchemy import select

from app.core.memory_compaction import canonical_key
from app.core.memory_background_service import SqlAlchemyMemoryBackgroundStore
from app.storage.db import build_engine, session_scope
from app.storage.models import MemoryItem
from app.storage.repositories import JobRepository


def _store(engine):
    return SqlAlchemyMemoryBackgroundStore(
        engine,
        max_attempts=3,
        memory_enabled_group_ids=None,
    )


def cleanup_jobs(engine, dry_run: bool) -> int:
    with session_scope(engine) as session:
        statuses = ["queued", "running", "failed"]
        jobs = JobRepository(session)
        count = jobs.count_active_jobs(
            job_type="memory_compaction",
            statuses=statuses,
        )
        if dry_run or not count:
            return count
        return jobs.cancel_jobs(
            job_type="memory_compaction",
            statuses=statuses,
            now=datetime.now(UTC),
        )


def requeue_episodes(engine, dry_run: bool) -> int:
    store = _store(engine)
    with session_scope(engine) as session:
        rows = session.execute(
            text(
                "SELECT id, group_id, compaction_version FROM conversation_episodes "
                "WHERE status='failed'"
            )
        ).all()
        count = 0
        for episode_id, group_id, compaction_version in rows:
            if dry_run:
                count += 1
                continue
            store.enqueue_episode_processing(
                episode_id=int(episode_id),
                group_id=int(group_id),
                compaction_generation=str(compaction_version),
                backfill_run_id=None,
                now=datetime.now(UTC),
            )
            count += 1
        return count


def backfill_canonical(engine, dry_run: bool) -> int:
    with session_scope(engine) as session:
        rows = session.scalars(
            select(MemoryItem).where(
                MemoryItem.status == "active",
                MemoryItem.canonical_key == "",
            )
        ).all()
        count = 0
        skipped = 0
        occupied: set[tuple[str, str, str]] = set()
        for existing in session.scalars(
            select(MemoryItem).where(MemoryItem.canonical_key != "")
        ):
            occupied.add(
                (
                    str(existing.scope_type or ""),
                    str(existing.scope_id or ""),
                    str(existing.canonical_key),
                )
            )
        for memory in rows:
            key = canonical_key(
                str(memory.memory_kind or ""),
                str(memory.subject_id or "group"),
                str(memory.predicate or ""),
                str(memory.object_text or ""),
            )
            if not key:
                continue
            identity = (
                str(memory.scope_type or ""),
                str(memory.scope_id or ""),
                key,
            )
            if identity in occupied:
                skipped += 1
                continue
            if dry_run:
                count += 1
                continue
            memory.canonical_key = key
            occupied.add(identity)
            count += 1
        print(f"  skipped conflicting: {skipped}")
        return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--tasks", default="all", choices=("all", "cleanup_jobs", "requeue_episodes", "backfill_canonical"))
    args = parser.parse_args()

    engine = build_engine(args.db)
    dry_run = args.command == "plan"
    tasks = (
        ("cleanup_jobs", cleanup_jobs),
        ("requeue_episodes", requeue_episodes),
        ("backfill_canonical", backfill_canonical),
    )
    for name, handler in tasks:
        if args.tasks != "all" and args.tasks != name:
            continue
        count = handler(engine, dry_run=dry_run)
        print(f"{name}: {'would process' if dry_run else 'processed'} {count}")
    engine.dispose()


if __name__ == "__main__":
    main()
