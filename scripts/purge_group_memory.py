"""Purge all memory-derived data for specific groups.

Deletes memory items, summaries, episodes, retrieval documents (raw_v3, memory,
summary kinds), FTS rows, semantic vectors and memory jobs for the given
groups. Raw `messages` are never touched. Run against a backup first, then
production. Idempotent: repeated runs delete nothing new.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Sequence

from sqlalchemy import text

from app.storage.db import build_engine, create_all


_VEC_TABLE_RE = re.compile(r"retrieval_documents_vec_g[1-9][0-9]*")


def _table_exists(connection, name: str) -> bool:
    row = connection.execute(
        text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name=:name"
        ),
        {"name": name},
    ).first()
    return row is not None


def _count(connection, table_where: str, parameters: dict[str, Any]) -> int:
    row = connection.execute(
        text(f"SELECT COUNT(*) FROM {table_where}"),
        parameters,
    ).first()
    return int(row[0]) if row else 0


def _group_operations(
    group_id: int,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    scope = str(group_id)
    like = f'%"{group_id}"%'
    operations: list[tuple[str, str, str, dict[str, Any]]] = [
        (
            "memory_item_semantic_vectors",
            "memory_item_semantic_vectors WHERE group_id = :g",
            "DELETE FROM memory_item_semantic_vectors WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "memory_items_vec",
            "memory_items_vec WHERE memory_id IN "
            "(SELECT id FROM memory_items WHERE scope_id = :scope)",
            "DELETE FROM memory_items_vec WHERE memory_id IN "
            "(SELECT id FROM memory_items WHERE scope_id = :scope)",
            {"scope": scope},
        ),
        (
            "memory_items_fts",
            "memory_items_fts WHERE scope_id = :scope",
            "DELETE FROM memory_items_fts WHERE scope_id = :scope",
            {"scope": scope},
        ),
        (
            "memory_items",
            "memory_items WHERE scope_id = :scope",
            "DELETE FROM memory_items WHERE scope_id = :scope",
            {"scope": scope},
        ),
        (
            "retrieval_documents_fts",
            "retrieval_documents_fts WHERE group_id = :g",
            "DELETE FROM retrieval_documents_fts WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "retrieval_document_messages",
            "retrieval_document_messages WHERE group_id = :g",
            "DELETE FROM retrieval_document_messages WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "retrieval_documents",
            "retrieval_documents WHERE group_id = :g",
            "DELETE FROM retrieval_documents WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "summaries",
            "summaries WHERE scope_id = :scope",
            "DELETE FROM summaries WHERE scope_id = :scope",
            {"scope": scope},
        ),
        (
            "episode_messages",
            "episode_messages WHERE group_id = :g",
            "DELETE FROM episode_messages WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "conversation_episodes",
            "conversation_episodes WHERE group_id = :g",
            "DELETE FROM conversation_episodes WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "memory_late_arrival_preparations",
            "memory_late_arrival_preparations WHERE group_id = :g",
            "DELETE FROM memory_late_arrival_preparations WHERE group_id = :g",
            {"g": group_id},
        ),
        (
            "jobs",
            "jobs WHERE json_extract(payload_json, '$.group_id') = :g "
            "OR payload_json LIKE :like",
            "DELETE FROM jobs WHERE json_extract(payload_json, '$.group_id') = :g "
            "OR payload_json LIKE :like",
            {"g": group_id, "like": like},
        ),
    ]
    return operations


def purge_groups(
    engine,
    group_ids: Sequence[int],
    *,
    dry_run: bool = False,
) -> dict[int, dict[str, int]]:
    """Delete memory-derived rows for each group; returns per-group counts."""
    results: dict[int, dict[str, int]] = {}
    with engine.begin() as connection:
        vec_tables = [
            row[0]
            for row in connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table'"
                )
            ).fetchall()
            if _VEC_TABLE_RE.fullmatch(str(row[0]))
        ]
        for group_id in group_ids:
            print(f"=== group {group_id} ===")
            per_table: dict[str, int] = {}
            operations = _group_operations(group_id)
            for table, count_where, statement, parameters in operations:
                if not _table_exists(connection, table):
                    print(f"  {table}: skip (missing)")
                    continue
                if table.startswith("retrieval_documents_vec_"):
                    continue
                count = _count(connection, count_where, parameters)
                per_table[table] = count
                print(
                    f"  {table}: before={count} "
                    f"deleted={0 if dry_run else count}"
                )
                if not dry_run and count:
                    connection.execute(text(statement), parameters)
            for vec_table in vec_tables:
                count_where = f"{vec_table} WHERE group_id = :g"
                statement = f"DELETE FROM {vec_table} WHERE group_id = :g"
                count = _count(connection, count_where, {"g": group_id})
                per_table[vec_table] = count
                print(
                    f"  {vec_table}: before={count} "
                    f"deleted={0 if dry_run else count}"
                )
                if not dry_run and count:
                    connection.execute(text(statement), {"g": group_id})
            results[group_id] = per_table
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Purge memory-derived data for specific groups."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", type=int, action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    engine = build_engine(args.database)
    create_all(engine)
    try:
        purge_groups(
            engine,
            args.group_id,
            dry_run=bool(args.dry_run),
        )
        print(f"done dry_run={bool(args.dry_run)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
