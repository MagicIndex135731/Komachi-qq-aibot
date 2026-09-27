"""Dry-run or apply one bounded, source-stable member fact replay window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from sqlalchemy import select

from app.config import AppSettings
from app.core.member_memory_backfill import MemberFactRefreshService
from app.providers.semantic_embeddings import build_embedding_provider
from app.storage.db import build_engine, session_scope
from app.storage.models import MemoryItem, Message
from app.storage.repositories import MemoryRepository


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sqlite_integrity(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if result is None or str(result[0]).casefold() != "ok":
        raise RuntimeError(f"SQLite integrity check failed for {path.name}")


def _window_ledger_digest(
    path: Path,
    *,
    group_id: int,
    user_id: int,
    start_message_id: int,
    end_message_id: int | None,
    max_messages: int,
) -> tuple[int, str]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        sql = (
            "SELECT id, platform_msg_id, group_id, user_id, timestamp, "
            "COALESCE(plain_text, '') FROM messages "
            "WHERE group_id = ? AND user_id = ? AND id >= ?"
        )
        parameters: list[int] = [int(group_id), int(user_id), int(start_message_id)]
        if end_message_id is not None:
            sql += " AND id <= ?"
            parameters.append(int(end_message_id))
        sql += " ORDER BY id LIMIT ?"
        parameters.append(int(max_messages) + 1)
        rows = connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return len(rows), digest.hexdigest()


def _verify_replay_backup(
    database: Path,
    backup: Path,
    *,
    group_id: int,
    user_id: int,
    start_message_id: int,
    end_message_id: int | None,
    max_messages: int,
) -> str:
    _verify_sqlite_integrity(database)
    _verify_sqlite_integrity(backup)
    live_ledger = _window_ledger_digest(
        database,
        group_id=group_id,
        user_id=user_id,
        start_message_id=start_message_id,
        end_message_id=end_message_id,
        max_messages=max_messages,
    )
    backup_ledger = _window_ledger_digest(
        backup,
        group_id=group_id,
        user_id=user_id,
        start_message_id=start_message_id,
        end_message_id=end_message_id,
        max_messages=max_messages,
    )
    if live_ledger != backup_ledger:
        raise RuntimeError("verified backup does not match the replay-window ledger")
    return _sha256(backup)


def _backfill_replayed_semantic_vectors(
    engine,
    *,
    embedder,
    group_id: int,
    user_id: int,
    start_message_id: int,
    end_message_id: int | None,
    max_messages: int,
) -> int:
    """Persist vectors only for active facts sourced from the replay window."""

    with session_scope(engine) as session:
        source_stmt = (
            select(Message.platform_msg_id)
            .where(
                Message.group_id == int(group_id),
                Message.user_id == int(user_id),
                Message.id >= int(start_message_id),
            )
            .order_by(Message.id)
            .limit(int(max_messages))
        )
        if end_message_id is not None:
            source_stmt = source_stmt.where(Message.id <= int(end_message_id))
        source_ids = {str(value) for value in session.scalars(source_stmt)}
        candidates = list(
            session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == str(group_id),
                    MemoryItem.subject_id == str(user_id),
                    MemoryItem.status == "active",
                )
            )
        )
        rows = [
            row
            for row in candidates
            if source_ids.intersection(
                str(value)
                for value in (row.source_msg_ids or [row.source_msg_id])
                if str(value).strip()
            )
        ]
        if not rows:
            return 0
        vectors = list(
            embedder.embed_documents([str(row.content or "") for row in rows]) or []
        )
        identity = embedder.identity
        if len(vectors) != len(rows):
            raise RuntimeError("embedding provider returned incomplete replay coverage")
        if any(len(vector) != int(identity.dimensions) for vector in vectors):
            raise RuntimeError("embedding provider returned an unexpected vector dimension")
        vector_rows = [
            {
                "memory_id": int(row.id),
                "group_id": int(group_id),
                "provider": str(identity.provider),
                "model": str(identity.model),
                "dimensions": int(identity.dimensions),
                "version": str(identity.version),
                "vector_json": json.dumps(
                    [float(value) for value in vector],
                    separators=(",", ":"),
                ),
            }
            for row, vector in zip(rows, vectors)
            if vector
        ]
        if vector_rows:
            MemoryRepository(session).upsert_memory_item_semantic_vectors(vector_rows)
        return len(vector_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--start-message-id", type=int, required=True)
    parser.add_argument("--end-message-id", type=int)
    parser.add_argument("--max-messages", type=int, default=200)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verified-backup", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    database = args.database.resolve()
    if not database.is_file():
        parser.error("--database must be an existing SQLite file")
    backup_hash = ""
    if args.apply:
        if args.verified_backup is None:
            parser.error("--apply requires --verified-backup")
        backup = args.verified_backup.resolve()
        if not backup.is_file() or backup == database:
            parser.error("--verified-backup must be a separate existing file")
        backup_hash = _verify_replay_backup(
            database,
            backup,
            group_id=int(args.group_id),
            user_id=int(args.user_id),
            start_message_id=int(args.start_message_id),
            end_message_id=args.end_message_id,
            max_messages=int(args.max_messages),
        )

    settings = AppSettings()
    engine = build_engine(database)
    embedder = None
    if args.apply:
        embedder = build_embedding_provider(
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
        if not embedder.available:
            raise RuntimeError("embedding provider is unavailable; replay was not applied")
    service = MemberFactRefreshService(
        engine=engine,
        settings=settings,
        group_ids={int(args.group_id)},
        bot_qq=int(settings.bot_qq),
        member_allowlist={int(args.user_id)},
    )
    report = service.replay_member_window(
        group_id=int(args.group_id),
        user_id=int(args.user_id),
        start_message_id=int(args.start_message_id),
        end_message_id=args.end_message_id,
        max_messages=int(args.max_messages),
        dry_run=not bool(args.apply),
    )
    semantic_vectors_indexed = 0
    if args.apply and embedder is not None:
        semantic_vectors_indexed = _backfill_replayed_semantic_vectors(
            engine,
            embedder=embedder,
            group_id=int(args.group_id),
            user_id=int(args.user_id),
            start_message_id=int(args.start_message_id),
            end_message_id=args.end_message_id,
            max_messages=int(args.max_messages),
        )
    report["semantic_vectors_indexed"] = int(semantic_vectors_indexed)
    payload = {
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry-run",
        "database_name": database.name,
        "verified_backup_sha256": backup_hash,
        "bounds": {
            "group_id": int(args.group_id),
            "user_id": int(args.user_id),
            "start_message_id": int(args.start_message_id),
            "end_message_id": args.end_message_id,
            "max_messages": int(args.max_messages),
        },
        "result": report,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
