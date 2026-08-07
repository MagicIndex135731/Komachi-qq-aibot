"""Persistent semantic vector backfill for memory_items.

Idempotent: vectors are upserted by memory_id. Run inside the container
(or against a backup) with the same embedding provider as production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, event as sa_event, text
from sqlalchemy.pool import NullPool

from app.config import AppSettings
from app.providers.semantic_embeddings import build_embedding_provider
from app.storage.db import create_all, session_scope
from app.storage.repositories import MemoryRepository


def _build_engine(database: Path):
    engine = create_engine(
        f"sqlite:///{database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    @sa_event.listens_for(engine, "connect")
    def _set_busy_timeout(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    create_all(engine)
    return engine


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill persistent semantic vectors for memory_items."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    settings = AppSettings()
    engine = _build_engine(args.database)
    try:
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
            raise RuntimeError("embedding provider is unavailable")
        identity = embedder.identity
        after_id = 0
        total = 0
        while True:
            with session_scope(engine) as session:
                rows = MemoryRepository(
                    session
                ).list_active_memory_items_for_indexing(
                    after_id=after_id,
                    limit=args.batch_size,
                )
                if not rows:
                    break
                contents = [str(row.content or "") for row in rows]
                vectors = embedder.embed_documents(contents) or []
                vector_rows: list[dict[str, Any]] = []
                for row, vector in zip(rows, vectors):
                    if not vector:
                        continue
                    vector_rows.append(
                        {
                            "memory_id": int(row.id),
                            "group_id": int(row.scope_id or 0),
                            "provider": str(identity.provider),
                            "model": str(identity.model),
                            "dimensions": int(identity.dimensions),
                            "version": str(identity.version),
                            "vector_json": json.dumps(
                                [float(value) for value in vector],
                                separators=(",", ":"),
                            ),
                        }
                    )
                if not args.dry_run and vector_rows:
                    MemoryRepository(session).upsert_memory_item_semantic_vectors(
                        vector_rows
                    )
                total += len(rows)
                after_id = int(rows[-1].id)
                print(
                    f"processed={len(rows)} indexed={len(vector_rows)} "
                    f"total={total} next_after={after_id}"
                )
                if args.limit is not None and total >= args.limit:
                    break
        print(f"done total={total} dry_run={bool(args.dry_run)}")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
