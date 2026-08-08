"""Backfill semantic vectors for episode/summary/memory retrieval documents.

These documents use document_family='' and previously had no active vector
generation, so semantic retrieval only covered raw_message_v3. This script
creates/reuses a building generation for that family, embeds the active
documents, and activates the generation when coverage is complete.

Usage (inside the production container):
    python scripts/backfill_summary_memory_vectors.py --db /workspace/data/bot.db
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/workspace")

from sqlalchemy import bindparam, select, text

from app.config import AppSettings
from app.core.time_utils import shanghai_now_naive
from app.providers.semantic_embeddings import build_embedding_provider
from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    ensure_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import RetrievalDocument


DOCUMENT_KINDS = ("episode", "episode_summary", "memory")
BATCH_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/data/bot.db")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

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
        raise SystemExit("embedding provider is unavailable")
    identity = provider.identity

    engine = build_engine(args.db)
    with session_scope(engine) as session:
        stale = session.execute(
            text(
                "UPDATE retrieval_index_state SET status='failed', updated_at=:now "
                "WHERE channel='vector' AND document_family='' AND status='building'"
            ),
            {"now": shanghai_now_naive()},
        )
        print(f"stale building generations failed: {stale.rowcount}")
    generation = ensure_retrieval_vector_generation(
        engine,
        provider=str(identity.provider),
        model=str(identity.model),
        dimensions=int(identity.dimensions),
        version=str(identity.version),
    )
    if generation is None:
        raise SystemExit("failed to create/reuse a building vector generation")
    print(f"generation={generation}")

    embedded_total = 0
    last_id = 0
    while True:
        with session_scope(engine) as session:
            query = (
                select(RetrievalDocument)
                .where(
                    RetrievalDocument.status == "active",
                    RetrievalDocument.document_kind.in_(DOCUMENT_KINDS),
                    RetrievalDocument.content != "",
                    RetrievalDocument.id > last_id,
                )
                .order_by(RetrievalDocument.id.asc())
                .limit(BATCH_SIZE)
            )
            documents = list(session.scalars(query))
        if not documents:
            break
        if args.limit is not None and embedded_total + len(documents) > args.limit:
            documents = documents[: max(0, args.limit - embedded_total)]
            if not documents:
                break
        with session_scope(engine) as session:
            session.execute(
                text(
                    "UPDATE retrieval_documents SET embedding_eligible=1, "
                    "embedding_status='pending' WHERE id IN :ids"
                ).bindparams(
                    bindparam("ids", expanding=True)
                ).params(
                    ids=tuple(int(document.id) for document in documents),
                )
            )
        vectors = provider.embed_documents(
            [str(document.content) for document in documents]
        )
        if vectors is None or len(vectors) != len(documents):
            raise RuntimeError("embedding provider returned an incomplete batch")
        embedded = write_retrieval_vector_embeddings(
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
        embedded_total += embedded
        last_id = int(documents[-1].id)
        with session_scope(engine) as session:
            session.execute(
                text(
                    "UPDATE retrieval_documents SET embedding_status='ready', "
                    "updated_at=:now WHERE id IN :ids"
                ).bindparams(
                    bindparam("ids", expanding=True),
                    now=shanghai_now_naive(),
                ).params(
                    ids=tuple(int(document.id) for document in documents),
                )
            )
        print(f"embedded {embedded_total} documents (last_id={last_id})")
        if args.limit is not None and embedded_total >= args.limit:
            break

    refreshed = refresh_retrieval_vector_generation(
        engine,
        generation=int(generation),
    )
    print(f"refreshed coverage: {refreshed}")
    with session_scope(engine) as session:
        active = session.execute(
            text(
                "SELECT generation FROM retrieval_index_state "
                "WHERE channel='vector' AND is_active=1 LIMIT 1"
            )
        ).scalar_one_or_none()
    activated = activate_retrieval_vector_generation(
        engine,
        generation=int(generation),
        expected_active_generation=int(active) if active is not None else None,
    )
    print(f"activated: {activated}")
    engine.dispose()


if __name__ == "__main__":
    main()
