"""Concrete, short-session adapters for V2 local retrieval channels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.hybrid_memory_retriever import (
    MemoryScopeViolation,
    RetrievalCandidate,
    RetrievalChannel,
)
from app.providers.semantic_embeddings import EmbeddingProvider
from app.storage.repositories import (
    RetrievalDocumentHit,
    RetrievalDocumentRepository,
)


SessionFactory = Callable[[], AbstractContextManager[Session]]


class ScopedMemoryRetrievalChannels:
    """Own concrete SQL-backed channels without sharing Sessions across threads."""

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: SessionFactory | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if session_factory is None:
            if engine is None:
                raise ValueError("engine or session_factory is required")
            maker = sessionmaker(
                bind=engine,
                class_=Session,
                expire_on_commit=False,
            )
            session_factory = maker
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider

    def as_mapping(self) -> Mapping[str, RetrievalChannel]:
        return {
            "bm25": self.bm25,
            "vector": self.vector,
            "temporal": self.temporal,
            "entity": self.entity,
            "fact": self.fact,
            "reply_graph": self.reply_graph,
            "exact_quote": self.exact_quote,
        }

    def bm25(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            hits = RetrievalDocumentRepository(
                session
            ).search_group_documents_fts_hits(
                group_id=group_id,
                query=self._query_text(resolved_query),
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def vector(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        provider = self._embedding_provider
        embedding = (
            provider.embed_query(self._query_text(resolved_query))
            if provider is not None and provider.available
            else None
        )
        with self._session_factory() as session:
            if provider is None or embedding is None:
                return ()
            identity = provider.identity
            hits = RetrievalDocumentRepository(
                session
            ).search_group_documents_vector_hits(
                group_id=group_id,
                embedding=embedding,
                provider=identity.provider,
                model=identity.model,
                dimensions=identity.dimensions,
                version=identity.version,
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def temporal(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            time_range = getattr(resolved_query, "time_range", None)
            hits = RetrievalDocumentRepository(
                session
            ).search_group_documents_temporal_hits(
                group_id=group_id,
                start_at=getattr(time_range, "start", None),
                end_at=getattr(time_range, "end", None),
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def entity(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            hits = RetrievalDocumentRepository(
                session
            ).search_group_documents_entity_hits(
                group_id=group_id,
                entities=self._string_tuple(
                    getattr(resolved_query, "entities", ())
                ),
                speaker_ids=self._string_tuple(
                    getattr(resolved_query, "speaker_ids", ())
                ),
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def fact(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            hits = RetrievalDocumentRepository(session).search_group_fact_hits(
                group_id=group_id,
                query=self._query_text(resolved_query),
                entities=self._string_tuple(
                    getattr(resolved_query, "entities", ())
                ),
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def reply_graph(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            hits = RetrievalDocumentRepository(
                session
            ).search_group_reference_hits(
                group_id=group_id,
                reference_msg_ids=self._string_tuple(
                    getattr(resolved_query, "reference_msg_ids", ())
                ),
                include_replies=True,
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    def exact_quote(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        with self._session_factory() as session:
            hits = RetrievalDocumentRepository(
                session
            ).search_group_reference_hits(
                group_id=group_id,
                reference_msg_ids=self._string_tuple(
                    getattr(resolved_query, "reference_msg_ids", ())
                ),
                include_replies=False,
                limit=limit,
            )
            return self._adapt(group_id=group_id, hits=hits)

    @staticmethod
    def _adapt(
        *,
        group_id: int,
        hits: Sequence[RetrievalDocumentHit],
    ) -> tuple[RetrievalCandidate, ...]:
        candidates: list[RetrievalCandidate] = []
        for hit in hits:
            if int(hit.group_id) != int(group_id) or not hit.source_msg_ids:
                raise MemoryScopeViolation(
                    f"unverified scoped retrieval document_id={hit.document_id}"
                )
            candidates.append(
                RetrievalCandidate(
                    document_id=hit.document_id,
                    group_id=hit.group_id,
                    document_kind=hit.document_kind,
                    episode_id=hit.episode_id,
                    source_msg_ids=hit.source_msg_ids,
                    start_at=hit.start_at,
                    end_at=hit.end_at,
                    channel_score=hit.score,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _query_text(resolved_query: Any) -> str:
        for attribute in (
            "retrieval_query",
            "resolved_query",
            "parsed_query",
            "original_query",
        ):
            value = getattr(resolved_query, attribute, None)
            if isinstance(value, str):
                return value
        return str(resolved_query or "")

    @staticmethod
    def _string_tuple(values: object) -> tuple[str, ...]:
        if not isinstance(values, (tuple, list)):
            return ()
        return tuple(
            dict.fromkeys(
                str(value).strip()
                for value in values
                if str(value).strip()
            )
        )


def build_memory_retrieval_channels(
    engine: Engine | None = None,
    *,
    session_factory: SessionFactory | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> Mapping[str, RetrievalChannel]:
    return ScopedMemoryRetrievalChannels(
        engine=engine,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
    ).as_mapping()
