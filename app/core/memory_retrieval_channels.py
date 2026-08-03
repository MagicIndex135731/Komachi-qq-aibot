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
        vector_generation: int | None = None,
        raw_message_v3_only: bool = False,
        legacy_v2_only: bool = False,
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
        self._vector_generation = (
            int(vector_generation) if vector_generation is not None else None
        )
        self._raw_message_v3_only = bool(raw_message_v3_only)
        self._legacy_v2_only = bool(legacy_v2_only)
        if self._raw_message_v3_only and self._legacy_v2_only:
            raise ValueError("retrieval document families are mutually exclusive")

    def as_mapping(self) -> Mapping[str, RetrievalChannel]:
        channels = {
            "bm25": self.bm25,
            "vector": self.vector,
            "temporal": self.temporal,
            "entity": self.entity,
            "reply_graph": self.reply_graph,
            "exact_quote": self.exact_quote,
        }
        if not self._raw_message_v3_only:
            channels["fact"] = self.fact
        return channels

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
                subject_ids=self._subject_ids(resolved_query),
                **self._hard_filters(resolved_query),
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
        with self._session_factory() as session:
            if provider is None or not provider.available:
                raise RuntimeError("vector embedding provider unavailable")
            embedding = provider.embed_query(self._query_text(resolved_query))
            if embedding is None:
                raise RuntimeError(
                    "vector embedding provider returned no embedding"
                )
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
                generation=self._vector_generation,
                limit=limit,
                subject_ids=self._subject_ids(resolved_query),
                **self._hard_filters(resolved_query),
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
                subject_ids=self._subject_ids(resolved_query),
                document_kinds=self._document_kinds(),
                speaker_ids=self._speaker_filter_ids(resolved_query),
                mentioned_user_ids=self._mentioned_user_ids(resolved_query),
                allow_unbounded=(
                    getattr(resolved_query, "coverage_mode", "relevance")
                    == "time_buckets"
                    or getattr(resolved_query, "answer_mode", "")
                    == "mention"
                ),
                sample_time_coverage=(
                    getattr(resolved_query, "coverage_mode", "relevance")
                    == "time_buckets"
                ),
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
                speaker_ids=self._speaker_filter_ids(resolved_query),
                limit=limit,
                subject_ids=self._subject_ids(resolved_query),
                document_kinds=self._document_kinds(),
                start_at=self._time_bound(resolved_query, "start"),
                end_at=self._time_bound(resolved_query, "end"),
                mentioned_user_ids=self._mentioned_user_ids(resolved_query),
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
                subject_ids=self._subject_ids(resolved_query),
                **self._hard_filters(resolved_query),
            )
            return self._adapt(group_id=group_id, hits=hits)

    def reply_graph(
        self,
        *,
        group_id: int,
        resolved_query: Any,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        hard_filters = self._hard_filters(resolved_query)
        # The reply author is not required to be the quoted message author.
        # Keeping the quoted speaker filter here silently drops normal
        # cross-speaker replies from deterministic reference retrieval.
        hard_filters["speaker_ids"] = None
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
                subject_ids=self._subject_ids(resolved_query),
                **hard_filters,
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
                subject_ids=self._subject_ids(resolved_query),
                **self._hard_filters(resolved_query),
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

    @classmethod
    def _subject_ids(cls, resolved_query: Any) -> tuple[str, ...] | None:
        values = getattr(resolved_query, "subject_ids", None)
        return None if values is None else cls._string_tuple(values)

    def _hard_filters(self, resolved_query: Any) -> dict[str, object]:
        return {
            "document_kinds": self._document_kinds(),
            "start_at": self._time_bound(resolved_query, "start"),
            "end_at": self._time_bound(resolved_query, "end"),
            "speaker_ids": self._speaker_filter_ids(resolved_query),
            "mentioned_user_ids": self._mentioned_user_ids(resolved_query),
        }

    def _document_kinds(self) -> tuple[str, ...] | None:
        if self._raw_message_v3_only:
            return ("raw_message_v3",)
        if self._legacy_v2_only:
            return ("episode", "episode_summary", "memory")
        return None

    @staticmethod
    def _time_bound(resolved_query: Any, name: str):
        time_range = getattr(resolved_query, "time_range", None)
        return getattr(time_range, name, None)

    def _speaker_filter_ids(self, resolved_query: Any) -> tuple[str, ...] | None:
        if getattr(resolved_query, "answer_mode", "") == "mention":
            return None
        subject_ids = self._subject_ids(resolved_query)
        if self._raw_message_v3_only and subject_ids is not None:
            return subject_ids
        speaker_ids = self._string_tuple(getattr(resolved_query, "speaker_ids", ()))
        return speaker_ids or None

    @classmethod
    def _mentioned_user_ids(cls, resolved_query: Any) -> tuple[str, ...] | None:
        if getattr(resolved_query, "answer_mode", "") != "mention":
            return None
        return cls._subject_ids(resolved_query)


def build_memory_retrieval_channels(
    engine: Engine | None = None,
    *,
    session_factory: SessionFactory | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_generation: int | None = None,
    raw_message_v3_only: bool = False,
    legacy_v2_only: bool = False,
) -> Mapping[str, RetrievalChannel]:
    return ScopedMemoryRetrievalChannels(
        engine=engine,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        vector_generation=vector_generation,
        raw_message_v3_only=raw_message_v3_only,
        legacy_v2_only=legacy_v2_only,
    ).as_mapping()
