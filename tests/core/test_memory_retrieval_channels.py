from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.hybrid_memory_retriever import HybridMemoryRetriever
from app.core.memory_query_resolver import ResolvedMemoryQuery, TimeRange
from app.core.memory_retrieval_channels import build_memory_retrieval_channels
from app.providers.semantic_embeddings import EmbeddingIdentity
from app.storage.db import (
    activate_retrieval_vector_generation,
    build_engine,
    create_all,
    ensure_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)


class _FakeEmbeddingProvider:
    def __init__(self, *, dimensions: int = 3) -> None:
        self.identity = EmbeddingIdentity(
            provider="fake",
            model="semantic-test",
            version="v1",
            dimensions=dimensions,
        )
        self.available = True

    def embed_query(self, query: str) -> list[float] | None:
        del query
        return [1.0, *([0.0] * (self.identity.dimensions - 1))]

    def embed_documents(self, documents):
        return [self.embed_query(str(document)) for document in documents]


def _seed_document(
    engine,
    *,
    group_id: int,
    user_id: int,
    platform_msg_id: str,
    content: str,
    embedding_eligible: bool = False,
):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=group_id,
            group_name=f"group-{group_id}",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=user_id,
            nickname=f"user-{user_id}",
            group_card="",
        )
        message = MessageRepository(session).add_group_message(
            platform_msg_id=platform_msg_id,
            group_id=group_id,
            user_id=user_id,
            timestamp=datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
            plain_text=content,
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        document = RetrievalDocumentRepository(session).upsert_document(
            scope_type="group",
            scope_id=str(group_id),
            group_id=group_id,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(message.id),
            start_at=message.timestamp,
            end_at=message.timestamp,
            content=content,
            metadata_json={},
            content_hash=f"hash-{platform_msg_id}",
            source_message_ids=[message.id],
            embedding_eligible=embedding_eligible,
        )
        session.flush()
        return document.id


def test_real_sqlite_parallel_channels_use_independent_short_sessions_and_scope_top_k(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "parallel-channels.db")
    create_all(engine)
    target_id = _seed_document(
        engine,
        group_id=10001,
        user_id=20001,
        platform_msg_id="target",
        content="杭州旅行路线",
    )
    for index in range(4):
        _seed_document(
            engine,
            group_id=10002,
            user_id=21000 + index,
            platform_msg_id=f"other-{index}",
            content=f"杭州旅行路线 杭州旅行路线 {index}",
        )

    created: list[Session] = []
    closed: list[int] = []

    class TrackingSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def close(self) -> None:
            closed.append(id(self))
            super().close()

    factory = sessionmaker(
        bind=engine,
        class_=TrackingSession,
        expire_on_commit=False,
    )
    channels = build_memory_retrieval_channels(session_factory=factory)
    result = HybridMemoryRetriever(
        channels=channels,
        candidate_limit=1,
        final_limit=5,
        channel_timeout_seconds=2.0,
    ).retrieve(
        group_id=10001,
        resolved_query=ResolvedMemoryQuery(
            original_query="杭州旅行",
            retrieval_query="杭州旅行",
            needs_history=True,
        ),
    )

    assert [candidate.document_id for candidate in result.candidates] == [target_id]
    assert all(candidate.group_id == 10001 for candidate in result.candidates)
    assert len(created) == len(channels)
    assert len({id(session) for session in created}) == len(channels)
    assert sorted(closed) == sorted(id(session) for session in created)


def test_vector_channel_keeps_semantic_hit_with_zero_lexical_overlap(tmp_path) -> None:
    engine = build_engine(tmp_path / "semantic-channel.db")
    create_all(engine)
    target_id = _seed_document(
        engine,
        group_id=10001,
        user_id=20001,
        platform_msg_id="semantic-target",
        embedding_eligible=True,
        content="项目代号是青鸟，最终决定迁移到周五",
    )
    other_id = _seed_document(
        engine,
        group_id=10002,
        user_id=20002,
        platform_msg_id="semantic-other",
        embedding_eligible=True,
        content="另一个群的高相似秘密",
    )
    provider = _FakeEmbeddingProvider()
    generation = ensure_retrieval_vector_generation(
        engine,
        provider=provider.identity.provider,
        model=provider.identity.model,
        dimensions=provider.identity.dimensions,
        version=provider.identity.version,
    )
    assert generation is not None
    assert (
        write_retrieval_vector_embeddings(
            engine,
            generation=generation,
            rows=[
                (target_id, 10001, [1.0, 0.0, 0.0]),
                (other_id, 10002, [1.0, 0.0, 0.0]),
            ],
        )
        == 2
    )
    assert (
        refresh_retrieval_vector_generation(
            engine,
            generation=generation,
            mark_ready=True,
        ).status
        == "ready"
    )
    assert activate_retrieval_vector_generation(
        engine,
        generation=generation,
        expected_active_generation=None,
    )
    channels = build_memory_retrieval_channels(
        engine,
        embedding_provider=provider,
    )
    result = HybridMemoryRetriever(
        channels={"bm25": channels["bm25"], "vector": channels["vector"]},
        candidate_limit=1,
        final_limit=5,
        channel_timeout_seconds=2.0,
    ).retrieve(
        group_id=10001,
        resolved_query=ResolvedMemoryQuery(
            original_query="完全无共同词",
            retrieval_query="完全无共同词",
            needs_history=True,
        ),
    )

    assert [candidate.document_id for candidate in result.candidates] == [target_id]
    assert result.candidates[0].routes == ("vector",)


def test_missing_vector_extension_leaves_bm25_channel_available(tmp_path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'fts-only-channels.db'}",
        future=True,
    )
    create_all(engine)
    target_id = _seed_document(
        engine,
        group_id=10001,
        user_id=20001,
        platform_msg_id="fts-only",
        content="杭州旅行路线",
    )
    provider = _FakeEmbeddingProvider()
    assert (
        ensure_retrieval_vector_generation(
            engine,
            provider=provider.identity.provider,
            model=provider.identity.model,
            dimensions=provider.identity.dimensions,
            version=provider.identity.version,
        )
        is None
    )
    channels = build_memory_retrieval_channels(
        engine,
        embedding_provider=provider,
    )
    result = HybridMemoryRetriever(
        channels={"bm25": channels["bm25"], "vector": channels["vector"]},
        candidate_limit=5,
        final_limit=5,
        channel_timeout_seconds=2.0,
    ).retrieve(
        group_id=10001,
        resolved_query=ResolvedMemoryQuery(
            original_query="杭州旅行",
            retrieval_query="杭州旅行",
            needs_history=True,
        ),
    )

    assert [candidate.document_id for candidate in result.candidates] == [target_id]
    assert result.candidates[0].routes == ("bm25",)


def test_vector_sql_failure_is_reported_as_a_failed_channel(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = build_engine(tmp_path / "vector-query-failure.db")
    create_all(engine)
    provider = _FakeEmbeddingProvider()

    def fail_vector_query(self, **_):
        raise SQLAlchemyError("sqlite-vec query failed")

    monkeypatch.setattr(
        RetrievalDocumentRepository,
        "search_group_documents_vector_hits",
        fail_vector_query,
    )
    channels = build_memory_retrieval_channels(
        engine,
        embedding_provider=provider,
    )

    result = HybridMemoryRetriever(
        channels={"vector": channels["vector"]},
        channel_timeout_seconds=2.0,
    ).retrieve(
        group_id=10001,
        resolved_query=ResolvedMemoryQuery(
            original_query="query",
            retrieval_query="query",
            needs_history=True,
        ),
    )

    assert result.candidates == ()
    assert result.failed_channels == ("vector",)
    assert result.channel_candidate_counts == (("vector", 0),)


def test_temporal_entity_fact_exact_and_reply_channels_use_scoped_provenance(
    tmp_path,
) -> None:
    engine = build_engine(tmp_path / "all-scoped-channels.db")
    create_all(engine)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="target",
            enabled=True,
            speak_enabled=True,
        )
        GroupRepository(session).upsert_group(
            group_id=10002,
            group_name="other",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="Alice",
            group_card="爱丽丝",
        )
        UserRepository(session).upsert_user(
            user_id=20002,
            nickname="Alice",
            group_card="爱丽丝",
        )
        messages = MessageRepository(session)
        quoted = messages.add_group_message(
            platform_msg_id="quoted-message",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
            plain_text="Alice 说发布计划改到周五",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        reply = messages.add_group_message(
            platform_msg_id="reply-message",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 23, 8, 1, tzinfo=UTC),
            plain_text="收到，按周五发布",
            raw_json={},
            msg_type="text",
            reply_to_msg_id="quoted-message",
            mentioned_bot=False,
        )
        other = messages.add_group_message(
            platform_msg_id="quoted-message-other-group",
            group_id=10002,
            user_id=20002,
            timestamp=datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
            plain_text="Alice 的另一个群秘密",
            raw_json={},
            msg_type="text",
            reply_to_msg_id="quoted-message",
            mentioned_bot=False,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        quoted_document = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(quoted.id),
            start_at=quoted.timestamp,
            end_at=quoted.timestamp,
            content=quoted.plain_text,
            metadata_json={"speaker": "Alice"},
            content_hash="quoted-document",
            source_message_ids=[quoted.id],
        )
        reply_document = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(reply.id),
            start_at=reply.timestamp,
            end_at=reply.timestamp,
            content=reply.plain_text,
            metadata_json={"speaker": "Alice"},
            content_hash="reply-document",
            source_message_ids=[reply.id],
        )
        documents.upsert_document(
            scope_type="group",
            scope_id="10002",
            group_id=10002,
            episode_id=None,
            document_kind="message",
            source_table="messages",
            source_id=str(other.id),
            start_at=other.timestamp,
            end_at=other.timestamp,
            content=other.plain_text,
            metadata_json={"speaker": "Alice"},
            content_hash="other-document",
            source_message_ids=[other.id],
        )
        memory = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="fact",
            content="Alice 的发布计划是周五",
            importance=5,
            confidence=0.9,
            source_msg_id="quoted-message",
        )
        fact_document = documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="memory",
            source_table="memory_items",
            source_id=str(memory.id),
            start_at=quoted.timestamp,
            end_at=quoted.timestamp,
            content=memory.content,
            metadata_json={"speaker": "Alice"},
            content_hash="fact-document",
            source_message_ids=[quoted.id],
        )
        quoted_document_id = quoted_document.id
        reply_document_id = reply_document.id
        fact_document_id = fact_document.id

    channels = build_memory_retrieval_channels(engine)
    resolved = ResolvedMemoryQuery(
        original_query="Alice 的发布计划",
        retrieval_query="Alice 发布计划",
        entities=("Alice",),
        speaker_ids=("20001",),
        time_range=TimeRange(
            datetime(2026, 7, 23, 0, 0, tzinfo=UTC),
            datetime(2026, 7, 24, 0, 0, tzinfo=UTC),
        ),
        reference_msg_ids=("quoted-message",),
        needs_history=True,
    )

    temporal_ids = {
        item.document_id
        for item in channels["temporal"](
            group_id=10001,
            resolved_query=resolved,
            limit=20,
        )
    }
    entity_ids = {
        item.document_id
        for item in channels["entity"](
            group_id=10001,
            resolved_query=resolved,
            limit=20,
        )
    }
    fact_ids = {
        item.document_id
        for item in channels["fact"](
            group_id=10001,
            resolved_query=resolved,
            limit=20,
        )
    }
    exact_ids = {
        item.document_id
        for item in channels["exact_quote"](
            group_id=10001,
            resolved_query=resolved,
            limit=20,
        )
    }
    reply_ids = {
        item.document_id
        for item in channels["reply_graph"](
            group_id=10001,
            resolved_query=resolved,
            limit=20,
        )
    }

    assert {quoted_document_id, reply_document_id, fact_document_id} <= temporal_ids
    assert quoted_document_id in entity_ids
    assert fact_ids == {fact_document_id}
    assert quoted_document_id in exact_ids
    assert reply_ids == {reply_document_id}
    for channel_ids in (temporal_ids, entity_ids, fact_ids, exact_ids, reply_ids):
        assert channel_ids
