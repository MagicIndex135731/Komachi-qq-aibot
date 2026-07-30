from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from app.group_main import _handle_group_recall_payload
from app.storage.db import (
    activate_retrieval_vector_generation,
    create_retrieval_vector_generation,
    refresh_retrieval_vector_generation,
    session_scope,
    write_retrieval_vector_embeddings,
)
from app.storage.models import Message, RetrievalDocument
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)


@pytest.mark.asyncio
async def test_group_recall_marks_source_deleted_and_revokes_existing_projection(
    sqlite_engine,
) -> None:
    generation = create_retrieval_vector_generation(
        sqlite_engine,
        provider="fake",
        model="recall-test",
        dimensions=2,
        version="v1",
        document_family="raw_message_v3",
    )
    assert generation is not None
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="test",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="member",
            group_card="member",
        )
        message = MessageRepository(session).add_group_message(
            platform_msg_id="991122",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 7, 30, tzinfo=UTC),
            plain_text="recalled evidence must disappear",
            raw_json={"post_type": "message"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        projected = RetrievalDocumentRepository(session).project_raw_message_v3(
            group_id=10001,
            message_id=message.id,
            embedding_generation=generation,
        )
        assert projected is not None
        internal_message_id = int(message.id)
        document_id = int(projected.id)
    assert (
        write_retrieval_vector_embeddings(
            sqlite_engine,
            generation=generation,
            rows=[(document_id, 10001, [1.0, 0.0])],
        )
        == 1
    )
    assert (
        refresh_retrieval_vector_generation(
            sqlite_engine,
            generation=generation,
            mark_ready=True,
        ).status
        == "ready"
    )
    assert activate_retrieval_vector_generation(
        sqlite_engine,
        generation=generation,
        expected_active_generation=None,
    )

    handled = await _handle_group_recall_payload(
        {
            "post_type": "notice",
            "notice_type": "group_recall",
            "group_id": 10001,
            "message_id": 991122,
            "user_id": 20001,
            "operator_id": 20001,
        },
        engine=sqlite_engine,
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        message = session.get(Message, internal_message_id)
        assert message is not None
        assert message.raw_json["delivery_state"] == "deleted"
        assert message.raw_json["deletion_reason"] == "group_recall"
        projected = session.scalars(
            select(RetrievalDocument).where(
                RetrievalDocument.document_kind == "raw_message_v3",
                RetrievalDocument.source_id == str(internal_message_id),
            )
        ).one()
        assert projected.status == "inactive"
        assert projected.embedding_status == "stale"
        assert (
            RetrievalDocumentRepository(session).search_group_documents_fts_hits(
                group_id=10001,
                query="recalled evidence",
                limit=10,
                document_kinds=("raw_message_v3",),
            )
            == []
        )
        assert (
            int(
                session.scalar(
                    text(
                        f"SELECT count(*) FROM "
                        f"retrieval_documents_vec_g{generation}"
                    )
                )
                or 0
            )
            == 0
        )


@pytest.mark.asyncio
async def test_non_recall_payload_is_not_consumed(sqlite_engine) -> None:
    assert (
        await _handle_group_recall_payload(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": 10001,
                "message_id": 1,
            },
            engine=sqlite_engine,
        )
        is False
    )
