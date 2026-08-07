from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.config import AppSettings
from app.core.memory_context_packer import MEMORY_GROUNDING_WITH_EVIDENCE
from app.main import build_memory_runtime
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)


class _NoopLlmClient:
    def generate_text(self, _prompt_lines: list[str]) -> str:
        return "{}"


def _settings(tmp_path) -> AppSettings:
    return AppSettings.model_construct(
        bot_qq=123456789,
        data_dir=tmp_path / "data",
        memory_compaction_enabled=True,
        memory_orchestration_v2_enabled=True,
        memory_orchestration_shadow_mode=True,
        memory_raw_v3_enabled=True,
        memory_layered_memory_enabled=True,
        memory_embedding_provider="disabled",
        memory_embedding_model="",
        memory_embedding_dimensions=8,
        memory_embedding_cache_dir=tmp_path / "models",
        memory_embedding_base_url="",
        memory_embedding_api_key="",
        memory_embedding_version="test-v1",
    )


PARAPHRASE_VARIANTS = (
    "阿渣喜欢喝什么？",
    "阿渣平时爱喝啥来着？",
    "阿渣的饮品偏好是啥？",
    "喝什么，阿渣喜欢？",
    "阿渣上次说他喜欢喝什么来着？",
    "20001喜欢喝什么？",
)

OPINION_VARIANTS = (
    "阿渣觉得八仙怎么样？",
    "阿渣感觉八仙如何？",
    "阿渣认为八仙咋样？",
    "阿渣怎么看八仙？",
    "阿渣对八仙什么看法？",
    "阿渣如何评价八仙？",
)

FIRST_PERSON_VARIANTS = (
    "我喜欢喝什么？",
    "我喜欢喝什么饮料？",
    "我爱喝什么？",
    "我想喝什么？",
    "我平时喜欢喝什么？",
    "我最喜欢喝什么？",
)

CLAIM_VARIANTS = (
    ("我什么时候说过我喜欢喝豆浆？", False),
    ("我哪句话说过我喜欢喝豆浆？", True),
    ("我自称喜欢喝豆浆是哪条？", True),
)


@pytest.fixture
def seeded(sqlite_engine) -> dict:
    """Seed one fact + one profile for member 20001 and variant queries."""
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        for group_id in (100, 200):
            GroupRepository(session).upsert_group(
                group_id=group_id,
                group_name=f"group-{group_id}",
                enabled=True,
                speak_enabled=True,
            )
        users = UserRepository(session)
        users.upsert_user(user_id=20001, nickname="A-Zha", group_card="阿渣")
        users.upsert_user(user_id=99, nickname="Questioner", group_card="提问者")
        messages = MessageRepository(session)
        fact_source = messages.add_group_message(
            platform_msg_id="para-fact-source",
            group_id=100,
            user_id=20001,
            timestamp=observed_at,
            plain_text="阿渣喜欢喝冰美式。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        query_ids: dict[str, int] = {}
        for index, variant in enumerate(
            (*PARAPHRASE_VARIANTS, *OPINION_VARIANTS)
        ):
            row = messages.add_group_message(
                platform_msg_id=f"para-query-{index}",
                group_id=100,
                user_id=99,
                timestamp=observed_at + timedelta(minutes=index + 1),
                plain_text=variant,
                raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            session.flush()
            query_ids[variant] = int(row.id)
        first_person_query_ids: dict[str, int] = {}
        for index, variant in enumerate(FIRST_PERSON_VARIANTS):
            row = messages.add_group_message(
                platform_msg_id=f"para-firstperson-query-{index}",
                group_id=100,
                user_id=99,
                timestamp=observed_at + timedelta(minutes=index + 1),
                plain_text=variant,
                raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            session.flush()
            first_person_query_ids[variant] = int(row.id)
        claim_query_ids: dict[str, int] = {}
        for index, (variant, _expect_detail) in enumerate(CLAIM_VARIANTS):
            row = messages.add_group_message(
                platform_msg_id=f"para-claim-query-{index}",
                group_id=100,
                user_id=99,
                timestamp=observed_at + timedelta(minutes=index + 1),
                plain_text=variant,
                raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            session.flush()
            claim_query_ids[variant] = int(row.id)
        first_person_source = messages.add_group_message(
            platform_msg_id="para-firstperson-source",
            group_id=100,
            user_id=99,
            timestamp=observed_at,
            plain_text="提问者喜欢喝豆浆。",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        cross_row = messages.add_group_message(
            platform_msg_id="para-cross-query",
            group_id=200,
            user_id=99,
            timestamp=observed_at,
            plain_text="阿渣喜欢喝什么？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        memories = MemoryRepository(session)
        fact = memories.add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣喜欢喝冰美式",
            importance=4,
            confidence=0.9,
            source_msg_id="para-fact-source",
            valid_from=observed_at,
        )
        profile = memories.add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="20001",
            memory_kind="profile",
            content="阿渣是咖啡爱好者",
            importance=3,
            confidence=0.8,
            source_msg_id="para-fact-source",
            valid_from=observed_at,
        )
        first_person_fact = memories.add_memory(
            scope_type="group",
            scope_id="100",
            subject_type="user",
            subject_id="99",
            memory_kind="preference",
            content="提问者喜欢喝豆浆",
            importance=4,
            confidence=0.9,
            source_msg_id="para-firstperson-source",
            valid_from=observed_at,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        for row in messages.list_group_messages_chronological(group_id=100):
            documents.project_raw_message_v3(
                group_id=100,
                message_id=int(row.id),
            )
        documents.project_raw_message_v3(
            group_id=200,
            message_id=int(cross_row.id),
        )
    return {
        "fact_source": "para-fact-source",
        "query_ids": query_ids,
        "first_person_source": "para-firstperson-source",
        "first_person_query_ids": first_person_query_ids,
        "claim_query_ids": claim_query_ids,
        "cross_group_query_id": int(cross_row.id),
        "fact": fact,
        "profile": profile,
        "first_person_fact": first_person_fact,
    }


def test_paraphrase_matrix_recalls_same_source_for_all_variants(
    sqlite_engine,
    tmp_path,
    seeded,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(tmp_path),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    bound_sources: list[set[str]] = []
    ambiguous_count = 0
    for variant in PARAPHRASE_VARIANTS:
        trace = runtime.v2_provider.evaluate(
            runtime.build_request(
                group_id=100,
                message_id=seeded["query_ids"][variant],
            )
        )
        packed = trace.result.packed_context
        subject_ids = trace.resolved_query.subject_ids
        fact_sources = {
            source_id
            for fact in packed.facts
            if "阿渣喜欢喝冰美式" in fact.text
            for source_id in fact.source_msg_ids
        }
        if subject_ids == ():
            ambiguous_count += 1
            assert fact_sources == set()
        elif subject_ids is not None:
            assert fact_sources == {seeded["fact_source"]}
            assert packed.grounding_policy == MEMORY_GROUNDING_WITH_EVIDENCE
            bound_sources.append(fact_sources)
        else:
            assert fact_sources <= {seeded["fact_source"]}
    assert len(bound_sources) >= 2
    assert ambiguous_count >= 0
    assert all(sources == bound_sources[0] for sources in bound_sources)


def test_opinion_variants_bind_member_and_recall_same_source(
    sqlite_engine,
    tmp_path,
    seeded,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(tmp_path),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    for variant in OPINION_VARIANTS:
        trace = runtime.v2_provider.evaluate(
            runtime.build_request(
                group_id=100,
                message_id=seeded["query_ids"][variant],
            )
        )
        assert trace.resolved_query.subject_ids == ("20001",), variant
        fact_sources = {
            source_id
            for fact in trace.result.packed_context.facts
            if "阿渣喜欢喝冰美式" in fact.text
            for source_id in fact.source_msg_ids
        }
        assert fact_sources == {seeded["fact_source"]}, variant


def test_first_person_variants_bind_requester_and_recall_same_source(
    sqlite_engine,
    tmp_path,
    seeded,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(tmp_path),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    for variant in FIRST_PERSON_VARIANTS:
        trace = runtime.v2_provider.evaluate(
            runtime.build_request(
                group_id=100,
                message_id=seeded["first_person_query_ids"][variant],
            )
        )
        assert trace.resolved_query.subject_ids == ("99",), variant
        assert trace.resolved_query.subject_binding == "requester", variant
        assert trace.resolved_query.answer_mode == "current_fact", variant
        fact_sources = {
            source_id
            for fact in trace.result.packed_context.facts
            if "提问者喜欢喝豆浆" in fact.text
            for source_id in fact.source_msg_ids
        }
        assert fact_sources == {seeded["first_person_source"]}, variant
        assert (
            trace.result.packed_context.grounding_policy
            == MEMORY_GROUNDING_WITH_EVIDENCE
        ), variant


def test_first_person_claim_variants_bind_requester_and_recall_same_source(
    sqlite_engine,
    tmp_path,
    seeded,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(tmp_path),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    for variant, expect_detail in CLAIM_VARIANTS:
        trace = runtime.v2_provider.evaluate(
            runtime.build_request(
                group_id=100,
                message_id=seeded["claim_query_ids"][variant],
            )
        )
        assert trace.resolved_query.subject_ids == ("99",), variant
        assert trace.resolved_query.subject_binding == "requester", variant
        assert trace.resolved_query.needs_history is True, variant
        assert trace.resolved_query.needs_detail is expect_detail, variant
        fact_sources = {
            source_id
            for fact in trace.result.packed_context.facts
            if "提问者喜欢喝豆浆" in fact.text
            for source_id in fact.source_msg_ids
        }
        assert fact_sources == {seeded["first_person_source"]}, variant


def test_paraphrase_variant_cross_group_fails_closed(
    sqlite_engine,
    tmp_path,
    seeded,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(tmp_path),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        runtime.build_request(
            group_id=200,
            message_id=seeded["cross_group_query_id"],
        )
    )
    assert all(
        source_id != seeded["fact_source"]
        for fact in trace.result.packed_context.facts
        for source_id in fact.source_msg_ids
    )
    assert all(
        source_id != seeded["fact_source"]
        for source_id in trace.result.selected_source_msg_ids
    )
