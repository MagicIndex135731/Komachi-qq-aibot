from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging
from types import SimpleNamespace

import pytest
import app.main as app_main
from app.config import AppSettings
from app.core.memory_background_service import ShadowJobRequest as BackgroundShadowJobRequest
from app.core.memory_context_packer import PackedMemoryContext
from app.core.memory_orchestrator import MemoryContextResult, ShadowJobRequest
from app.core.memory_v2_context import MemoryV2Request
from app.main import build_memory_runtime
from app.providers.semantic_embeddings import EmbeddingIdentity
from app.storage.db import session_scope
from app.storage.repositories import (
    EpisodeRepository,
    GroupRepository,
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    SummaryRepository,
    UserRepository,
)


class _NoopLlmClient:
    def generate_text(self, _prompt_lines: list[str]) -> str:
        return "{}"


def test_onebot_mentions_are_projected_into_evidence_identity() -> None:
    assert app_main._mentioned_uins(
        {
            "message": [
                {"type": "text", "data": {"text": "hello"}},
                {"type": "at", "data": {"qq": "123"}},
                {"type": "at", "data": {"qq": "all"}},
                {"type": "at", "data": {"qq": 456}},
                {"type": "at", "data": {"uin": "789"}},
                {"type": "at", "data": {"target": "987"}},
                {"type": "at", "data": {"qq": "123"}},
            ]
        }
    ) == ("123", "456", "789", "987")


class _AvailableEmbeddingProvider:
    identity = EmbeddingIdentity(
        provider="fake",
        model="fake-model",
        version="fake-v1",
        dimensions=8,
    )
    available = True

    def embed_query(self, _query: str):
        return [0.0] * 8

    def embed_documents(self, documents):
        return [[0.0] * 8 for _ in documents]


def _settings(tmp_path, *, v2_enabled: bool, shadow_mode: bool, compaction_enabled: bool) -> AppSettings:
    return AppSettings.model_construct(
        bot_qq=123456789,
        data_dir=tmp_path / "data",
        memory_compaction_enabled=compaction_enabled,
        memory_orchestration_v2_enabled=v2_enabled,
        memory_orchestration_shadow_mode=shadow_mode,
        memory_embedding_provider="disabled",
        memory_embedding_model="",
        memory_embedding_dimensions=8,
        memory_embedding_cache_dir=tmp_path / "models",
        memory_embedding_base_url="",
        memory_embedding_api_key="",
        memory_embedding_version="test-v1",
    )


def test_build_memory_runtime_shares_one_lazy_embedding_provider_and_background(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    )

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    assert runtime.background_service is not None
    assert runtime.background_service.idle_minutes == 10
    assert runtime.embedding_provider is runtime.background_service.embedder
    assert runtime.memory_compaction_service is not None
    assert runtime.memory_compaction_service.background_service is runtime.background_service
    assert runtime.memory_compaction_service.legacy_enabled is False
    assert runtime.memory_orchestrator.v2_provider is runtime.v2_provider
    assert runtime.memory_orchestrator.legacy_provider == runtime.legacy_provider.build_context
    assert runtime.embedding_generation is None


def test_raw_v3_activation_switch_controls_strict_provider_and_document_family(
    sqlite_engine,
    tmp_path,
) -> None:
    legacy_settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=False,
        compaction_enabled=True,
    )
    legacy_runtime = build_memory_runtime(
        settings=legacy_settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    assert legacy_runtime.memory_orchestrator.strict_scoped_fallback is False
    assert "fact" in legacy_runtime.v2_provider._retriever.channels

    raw_settings = legacy_settings.model_copy(
        update={"memory_raw_v3_enabled": True}
    )
    raw_runtime = build_memory_runtime(
        settings=raw_settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    assert raw_runtime.memory_orchestrator.strict_scoped_fallback is True
    assert "fact" not in raw_runtime.v2_provider._retriever.channels
    assert raw_runtime.v2_provider._retriever.candidate_limit == 600
    assert raw_runtime.v2_provider._retriever.final_limit == 300
    assert raw_runtime.v2_provider._historical_no_hit_omit_recent is True

    adaptive_runtime = build_memory_runtime(
        settings=raw_settings.model_copy(
            update={"memory_adaptive_context_enabled": True}
        ),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    assert adaptive_runtime.v2_provider._retriever.candidate_limit == 300
    assert adaptive_runtime.v2_provider._retriever.final_limit == 300


def test_build_memory_runtime_passes_resolved_vector_generation_to_background_store(
    sqlite_engine,
    tmp_path,
    monkeypatch,
) -> None:
    provider = _AvailableEmbeddingProvider()
    monkeypatch.setattr(app_main, "build_embedding_provider", lambda **_kwargs: provider)
    monkeypatch.setattr(
        app_main,
        "ensure_retrieval_vector_generation",
        lambda *_args, **_kwargs: 7,
    )

    runtime = build_memory_runtime(
        settings=_settings(
            tmp_path,
            v2_enabled=True,
            shadow_mode=True,
            compaction_enabled=True,
        ),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    assert runtime.embedding_generation == 7
    assert runtime.background_service is not None
    assert runtime.background_service.store.embedding_generation == 7
    assert runtime.background_service.embedder is provider


def test_build_memory_runtime_can_bind_prepared_raw_generation_for_offline_eval(
    sqlite_engine,
    tmp_path,
    monkeypatch,
) -> None:
    provider = _AvailableEmbeddingProvider()
    monkeypatch.setattr(
        app_main,
        "build_embedding_provider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        app_main,
        "find_active_retrieval_vector_generation",
        lambda *_args, **_kwargs: pytest.fail(
            "offline evaluation must not resolve the active generation"
        ),
    )

    runtime = build_memory_runtime(
        settings=_settings(
            tmp_path,
            v2_enabled=True,
            shadow_mode=False,
            compaction_enabled=True,
        ).model_copy(update={"memory_raw_v3_enabled": True}),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
        raw_message_embedding_generation_override=17,
    )

    assert runtime.embedding_generation == 17
    assert runtime.background_service is not None
    assert runtime.background_service.store.raw_message_embedding_generation == 17


def test_production_runtime_does_not_pin_vector_queries_to_startup_generation(
    sqlite_engine,
    tmp_path,
    monkeypatch,
) -> None:
    provider = _AvailableEmbeddingProvider()
    monkeypatch.setattr(
        app_main,
        "build_embedding_provider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        app_main,
        "find_active_retrieval_vector_generation",
        lambda *_args, **_kwargs: 23,
    )

    runtime = build_memory_runtime(
        settings=_settings(
            tmp_path,
            v2_enabled=True,
            shadow_mode=False,
            compaction_enabled=True,
        ).model_copy(update={"memory_raw_v3_enabled": True}),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    vector_channel = runtime.v2_provider._retriever.channels["vector"]
    assert runtime.embedding_generation == 23
    assert vector_channel.__self__._vector_generation is None


def test_disabled_runtime_keeps_v1_and_does_not_construct_group_background_worker(
    sqlite_engine,
    tmp_path,
) -> None:
    runtime = build_memory_runtime(
        settings=_settings(
            tmp_path,
            v2_enabled=False,
            shadow_mode=True,
            compaction_enabled=False,
        ),
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    assert runtime.background_service is None
    assert runtime.memory_compaction_service is None
    assert runtime.memory_orchestrator.v2_enabled is False
    assert runtime.memory_orchestrator.shadow_mode is False


@pytest.mark.asyncio
async def test_shadow_enqueue_translates_platform_id_to_content_free_canonical_job(
    sqlite_engine,
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    )
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=42,
            nickname="user",
            group_card="",
        )
        row = MessageRepository(session).add_group_message(
            platform_msg_id="platform-77",
            group_id=10001,
            user_id=42,
            timestamp=datetime(2026, 7, 23, tzinfo=UTC),
            plain_text="不应进入 shadow payload 的正文",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        canonical_id = int(row.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    captured: list[BackgroundShadowJobRequest] = []
    assert runtime.background_service is not None
    monkeypatch.setattr(
        runtime.background_service,
        "enqueue_shadow",
        lambda request: captured.append(request),
    )
    assert runtime.memory_compaction_service is not None
    await runtime.memory_compaction_service.start()

    await asyncio.to_thread(
        runtime.memory_orchestrator.shadow_enqueue,
        ShadowJobRequest(
            group_id=10001,
            current_msg_id="platform-77",
            config_version="config-v3",
            index_generation="index-v9",
        )
    )
    for _ in range(20):
        if captured:
            break
        await asyncio.sleep(0.01)
    await runtime.memory_compaction_service.stop()

    assert captured == [
        BackgroundShadowJobRequest(
            group_id=10001,
            message_id=canonical_id,
            config_generation="config-v3",
            index_generation="index-v9",
        )
    ]
    assert "正文" not in repr(captured[0])


def test_runtime_request_loader_and_v2_evaluation_use_persisted_ids(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    )
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=42,
            nickname="user",
            group_card="",
        )
        row = MessageRepository(session).add_group_message(
            platform_msg_id="platform-88",
            group_id=10001,
            user_id=42,
            timestamp=datetime(2026, 7, 23, tzinfo=UTC),
            plain_text="测试查询",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        canonical_id = int(row.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    request = runtime.build_request(group_id=10001, message_id=canonical_id)
    trace = runtime.v2_provider.evaluate(request)

    assert request.group_id == 10001
    assert request.target_message_id == "platform-88"
    assert trace.result.group_id == 10001
    assert trace.resolved_query.original_query == "测试查询"


def test_runtime_v2_fact_loader_filters_direct_member_queries_by_resolved_subject(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(update={"memory_raw_v3_enabled": True})
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        users = UserRepository(session)
        users.upsert_user(user_id=42, nickname="A-Zha", group_card="阿渣")
        users.upsert_user(user_id=43, nickname="Garfield", group_card="加菲猫")
        users.upsert_user(user_id=44, nickname="Xiao Ming", group_card="小明")
        users.upsert_user(user_id=99, nickname="Questioner", group_card="提问者")
        messages = MessageRepository(session)

        def add_message(
            *,
            platform_msg_id: str,
            user_id: int,
            plain_text: str,
            nickname: str,
            card: str,
            minutes: int,
        ):
            return messages.add_group_message(
                platform_msg_id=platform_msg_id,
                group_id=10001,
                user_id=user_id,
                timestamp=observed_at + timedelta(minutes=minutes),
                plain_text=plain_text,
                raw_json={"sender": {"nickname": nickname, "card": card}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

        add_message(
            platform_msg_id="azha-fact-source",
            user_id=42,
            plain_text="我最喜欢葫芦兄弟动画。",
            nickname="A-Zha",
            card="阿渣",
            minutes=0,
        )
        add_message(
            platform_msg_id="garfield-member-source",
            user_id=43,
            plain_text="今天聊了动画。",
            nickname="Garfield",
            card="加菲猫",
            minutes=1,
        )
        add_message(
            platform_msg_id="other-fact-source",
            user_id=44,
            plain_text="我最喜欢猫和老鼠动画。",
            nickname="Xiao Ming",
            card="小明",
            minutes=2,
        )
        memories = MemoryRepository(session)
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="阿渣最喜欢葫芦兄弟动画。",
            importance=4,
            confidence=0.9,
            source_msg_id="azha-fact-source",
            valid_from=observed_at,
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="44",
            memory_kind="preference",
            content="小明最喜欢猫和老鼠动画。",
            importance=5,
            confidence=0.95,
            source_msg_id="other-fact-source",
            valid_from=observed_at,
        )
        azha_query = add_message(
            platform_msg_id="query-azha",
            user_id=99,
            plain_text="阿渣最喜欢什么动画？",
            nickname="Questioner",
            card="提问者",
            minutes=3,
        )
        garfield_query = add_message(
            platform_msg_id="query-garfield",
            user_id=99,
            plain_text="加菲猫最喜欢什么动画？",
            nickname="Questioner",
            card="提问者",
            minutes=4,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        for row in messages.list_group_messages_chronological(group_id=10001):
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=int(row.id),
                embedding_generation=None,
            )
        azha_query_id = int(azha_query.id)
        garfield_query_id = int(garfield_query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    azha_trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=azha_query_id)
    )
    garfield_trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=garfield_query_id)
    )

    assert azha_trace.resolved_query.speaker_ids == ("42",)
    assert azha_trace.result.packed_context.facts == ()
    assert "azha-fact-source" in azha_trace.result.selected_source_msg_ids
    assert garfield_trace.resolved_query.speaker_ids == ("43",)
    assert garfield_trace.result.packed_context.facts == ()


def test_runtime_v2_fact_loader_supplements_group_scoped_running_jokes(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(
        update={
            "memory_raw_v3_enabled": True,
            "memory_layered_memory_enabled": True,
        }
    )
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=99, nickname="Questioner", group_card="提问者"
        )
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="group-joke-source",
            group_id=10001,
            user_id=99,
            timestamp=observed_at,
            plain_text="群里的固定梗是周一也叫小周末。",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="group",
            subject_id="group",
            memory_kind="running_joke",
            content="群里的固定梗是周一也叫小周末。",
            importance=4,
            confidence=0.9,
            source_msg_id="group-joke-source",
            valid_from=observed_at,
        )
        query = messages.add_group_message(
            platform_msg_id="group-joke-query",
            group_id=10001,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=1),
            plain_text="群里有什么梗？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        query_id = int(query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=query_id)
    )

    assert trace.resolved_query.subject_role == "group"
    assert any(
        fact.source_msg_ids == ("group-joke-source",)
        for fact in trace.result.packed_context.facts
    )


def test_runtime_v2_fact_loader_distinguishes_ambiguous_and_unbound_member_queries(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(update={"memory_raw_v3_enabled": True})
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        memories = MemoryRepository(session)
        for group_id in (10001, 10002):
            groups.upsert_group(
                group_id=group_id,
                group_name=str(group_id),
                enabled=True,
                speak_enabled=True,
            )

        def seed_member(
            *,
            group_id: int,
            user_id: int,
            card: str,
            platform_msg_id: str,
            content: str,
        ):
            users.upsert_user(user_id=user_id, nickname=f"user-{user_id}", group_card=card)
            return messages.add_group_message(
                platform_msg_id=platform_msg_id,
                group_id=group_id,
                user_id=user_id,
                timestamp=observed_at,
                plain_text=content,
                raw_json={"sender": {"nickname": f"user-{user_id}", "card": card}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

        seed_member(
            group_id=10001,
            user_id=42,
            card="阿渣",
            platform_msg_id="duplicate-azha-42",
            content="我最喜欢葫芦兄弟动画。",
        )
        seed_member(
            group_id=10001,
            user_id=43,
            card="阿渣",
            platform_msg_id="duplicate-azha-43",
            content="同名成员发言。",
        )
        for user_id, card, source_id in (
            (42, "新名甲", "renamed-42"),
            (43, "新名乙", "renamed-43"),
        ):
            messages.add_group_message(
                platform_msg_id=source_id,
                group_id=10001,
                user_id=user_id,
                timestamp=observed_at + timedelta(minutes=1),
                plain_text="改名后的消息",
                raw_json={
                    "sender": {
                        "nickname": f"user-{user_id}",
                        "card": card,
                    }
                },
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="阿渣最喜欢葫芦兄弟动画。",
            importance=4,
            confidence=0.9,
            source_msg_id="duplicate-azha-42",
            valid_from=observed_at,
        )

        multi_messages = []
        multi_memories = []
        for user_id, card, source_id, animation in (
            (52, "阿渣", "multi-azha", "葫芦兄弟"),
            (53, "加菲猫", "multi-garfield", "猫和老鼠"),
        ):
            message = seed_member(
                group_id=10002,
                user_id=user_id,
                card=card,
                platform_msg_id=source_id,
                content=f"我最喜欢{animation}动画。",
            )
            memory = memories.add_memory(
                scope_type="group",
                scope_id="10002",
                subject_type="user",
                subject_id=str(user_id),
                memory_kind="preference",
                content=f"{card}最喜欢{animation}动画。",
                importance=4,
                confidence=0.9,
                source_msg_id=source_id,
                valid_from=observed_at,
            )
            multi_messages.append(message)
            multi_memories.append(memory)
        session.flush()
        episodes = EpisodeRepository(session)
        episode = episodes.create_episode(
            group_id=10002,
            start_message_id=multi_messages[0].id,
            started_at=observed_at,
            segmentation_version="test-v1",
        )
        session.flush()
        for ordinal, message in enumerate(multi_messages):
            episodes.add_message(
                episode_id=episode.id,
                group_id=10002,
                message_id=message.id,
                ordinal=ordinal,
                estimated_tokens=10,
            )
        episodes.close_episode(
            episode_id=episode.id,
            ended_at=observed_at,
            end_message_id=multi_messages[-1].id,
            boundary_reason="test",
            content_hash="runtime-subject-episode",
        )
        documents = RetrievalDocumentRepository(session)
        memory_document_ids: dict[str, int] = {}
        for message, memory in zip(multi_messages, multi_memories, strict=True):
            document = documents.upsert_document(
                scope_type="group",
                scope_id="10002",
                group_id=10002,
                episode_id=episode.id,
                document_kind="memory",
                source_table="memory_items",
                source_id=str(memory.id),
                start_at=observed_at,
                end_at=observed_at,
                content=memory.content,
                metadata_json={"subject_id": memory.subject_id},
                content_hash=f"runtime-memory-{memory.subject_id}",
                source_message_ids=[message.id],
            )
            memory_document_ids[str(memory.subject_id)] = int(document.id)
        episode_document = documents.upsert_document(
            scope_type="group",
            scope_id="10002",
            group_id=10002,
            episode_id=episode.id,
            document_kind="episode",
            source_table="conversation_episodes",
            source_id=str(episode.id),
            start_at=observed_at,
            end_at=observed_at,
            content="阿渣和加菲猫讨论了最喜欢的动画。",
            metadata_json={"episode_id": episode.id},
            content_hash="runtime-episode-document",
            source_message_ids=[message.id for message in multi_messages],
        )
        episode_document_id = int(episode_document.id)
        raw_document_ids: dict[str, int] = {}
        for group_id in (10001, 10002):
            for row in messages.list_group_messages_chronological(group_id=group_id):
                raw_document = documents.project_raw_message_v3(
                    group_id=group_id,
                    message_id=int(row.id),
                    embedding_generation=None,
                )
                if raw_document is not None:
                    raw_document_ids[str(row.platform_msg_id)] = int(raw_document.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )

    def evaluate(group_id: int, query: str):
        return runtime.v2_provider.evaluate(
            MemoryV2Request(
                group_id=group_id,
                query=query,
                recent_messages=(),
                quoted_message=None,
                target_message_id=None,
                available_input=10_000,
                now=observed_at,
            )
        )

    duplicate = evaluate(10001, "阿渣最喜欢什么动画？")
    unique = evaluate(10002, "加菲猫最喜欢什么动画？")
    multiple = evaluate(10002, "阿渣和加菲猫最喜欢什么动画？")
    unknown = evaluate(10002, "陌生猫最喜欢什么动画？")
    first_person = evaluate(10002, "我最喜欢什么动画？")
    second_person = evaluate(10002, "你最喜欢什么动画？")
    first_person_likes = evaluate(10002, "我喜欢什么动画？")
    subjectless = evaluate(10002, "最喜欢什么动画？")
    unbound = evaluate(10002, "动画有什么推荐？")

    assert duplicate.resolved_query.subject_ids == ()
    assert duplicate.result.packed_context.facts == ()
    unique_candidate_ids = {document_id for document_id, _score in unique.candidate_scores}
    assert raw_document_ids["multi-garfield"] in unique_candidate_ids
    assert raw_document_ids["multi-azha"] not in unique_candidate_ids
    assert episode_document_id not in unique_candidate_ids
    assert multiple.resolved_query.subject_ids == ()
    assert multiple.result.packed_context.facts == ()
    assert unknown.resolved_query.subject_ids == ()
    assert unknown.result.packed_context.facts == ()
    assert first_person.resolved_query.subject_ids is None
    assert second_person.resolved_query.subject_ids is None
    assert first_person_likes.resolved_query.subject_ids is None
    assert subjectless.resolved_query.subject_ids is None
    assert first_person.result.packed_context.facts == ()
    assert second_person.result.packed_context.facts == ()
    assert first_person_likes.result.packed_context.facts == ()
    assert subjectless.result.packed_context.facts == ()
    assert unbound.resolved_query.subject_ids is None
    assert unbound.result.packed_context.facts == ()
    for trace in (multiple, unknown):
        candidate_ids = {document_id for document_id, _score in trace.candidate_scores}
        assert candidate_ids == set()
    unbound_candidate_ids = {document_id for document_id, _score in unbound.candidate_scores}
    assert unbound_candidate_ids <= set(raw_document_ids.values())
    unique_segment_ids = {
        int(segment.document_id)
        for segment in unique.result.packed_context.evidence_segments
        if segment.document_id is not None
    }
    assert raw_document_ids["multi-azha"] not in unique_segment_ids
    for trace in (multiple, unknown):
        segment_ids = {
            int(segment.document_id)
            for segment in trace.result.packed_context.evidence_segments
            if segment.document_id is not None
        }
        assert segment_ids == set()


def test_runtime_member_loader_uses_latest_eligible_target_group_snapshot(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(update={"memory_raw_v3_enabled": True})
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        for group_id in (10001, 10002):
            groups.upsert_group(
                group_id=group_id,
                group_name=str(group_id),
                enabled=True,
                speak_enabled=True,
            )
        for user_id in (42, 43, 44, 45, 46):
            users.upsert_user(user_id=user_id, nickname=f"user-{user_id}", group_card="")

        def seed(
            message_id: str,
            *,
            group_id: int,
            user_id: int,
            card: str,
            offset: int,
            delivery_state: str = "",
        ) -> None:
            messages.add_group_message(
                platform_msg_id=message_id,
                group_id=group_id,
                user_id=user_id,
                timestamp=observed_at + timedelta(minutes=offset),
                plain_text="sender snapshot",
                raw_json={
                    "sender": {"nickname": f"user-{user_id}", "card": card},
                    "delivery_state": delivery_state,
                },
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

        seed("eligible-azha", group_id=10001, user_id=42, card="阿渣", offset=0)
        seed(
            "blocked-renamed",
            group_id=10001,
            user_id=42,
            card="污染名",
            offset=3,
            delivery_state="blocked",
        )
        seed(
            "reserved-yesterday",
            group_id=10001,
            user_id=44,
            card="昨天",
            offset=2,
            delivery_state="reserved",
        )
        seed("cross-group-azha", group_id=10002, user_id=43, card="阿渣", offset=1)
        messages.add_private_message(
            platform_msg_id="private-sender-snapshot",
            user_id=46,
            timestamp=observed_at + timedelta(minutes=6),
            plain_text="private sender snapshot",
            raw_json={"sender": {"nickname": "private", "card": "私聊别名"}},
        )
        seed(
            "uncertain-result",
            group_id=10001,
            user_id=45,
            card="结果",
            offset=4,
            delivery_state="uncertain",
        )
        seed(
            "deleted-message",
            group_id=10001,
            user_id=46,
            card="消息",
            offset=5,
            delivery_state="deleted",
        )
        member_snapshots = messages.list_recent_group_member_messages(
            group_id=10001,
            limit=20,
        )
        assert {row.user_id for row in member_snapshots} == {42}
        all_snapshots = messages.list_recent_group_member_messages(
            group_id=None,
            limit=20,
        )
        assert {row.user_id for row in all_snapshots} == {42, 43}

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        MemoryV2Request(
            group_id=10001,
            query="阿渣昨天的劲爆发言",
            recent_messages=(),
            quoted_message=None,
            target_message_id=None,
            available_input=10_000,
            now=observed_at,
        )
    )

    assert trace.resolved_query.speaker_ids == ("42",)
    assert trace.resolved_query.subject_ids == ("42",)

    with session_scope(sqlite_engine) as session:
        MessageRepository(session).add_group_message(
            platform_msg_id="new-cross-group-superstring",
            group_id=10002,
            user_id=43,
            timestamp=observed_at + timedelta(minutes=10),
            plain_text="sender snapshot",
            raw_json={"sender": {"nickname": "user-43", "card": "王阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    refreshed_trace = runtime.v2_provider.evaluate(
        MemoryV2Request(
            group_id=10001,
            query="王阿渣昨天的劲爆发言",
            recent_messages=(),
            quoted_message=None,
            target_message_id=None,
            available_input=10_000,
            now=observed_at,
        )
    )

    assert refreshed_trace.resolved_query.speaker_ids == ()
    assert refreshed_trace.resolved_query.subject_ids == ()


def test_shadow_evaluator_records_rewrite_flag_from_real_v2_trace(
    sqlite_engine,
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    )
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=42,
            nickname="user",
            group_card="",
        )
        row = MessageRepository(session).add_group_message(
            platform_msg_id="platform-99",
            group_id=10001,
            user_id=42,
            timestamp=datetime(2026, 7, 23, tzinfo=UTC),
            plain_text="后来呢",
            raw_json={},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        canonical_id = int(row.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    assert runtime.background_service is not None
    empty_packed = PackedMemoryContext(
        mode="normal",
        budget=100,
        estimated_tokens=0,
        text="",
    )
    monkeypatch.setattr(
        runtime.v2_provider,
        "evaluate",
        lambda _request: SimpleNamespace(
            result=MemoryContextResult(
                group_id=10001,
                packed_context=empty_packed,
                selected_source_msg_ids=(),
                estimated_tokens=0,
                mode="v2",
            ),
            resolved_query=SimpleNamespace(rewrite_used=True),
            candidate_scores=(),
        ),
    )

    evaluation = runtime.background_service.shadow_evaluator.evaluate(
        BackgroundShadowJobRequest(
            group_id=10001,
            message_id=canonical_id,
            config_generation="config-v1",
            index_generation="index-v1",
        )
    )

    assert evaluation.rewrite_used is True


def test_layered_raw_v3_restores_fact_channel_and_logs_runtime_flags(
    sqlite_engine,
    tmp_path,
    caplog,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(
        update={
            "memory_raw_v3_enabled": True,
            "memory_layered_memory_enabled": True,
        }
    )
    with caplog.at_level(logging.INFO):
        runtime = build_memory_runtime(
            settings=settings,
            engine=sqlite_engine,
            llm_client=_NoopLlmClient(),
            bot_display_name="bot",
        )

    assert "fact" in runtime.v2_provider._retriever.channels
    assert runtime.v2_provider._retriever.channels["fact"] is not None
    assert any(
        "layered_enabled=True" in record.message
        and "tools_enabled=False" in record.message
        and "raw_enabled=True" in record.message
        for record in caplog.records
    )


def test_layered_raw_v3_facts_and_summaries_reach_packed_context(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(
        update={
            "memory_raw_v3_enabled": True,
            "memory_layered_memory_enabled": True,
        }
    )
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=42,
            nickname="A-Zha",
            group_card="阿渣",
        )
        UserRepository(session).upsert_user(
            user_id=99,
            nickname="Questioner",
            group_card="提问者",
        )
        messages = MessageRepository(session)
        fact_source = messages.add_group_message(
            platform_msg_id="layered-fact-source",
            group_id=10001,
            user_id=42,
            timestamp=observed_at,
            plain_text="我最喜欢喝冰美式。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        query = messages.add_group_message(
            platform_msg_id="layered-query",
            group_id=10001,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=1),
            plain_text="阿渣以前说过喜欢喝什么？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        memory = MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="阿渣喜欢喝冰美式",
            importance=4,
            confidence=0.9,
            source_msg_id="layered-fact-source",
            valid_from=observed_at,
        )
        SummaryRepository(session).upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="episode",
            summary_key="episode:1:test",
            start_at=observed_at,
            end_at=observed_at + timedelta(hours=1),
            content="阿渣饮品偏好：冰美式",
            source_count=1,
            source_start_msg_id="layered-fact-source",
            source_end_msg_id="layered-fact-source",
        )
        SummaryRepository(session).upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="window",
            summary_key="window:1:test",
            start_at=observed_at,
            end_at=observed_at + timedelta(hours=1),
            content="兼容历史窗口摘要：冰美式",
            source_count=1,
            source_start_msg_id="layered-fact-source",
            source_end_msg_id="layered-fact-source",
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        documents.upsert_document(
            scope_type="group",
            scope_id="10001",
            group_id=10001,
            episode_id=None,
            document_kind="memory",
            source_table="memory_items",
            source_id=str(memory.id),
            start_at=observed_at,
            end_at=observed_at,
            content="阿渣喜欢喝冰美式",
            metadata_json={"subject_id": "42", "kind": "preference"},
            content_hash="hash-layered-memory",
            source_message_ids=[int(fact_source.id)],
        )
        for row in messages.list_group_messages_chronological(group_id=10001):
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=int(row.id),
                embedding_generation=None,
            )
        query_id = int(query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=query_id)
    )
    packed = trace.result.packed_context

    assert any(
        "阿渣喜欢喝冰美式" in fact.text
        and "layered-fact-source" in fact.source_msg_ids
        for fact in packed.facts
    )
    assert any(
        "阿渣饮品偏好" in summary.text
        and "layered-fact-source" in summary.source_msg_ids
        for summary in packed.summaries
    )
    assert any(
        "兼容历史窗口摘要" in summary.text
        and "layered-fact-source" in summary.source_msg_ids
        for summary in packed.summaries
    )
    assert "layered-fact-source" in trace.result.selected_source_msg_ids


def test_raw_v3_without_layered_keeps_facts_and_summaries_empty(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(update={"memory_raw_v3_enabled": True})
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=42,
            nickname="A-Zha",
            group_card="阿渣",
        )
        UserRepository(session).upsert_user(
            user_id=99,
            nickname="Questioner",
            group_card="提问者",
        )
        messages = MessageRepository(session)
        fact_source = messages.add_group_message(
            platform_msg_id="non-layered-fact-source",
            group_id=10001,
            user_id=42,
            timestamp=observed_at,
            plain_text="我最喜欢喝冰美式。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        query = messages.add_group_message(
            platform_msg_id="non-layered-query",
            group_id=10001,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=1),
            plain_text="阿渣以前说过喜欢喝什么？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        MemoryRepository(session).add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="42",
            memory_kind="preference",
            content="阿渣喜欢喝冰美式",
            importance=4,
            confidence=0.9,
            source_msg_id="non-layered-fact-source",
            valid_from=observed_at,
        )
        SummaryRepository(session).upsert_summary(
            scope_type="group",
            scope_id="10001",
            summary_level="episode",
            summary_key="episode:2:test",
            start_at=observed_at,
            end_at=observed_at + timedelta(hours=1),
            content="阿渣饮品偏好：冰美式",
            source_count=1,
            source_start_msg_id="non-layered-fact-source",
            source_end_msg_id="non-layered-fact-source",
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        for row in messages.list_group_messages_chronological(group_id=10001):
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=int(row.id),
                embedding_generation=None,
            )
        query_id = int(query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=query_id)
    )
    packed = trace.result.packed_context

    assert packed.facts == ()
    assert packed.summaries == ()


def test_member_fact_supplement_prefers_query_relevant_facts(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(
        update={
            "memory_raw_v3_enabled": True,
            "memory_layered_memory_enabled": True,
        }
    )
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="A-Zha",
            group_card="阿渣",
        )
        UserRepository(session).upsert_user(
            user_id=99,
            nickname="Questioner",
            group_card="提问者",
        )
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="unrelated-fact-src",
            group_id=10001,
            user_id=20001,
            timestamp=observed_at,
            plain_text="我在给 agent 做前后端。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="anime-fact-src",
            group_id=10001,
            user_id=20001,
            timestamp=observed_at + timedelta(minutes=1),
            plain_text="我一直在看海贼王。",
            raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        query = messages.add_group_message(
            platform_msg_id="anime-query",
            group_id=10001,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=2),
            plain_text="阿渣喜欢什么动画？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        memories = MemoryRepository(session)
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="profile",
            content="阿渣在给 agent 做前后端",
            importance=5,
            confidence=0.95,
            source_msg_id="unrelated-fact-src",
            valid_from=observed_at,
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣一直在看海贼王动画",
            importance=2,
            confidence=0.8,
            source_msg_id="anime-fact-src",
            valid_from=observed_at,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        for row in messages.list_group_messages_chronological(group_id=10001):
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=int(row.id),
            )
        query_id = int(query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=query_id)
    )
    packed = trace.result.packed_context
    texts = [fact.text for fact in packed.facts]
    assert any("海贼王" in text for text in texts)
    assert any("前后端" in text for text in texts)
    assert texts.index(next(t for t in texts if "海贼王" in t)) < texts.index(
        next(t for t in texts if "前后端" in t)
    )
    anime_fact = next(fact for fact in packed.facts if "海贼王" in fact.text)
    assert anime_fact.memory_kind == "preference"
    assert anime_fact.observed_at is not None
    assert "kind: preference; observed_at:" in packed.text


def test_preference_question_prefers_preference_kind_over_current(
    sqlite_engine,
    tmp_path,
) -> None:
    settings = _settings(
        tmp_path,
        v2_enabled=True,
        shadow_mode=True,
        compaction_enabled=True,
    ).model_copy(
        update={
            "memory_raw_v3_enabled": True,
            "memory_layered_memory_enabled": True,
        }
    )
    observed_at = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001,
            group_name="group",
            enabled=True,
            speak_enabled=True,
        )
        UserRepository(session).upsert_user(
            user_id=20001,
            nickname="A-Zha",
            group_card="阿渣",
        )
        UserRepository(session).upsert_user(
            user_id=99,
            nickname="Questioner",
            group_card="提问者",
        )
        messages = MessageRepository(session)
        for platform_id, user_id, text, minutes in (
            ("current-src", 20001, "我在做前后端。", 0),
            ("anime-src", 20001, "我一直在看海贼王。", 1),
            ("matched-src", 20001, "我喜欢看动画。", 2),
        ):
            messages.add_group_message(
                platform_msg_id=platform_id,
                group_id=10001,
                user_id=user_id,
                timestamp=observed_at + timedelta(minutes=minutes),
                plain_text=text,
                raw_json={"sender": {"nickname": "A-Zha", "card": "阿渣"}},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        query = messages.add_group_message(
            platform_msg_id="pref-query",
            group_id=10001,
            user_id=99,
            timestamp=observed_at + timedelta(minutes=3),
            plain_text="阿渣喜欢什么动画？",
            raw_json={"sender": {"nickname": "Questioner", "card": "提问者"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.flush()
        memories = MemoryRepository(session)
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="current",
            content="阿渣在给agent做前后端",
            importance=5,
            confidence=0.95,
            source_msg_id="current-src",
            valid_from=observed_at,
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣一直在看海贼王",
            importance=2,
            confidence=0.8,
            source_msg_id="anime-src",
            valid_from=observed_at,
        )
        memories.add_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="20001",
            memory_kind="preference",
            content="阿渣喜欢看动画",
            importance=1,
            confidence=0.7,
            source_msg_id="matched-src",
            valid_from=observed_at,
        )
        session.flush()
        documents = RetrievalDocumentRepository(session)
        for row in messages.list_group_messages_chronological(group_id=10001):
            documents.project_raw_message_v3(
                group_id=10001,
                message_id=int(row.id),
            )
        query_id = int(query.id)

    runtime = build_memory_runtime(
        settings=settings,
        engine=sqlite_engine,
        llm_client=_NoopLlmClient(),
        bot_display_name="bot",
    )
    trace = runtime.v2_provider.evaluate(
        runtime.build_request(group_id=10001, message_id=query_id)
    )
    texts = [fact.text for fact in trace.result.packed_context.facts]
    assert any("喜欢看动画" in text for text in texts)
    assert any("海贼王" in text for text in texts)
    assert any("前后端" in text for text in texts)
    assert texts.index(next(t for t in texts if "海贼王" in t)) < texts.index(
        next(t for t in texts if "前后端" in t)
    )
