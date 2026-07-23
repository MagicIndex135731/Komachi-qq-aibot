from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import app.main as app_main
from app.config import AppSettings
from app.core.memory_background_service import ShadowJobRequest as BackgroundShadowJobRequest
from app.core.memory_context_packer import PackedMemoryContext
from app.core.memory_orchestrator import MemoryContextResult, ShadowJobRequest
from app.main import build_memory_runtime
from app.providers.semantic_embeddings import EmbeddingIdentity
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


class _NoopLlmClient:
    def generate_text(self, _prompt_lines: list[str]) -> str:
        return "{}"


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
    assert runtime.embedding_provider is runtime.background_service.embedder
    assert runtime.memory_compaction_service is not None
    assert runtime.memory_compaction_service.background_service is runtime.background_service
    assert runtime.memory_compaction_service.legacy_enabled is False
    assert runtime.memory_orchestrator.v2_provider is runtime.v2_provider
    assert runtime.memory_orchestrator.legacy_provider == runtime.legacy_provider.build_context
    assert runtime.embedding_generation is None


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
