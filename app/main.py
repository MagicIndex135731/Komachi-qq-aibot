"""Shared factory facades and the compatible group entry command.

Split production processes use app.group_main and app.private_main.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable
from app.adapters.onebot_models import resolve_message_type
from app.config import AppSettings
from app.core.group_image_generation import GroupImageGenerationService
from app.core.memory_background_service import MemoryBackgroundService
from app.core.memory_compaction_service import MemoryCompactionService
from app.core.memory_context_packer import EvidenceMessage
from app.core.message_archive import sync_group_message_archives_from_db
from app.providers.llm_client import LlmClient
from app.providers.semantic_embeddings import build_embedding_provider
from app.providers.web_search import WebSearchClient
from app.storage.db import (
    ensure_retrieval_vector_generation,
    find_active_retrieval_vector_generation,
)
from app.storage.repositories import MessageRepository
from app.runtime import providers as _provider_composition
from app.runtime import memory_composition as _memory_composition
from app.runtime.memory_composition import (
    MemoryRuntimeComposition,
    _DatabaseShadowEvaluator,
    MEMORY_SEGMENTATION_GENERATION,
    MEMORY_COMPACTION_GENERATION,
    MEMORY_CONFIG_GENERATION,
)


def build_web_search_client(settings: AppSettings) -> WebSearchClient | None:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_web_search_client(settings=settings)


def build_image_reference_search_client(settings: AppSettings) -> WebSearchClient | None:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_image_reference_search_client(settings=settings)


def build_usage_recorder(engine):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_usage_recorder(engine=engine)


def resolve_llm_transport_models(*, model: str, fallback_model: str | None) -> tuple[str, str]:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.resolve_llm_transport_models(
        model=model, fallback_model=fallback_model
    )


def resolve_primary_chat_completions_model(*, model: str, fallback_model: str | None) -> str:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.resolve_primary_chat_completions_model(
        model=model, fallback_model=fallback_model
    )


def build_llm_client(*, settings: AppSettings, engine) -> LlmClient:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_llm_client(
        settings=settings, engine=engine, client_factory=LlmClient
    )


def build_proactive_judge_client(*, settings: AppSettings, llm_client, engine=None):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_proactive_judge_client(
        settings=settings, llm_client=llm_client, engine=engine, client_factory=LlmClient
    )


def build_episode_topic_judge_client(*, settings: AppSettings, llm_client, engine=None):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_episode_topic_judge_client(
        settings=settings, llm_client=llm_client, engine=engine, client_factory=LlmClient
    )


def build_episode_post_segment_client(*, settings: AppSettings, llm_client, engine=None):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_episode_post_segment_client(
        settings=settings, llm_client=llm_client, engine=engine, client_factory=LlmClient
    )


def build_memory_compaction_client(*, settings: AppSettings, llm_client, engine=None):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_memory_compaction_client(
        settings=settings, llm_client=llm_client, engine=engine, client_factory=LlmClient
    )


def build_group_image_llm_client(*, settings: AppSettings, engine, llm_client):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_group_image_llm_client(
        settings=settings, engine=engine, llm_client=llm_client, client_factory=LlmClient
    )


def build_group_image_reference_planner_client(*, settings: AppSettings, llm_client):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_group_image_reference_planner_client(
        settings=settings, llm_client=llm_client, client_factory=LlmClient
    )


def build_group_image_service(
    *,
    settings: AppSettings,
    llm_client,
    sender,
    web_search_client=None,
    image_reference_planner_client=None,
) -> GroupImageGenerationService:
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition.build_group_image_service(
        settings=settings,
        llm_client=llm_client,
        sender=sender,
        web_search_client=web_search_client,
        image_reference_planner_client=image_reference_planner_client,
        service_factory=GroupImageGenerationService,
    )


def _build_query_rewrite_provider(*, settings: AppSettings, llm_client, engine=None):
    """Compatibility facade; implementation lives in app.runtime."""
    return _provider_composition._build_query_rewrite_provider(
        settings=settings, llm_client=llm_client, engine=engine, client_factory=LlmClient
    )


def _evidence_messages_from_rows(
    *,
    rows,
    users_by_id: dict[int, object],
    messages: MessageRepository,
    settings: AppSettings,
    bot_display_name: str,
) -> tuple[EvidenceMessage, ...]:
    """Compatibility facade; implementation lives in app.runtime."""
    return _memory_composition._evidence_messages_from_rows(
        rows=rows,
        users_by_id=users_by_id,
        messages=messages,
        settings=settings,
        bot_display_name=bot_display_name,
    )


def _mentioned_uins(raw_json: object) -> tuple[str, ...]:
    """Compatibility facade; implementation lives in app.runtime."""
    return _memory_composition._mentioned_uins(raw_json=raw_json)


def build_memory_compaction_service(
    *,
    settings: AppSettings,
    engine,
    llm_client,
    background_service: MemoryBackgroundService | None = None,
) -> MemoryCompactionService | None:
    """Compatibility facade; implementation lives in app.runtime."""
    return _memory_composition.build_memory_compaction_service(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        background_service=background_service,
    )


def build_memory_runtime(
    *,
    settings: AppSettings,
    engine,
    llm_client,
    bot_display_name: str,
    raw_message_embedding_generation_override: int | None = None,
    evaluation_candidate_filter: Callable[..., tuple[object, ...]] | None = None,
    memory_enabled_group_ids: frozenset[int] | None = None,
) -> MemoryRuntimeComposition:
    """Compatibility facade; implementation lives in app.runtime."""
    return _memory_composition.build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name=bot_display_name,
        raw_message_embedding_generation_override=raw_message_embedding_generation_override,
        evaluation_candidate_filter=evaluation_candidate_filter,
        memory_enabled_group_ids=memory_enabled_group_ids,
        embedding_provider_factory=build_embedding_provider,
        ensure_vector_generation=ensure_retrieval_vector_generation,
        find_vector_generation=find_active_retrieval_vector_generation,
        llm_client_factory=LlmClient,
    )


def create_runtime_banner(*, bot_qq: int, model: str) -> str:
    return f"qq-ai-bot starting with bot={bot_qq} model={model}"


def _group_policy_entry(*, group_id: int, group_policy: dict[str, Any]) -> dict[str, Any]:
    defaults = group_policy.get("default_group_behavior", {})
    configured = group_policy.get("groups", {}).get(str(group_id), {})
    return {**defaults, **configured}


def should_ingest_group_message(*, group_id: int, group_policy: dict[str, Any]) -> bool:
    return should_speak_in_group(group_id=group_id, group_policy=group_policy)


def should_speak_in_group(*, group_id: int, group_policy: dict[str, Any]) -> bool:
    entry = _group_policy_entry(group_id=group_id, group_policy=group_policy)
    return bool(entry.get("enabled", False) and entry.get("speak", False))


def should_archive_group_history(*, group_id: int, group_policy: dict[str, Any]) -> bool:
    entry = _group_policy_entry(group_id=group_id, group_policy=group_policy)
    return bool(
        entry.get("enabled", False) and entry.get("speak", False) and entry.get("archive", False)
    )


def should_enable_memory_in_group(*, group_id: int, group_policy: dict[str, Any]) -> bool:
    entry = _group_policy_entry(group_id=group_id, group_policy=group_policy)
    return bool(entry.get("memory_enabled", False))


def sync_history_archives(engine, runtime) -> dict[int, int]:
    allowed_group_ids = {
        int(group_id)
        for group_id in runtime.group_policy.get("groups", {})
        if should_archive_group_history(group_id=int(group_id), group_policy=runtime.group_policy)
    }
    return sync_group_message_archives_from_db(
        engine=engine,
        history_dir=runtime.settings.data_dir / "history",
        allowed_group_ids=allowed_group_ids,
    )


async def run() -> None:
    """Compatibility command that starts the supported group process only.

    Private chat has a separate composition root (``app.private_main``).
    Keeping this alias avoids silently running the removed combined runtime.
    """
    from app.group_main import run as group_run

    await group_run()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
