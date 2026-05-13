from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_group_message_event, parse_private_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.group_image_generation import GroupImageGenerationService
from app.core.message_archive import sync_group_message_archives_from_db
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.dev_control.service import DevControlService
from app.providers.llm_client import LlmClient
from app.providers.web_search import WebSearchClient
from app.storage.db import build_engine, create_all
from app.storage.db import session_scope
from app.storage.repositories import UsageRepository


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


def sync_history_archives(engine, runtime) -> dict[int, int]:
    allowed_group_ids = {
        int(group_id)
        for group_id in runtime.group_policy.get("groups", {})
        if should_ingest_group_message(group_id=int(group_id), group_policy=runtime.group_policy)
    }
    return sync_group_message_archives_from_db(
        engine=engine,
        history_dir=runtime.settings.data_dir / "history",
        allowed_group_ids=allowed_group_ids,
    )


def build_web_search_client(settings: AppSettings) -> WebSearchClient | None:
    provider = settings.search_provider.strip().lower()
    if provider != "ddgs" and not settings.search_api_key.strip():
        return None
    return WebSearchClient(
        provider=provider,
        base_url=settings.search_base_url,
        api_key=settings.search_api_key,
        timeout_seconds=settings.search_timeout_seconds,
        region=settings.search_region,
        backend=settings.search_backend,
    )


def build_usage_recorder(engine):
    def recorder(usage) -> None:
        with session_scope(engine) as session:
            UsageRepository(session).add_usage(
                timestamp=usage.timestamp,
                model=usage.model,
                endpoint=usage.endpoint,
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
            )

    return recorder


def resolve_llm_transport_models(*, model: str, fallback_model: str | None) -> tuple[str, str]:
    compat_model = model.strip()
    fallback = (fallback_model or "").strip()
    if fallback and not fallback.startswith("cc-"):
        return fallback, compat_model
    if compat_model and not compat_model.startswith("cc-"):
        return compat_model, compat_model
    return "", compat_model


def build_llm_client(*, settings: AppSettings, engine) -> LlmClient:
    responses_model, compat_model = resolve_llm_transport_models(
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
    )
    return LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_model=responses_model,
        compat_model=compat_model,
        usage_recorder=build_usage_recorder(engine),
    )


def build_group_image_llm_client(*, settings: AppSettings, engine, llm_client):
    if not settings.group_image_base_url.strip() and not settings.group_image_api_key.strip():
        return llm_client
    responses_model, compat_model = resolve_llm_transport_models(
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
    )
    return LlmClient(
        base_url=settings.group_image_base_url.strip() or settings.llm_base_url,
        api_key=settings.group_image_api_key.strip() or settings.llm_api_key,
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
        responses_model=responses_model,
        compat_model=compat_model,
        http_client=httpx.Client(timeout=30.0, trust_env=False),
        usage_recorder=build_usage_recorder(engine),
    )


def build_group_image_service(*, settings: AppSettings, llm_client, sender) -> GroupImageGenerationService:
    return GroupImageGenerationService(
        llm_client=llm_client,
        sender=sender,
        output_dir=settings.data_dir / "generated_images",
        model=settings.group_image_model,
        size=settings.group_image_size,
        quality=settings.group_image_quality,
        background=settings.group_image_background,
        output_format=settings.group_image_output_format,
        output_compression=settings.group_image_output_compression,
        moderation=settings.group_image_moderation,
        max_slots=settings.group_image_queue_capacity,
    )


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)
    sync_history_archives(engine, runtime)

    gateway = NapCatGateway(ws_url=settings.napcat_ws_url)
    sender = Sender(gateway)
    llm_client = build_llm_client(settings=settings, engine=engine)
    group_image_llm_client = build_group_image_llm_client(settings=settings, engine=engine, llm_client=llm_client)
    group_image_service = build_group_image_service(
        settings=settings,
        llm_client=group_image_llm_client,
        sender=sender,
    )
    web_search_client = build_web_search_client(settings)
    dev_control_service = DevControlService(
        engine=engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=settings.owner_qq,
        bot_qq=settings.bot_qq,
        private_chat_qqs=settings.private_chat_whitelist,
        admin_qqs=settings.admin_whitelist,
        repo_root=Path(__file__).resolve().parent.parent,
        data_dir=settings.data_dir,
        assistant_name=str(runtime.persona.get("name", "Codex")),
        persona=runtime.persona,
        safety=runtime.safety,
    )
    await dev_control_service.start()
    router = InboundRouter(
        engine=engine,
        runtime=runtime,
        sender=sender,
        llm_client=llm_client,
        reply_policy=ReplyPolicy(),
        context_builder=ContextBuilder(),
        admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
        web_search_client=web_search_client,
        dev_control_service=dev_control_service,
        group_image_service=group_image_service,
    )

    async def handle_payload(payload: dict) -> None:
        if payload.get("post_type") != "message":
            return

        message_type = payload.get("message_type")
        if message_type == "private":
            event = parse_private_message_event(payload)
            await router.handle_private_message(event)
            return

        if message_type != "group":
            return
        group_id = int(payload["group_id"])
        if not should_ingest_group_message(group_id=group_id, group_policy=runtime.group_policy):
            return

        event = parse_group_message_event(
            payload,
            bot_qq=settings.bot_qq,
            bot_name=str(runtime.persona.get("name", settings.bot_qq)),
        )
        await router.handle_group_message(event)

    logging.info(create_runtime_banner(bot_qq=settings.bot_qq, model=settings.llm_model))
    try:
        await gateway.connect_and_consume(handle_payload)
    finally:
        await dev_control_service.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
