from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_group_message_event, parse_private_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.group_image_generation import GroupImageGenerationService
from app.core.group_history_backfill import backfill_recent_group_history
from app.core.message_archive import sync_group_message_archives_from_db
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.dev_control.service import DevControlService
from app.providers.llm_client import LlmClient
from app.providers.web_search import WebSearchClient
from app.runtime_heartbeat import RuntimeHeartbeat
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


def should_archive_group_history(*, group_id: int, group_policy: dict[str, Any]) -> bool:
    entry = _group_policy_entry(group_id=group_id, group_policy=group_policy)
    return bool(entry.get("enabled", False) and entry.get("speak", False) and entry.get("archive", False))


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


def build_web_search_client(settings: AppSettings) -> WebSearchClient | None:
    if settings.llm_builtin_web_search and settings.llm_text_endpoint == "responses":
        return None
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


def resolve_primary_chat_completions_model(*, model: str, fallback_model: str | None) -> str:
    del fallback_model
    compat_model = model.strip()
    if compat_model.startswith("cc-"):
        stripped = compat_model[3:].strip()
        if stripped:
            return stripped
    return compat_model


def build_llm_client(*, settings: AppSettings, engine) -> LlmClient:
    chat_model = resolve_primary_chat_completions_model(
        model=settings.llm_model,
        fallback_model=settings.llm_fallback_model,
    )
    fallback_model = (settings.llm_fallback_model or "").strip()
    if fallback_model == chat_model:
        fallback_model = ""
    responses_model = chat_model if settings.llm_text_endpoint == "responses" else ""
    tool_event_log = settings.log_dir / "responses-tool-events.jsonl"

    def record_tool_event(event: dict) -> None:
        payload = {"timestamp": datetime.now().astimezone().isoformat(), **event}
        with tool_event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    return LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=chat_model,
        fallback_model=fallback_model,
        vision_model=(settings.llm_vision_model or "").strip(),
        responses_model=responses_model,
        image_responses_model=chat_model,
        compat_model=chat_model,
        builtin_web_search=settings.llm_builtin_web_search and settings.llm_text_endpoint == "responses",
        web_search_context_size=settings.llm_builtin_web_search_context_size,
        reasoning_effort=settings.llm_reasoning_effort if settings.llm_text_endpoint == "responses" else "",
        max_output_tokens=settings.llm_max_output_tokens,
        usage_recorder=build_usage_recorder(engine),
        tool_event_recorder=record_tool_event,
    )


def build_group_image_llm_client(*, settings: AppSettings, engine, llm_client):
    del llm_client
    required = {
        "GROUP_IMAGE_BASE_URL": settings.group_image_base_url.strip(),
        "GROUP_IMAGE_API_KEY": settings.group_image_api_key.strip(),
        "GROUP_IMAGE_MODEL": settings.group_image_model.strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Dedicated image generation configuration is missing: {', '.join(missing)}")

    return LlmClient(
        base_url=required["GROUP_IMAGE_BASE_URL"],
        api_key=required["GROUP_IMAGE_API_KEY"],
        model=required["GROUP_IMAGE_MODEL"],
        responses_model="",
        image_responses_model="",
        compat_model=required["GROUP_IMAGE_MODEL"],
        image_generations_endpoint=settings.group_image_generations_endpoint,
        image_edits_endpoint=settings.group_image_edits_endpoint,
        max_output_tokens=settings.llm_max_output_tokens,
        usage_recorder=build_usage_recorder(engine),
    )


def build_group_image_service(
    *,
    settings: AppSettings,
    llm_client,
    sender,
    web_search_client=None,
) -> GroupImageGenerationService:
    return GroupImageGenerationService(
        llm_client=llm_client,
        sender=sender,
        web_search_client=web_search_client,
        output_dir=settings.data_dir / "generated_images",
        model=settings.group_image_model,
        size=settings.group_image_size,
        quality=settings.group_image_quality,
        background=None,
        output_format=settings.group_image_output_format,
        output_compression=None,
        moderation=None,
        max_slots=settings.group_image_queue_capacity,
        image_max_attempts=1,
        image_timeout_seconds=settings.group_image_timeout_seconds,
    )


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "app.heartbeat.json")
    group_image_service = None
    dev_control_service = None
    try:
        await heartbeat.start()
        engine = await asyncio.to_thread(build_engine, settings.sqlite_path)
        await asyncio.to_thread(create_all, engine)
        await asyncio.to_thread(sync_history_archives, engine, runtime)

        gateway = NapCatGateway(ws_url=settings.napcat_ws_url, reconnect_forever=True)
        sender = Sender(gateway)
        llm_client = build_llm_client(settings=settings, engine=engine)
        group_image_llm_client = build_group_image_llm_client(settings=settings, engine=engine, llm_client=llm_client)
        web_search_client = build_web_search_client(settings)
        group_image_service = build_group_image_service(
            settings=settings,
            llm_client=group_image_llm_client,
            sender=sender,
            web_search_client=web_search_client,
        )
        persistent_group_engine = engine if hasattr(engine, "connect") else None
        if hasattr(group_image_service, "engine") and getattr(group_image_service, "engine", None) is None:
            group_image_service.engine = persistent_group_engine
        if hasattr(group_image_service, "start") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.start()
        dev_control_service = DevControlService(
            engine=engine,
            sender=sender,
            llm_client=llm_client,
            image_llm_client=group_image_llm_client,
            owner_qq=settings.owner_qq,
            bot_qq=settings.bot_qq,
            private_chat_qqs=settings.private_chat_whitelist,
            admin_qqs=settings.admin_whitelist,
            repo_root=Path(__file__).resolve().parent.parent,
            data_dir=settings.data_dir,
            web_search_client=web_search_client,
            image_model=settings.llm_model,
            image_size="auto",
            image_quality="high",
            image_background=None,
            image_output_format="png",
            image_output_compression=None,
            image_moderation=None,
            image_queue_capacity=settings.group_image_queue_capacity,
            image_max_attempts=1,
            image_timeout_seconds=settings.group_image_timeout_seconds,
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

        async def backfill_group_history_on_connect() -> None:
            await backfill_recent_group_history(
                router=router,
                gateway=gateway,
                bot_qq=settings.bot_qq,
                bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            )

        logging.info(create_runtime_banner(bot_qq=settings.bot_qq, model=settings.llm_model))
        await gateway.connect_and_consume(handle_payload, on_connect=backfill_group_history_on_connect)
    finally:
        if group_image_service is not None and hasattr(group_image_service, "stop") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.stop()
        if dev_control_service is not None:
            await dev_control_service.stop()
        await heartbeat.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
