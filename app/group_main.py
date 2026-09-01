from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path

from sqlalchemy import bindparam, text

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_group_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.time_utils import ASIA_SHANGHAI
from app.core.context_builder import ContextBuilder
from app.core.group_history_backfill import backfill_recent_group_history
from app.core.member_memory_backfill import MemberFactRefreshService
from app.core.persona_live_sync import PersonaLiveSyncService
from app.core.persona_switch import PersonaManager, PersonaSwitchService
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.main import (
    build_group_image_llm_client,
    build_group_image_reference_planner_client,
    build_group_image_service,
    build_llm_client,
    build_memory_runtime,
    build_proactive_judge_client,
    build_web_search_client,
    create_runtime_banner,
    should_enable_memory_in_group,
    should_ingest_group_message,
    sync_history_archives,
)
from app.providers.semantic_embeddings import EmbeddingProvider
from app.runtime_heartbeat import RuntimeHeartbeat
from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import MessageRepository, RetrievalDocumentRepository


def _write_group_ready_marker(*, log_dir: Path, state: str) -> None:
    """Persist gateway readiness for external start/status scripts.

    ``connected`` is written as soon as the OneBot websocket is up (the bot is
    already consuming messages at that point); ``ready`` is written after the
    startup backfill and startup-window mention replay complete. The marker is
    refreshed on every reconnect, so start/status scripts can treat a fresh
    marker as proof the bot is accepting messages.
    """
    payload = {
        "pid": os.getpid(),
        "state": state,
        "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "group.ready.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _prewarm_memory_embedding(
    *,
    log_dir: Path,
    provider: EmbeddingProvider | None,
    requested_device: str,
) -> None:
    """Load and verify the reply process' embedding runtime before readiness."""
    marker_path = log_dir / "memory.embedding.ready.json"
    marker_path.unlink(missing_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if provider is None:
        raise RuntimeError("memory embedding provider is missing")
    identity = provider.identity
    if identity.provider == "disabled":
        payload = {
            "pid": os.getpid(),
            "state": "disabled",
            "provider": identity.provider,
            "model": identity.model,
            "dimensions": identity.dimensions,
            "accelerator": "disabled",
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }
    else:
        if not provider.available:
            raise RuntimeError("memory embedding provider is unavailable")
        logging.info(
            "memory_embedding_prewarm state=starting provider=%s model=%s requested_device=%s",
            identity.provider,
            identity.model,
            requested_device,
        )
        vector = provider.embed_query("小町记忆检索启动预热")
        if vector is None or len(vector) != identity.dimensions:
            raise RuntimeError("memory embedding prewarm returned an invalid vector")
        accelerator = str(
            getattr(
                provider,
                "active_accelerator",
                "remote" if identity.provider == "openai_compatible" else "unknown",
            )
        )
        if (
            identity.provider == "local"
            and requested_device.strip().lower() == "cuda"
            and accelerator != "cuda"
        ):
            raise RuntimeError("memory embedding prewarm did not activate CUDA")
        payload = {
            "pid": os.getpid(),
            "state": "ready",
            "provider": identity.provider,
            "model": identity.model,
            "dimensions": identity.dimensions,
            "accelerator": accelerator,
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }

    temporary_path = marker_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(marker_path)
    logging.info(
        "memory_embedding_prewarm state=%s provider=%s model=%s dimensions=%s accelerator=%s",
        payload["state"],
        payload["provider"],
        payload["model"],
        payload["dimensions"],
        payload["accelerator"],
    )


def _prewarm_persona_example_vectors(
    manager,
    personas: dict,
    *,
    log_dir: Path,
) -> None:
    """Warm the persona example vector caches for live-refresh personas."""

    marker = Path(log_dir) / "persona.embedding.ready.json"
    marker.unlink(missing_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        prewarmed: list[str] = []
        for persona_key, persona in personas.items():
            if not isinstance(persona, dict) or not persona.get("live_refresh"):
                continue
            try:
                group_id = int(persona.get("source_group_id") or 0)
            except (TypeError, ValueError):
                continue
            if group_id <= 0:
                continue
            count = manager.prewarm_examples(int(group_id), persona_key)
            prewarmed.append(f"{persona_key}={count}")
            logging.info(
                "persona_embedding_prewarm persona=%s samples=%s",
                persona_key,
                count,
            )
        payload = {
            "state": "ready",
            "personas": prewarmed,
            "updated_at": datetime.now(ASIA_SHANGHAI).isoformat(),
        }
    except Exception:
        logging.exception("persona_embedding_prewarm_failed")
        payload = {"state": "failed"}
    marker.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _max_message_id(engine) -> int:
    with session_scope(engine) as session:
        row = session.execute(
            text("SELECT COALESCE(MAX(id), 0) FROM messages")
        ).scalar_one()
        return int(row)


def _startup_window_mention_rows(
    engine,
    *,
    watermark_message_id: int,
    enabled_group_ids: tuple[int, ...],
    bot_qq: int,
    limit: int = 5,
):
    """Return @小町 messages persisted by startup backfill but never replied."""
    if not enabled_group_ids:
        return []
    with session_scope(engine) as session:
        return session.execute(
            text(
                "SELECT id, group_id, user_id, platform_msg_id, raw_json, "
                "plain_text, reply_to_msg_id, timestamp FROM messages "
                "WHERE id > :wm AND mentioned_bot = 1 AND user_id != :bot "
                "AND group_id IN :groups ORDER BY id LIMIT :limit"
            ).bindparams(bindparam("groups", expanding=True)),
            {
                "wm": int(watermark_message_id),
                "bot": int(bot_qq),
                "groups": tuple(enabled_group_ids),
                "limit": max(1, int(limit)),
            },
        ).all()


async def _replay_startup_window_mentions(
    *,
    engine,
    router: InboundRouter,
    settings: AppSettings,
    runtime,
    watermark_message_id: int,
) -> None:
    """补处理启动窗口内被回填为历史、但实际是 @小町 的提问。"""
    enabled_group_ids = tuple(
        int(group_id)
        for group_id in runtime.group_policy.get("groups", {})
        if should_ingest_group_message(
            group_id=int(group_id),
            group_policy=runtime.group_policy,
        )
    )
    rows = _startup_window_mention_rows(
        engine,
        watermark_message_id=watermark_message_id,
        enabled_group_ids=enabled_group_ids,
        bot_qq=settings.bot_qq,
    )
    bot_name = str(runtime.persona.get("name", settings.bot_qq))
    for row in rows:
        raw_json = row.raw_json if isinstance(row.raw_json, dict) else {}
        message_segments: list[dict] = []
        raw_message = raw_json.get("message")
        if isinstance(raw_message, list) and raw_message:
            message_segments = list(raw_message)
        else:
            if row.reply_to_msg_id:
                message_segments.append(
                    {"type": "reply", "data": {"id": row.reply_to_msg_id}}
                )
            if row.plain_text:
                message_segments.append(
                    {"type": "text", "data": {"text": row.plain_text}}
                )
        if not any(
            isinstance(segment, dict) and segment.get("type") == "at"
            for segment in message_segments
        ):
            message_segments.append(
                {"type": "at", "data": {"qq": str(settings.bot_qq)}}
            )
        payload = {
            "post_type": "message",
            "message_id": row.platform_msg_id,
            "group_id": row.group_id,
            "user_id": row.user_id,
            "sender": raw_json.get("sender")
            or {"nickname": str(row.user_id), "card": ""},
            "time": int(datetime.fromisoformat(str(row.timestamp)).timestamp()),
            "message": message_segments,
        }
        event = parse_group_message_event(
            payload,
            bot_qq=settings.bot_qq,
            bot_name=bot_name,
        )
        logging.info(
            "startup_window_replay group_id=%s msg_id=%s",
            event.group_id,
            event.platform_msg_id,
        )
        try:
            await router._handle_persisted_group_message(event)
        except Exception:
            logging.exception(
                "startup_window_replay_failed group_id=%s msg_id=%s",
                event.group_id,
                event.platform_msg_id,
            )
        await asyncio.sleep(0.5)


def _revoke_group_message_projection(
    engine,
    *,
    group_id: int,
    platform_msg_id: str,
) -> tuple[int | None, int]:
    with session_scope(engine) as session:
        message = MessageRepository(session).mark_group_message_deleted(
            group_id=int(group_id),
            platform_msg_id=str(platform_msg_id),
            reason="group_recall",
        )
        if message is None:
            return None, 0
        session.flush()
        revoked = RetrievalDocumentRepository(
            session
        ).deactivate_raw_message_v3(
            group_id=int(group_id),
            message_id=int(message.id),
        )
        return int(message.id), revoked


async def _handle_group_recall_payload(payload: dict, *, engine) -> bool:
    if (
        payload.get("post_type") != "notice"
        or payload.get("notice_type") != "group_recall"
    ):
        return False
    try:
        group_id = int(payload["group_id"])
        platform_msg_id = str(payload["message_id"])
    except (KeyError, TypeError, ValueError):
        logging.warning("group_recall_invalid_payload")
        return True
    message_id, revoked = await asyncio.to_thread(
        _revoke_group_message_projection,
        engine,
        group_id=group_id,
        platform_msg_id=platform_msg_id,
    )
    logging.info(
        "group_recall_processed group_id=%s message_id=%s found=%s revoked=%s",
        group_id,
        platform_msg_id,
        message_id is not None,
        revoked,
    )
    return True


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "group.heartbeat.json")
    await heartbeat.start()
    engine = None
    group_image_service = None
    memory_compaction_service = None
    persona_sync_task = None
    member_fact_refresh_task = None
    persona_prewarm_task = None
    try:
        engine = await asyncio.to_thread(build_engine, settings.sqlite_path)
        logging.info("startup_step build_engine done")
        await asyncio.to_thread(create_all, engine)
        logging.info("startup_step create_all done")
        await asyncio.to_thread(sync_history_archives, engine, runtime)
        logging.info("startup_step sync_history done")

        gateway = NapCatGateway(
            ws_url=settings.napcat_ws_url,
            reconnect_forever=True,
        )
        sender = Sender(gateway)
        llm_client = build_llm_client(settings=settings, engine=engine)
        proactive_judge_client = build_proactive_judge_client(
            settings=settings,
            llm_client=llm_client,
            engine=engine,
        )
        group_image_llm_client = build_group_image_llm_client(settings=settings, engine=engine, llm_client=llm_client)
        web_search_client = build_web_search_client(settings)
        image_reference_planner_client = build_group_image_reference_planner_client(
            settings=settings,
            llm_client=llm_client,
        )
        group_image_service = build_group_image_service(
            settings=settings,
            llm_client=group_image_llm_client,
            sender=sender,
            web_search_client=web_search_client,
            image_reference_planner_client=image_reference_planner_client,
        )
        memory_runtime = await asyncio.to_thread(
            build_memory_runtime,
            settings=settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
            memory_enabled_group_ids=frozenset(
                int(group_id)
                for group_id in runtime.group_policy.get("groups", {})
                if should_enable_memory_in_group(
                    group_id=int(group_id),
                    group_policy=runtime.group_policy,
                )
            ),
        )
        logging.info("startup_step build_memory_runtime done")
        await asyncio.to_thread(
            _prewarm_memory_embedding,
            log_dir=settings.log_dir,
            provider=memory_runtime.embedding_provider,
            requested_device=settings.memory_embedding_device,
        )
        memory_compaction_service = memory_runtime.memory_compaction_service
        persistent_group_engine = engine if hasattr(engine, "connect") else None
        if hasattr(group_image_service, "engine") and getattr(group_image_service, "engine", None) is None:
            group_image_service.engine = persistent_group_engine
        if hasattr(group_image_service, "start") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.start()
        if memory_compaction_service is not None:
            await memory_compaction_service.start()
        persona_manager = PersonaManager(
            engine=engine,
            personas=getattr(runtime, "personas", {}) or {},
            default_persona=runtime.persona,
            embedding_provider=memory_runtime.embedding_provider,
        )
        await asyncio.to_thread(persona_manager.load_state)
        persona_prewarm_task = asyncio.create_task(
            asyncio.to_thread(
                _prewarm_persona_example_vectors,
                persona_manager,
                getattr(runtime, "personas", {}) or {},
                log_dir=settings.log_dir,
            )
        )
        persona_switch_service = PersonaSwitchService(
            manager=persona_manager,
            sender=sender,
            bot_qq=settings.bot_qq,
        )
        persona_sync_service = PersonaLiveSyncService(
            engine=engine,
            settings=settings,
            personas=getattr(runtime, "personas", {}) or {},
            manager=persona_manager,
        )
        persona_sync_task = asyncio.create_task(persona_sync_service.run())
        memory_group_ids = {
            int(group_id)
            for group_id in runtime.group_policy.get("groups", {})
            if should_enable_memory_in_group(
                group_id=int(group_id),
                group_policy=runtime.group_policy,
            )
        }
        member_fact_refresh_service = MemberFactRefreshService(
            engine=engine,
            settings=settings,
            group_ids=memory_group_ids,
            bot_qq=settings.bot_qq,
            bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            member_allowlist={
                int(persona["source_user_id"])
                for persona in getattr(runtime, "personas", {}).values()
                if isinstance(persona, dict)
                and persona.get("live_refresh")
                and str(persona.get("source_user_id") or "").isdigit()
            },
        )
        member_fact_refresh_task = asyncio.create_task(
            member_fact_refresh_service.run()
        )
        router = InboundRouter(
            engine=engine,
            runtime=runtime,
            sender=sender,
            llm_client=llm_client,
            proactive_judge_client=proactive_judge_client,
            reply_policy=ReplyPolicy(),
            context_builder=ContextBuilder(),
            admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
            web_search_client=web_search_client,
            dev_control_service=None,
            group_image_service=group_image_service,
            memory_compaction_service=memory_compaction_service,
            memory_orchestrator=memory_runtime.memory_orchestrator,
            persona_manager=persona_manager,
            persona_switch_service=persona_switch_service,
        )

        async def handle_payload(payload: dict) -> None:
            if await _handle_group_recall_payload(payload, engine=engine):
                return
            if payload.get("post_type") != "message":
                return
            if payload.get("message_type") != "group":
                return

            group_id = int(payload["group_id"])
            if group_id == 10001:
                logging.info(
                    "group_payload_received group_id=%s msg_id=%s user_id=%s",
                    group_id,
                    payload.get("message_id"),
                    payload.get("user_id"),
                )
            if not should_ingest_group_message(group_id=group_id, group_policy=runtime.group_policy):
                return

            event = parse_group_message_event(
                payload,
                bot_qq=settings.bot_qq,
                bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            )
            await router.handle_group_message(event)

        async def backfill_group_history_on_connect() -> None:
            _write_group_ready_marker(log_dir=settings.log_dir, state="connected")
            try:
                watermark_message_id = _max_message_id(engine)
                await backfill_recent_group_history(
                    router=router,
                    gateway=gateway,
                    bot_qq=settings.bot_qq,
                    bot_name=str(runtime.persona.get("name", settings.bot_qq)),
                )
                await _replay_startup_window_mentions(
                    engine=engine,
                    router=router,
                    settings=settings,
                    runtime=runtime,
                    watermark_message_id=watermark_message_id,
                )
            except Exception:
                logging.exception("startup_backfill_failed; continuing to accept messages")
            _write_group_ready_marker(log_dir=settings.log_dir, state="ready")
            logging.info("group_ready accepting_messages=True state=ready")

        logging.info(create_runtime_banner(bot_qq=settings.bot_qq, model=f"{settings.llm_model} [group]"))
        await gateway.connect_and_consume(handle_payload, on_connect=backfill_group_history_on_connect)
    finally:
        if member_fact_refresh_task is not None:
            member_fact_refresh_task.cancel()
        if persona_prewarm_task is not None:
            persona_prewarm_task.cancel()
        if persona_sync_task is not None:
            persona_sync_task.cancel()
        if group_image_service is not None and hasattr(group_image_service, "stop") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.stop()
        if memory_compaction_service is not None:
            await memory_compaction_service.stop()
        await heartbeat.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
