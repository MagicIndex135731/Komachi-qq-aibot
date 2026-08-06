from __future__ import annotations

import asyncio
import logging

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_group_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.group_history_backfill import backfill_recent_group_history
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.main import (
    build_group_image_llm_client,
    build_group_image_service,
    build_llm_client,
    build_memory_runtime,
    build_web_search_client,
    create_runtime_banner,
    should_ingest_group_message,
    sync_history_archives,
)
from app.runtime_heartbeat import RuntimeHeartbeat
from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import MessageRepository, RetrievalDocumentRepository


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
    try:
        engine = await asyncio.to_thread(build_engine, settings.sqlite_path)
        await asyncio.to_thread(create_all, engine)
        await asyncio.to_thread(sync_history_archives, engine, runtime)

        gateway = NapCatGateway(
            ws_url=settings.napcat_ws_url,
            reconnect_forever=True,
        )
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
        memory_runtime = await asyncio.to_thread(
            build_memory_runtime,
            settings=settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name=str(runtime.persona.get("name", settings.bot_qq)),
        )
        memory_compaction_service = memory_runtime.memory_compaction_service
        persistent_group_engine = engine if hasattr(engine, "connect") else None
        if hasattr(group_image_service, "engine") and getattr(group_image_service, "engine", None) is None:
            group_image_service.engine = persistent_group_engine
        if hasattr(group_image_service, "start") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.start()
        if memory_compaction_service is not None:
            await memory_compaction_service.start()
        router = InboundRouter(
            engine=engine,
            runtime=runtime,
            sender=sender,
            llm_client=llm_client,
            reply_policy=ReplyPolicy(),
            context_builder=ContextBuilder(),
            admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
            web_search_client=web_search_client,
            dev_control_service=None,
            group_image_service=group_image_service,
            memory_compaction_service=memory_compaction_service,
            memory_orchestrator=memory_runtime.memory_orchestrator,
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
            await backfill_recent_group_history(
                router=router,
                gateway=gateway,
                bot_qq=settings.bot_qq,
                bot_name=str(runtime.persona.get("name", settings.bot_qq)),
            )

        logging.info(create_runtime_banner(bot_qq=settings.bot_qq, model=f"{settings.llm_model} [group]"))
        await gateway.connect_and_consume(handle_payload, on_connect=backfill_group_history_on_connect)
    finally:
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
