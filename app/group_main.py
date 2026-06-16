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
    build_web_search_client,
    create_runtime_banner,
    should_ingest_group_message,
    sync_history_archives,
)
from app.runtime_heartbeat import RuntimeHeartbeat
from app.storage.db import build_engine, create_all


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)
    sync_history_archives(engine, runtime)

    gateway = NapCatGateway(
        ws_url=settings.napcat_ws_url,
        reconnect_forever=True,
    )
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "group.heartbeat.json")
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
    )

    async def handle_payload(payload: dict) -> None:
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
    try:
        await heartbeat.start()
        await gateway.connect_and_consume(handle_payload, on_connect=backfill_group_history_on_connect)
    finally:
        if hasattr(group_image_service, "stop") and getattr(group_image_service, "engine", None) is not None:
            await group_image_service.stop()
        await heartbeat.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
