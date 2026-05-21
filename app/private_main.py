from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.onebot_models import parse_private_message_event
from app.adapters.sender import Sender
from app.admin.commands import AdminCommandParser
from app.config import AppSettings, load_runtime_config
from app.core.context_builder import ContextBuilder
from app.core.reply_policy import ReplyPolicy
from app.core.router import InboundRouter
from app.dev_control.service import DevControlService
from app.main import build_llm_client, build_web_search_client
from app.private_reminders import PrivateReminderScheduler, load_private_reminders
from app.storage.db import build_engine, create_all


async def _wait_for_gateway_ready(gateway: NapCatGateway, *, timeout_seconds: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while gateway.websocket is None:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("private gateway did not connect in time")
        await asyncio.sleep(0.1)


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)

    gateway = NapCatGateway(
        ws_url=settings.napcat_ws_url,
        reconnect_forever=True,
    )
    sender = Sender(gateway)
    llm_client = build_llm_client(settings=settings, engine=engine)
    web_search_client = build_web_search_client(settings)
    dev_control_service = DevControlService(
        engine=engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=settings.owner_qq,
        bot_qq=settings.bot_qq,
        private_chat_qqs=settings.private_chat_whitelist,
        repo_root=Path(__file__).resolve().parent.parent,
        data_dir=settings.data_dir,
        enable_local_worker=False,
        web_search_client=web_search_client,
        assistant_name=str(runtime.persona.get("name", "Codex")),
        persona=runtime.persona,
        safety=runtime.safety,
    )
    reminder_scheduler = PrivateReminderScheduler(
        sender=sender,
        data_dir=settings.data_dir,
        reminders=load_private_reminders(config_dir=settings.config_dir),
        allowed_user_ids=settings.private_chat_whitelist,
    )
    router = InboundRouter(
        engine=engine,
        runtime=runtime,
        sender=sender,
        llm_client=llm_client,
        reply_policy=ReplyPolicy(),
        context_builder=ContextBuilder(),
        admin_parser=AdminCommandParser(admin_whitelist=settings.admin_whitelist),
        web_search_client=None,
        dev_control_service=dev_control_service,
    )

    async def handle_payload(payload: dict) -> None:
        if payload.get("post_type") != "message":
            return
        if payload.get("message_type") != "private":
            return

        event = parse_private_message_event(payload)
        await router.handle_private_message(event)

    logging.info(f"qq-ai-private starting with owner={settings.owner_qq} model={settings.llm_model}")
    gateway_task = asyncio.create_task(gateway.connect_and_consume(handle_payload))
    try:
        await _wait_for_gateway_ready(gateway)
        await dev_control_service.start()
        await reminder_scheduler.start()
        await gateway_task
    finally:
        gateway_task.cancel()
        await asyncio.gather(gateway_task, return_exceptions=True)
        await reminder_scheduler.stop()
        await dev_control_service.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
