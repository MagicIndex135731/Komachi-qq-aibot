from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.sender import Sender
from app.config import AppSettings, load_runtime_config
from app.dev_control.service import DevControlService
from app.main import build_llm_client
from app.runtime_heartbeat import RuntimeHeartbeat
from app.storage.db import build_engine, create_all


async def _wait_for_gateway_ready(gateway: NapCatGateway, gateway_task: asyncio.Task) -> None:
    while gateway.websocket is None:
        if gateway_task.done():
            await gateway_task
        await asyncio.sleep(0.1)


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)

    gateway = NapCatGateway(ws_url=settings.napcat_ws_url, reconnect_forever=True)
    heartbeat = RuntimeHeartbeat(heartbeat_file=settings.log_dir / "worker.heartbeat.json")
    sender = Sender(gateway)
    llm_client = build_llm_client(settings=settings, engine=engine)
    service = DevControlService(
        engine=engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=settings.owner_qq,
        bot_qq=settings.bot_qq,
        repo_root=Path(__file__).resolve().parent.parent,
        data_dir=settings.data_dir,
        enable_local_worker=True,
        assistant_name=str(runtime.persona.get("name", "Codex")),
        persona=runtime.persona,
        safety=runtime.safety,
    )

    async def ignore_payload(payload: dict) -> None:
        if payload.get("post_type") == "message" and payload.get("message_type") == "group":
            if int(payload.get("group_id", 0) or 0) == 10001:
                logging.info(
                    "worker_process_observed_group_payload group_id=%s msg_id=%s user_id=%s",
                    payload.get("group_id"),
                    payload.get("message_id"),
                    payload.get("user_id"),
                )
        return None

    gateway_task = asyncio.create_task(gateway.connect_and_consume(ignore_payload))
    try:
        await heartbeat.start()
        await _wait_for_gateway_ready(gateway, gateway_task)
        await service.start()
        logging.info(f"qq-ai-dev-worker starting with owner={settings.owner_qq} model={settings.llm_model}")
        await gateway_task
    finally:
        gateway_task.cancel()
        await asyncio.gather(gateway_task, return_exceptions=True)
        await service.stop()
        await heartbeat.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
