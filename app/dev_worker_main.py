from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.adapters.napcat_ws import NapCatGateway
from app.adapters.sender import Sender
from app.config import AppSettings, load_runtime_config
from app.dev_control.service import DevControlService
from app.main import build_llm_client
from app.storage.db import build_engine, create_all


async def _wait_for_gateway_ready(gateway: NapCatGateway, *, timeout_seconds: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while gateway.websocket is None:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("worker gateway did not connect in time")
        await asyncio.sleep(0.1)


async def run() -> None:
    settings = AppSettings()
    runtime = load_runtime_config(settings)
    engine = build_engine(settings.sqlite_path)
    create_all(engine)

    gateway = NapCatGateway(ws_url=settings.napcat_ws_url)
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

    async def ignore_payload(_payload: dict) -> None:
        return None

    gateway_task = asyncio.create_task(gateway.connect_and_consume(ignore_payload))
    try:
        await _wait_for_gateway_ready(gateway)
        await service.start()
        logging.info(f"qq-ai-dev-worker starting with owner={settings.owner_qq} model={settings.llm_model}")
        await gateway_task
    finally:
        gateway_task.cancel()
        await asyncio.gather(gateway_task, return_exceptions=True)
        await service.stop()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
