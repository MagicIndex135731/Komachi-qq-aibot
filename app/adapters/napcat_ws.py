from __future__ import annotations

import asyncio
import logging
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import websockets


class NapCatGateway:
    REQUEST_TIMEOUT_SECONDS = 20.0

    def __init__(
        self,
        *,
        ws_url: str,
        access_token: str | None = None,
        reconnect_forever: bool = False,
        reconnect_delay_seconds: float = 2.0,
    ) -> None:
        self.ws_url = ws_url
        self.access_token = access_token
        self.reconnect_forever = reconnect_forever
        self.reconnect_delay_seconds = max(0.0, float(reconnect_delay_seconds))
        self.websocket = None
        self._pending_calls: dict[str, asyncio.Future] = {}
        self._handler_tasks: set[asyncio.Task] = set()

    async def connect_and_consume(
        self, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else None
        while True:
            disconnect_error = ConnectionError("NapCat websocket disconnected")
            should_reconnect = False
            try:
                async with websockets.connect(self.ws_url, additional_headers=headers) as websocket:
                    self.websocket = websocket
                    async for raw_message in websocket:
                        payload = json.loads(raw_message)
                        echo = payload.get("echo")
                        if echo and echo in self._pending_calls:
                            future = self._pending_calls.pop(echo)
                            if not future.done():
                                future.set_result(payload)
                            continue
                        task = asyncio.create_task(self._run_handler(handler, payload))
                        self._handler_tasks.add(task)
                        task.add_done_callback(self._handler_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self.reconnect_forever:
                    raise
                should_reconnect = True
                logging.exception("NapCat websocket loop failed; reconnecting")
            else:
                should_reconnect = self.reconnect_forever
                if should_reconnect:
                    logging.warning("NapCat websocket closed; reconnecting")
            finally:
                self.websocket = None
                self._fail_pending_calls(disconnect_error)
                pending_tasks = list(self._handler_tasks)
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                self._handler_tasks.clear()
            if not should_reconnect:
                return
            await asyncio.sleep(self.reconnect_delay_seconds)

    async def _run_handler(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        payload: dict[str, Any],
    ) -> None:
        try:
            await handler(payload)
        except Exception:
            logging.exception("NapCat payload handler failed")

    async def call_api(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.websocket is None:
            raise RuntimeError("NapCat websocket is not connected")
        echo = uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[echo] = future
        envelope = {"action": action, "params": params, "echo": echo}
        try:
            await self.websocket.send(json.dumps(envelope, ensure_ascii=False))
        except Exception:
            self._discard_pending_call(echo, future)
            raise
        try:
            return await asyncio.wait_for(future, timeout=self.REQUEST_TIMEOUT_SECONDS)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self._discard_pending_call(echo, future)
            raise

    def _discard_pending_call(self, echo: str, future: asyncio.Future) -> None:
        self._pending_calls.pop(echo, None)
        if not future.done():
            future.cancel()

    def _fail_pending_calls(self, error: Exception) -> None:
        pending_calls = self._pending_calls
        self._pending_calls = {}
        for future in pending_calls.values():
            if not future.done():
                future.set_exception(error)
