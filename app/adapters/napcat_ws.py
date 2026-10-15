from __future__ import annotations

import asyncio
import logging
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import websockets


class NapCatDeliveryUncertainError(ConnectionError):
    """A send frame may have reached NapCat before the websocket failed."""


def _contains_outbound_video(params):
    pending = [params.get('message'), params.get('messages')]
    checked = 0
    while pending and checked < 1000:
        value = pending.pop()
        checked += 1
        if isinstance(value, list):
            pending.extend(value[:100])
        elif isinstance(value, dict):
            if value.get('type') == 'video':
                return True
            if value.get('type') == 'node' and isinstance(value.get('data'), dict):
                pending.append(value['data'].get('content'))
    return False


class NapCatGateway:
    REQUEST_TIMEOUT_SECONDS = 20.0

    def __init__(
        self,
        *,
        ws_url: str,
        access_token: str | None = None,
        reconnect_forever: bool = False,
        reconnect_delay_seconds: float = 2.0,
        max_handler_tasks: int = 128,
    ) -> None:
        self.ws_url = ws_url
        self.access_token = access_token
        self.reconnect_forever = reconnect_forever
        self.reconnect_delay_seconds = max(0.0, float(reconnect_delay_seconds))
        self.websocket = None
        self._pending_calls: dict[str, asyncio.Future] = {}
        self._handler_tasks: set[asyncio.Task] = set()
        self.max_handler_tasks = max(1, int(max_handler_tasks))
        self.on_connection_change: Callable[[bool], None] | None = None

    async def connect_and_consume(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        on_connect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else None
        while True:
            disconnect_error = ConnectionError("NapCat websocket disconnected")
            should_reconnect = False
            on_connect_task: asyncio.Task | None = None
            cancelled = False
            try:
                async with websockets.connect(self.ws_url, additional_headers=headers) as websocket:
                    self.websocket = websocket
                    if self.on_connection_change is not None:
                        self.on_connection_change(True)
                    if on_connect is not None:
                        on_connect_task = asyncio.create_task(self._run_on_connect(on_connect))
                        await asyncio.sleep(0)
                    async for raw_message in websocket:
                        payload = json.loads(raw_message)
                        echo = payload.get("echo")
                        if echo and echo in self._pending_calls:
                            future = self._pending_calls.pop(echo)
                            if not future.done():
                                future.set_result(payload)
                            continue
                        if len(self._handler_tasks) >= self.max_handler_tasks:
                            # Reconnect/backfill reports a gap instead of growing without bound.
                            raise RuntimeError("OneBot inbound handler capacity exceeded")
                        task = asyncio.create_task(self._run_handler(handler, payload))
                        self._handler_tasks.add(task)
                        task.add_done_callback(self._handler_tasks.discard)
                        # Let fast durable-ingest handlers complete during bursts.
                        # The reader still keeps running while slow handlers await RPCs.
                        await asyncio.sleep(0)
            except asyncio.CancelledError:
                cancelled = True
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
                if self.on_connection_change is not None:
                    self.on_connection_change(False)
                self._fail_pending_calls(disconnect_error)
                if on_connect_task is not None and not on_connect_task.done():
                    on_connect_task.cancel()
                pending_tasks = list(self._handler_tasks)
                if pending_tasks:
                    # Allow already received messages to reach the durable inbox, but
                    # never let a slow model handler delay reconnect/shutdown forever.
                    if cancelled:
                        unfinished = pending_tasks
                    else:
                        _, unfinished = await asyncio.wait(pending_tasks, timeout=2.0)
                    for task in unfinished:
                        task.cancel()
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                self._handler_tasks.clear()
                if on_connect_task is not None:
                    await asyncio.gather(on_connect_task, return_exceptions=True)
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

    async def _run_on_connect(
        self,
        on_connect: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            await on_connect()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("NapCat on_connect hook failed")

    async def call_api(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.websocket is None:
            raise RuntimeError("NapCat websocket is not connected")
        echo = uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[echo] = future
        envelope = {"action": action, "params": params, "echo": echo}
        try:
            await self.websocket.send(json.dumps(envelope, ensure_ascii=False))
        except Exception as exc:
            self._discard_pending_call(echo, future)
            raise NapCatDeliveryUncertainError("NapCat websocket failed while sending request") from exc
        try:
            video = action in {'send_group_msg','send_private_msg','send_group_forward_msg','send_private_forward_msg'} and _contains_outbound_video(params)
            return await asyncio.wait_for(future, timeout=90.0 if video else self.REQUEST_TIMEOUT_SECONDS)
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
