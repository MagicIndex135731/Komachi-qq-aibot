import asyncio
import json

import pytest

from app.adapters.napcat_ws import NapCatGateway


class FakeWebSocket:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.sent = []
        self.fail_send = fail_send

    async def send(self, data: str) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(data)


class FakeIncomingWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


class BlockingIncomingWebSocket:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await asyncio.Future()
        raise StopAsyncIteration


class FakeDuplexWebSocket:
    def __init__(self, initial_messages: list[dict]) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(json.dumps(message))
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, data: str) -> None:
        self.sent.append(data)
        envelope = json.loads(data)
        echo = envelope["echo"]
        await self.queue.put(json.dumps({"status": "ok", "echo": echo, "data": {"message_id": 42}}))
        await self.queue.put(None)


def test_call_api_clears_pending_call_when_send_fails() -> None:
    gateway = NapCatGateway(ws_url="ws://example")
    gateway.websocket = FakeWebSocket(fail_send=True)

    with pytest.raises(RuntimeError):
        asyncio.run(gateway.call_api("send_group_msg", {"group_id": 1, "message": "hi"}))

    assert gateway._pending_calls == {}


def test_connect_and_consume_keeps_running_when_handler_raises(monkeypatch) -> None:
    gateway = NapCatGateway(ws_url="ws://example")
    fake_socket = FakeIncomingWebSocket(
        [
            '{"post_type":"message","message_type":"group","message_id":"1"}',
            '{"post_type":"message","message_type":"group","message_id":"2"}',
        ]
    )
    seen = []

    monkeypatch.setattr("app.adapters.napcat_ws.websockets.connect", lambda *args, **kwargs: fake_socket)

    async def handler(payload: dict) -> None:
        seen.append(payload["message_id"])
        if payload["message_id"] == "1":
            raise RuntimeError("boom")

    asyncio.run(gateway.connect_and_consume(handler))

    assert seen == ["1", "2"]


def test_call_api_clears_pending_call_when_reply_never_arrives() -> None:
    gateway = NapCatGateway(ws_url="ws://example")
    gateway.websocket = FakeWebSocket()

    async def invoke() -> None:
        task = asyncio.create_task(
            gateway.call_api("send_group_msg", {"group_id": 1, "message": "hi"})
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, 0.01)

    asyncio.run(invoke())

    assert gateway._pending_calls == {}


@pytest.mark.asyncio
async def test_connect_and_consume_can_process_api_echo_while_handler_waits(monkeypatch) -> None:
    gateway = NapCatGateway(ws_url="ws://example")
    fake_socket = FakeDuplexWebSocket(
        [
            {
                "post_type": "message",
                "message_type": "group",
                "message_id": "in-1",
            }
        ]
    )
    seen = {"result": None}

    monkeypatch.setattr("app.adapters.napcat_ws.websockets.connect", lambda *args, **kwargs: fake_socket)

    async def handler(payload: dict) -> None:
        assert payload["message_id"] == "in-1"
        seen["result"] = await gateway.call_api("send_group_msg", {"group_id": 1, "message": "hi"})

    await gateway.connect_and_consume(handler)

    assert seen["result"] == {"status": "ok", "echo": json.loads(fake_socket.sent[0])["echo"], "data": {"message_id": 42}}


@pytest.mark.asyncio
async def test_connect_and_consume_reconnects_after_disconnect_when_enabled(monkeypatch) -> None:
    gateway = NapCatGateway(
        ws_url="ws://example",
        reconnect_forever=True,
        reconnect_delay_seconds=0.0,
    )
    sockets = [
        FakeIncomingWebSocket(
            ['{"post_type":"message","message_type":"private","message_id":"1"}']
        ),
        FakeIncomingWebSocket(
            ['{"post_type":"message","message_type":"private","message_id":"2"}']
        ),
    ]
    connect_attempts: list[str] = []
    seen: list[str] = []
    second_message_seen = asyncio.Event()

    def fake_connect(*args, **kwargs):
        del args, kwargs
        connect_attempts.append("connect")
        if sockets:
            return sockets.pop(0)
        return BlockingIncomingWebSocket()

    monkeypatch.setattr("app.adapters.napcat_ws.websockets.connect", fake_connect)

    async def handler(payload: dict) -> None:
        seen.append(payload["message_id"])
        if payload["message_id"] == "2":
            second_message_seen.set()

    task = asyncio.create_task(gateway.connect_and_consume(handler))
    await asyncio.wait_for(second_message_seen.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen == ["1", "2"]
    assert len(connect_attempts) >= 2


@pytest.mark.asyncio
async def test_connect_and_consume_runs_on_connect_for_initial_connect_and_reconnect(monkeypatch) -> None:
    gateway = NapCatGateway(
        ws_url="ws://example",
        reconnect_forever=True,
        reconnect_delay_seconds=0.0,
    )
    sockets = [
        FakeIncomingWebSocket(
            ['{"post_type":"message","message_type":"private","message_id":"1"}']
        ),
        FakeIncomingWebSocket(
            ['{"post_type":"message","message_type":"private","message_id":"2"}']
        ),
    ]
    on_connect_calls: list[str] = []
    second_message_seen = asyncio.Event()

    def fake_connect(*args, **kwargs):
        del args, kwargs
        if sockets:
            return sockets.pop(0)
        return BlockingIncomingWebSocket()

    monkeypatch.setattr("app.adapters.napcat_ws.websockets.connect", fake_connect)

    async def handler(payload: dict) -> None:
        if payload["message_id"] == "2":
            second_message_seen.set()

    async def on_connect() -> None:
        on_connect_calls.append("connected")

    task = asyncio.create_task(gateway.connect_and_consume(handler, on_connect=on_connect))
    await asyncio.wait_for(second_message_seen.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert on_connect_calls == ["connected", "connected"]


@pytest.mark.asyncio
async def test_connect_and_consume_does_not_block_live_messages_while_on_connect_runs(monkeypatch) -> None:
    gateway = NapCatGateway(ws_url="ws://example")
    fake_socket = FakeIncomingWebSocket(
        ['{"post_type":"message","message_type":"group","message_id":"live-1"}']
    )
    handler_called = asyncio.Event()
    release_on_connect = asyncio.Event()

    monkeypatch.setattr("app.adapters.napcat_ws.websockets.connect", lambda *args, **kwargs: fake_socket)

    async def handler(payload: dict) -> None:
        if payload["message_id"] == "live-1":
            handler_called.set()

    async def on_connect() -> None:
        await release_on_connect.wait()

    task = asyncio.create_task(gateway.connect_and_consume(handler, on_connect=on_connect))

    await asyncio.wait_for(handler_called.wait(), timeout=0.2)
    release_on_connect.set()
    await task
