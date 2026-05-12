import asyncio

from app.adapters.sender import OutboundMessage, OutboundPrivateMessage, Sender


class FakeGateway:
    def __init__(self) -> None:
        self.calls = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        return {"status": "ok"}


def test_sender_uses_send_group_msg() -> None:
    gateway = FakeGateway()
    sender = Sender(gateway)
    asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text="hello")))

    assert gateway.calls == [("send_group_msg", {"group_id": 10001, "message": "hello"})]


def test_sender_uses_send_private_msg() -> None:
    gateway = FakeGateway()
    sender = Sender(gateway)
    asyncio.run(sender.send_private_text(OutboundPrivateMessage(user_id=987654321, text="ok")))

    assert gateway.calls == [("send_private_msg", {"user_id": 987654321, "message": "ok"})]
