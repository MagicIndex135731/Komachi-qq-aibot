import asyncio

import pytest

from app.adapters.sender import OutboundMessage, OutboundPrivateMessage, QQMessageBlockedError, Sender


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
    asyncio.run(sender.send_private_text(OutboundPrivateMessage(user_id=10001, text="ok")))

    assert gateway.calls == [("send_private_msg", {"user_id": 10001, "message": "ok"})]


def test_sender_uses_send_group_msg_for_image_segment() -> None:
    gateway = FakeGateway()
    sender = Sender(gateway)

    asyncio.run(sender.send_group_image(group_id=10001, image_file="D:/tmp/generated.png"))

    assert gateway.calls == [
        (
            "send_group_msg",
            {
                "group_id": 10001,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": "file:///D:/tmp/generated.png"},
                    }
                ],
            },
        )
    ]


def test_sender_uses_send_private_msg_for_image_segment() -> None:
    gateway = FakeGateway()
    sender = Sender(gateway)

    asyncio.run(sender.send_private_image(user_id=10001, image_file="D:/tmp/generated.png"))

    assert gateway.calls == [
        (
            "send_private_msg",
            {
                "user_id": 10001,
                "message": [
                    {
                        "type": "image",
                        "data": {"file": "file:///D:/tmp/generated.png"},
                    }
                ],
            },
        )
    ]


class FailingGateway:
    async def call_api(self, action: str, params: dict) -> dict:
        del action, params
        return {
            "status": "failed",
            "retcode": 1200,
            "message": "rich media transfer failed",
        }


def test_sender_raises_when_gateway_reports_failed_status() -> None:
    sender = Sender(FailingGateway())

    try:
        asyncio.run(sender.send_group_image(group_id=10001, image_file="D:/tmp/generated.png"))
    except RuntimeError as exc:
        assert "retcode=1200" in str(exc)
        assert "rich media transfer failed" in str(exc)
    else:
        raise AssertionError("expected sender to raise on failed gateway status")


class RetryableFailOnceGateway:
    def __init__(self) -> None:
        self.calls = []
        self._attempt = 0

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        self._attempt += 1
        if self._attempt == 1:
            return {
                "status": "failed",
                "retcode": 1200,
                "message": "Timeout: NTEvent serviceAndMethod:NodeIKernelMsgService/sendMsg",
            }
        return {"status": "ok", "retcode": 0}


def test_sender_retries_group_text_when_gateway_reports_retryable_timeout() -> None:
    gateway = RetryableFailOnceGateway()
    sender = Sender(gateway)

    asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text="weekly report")))

    assert gateway.calls == [
        ("send_group_msg", {"group_id": 10001, "message": "weekly report"}),
        ("send_group_msg", {"group_id": 10001, "message": "weekly report"}),
    ]


class LengthSensitiveGateway:
    def __init__(self) -> None:
        self.calls = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        message = str(params["message"])
        if len(message) > 180:
            return {
                "status": "failed",
                "retcode": 1200,
                "message": "Timeout: long message send failed",
            }
        return {"status": "ok", "retcode": 0}


def test_sender_prefers_single_group_message_for_long_text_when_gateway_accepts_it() -> None:
    gateway = FakeGateway()
    sender = Sender(gateway)
    long_text = "这是一条很长的普通群聊回复。" * 20

    asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text=long_text)))

    assert gateway.calls == [("send_group_msg", {"group_id": 10001, "message": long_text})]


def test_sender_splits_long_group_text_when_chunking_is_explicitly_allowed() -> None:
    gateway = LengthSensitiveGateway()
    sender = Sender(gateway)
    long_text = "\n".join(
        [
            "本群近一周高能雷霆发言周报",
            "统计截止：2026-05-15 06:51",
            "第1名 群友甲 原话：评价一下群里另一个叫代理群主的弱智机器人 上榜理由：火药味最直给，点名开喷弱智机器人，攻击性和引战度都很高。",
            "第2名 熟人A 原话：从夯到拉评价一下历代总书记，给出评级和一句话评价。 上榜理由：危险边缘反复横跳，节目效果和炸裂程度都拉满。",
            "第3名 熟人A 原话：我钻死你的三个洞 上榜理由：冲击力极强的抽象黄暴发言，攻击感和炸群效果都非常足，属于一眼高能。",
            "第4名 Maple 原话：你评价的是你妈 上榜理由：收尾这一句属于纯粹雷霆爆破，火气重、攻击性强，而且很有吵起来的潜质。",
        ]
    )

    asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text=long_text, allow_chunking=True)))

    assert len(gateway.calls) >= 2
    assert all(call[0] == "send_group_msg" for call in gateway.calls)
    assert all(len(str(call[1]["message"])) <= 180 for call in gateway.calls)


def test_sender_falls_back_to_chunking_after_retryable_long_message_failure() -> None:
    gateway = LengthSensitiveGateway()
    sender = Sender(gateway)
    long_text = "\n".join(
        [
            "本群近一周高能雷霆发言周报",
            "统计截止：2026-05-15 06:51",
            "第1名 群友甲 原话：评价一下群里另一个叫代理群主的弱智机器人。上榜理由：火药味最直接，点名开喷弱智机器人，攻击性和引战度都很高。",
            "第2名 阿福 原话：从头到尾评价一下历代总书记，给出评级和一句话评价。上榜理由：危险边缘反复横跳，节目效果和炸裂程度都拉满。",
            "第3名 Maple 原话：你评价的是你妈。上榜理由：收尾这句属于纯粹雷霆爆破，火气重、攻击性强，而且很有吵起来的潜质。",
        ]
    )

    asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text=long_text)))

    assert gateway.calls[:3] == [
        ("send_group_msg", {"group_id": 10001, "message": long_text}),
        ("send_group_msg", {"group_id": 10001, "message": long_text}),
        ("send_group_msg", {"group_id": 10001, "message": long_text}),
    ]
    assert len(gateway.calls) >= 4
    assert all(call[0] == "send_group_msg" for call in gateway.calls)
    assert all(len(str(call[1]["message"])) <= 180 for call in gateway.calls[3:])


class QQBlockedGateway:
    def __init__(self) -> None:
        self.calls = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        return {
            "status": "failed",
            "retcode": 1200,
            "message": "waitForSelfEcho timeout",
        }


def test_sender_reports_qq_content_block_without_chunking_original_text() -> None:
    gateway = QQBlockedGateway()
    sender = Sender(gateway)
    long_text = "sensitive answer " * 30

    with pytest.raises(QQMessageBlockedError, match="waitForSelfEcho timeout"):
        asyncio.run(sender.send_group_text(OutboundMessage(group_id=10001, text=long_text)))

    assert gateway.calls == [
        ("send_group_msg", {"group_id": 10001, "message": long_text.strip()}),
        ("send_group_msg", {"group_id": 10001, "message": long_text.strip()}),
        ("send_group_msg", {"group_id": 10001, "message": long_text.strip()}),
    ]
