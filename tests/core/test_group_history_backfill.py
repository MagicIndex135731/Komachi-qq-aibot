from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.core.group_history_backfill import backfill_recent_group_history
from app.core.router import InboundRouter
from app.storage.db import session_scope
from app.storage.models import Message


class FakeSender:
    def __init__(self, gateway) -> None:
        self.gateway = gateway
        self.sent = []

    async def send_group_text(self, outbound) -> None:
        self.sent.append(outbound)


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        del conversation_key
        self.calls.append(prompt_lines)
        return "unused"


class FakeGateway:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, dict(params)))
        return self.responses.pop(0)


def _history_payload(*, message_id: str, group_id: int, user_id: int, text: str, at: datetime) -> dict:
    return {
        "message_id": message_id,
        "group_id": group_id,
        "user_id": user_id,
        "time": int(at.timestamp()),
        "sender": {"nickname": f"user-{user_id}", "card": ""},
        "message": [{"type": "text", "data": {"text": text}}],
    }


@pytest.mark.asyncio
async def test_backfill_recent_group_history_persists_missing_messages_without_reply(sqlite_engine) -> None:
    gateway = FakeGateway(
        [
            {
                "status": "ok",
                "retcode": 0,
                "data": {
                    "messages": [
                        _history_payload(
                            message_id="hist-2",
                            group_id=10001,
                            user_id=20002,
                            text="@Mira 这条是断网期间的历史消息",
                            at=datetime(2026, 5, 9, 12, 1, tzinfo=UTC),
                        ),
                        _history_payload(
                            message_id="hist-1",
                            group_id=10001,
                            user_id=20001,
                            text="更早一点",
                            at=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
                        ),
                    ]
                },
            }
        ]
    )
    sender = FakeSender(gateway)
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    persisted = await backfill_recent_group_history(
        router=router,
        gateway=gateway,
        bot_qq=router.runtime.settings.bot_qq,
        bot_name=str(router.runtime.persona.get("name", router.runtime.settings.bot_qq)),
    )

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.timestamp, Message.id)
        ).scalars().all()

    assert persisted == 2
    assert sender.sent == []
    assert llm.calls == []
    assert [message.platform_msg_id for message in stored_messages] == ["hist-1", "hist-2"]
    assert gateway.calls == [("get_group_msg_history", {"group_id": 10001, "count": 50})]


@pytest.mark.asyncio
async def test_backfill_recent_group_history_stops_after_reaching_saved_overlap(sqlite_engine) -> None:
    gateway = FakeGateway(
        [
            {
                "status": "ok",
                "retcode": 0,
                "data": {
                    "messages": [
                        _history_payload(
                            message_id="hist-3",
                            group_id=10001,
                            user_id=20003,
                            text="漏掉的新消息",
                            at=datetime(2026, 5, 9, 12, 2, tzinfo=UTC),
                        ),
                        _history_payload(
                            message_id="hist-2",
                            group_id=10001,
                            user_id=20002,
                            text="已存在的重叠消息",
                            at=datetime(2026, 5, 9, 12, 1, tzinfo=UTC),
                        ),
                    ]
                },
            }
        ]
    )
    sender = FakeSender(gateway)
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    existing_event = SimpleNamespace(
        platform_msg_id="hist-2",
        group_id=10001,
        user_id=20002,
        nickname="user-20002",
        group_card="",
        plain_text="已存在的重叠消息",
        raw_payload=_history_payload(
            message_id="hist-2",
            group_id=10001,
            user_id=20002,
            text="已存在的重叠消息",
            at=datetime(2026, 5, 9, 12, 1, tzinfo=UTC),
        ),
        timestamp=datetime(2026, 5, 9, 12, 1, tzinfo=UTC),
        msg_type="text",
        images=[],
        mentioned_bot=False,
        reply_to_msg_id=None,
    )
    assert router.ingest_historical_group_message(existing_event) is True

    persisted = await backfill_recent_group_history(
        router=router,
        gateway=gateway,
        bot_qq=router.runtime.settings.bot_qq,
        bot_name=str(router.runtime.persona.get("name", router.runtime.settings.bot_qq)),
    )

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.timestamp, Message.id)
        ).scalars().all()

    assert persisted == 1
    assert [message.platform_msg_id for message in stored_messages] == ["hist-2", "hist-3"]
