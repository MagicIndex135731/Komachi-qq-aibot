from __future__ import annotations

from datetime import UTC, datetime

import pytest

import app.core.router as router_module
from app.adapters.onebot_models import PrivateMessageEvent
from app.core.router import InboundRouter
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


class FakeSender:
    def __init__(self) -> None:
        self.private_sent = []

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)


class FakeLlm:
    def generate_text(self, prompt_lines, *, images=None, conversation_key=None):
        del prompt_lines, images, conversation_key
        return "unused"


class FakeDevControlService:
    def __init__(self) -> None:
        self.events = []

    async def handle_private_message(self, event: PrivateMessageEvent) -> bool:
        self.events.append(event)
        return True


def make_private_event(
    *,
    user_id: int,
    text: str,
    raw_payload: dict | None = None,
    msg_type: str = "text",
    reply_to_msg_id: str | None = None,
    images: list | None = None,
) -> PrivateMessageEvent:
    return PrivateMessageEvent(
        platform_msg_id="p-1",
        user_id=user_id,
        nickname="owner",
        plain_text=text,
        raw_payload=raw_payload or {},
        timestamp=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
        msg_type=msg_type,
        reply_to_msg_id=reply_to_msg_id,
        images=list(images or []),
    )


@pytest.mark.asyncio
async def test_router_forwards_owner_private_message_to_dev_service(sqlite_engine) -> None:
    sender = FakeSender()
    dev_service = FakeDevControlService()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlm(),
        dev_control_service=dev_service,
    )
    router.runtime.settings.owner_qq = 10001

    await router.handle_private_message(make_private_event(user_id=10001, text="check logs"))

    assert [event.plain_text for event in dev_service.events] == ["check logs"]
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_router_ignores_duplicate_owner_private_message_delivery(sqlite_engine) -> None:
    sender = FakeSender()
    dev_service = FakeDevControlService()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlm(),
        dev_control_service=dev_service,
    )
    router.runtime.settings.owner_qq = 10001
    event = make_private_event(user_id=10001, text="check logs")

    await router.handle_private_message(event)
    await router.handle_private_message(event)

    assert [payload.platform_msg_id for payload in dev_service.events] == ["p-1"]
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_router_private_message_dedup_does_not_conflict_with_group_message_ids(sqlite_engine) -> None:
    sender = FakeSender()
    dev_service = FakeDevControlService()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlm(),
        dev_control_service=dev_service,
    )
    router.runtime.settings.owner_qq = 10001

    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=20001, nickname="Alice", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id="p-1",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 10, 11, 59, tzinfo=UTC),
            plain_text="group message",
            raw_json={"message_id": "p-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_private_message(make_private_event(user_id=10001, text="check logs"))

    assert [event.plain_text for event in dev_service.events] == ["check logs"]
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_router_handles_private_group_allow_command_without_dev_service(sqlite_engine) -> None:
    sender = FakeSender()
    dev_service = FakeDevControlService()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlm(),
        dev_control_service=dev_service,
    )
    router.runtime.settings.owner_qq = 10001

    await router.handle_private_message(make_private_event(user_id=987654321, text="/bot group allow 10086"))

    with session_scope(sqlite_engine) as session:
        group = GroupRepository(session).get_group(10086)

    assert group is not None
    assert group.speak_enabled is True
    assert [outbound.text for outbound in sender.private_sent] == ["已允许群 10086 发言。"]
    assert dev_service.events == []


@pytest.mark.asyncio
async def test_router_persists_private_images_and_forwards_cached_event(sqlite_engine, tmp_path, monkeypatch) -> None:
    sender = FakeSender()
    dev_service = FakeDevControlService()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlm(),
        dev_control_service=dev_service,
    )
    router.runtime.settings.owner_qq = 10001

    def fake_cache(raw_payload, *, cache_dir, http_client=None) -> None:
        del cache_dir, http_client
        raw_payload["message"][1]["data"]["local_path"] = str(tmp_path / "private-cat.png")

    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", fake_cache)

    event = make_private_event(
        user_id=10001,
        text="看这个",
        raw_payload={
            "message_id": "p-1",
            "message": [
                {"type": "text", "data": {"text": "看这个"}},
                {
                    "type": "image",
                    "data": {
                        "file": "cat.png",
                        "url": "https://img.example.test/cat.png",
                    },
                },
            ],
        },
        msg_type="mixed",
        images=[object()],
    )

    await router.handle_private_message(event)

    assert len(dev_service.events) == 1
    assert len(dev_service.events[0].images) == 1
    assert dev_service.events[0].images[0].local_path == str(tmp_path / "private-cat.png")

    with session_scope(sqlite_engine) as session:
        stored = MessageRepository(session).get_by_platform_msg_id("private-inbound-10001-p-1")

    assert stored is not None
    assert stored.msg_type == "mixed"
    assert stored.reply_to_msg_id is None
    assert stored.raw_json["message"][1]["data"]["local_path"] == str(tmp_path / "private-cat.png")
