from __future__ import annotations

import asyncio

import pytest

from app.core.persona_switch import (
    DEFAULT_PERSONA_KEY,
    PersonaManager,
    PersonaSwitchService,
    parse_switch_command,
    persona_aliases,
)


def _personas() -> dict[str, dict]:
    return {
        "default": {
            "name": "测试小町",
            "identity": "AI assistant",
            "core_traits": ["calm"],
            "speaking_style": {"tone": "natural"},
        },
        "test_self": {
            "name": "测试君",
            "identity": "group member",
            "core_traits": ["casual"],
            "speaking_style": {"tone": "casual"},
            "group_card": "测试君",
            "source_user_id": 123456789,
        },
    }


def test_parse_switch_command_accepts_half_and_full_width_colon() -> None:
    personas = _personas()
    assert parse_switch_command("切换人格为:测试君", personas) == "test_self"
    assert parse_switch_command("切换人格为：测试君", personas) == "test_self"
    assert parse_switch_command("  切换人格为 :  测试君  ", personas) == "test_self"
    assert parse_switch_command("切换人格为:测试小町", personas) == "default"
    assert parse_switch_command("切换人格为:小町", personas) == "default"


def test_parse_switch_command_ignores_non_commands_and_unknown_targets() -> None:
    personas = _personas()
    assert parse_switch_command("你好", personas) is None
    assert parse_switch_command("切换人格为:不存在", personas) is None
    assert parse_switch_command("切换人格为", personas) is None


def test_persona_aliases_include_short_cjk_suffix() -> None:
    assert "小町" in persona_aliases({"name": "测试小町"})


def test_persona_manager_persists_per_group_keys(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    assert manager.active_key(10001) == DEFAULT_PERSONA_KEY
    assert manager.active_name(10001) == "测试小町"

    manager.set_persona_key(10001, "test_self")
    manager.set_card_snapshot(10001, "原名")
    manager.set_account_avatar_snapshot("avatar://original")

    reloaded = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    reloaded.load_state()
    assert reloaded.active_key(10001) == "test_self"
    assert reloaded.active_name(10001) == "测试君"
    assert reloaded.card_snapshot(10001) == "原名"
    assert reloaded.account_avatar_snapshot() == "avatar://original"
    assert reloaded.active_key(10002) == DEFAULT_PERSONA_KEY


def test_persona_manager_bot_label_marked_when_impersonating(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    assert manager.bot_transcript_label(10001) == "测试小町"
    manager.set_persona_key(10001, "test_self")
    assert manager.bot_transcript_label(10001) == "测试君（小町扮演）"


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.avatar_map: dict[int, str] = {}

    async def get_qq_avatar(self, *, user_id: int) -> str:
        self.calls.append(("get_qq_avatar", {"user_id": int(user_id)}))
        return self.avatar_map.get(int(user_id), "avatar://fallback")

    async def set_qq_avatar(self, *, file: str) -> None:
        self.calls.append(("set_qq_avatar", {"file": file}))

    async def set_group_card(self, *, group_id: int, user_id: int, card: str) -> None:
        self.calls.append(
            ("set_group_card", {"group_id": group_id, "user_id": user_id, "card": card})
        )

    async def get_group_member_info(self, *, group_id: int, user_id: int) -> dict:
        self.calls.append(("get_group_member_info", {"group_id": group_id, "user_id": user_id}))
        return {"status": "ok", "data": {"card": "原名"}}


def test_switch_service_applies_and_restores_profile(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    sender = FakeSender()
    sender.avatar_map = {
        123456789: "avatar://target",
        987654321: "avatar://original",
    }
    service = PersonaSwitchService(manager=manager, sender=sender, bot_qq=987654321)

    confirmation = asyncio.run(service.switch(group_id=10001, target_key="test_self"))
    assert manager.active_key(10001) == "test_self"
    assert "已切换为测试君人格" in confirmation
    assert ("set_qq_avatar", {"file": "avatar://target"}) in sender.calls
    assert ("set_group_card", {"group_id": 10001, "user_id": 987654321, "card": "测试君"}) in sender.calls

    sender.calls.clear()
    confirmation = asyncio.run(service.switch(group_id=10001, target_key="default"))
    assert manager.active_key(10001) == DEFAULT_PERSONA_KEY
    assert ("set_qq_avatar", {"file": "avatar://original"}) in sender.calls
    assert ("set_group_card", {"group_id": 10001, "user_id": 987654321, "card": "原名"}) in sender.calls


def test_switch_service_confirms_noop_when_already_active(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    sender = FakeSender()
    service = PersonaSwitchService(manager=manager, sender=sender, bot_qq=987654321)

    confirmation = asyncio.run(service.switch(group_id=10001, target_key="default"))
    assert "无需切换" in confirmation
    assert sender.calls == []


def test_router_impersonation_guardrail_and_memory_filter(sqlite_engine) -> None:
    from app.core.router import InboundRouter

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )
    assert InboundRouter._filter_bot_identity_memory_lines(
        ["主人今天心情不错", "晚上吃什么"]
    ) == ["晚上吃什么"]

    router.persona_manager.personas["test_self"] = {
        "name": "测试君",
        "identity": "group member",
    }
    router.persona_manager.set_persona_key(10001, "test_self")
    text = router._persona_text_for(router._active_persona(10001), 10001)
    assert "完整扮演群成员 测试君" in text
    assert "不是小町" in text
