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


def test_switch_service_switches_without_touching_display(sqlite_engine) -> None:
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
    assert sender.calls == []

    sender.calls.clear()
    confirmation = asyncio.run(service.switch(group_id=10001, target_key="default"))
    assert manager.active_key(10001) == DEFAULT_PERSONA_KEY
    assert sender.calls == []


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
    router.persona_manager.personas["test_self"] = {
        "name": "测试君",
        "identity": "group member",
    }
    router.persona_manager.set_persona_key(10001, "test_self")
    text = router._persona_text_for(router._active_persona(10001), 10001)
    assert "完整扮演群成员 测试君" in text
    assert "不是任何其他身份" in text


def test_router_sanitizes_impersonation_context_lines(sqlite_engine) -> None:
    from app.core.router import InboundRouter

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )
    router.runtime.persona["name"] = "测试小町"

    router.persona_manager.personas["test_self"] = {
        "name": "测试君",
        "identity": "group member",
    }
    router.persona_manager.set_persona_key(10001, "test_self")

    scrubbed = router._sanitize_impersonation_lines(
        [
            "阿渣（小町扮演）: 主人，来了",
            "测试小町: 大人您稍等",
            "路人: 谁是你的主人",
            "测试君: 大人您稍等",
            "小町今天毒舌了一整天",
        ],
        group_id=10001,
    )

    assert scrubbed == [
        "测试君: 你稍等",
    ]


def test_router_formats_clean_bot_label_for_prompt_lines(sqlite_engine) -> None:
    from app.core.router import InboundRouter

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )
    router.persona_manager.personas["test_self"] = {
        "name": "测试君",
        "identity": "group member",
    }
    router.persona_manager.set_persona_key(10001, "test_self")

    line = router._format_message_line(
        user_id=router.runtime.settings.bot_qq,
        plain_text="来了",
        users_by_id={},
        group_id=10001,
    )

    assert line == "测试君（Mira扮演）: 来了"


def test_router_appends_relevant_examples_while_impersonating(sqlite_engine) -> None:
    from app.core.router import InboundRouter

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )

    persona = {
        "name": "测试君",
        "identity": "group member",
        "example_bank": ["明天看球", "上号"],
    }
    text = router._with_relevant_examples(
        "persona-text",
        persona,
        ["甲: 明天看球吗"],
        10001,
    )

    assert "明天看球" in text
    assert "上号" not in text


def test_router_appends_relevant_facts_from_shared_memory(sqlite_engine) -> None:
    from app.core.router import InboundRouter
    from app.storage.db import session_scope
    from app.storage.repositories import MemoryRepository

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )

    router.persona_manager.personas["test_self"] = {
        "name": "测试君",
        "identity": "group member",
        "source_user_id": 222,
    }
    router.persona_manager.set_persona_key(10001, "test_self")
    with session_scope(sqlite_engine) as session:
        MemoryRepository(session).upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="222",
            memory_kind="fact",
            canonical_key="主玩英雄联盟手游",
            predicate="游戏",
            object_text="",
            content="主玩英雄联盟手游",
            importance=3,
            confidence=0.8,
            source_msg_ids=[],
        )

    text = router._with_relevant_facts(
        "persona-text",
        router._active_persona(10001),
        ["你最擅长什么lol英雄"],
        10001,
    )
    assert "主玩英雄联盟手游" in text
    assert "不等于'讨厌'" in text


def test_persona_manager_retrieves_facts_from_memory(sqlite_engine) -> None:
    from app.storage.db import session_scope
    from app.storage.repositories import MemoryRepository

    personas = {
        "default": {"name": "测试小町"},
        "test_self": {"name": "测试君", "identity": "group member", "source_user_id": 222},
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    manager.set_persona_key(10001, "test_self")
    with session_scope(sqlite_engine) as session:
        repo = MemoryRepository(session)
        for category, fact in (
            ("游戏", "主玩英雄联盟手游"),
            ("工作", "在快手实习"),
        ):
            repo.upsert_canonical_memory(
                scope_type="group",
                scope_id="10001",
                subject_type="user",
                subject_id="222",
                memory_kind="fact",
                canonical_key=fact,
                predicate=category,
                object_text="",
                content=fact,
                importance=3,
                confidence=0.8,
                source_msg_ids=[],
            )

    picked = manager.retrieve_facts(10001, ["你最擅长什么英雄"], limit=2)

    assert any(item["fact"] == "主玩英雄联盟手游" for item in picked)


def test_live_persona_refreshes_relationship_labels(sqlite_engine) -> None:
    from app.storage.db import session_scope
    from app.storage.repositories import UserRepository

    personas = {
        "default": {"name": "测试小町"},
        "test_self": {
            "name": "测试君",
            "identity": "group member",
            "source_user_id": 222,
            "relationships": [
                {"member": "999", "member_user_id": 999, "relation": "朋友"},
                {"member": "888", "relation": "球友"},
            ],
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(
            user_id=999, nickname="旧昵称", group_card="路人卡"
        )
        UserRepository(session).upsert_user(
            user_id=888, nickname="球友甲", group_card=""
        )
    manager.set_persona_key(10001, "test_self")

    live = manager.live_persona(10001)

    assert live["relationships"][0]["member"] == "路人卡"
    assert live["relationships"][1]["member"] == "球友甲"


def test_example_vectors_persist_across_manager_instances(sqlite_engine) -> None:
    from app.core.persona_switch import PersonaManager
    from app.storage.db import session_scope
    from app.storage.models import PersonaExampleVector
    from app.storage.repositories import PersonaStyleExampleRepository

    class _FakeEmbedding:
        available = True

        class _Identity:
            provider = "test"
            model = "fake"
            dimensions = 4

        identity = _Identity()

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    with session_scope(sqlite_engine) as session:
        PersonaStyleExampleRepository(session).insert_many(
            [
                {
                    "group_id": 10001,
                    "user_id": 222,
                    "msg_id": "m-vec-1",
                    "text": "在吗",
                    "context_before": [],
                    "context_after": [],
                    "reply_target": None,
                }
            ]
        )
    personas = {
        "default": {"name": "测试小町"},
        "test_self": {
            "name": "测试君",
            "identity": "group member",
            "source_user_id": 222,
            "source_group_id": 10001,
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
        embedding_provider=_FakeEmbedding(),
    )
    manager.load_state()
    manager._group_keys[10001] = "test_self"
    picked = manager.retrieve_examples(10001, ["在吗"], limit=1)
    assert picked

    with session_scope(sqlite_engine) as session:
        assert session.query(PersonaExampleVector).filter_by(user_id=222).count() == 1

    fresh = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
        embedding_provider=_FakeEmbedding(),
    )
    fresh.load_state()
    fresh._group_keys[10001] = "test_self"
    picked_again = fresh.retrieve_examples(10001, ["在吗"], limit=1)
    assert picked_again
