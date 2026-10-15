from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

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
            "aliases": ["粘人小町"],
            "core_traits": ["calm"],
            "speaking_style": {"tone": "natural"},
        },
        "mouthy_komachi": {
            "name": "测试小町",
            "identity": "AI assistant",
            "aliases": ["嘴臭小町"],
            "komachi_variant": True,
            "switch_label": "嘴臭小町",
            "core_traits": ["sharp"],
            "speaking_style": {"tone": "sharp"},
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


def _add_fact_source(session, source_id, content, *, group_id=10001, user_id=222, state=""):
    from app.storage.repositories import GroupRepository, MessageRepository, UserRepository

    GroupRepository(session).upsert_group(group_id=group_id, group_name="test", enabled=True, speak_enabled=True)
    UserRepository(session).upsert_user(user_id=user_id, nickname="test", group_card="")
    return MessageRepository(session).add_group_message(
        platform_msg_id=source_id, group_id=group_id, user_id=user_id,
        timestamp=datetime(2026, 9, 19, 12, tzinfo=UTC), plain_text=content,
        raw_json={"delivery_state": state} if state else {}, msg_type="text",
        reply_to_msg_id=None, mentioned_bot=False,
    )


def test_parse_switch_command_accepts_half_and_full_width_colon() -> None:
    personas = _personas()
    assert parse_switch_command("切换人格为:测试君", personas) == "test_self"
    assert parse_switch_command("切换人格为：测试君", personas) == "test_self"
    assert parse_switch_command("  切换人格为 :  测试君  ", personas) == "test_self"
    assert parse_switch_command("切换人格为:测试小町", personas) == "default"
    assert parse_switch_command("切换人格为:小町", personas) == "default"
    assert parse_switch_command("切换人格为:粘人小町", personas) == "default"
    assert parse_switch_command("切换人格为:嘴臭小町", personas) == "mouthy_komachi"


def test_parse_switch_command_ignores_non_commands_and_unknown_targets() -> None:
    personas = _personas()
    assert parse_switch_command("你好", personas) is None
    assert parse_switch_command("切换人格为:不存在", personas) is None
    assert parse_switch_command("切换人格为", personas) is None


def test_persona_aliases_include_short_cjk_suffix() -> None:
    assert "小町" in persona_aliases({"name": "测试小町"})


def test_persona_manager_resets_keys_on_restart_but_preserves_snapshots(
    sqlite_engine,
) -> None:
    from app.storage.db import session_scope
    from app.storage.repositories import GroupPersonaStateRepository

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
    assert reloaded.active_key(10001) == DEFAULT_PERSONA_KEY
    assert reloaded.active_name(10001) == "测试小町"
    assert reloaded.card_snapshot(10001) == "原名"
    assert reloaded.account_avatar_snapshot() == "avatar://original"
    assert reloaded.active_key(10002) == DEFAULT_PERSONA_KEY
    with session_scope(sqlite_engine) as session:
        state = GroupPersonaStateRepository(session).get(10001)
        assert state is not None
        assert state.persona_key == DEFAULT_PERSONA_KEY


def test_persona_prewarm_does_not_change_active_persona(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()

    assert manager.prewarm_examples(10001, "test_self") == 0
    assert manager.active_key(10001) == DEFAULT_PERSONA_KEY

    manager.set_persona_key(10001, "test_self")
    assert manager.prewarm_examples(10001, DEFAULT_PERSONA_KEY) == 0
    assert manager.active_key(10001) == "test_self"


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


def test_komachi_variant_is_not_treated_as_member_impersonation(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    manager.set_persona_key(10001, "mouthy_komachi")

    assert manager.is_impersonating(10001) is False
    assert manager.bot_transcript_label(10001) == "测试小町"


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


def test_switch_service_uses_komachi_variant_label(sqlite_engine) -> None:
    personas = _personas()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    service = PersonaSwitchService(manager=manager, sender=FakeSender(), bot_qq=987654321)

    confirmation = asyncio.run(
        service.switch(group_id=10001, target_key="mouthy_komachi")
    )

    assert manager.active_key(10001) == "mouthy_komachi"
    assert confirmation == "已切换为嘴臭小町人格。"


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


def test_router_keeps_komachi_variant_on_komachi_path(sqlite_engine) -> None:
    from app.core.router import InboundRouter

    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=object(),
        llm_client=object(),
    )
    router.persona_manager.personas["mouthy_komachi"] = {
        "name": "测试小町",
        "identity": "AI assistant",
        "komachi_variant": True,
    }
    router.persona_manager.set_persona_key(10001, "mouthy_komachi")

    assert router._impersonating(10001) is False
    assert "完整扮演群成员" not in router._persona_text_for(
        router._active_persona(10001), 10001
    )


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
        _add_fact_source(session, "router-fact-source", "主玩英雄联盟手游")
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
            source_msg_ids=["router-fact-source"],
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
            _add_fact_source(session, fact, fact)
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
                source_msg_ids=[fact],
            )

    picked = manager.retrieve_facts(10001, ["你最擅长什么英雄"], limit=2)

    assert any(item["fact"] == "主玩英雄联盟手游" for item in picked)


def test_persona_manager_selects_dynamic_fact_kinds_by_intent_and_validity(
    sqlite_engine,
) -> None:
    from datetime import UTC, datetime, timedelta

    from app.storage.db import session_scope
    from app.storage.repositories import MemoryRepository

    personas = {
        "default": {"name": "测试小町"},
        "test_self": {
            "name": "测试君",
            "identity": "group member",
            "source_user_id": 222,
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    manager.set_persona_key(10001, "test_self")
    now = datetime(2026, 9, 19, 12, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        repo = MemoryRepository(session)
        for kind, key, content, valid_until in (
            ("current", "current-new", "最近在学日语", now + timedelta(days=2)),
            ("current", "current-old", "之前在学法语", now - timedelta(days=1)),
            ("plan", "plan", "接下来准备去登山", now + timedelta(days=3)),
            ("preference", "preference", "喜欢爵士乐", None),
            ("relationship", "relationship", "和甲是同学", None),
        ):
            _add_fact_source(session, key, content)
            repo.upsert_canonical_memory(
                scope_type="group",
                scope_id="10001",
                subject_type="user",
                subject_id="222",
                memory_kind=kind,
                canonical_key=key,
                predicate=kind,
                object_text="",
                content=content,
                importance=3,
                confidence=0.9,
                source_msg_ids=[key],
                valid_from=now - timedelta(days=2),
                valid_until=valid_until,
            )
        _add_fact_source(session, "anime-abbreviation", "刚看完一集 RW0")
        repo.upsert_canonical_memory(
            scope_type="group",
            scope_id="10001",
            subject_type="user",
            subject_id="222",
            memory_kind="event",
            canonical_key="anime-abbreviation",
            predicate="动漫",
            object_text="",
            content="刚看完一集 RW0",
            importance=3,
            confidence=0.9,
            source_msg_ids=["anime-abbreviation"],
            valid_from=now - timedelta(hours=1),
        )

    current = manager.retrieve_facts(
        10001, ["你最近在学什么"], now=now, limit=5
    )
    plans = manager.retrieve_facts(
        10001, ["你接下来有什么计划"], now=now, limit=5
    )
    preferences = manager.retrieve_facts(
        10001, ["你喜欢什么音乐"], now=now, limit=5
    )
    relationships = manager.retrieve_facts(
        10001, ["你和甲是什么关系"], now=now, limit=5
    )
    current_anime = manager.retrieve_facts(
        10001, ["你最近在看什么动画"], now=now, limit=5
    )

    assert [item["fact"] for item in current] == ["最近在学日语"]
    assert [item["fact"] for item in plans] == ["接下来准备去登山"]
    assert [item["fact"] for item in preferences] == ["喜欢爵士乐"]
    assert [item["fact"] for item in relationships] == ["和甲是同学"]
    assert [item["fact"] for item in current_anime] == ["刚看完一集 RW0"]


@pytest.mark.parametrize("change", ["reserved", "blocked", "uncertain", "deleted", "missing", "cross_group", "no_sources"])
def test_persona_facts_revalidate_all_current_group_provenance(sqlite_engine, change):
    from app.storage.db import session_scope
    from app.storage.repositories import MemoryRepository

    personas = {"default": {"name": "test bot"}, "actor": {"name": "test member", "source_user_id": 222}}
    manager = PersonaManager(engine=sqlite_engine, personas=personas, default_persona=personas["default"])
    manager.set_persona_key(10001, "actor")
    with session_scope(sqlite_engine) as session:
        _add_fact_source(session, "valid-fact-source", "喜欢爵士乐")
        if change not in {"missing", "no_sources"}:
            _add_fact_source(session, "bad-fact-source", "喜欢被撤回的音乐", group_id=20001 if change == "cross_group" else 10001, state=change if change != "cross_group" else "")
        for marker, sources in (
            ("喜欢爵士乐", ["valid-fact-source"]),
            ("喜欢被撤回的音乐", [] if change == "no_sources" else ["valid-fact-source", "bad-fact-source"]),
        ):
            MemoryRepository(session).upsert_canonical_memory(
                scope_type="group", scope_id="10001", subject_type="user", subject_id="222",
                memory_kind="preference", canonical_key=marker, predicate="likes", object_text=marker,
                content=marker, importance=3, confidence=0.9, source_msg_ids=sources,
            )
    picked = manager.retrieve_facts(10001, ["你喜欢什么音乐"], limit=5)
    assert [item["fact"] for item in picked] == ["喜欢爵士乐"]


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


def test_member_alias_cache_is_independent_for_each_group(sqlite_engine):
    from datetime import UTC, datetime
    from app.storage.db import session_scope
    from app.storage.repositories import GroupRepository, MessageRepository, UserRepository
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=777, nickname="common", group_card="")
        for group, card in ((100, "group-a-card"), (200, "group-b-card")):
            GroupRepository(session).upsert_group(group_id=group, group_name="test", enabled=True, speak_enabled=True)
            MessageRepository(session).add_group_message(platform_msg_id=f"alias-{group}", group_id=group, user_id=777, timestamp=datetime.now(UTC), plain_text="hello", raw_json={"sender": {"card": card}}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False)
    manager = PersonaManager(engine=sqlite_engine, personas={}, default_persona={})
    assert manager.member_label_for_user(777, 100) == "group-a-card"
    assert manager.member_label_for_user(777, 200) == "group-b-card"
    assert manager.member_label_for_user(777, 100) == "group-a-card"


def test_style_bank_does_not_keep_a_recalled_local_source(sqlite_engine):
    from datetime import UTC, datetime
    from app.storage.db import session_scope
    from app.storage.repositories import GroupRepository, MessageRepository, PersonaStyleExampleRepository, UserRepository
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=100, group_name="test", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=777, nickname="test", group_card="")
        MessageRepository(session).add_group_message(platform_msg_id="style-source", group_id=100, user_id=777, timestamp=datetime.now(UTC), plain_text="old-style", raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False)
        PersonaStyleExampleRepository(session).insert_many([{"group_id": 100, "user_id": 777, "msg_id": "style-source", "text": "old-style", "context_before": [], "reply_target": None}])
    personas = {"member": {"source_user_id": 777, "source_group_id": 100}}
    manager = PersonaManager(engine=sqlite_engine, personas=personas, default_persona={})
    manager.load_style_banks()
    assert manager.style_bank(100, persona_key="member")
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(group_id=100, platform_msg_id="style-source")
    assert manager.style_bank(100, persona_key="member") == []


@pytest.mark.parametrize("situation_kind", ["quoted", "adjacent"])
@pytest.mark.parametrize("state", ["reserved", "blocked", "uncertain", "deleted", "outbound", "serialized_outbound"])
def test_style_bank_revalidates_live_situation_sources(sqlite_engine, situation_kind, state):
    from app.core.persona_live_sync import _build_examples
    from app.core.persona_style_retrieval import build_style_document, build_style_retrieval_query
    from app.storage.db import session_scope
    from app.storage.repositories import MessageRepository, PersonaStyleExampleRepository

    rows = [
        {"id": 1, "platform_msg_id": "situation-source", "group_id": 10001, "user_id": 111,
         "timestamp": datetime(2026, 9, 19, 12, tzinfo=UTC), "plain_text": "游戏更新怎么延期了",
         "msg_type": "text", "reply_to_msg_id": None, "raw_json": {}},
        {"id": 2, "platform_msg_id": "style-response", "group_id": 10001, "user_id": 222,
         "timestamp": datetime(2026, 9, 19, 12, 1, tzinfo=UTC), "plain_text": "游戏延期只能继续等",
         "msg_type": "text", "reply_to_msg_id": "situation-source" if situation_kind == "quoted" else None,
         "raw_json": {}},
    ]
    with session_scope(sqlite_engine) as session:
        for row in rows:
            source = _add_fact_source(session, row["platform_msg_id"], row["plain_text"], user_id=row["user_id"])
            source.reply_to_msg_id = row["reply_to_msg_id"]
        PersonaStyleExampleRepository(session).insert_many(_build_examples(
            rows, user_id=222, group_id=10001, bot_qqs={900001}, bot_text_names={"test bot"},
        ))
    personas = {"member": {"source_user_id": 222, "source_group_id": 10001}}
    manager = PersonaManager(engine=sqlite_engine, personas=personas, default_persona={})
    manager.load_style_banks()
    before = build_style_document(manager.style_bank(10001, persona_key="member")[0])
    assert before.situation == "游戏更新怎么延期了"
    with session_scope(sqlite_engine) as session:
        source = MessageRepository(session).get_by_platform_msg_id("situation-source")
        source.raw_json = (
            '{"direction":"outbound"}' if state == "serialized_outbound"
            else {"direction": "outbound"} if state == "outbound"
            else {"delivery_state": state}
        )
    after = build_style_document(manager.style_bank(10001, persona_key="member")[0])
    assert after.situation == ""
    assert before.document_hash != after.document_hash

    class FakeEmbedding:
        identity = type("Identity", (), {"provider": "test", "model": "fake", "version": "v1", "dimensions": 2})()
        def embed_query(self, text):
            return [1.0, 0.0]

    manager.embedding_provider = FakeEmbedding()
    manager._example_vectors[manager._example_cache_key(222, 10001)] = (
        {before.msg_id: before}, {before.msg_id: [1.0, 0.0]},
    )
    assert manager.retrieve_examples(
        10001, build_style_retrieval_query(current_text="游戏更新怎么延期了"), persona_key="member",
    ) == []


def test_style_bank_drops_legacy_unverifiable_adjacent_situation(sqlite_engine):
    from app.storage.db import session_scope
    from app.storage.repositories import PersonaStyleExampleRepository

    with session_scope(sqlite_engine) as session:
        _add_fact_source(session, "live-style", "游戏延期只能继续等")
        PersonaStyleExampleRepository(session).insert_many([
            {"group_id": 10001, "user_id": 222, "msg_id": "live-style", "text": "游戏延期只能继续等",
             "context_before": [{"speaker": "unknown", "text": "unverifiable live situation"}], "reply_target": None},
            {"group_id": 10001, "user_id": 222, "msg_id": "imported-style", "text": "imported response",
             "context_before": [{"speaker": "unknown", "text": "compatible imported situation"}], "reply_target": None},
        ])
    manager = PersonaManager(engine=sqlite_engine, personas={"member": {"source_user_id": 222, "source_group_id": 10001}}, default_persona={})
    manager.load_style_banks()
    bank = {entry["msg_id"]: entry for entry in manager.style_bank(10001, persona_key="member")}
    assert bank["live-style"]["context_before"] == []
    assert bank["imported-style"]["context_before"][0]["text"] == "compatible imported situation"


def test_example_vectors_persist_across_manager_instances(sqlite_engine) -> None:
    from app.core.persona_switch import PersonaManager
    from app.storage.db import session_scope
    from app.storage.models import PersonaExampleVector
    from app.storage.repositories import PersonaStyleExampleRepository

    class _FakeEmbedding:
        available = True

        def __init__(self):
            self.document_calls = 0

        class _Identity:
            provider = "test"
            model = "fake"
            version = "v1"
            dimensions = 4

        identity = _Identity()

        def embed_documents(self, texts):
            self.document_calls += 1
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
                    "text": "我在",
                    "context_before": [],
                    "context_after": [],
                    "reply_target": "在吗",
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
    first_provider = _FakeEmbedding()
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
        embedding_provider=first_provider,
    )
    manager.load_state()
    manager._group_keys[10001] = "test_self"
    assert manager.prewarm_examples(10001, "test_self") == 1
    assert first_provider.document_calls == 1
    picked = manager.retrieve_examples(10001, ["在吗"], limit=1)
    assert picked

    with session_scope(sqlite_engine) as session:
        assert session.query(PersonaExampleVector).filter_by(user_id=222).count() == 1

    fresh_provider = _FakeEmbedding()
    fresh = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
        embedding_provider=fresh_provider,
    )
    fresh.load_state()
    fresh._group_keys[10001] = "test_self"
    assert fresh.prewarm_examples(10001, "test_self") == 1
    assert fresh_provider.document_calls == 0
    picked_again = fresh.retrieve_examples(10001, ["在吗"], limit=1)
    assert picked_again


def test_example_vectors_ignore_and_purge_stale_persisted_rows(sqlite_engine) -> None:
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
                    "text": "我在",
                    "context_before": [],
                    "context_after": [],
                    "reply_target": "在吗",
                }
            ]
        )
        session.add(
            PersonaExampleVector(
                msg_id="stale-msg",
                user_id=222,
                group_id=10001,
                vector_json=json.dumps([1.0, 0.0, 0.0, 0.0]),
            )
        )
        session.commit()
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

    assert manager.prewarm_examples(10001, "test_self") == 1
    picked = manager.retrieve_examples(10001, ["在吗"], limit=1)

    assert picked
    with session_scope(sqlite_engine) as session:
        assert (
            session.query(PersonaExampleVector)
            .filter_by(user_id=222, msg_id="stale-msg")
            .count()
            == 0
        )


def test_style_bank_is_scoped_by_source_group(sqlite_engine) -> None:
    from app.core.persona_switch import PersonaManager
    from app.storage.db import session_scope
    from app.storage.repositories import PersonaStyleExampleRepository

    with session_scope(sqlite_engine) as session:
        PersonaStyleExampleRepository(session).insert_many(
            [
                {
                    "group_id": 10001,
                    "user_id": 222,
                    "msg_id": "source-group",
                    "text": "本群说法",
                    "context_before": [],
                },
                {
                    "group_id": 10002,
                    "user_id": 222,
                    "msg_id": "other-group",
                    "text": "其他群说法",
                    "context_before": [],
                },
            ]
        )
    personas = {
        "default": {"name": "测试小町"},
        "member": {
            "name": "测试君",
            "source_user_id": 222,
            "source_group_id": 10001,
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )

    manager.load_style_banks()
    bank = manager.style_bank(10001, persona_key="member")

    assert [entry["msg_id"] for entry in bank] == ["source-group"]


def test_style_retrieval_log_contains_metrics_but_not_chat_text(
    sqlite_engine,
    caplog,
) -> None:
    from app.core.persona_style_retrieval import build_style_retrieval_query
    from app.core.persona_switch import PersonaManager
    from app.storage.db import session_scope
    from app.storage.repositories import PersonaStyleExampleRepository

    class _FakeEmbedding:
        available = True
        identity = type(
            "Identity",
            (),
            {"provider": "test", "model": "fake", "version": "v1", "dimensions": 2},
        )()

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    with session_scope(sqlite_engine) as session:
        PersonaStyleExampleRepository(session).insert_many(
            [
                {
                    "group_id": 10001,
                    "user_id": 222,
                    "msg_id": "safe-log-id",
                    "text": "这是绝密候选正文",
                    "reply_target": "群友: 绝密查询词相关吗",
                    "context_before": [],
                }
            ]
        )
    personas = {
        "default": {"name": "测试小町"},
        "member": {
            "name": "测试君",
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
    manager.load_style_banks()
    manager.prewarm_examples(10001, "member")
    caplog.clear()

    with caplog.at_level("INFO"):
        manager.retrieve_examples(
            10001,
            build_style_retrieval_query(
                current_text="绝密查询词是什么",
                current_message_id="query-id",
            ),
            persona_key="member",
        )

    output = caplog.text
    assert "persona_style_retrieval" in output
    assert "selected_count=" in output
    assert "query-id" in output
    assert "绝密查询词是什么" not in output
    assert "这是绝密候选正文" not in output
