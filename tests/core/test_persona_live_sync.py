from __future__ import annotations

from datetime import UTC, datetime

from app.core.persona_live_sync import (
    PersonaLiveSyncService,
    _build_examples,
    _merge_profile,
)
from app.core.persona_switch import PersonaManager
from app.storage.db import session_scope
from app.storage.repositories import (
    GroupRepository,
    MessageRepository,
    PersonaStyleExampleRepository,
    PersonaStyleSyncStateRepository,
    UserRepository,
)


def _row(
    row_id: int,
    msg_id: str,
    user_id: int,
    text: str,
    *,
    reply_to: str | None = None,
    card: str = "",
    nickname: str = "",
) -> dict:
    return {
        "id": row_id,
        "platform_msg_id": msg_id,
        "group_id": 10001,
        "user_id": user_id,
        "plain_text": text,
        "msg_type": "text",
        "reply_to_msg_id": reply_to,
        "raw_json": {"sender": {"card": card, "nickname": nickname}},
        "timestamp": datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
    }


def test_build_examples_keeps_context_and_reply_target() -> None:
    rows = [
        _row(1, "m1", 111, "在吗", card="路人甲"),
        _row(2, "m2", 222, "老哥我在", reply_to="m1", card="测试君"),
        _row(3, "m3", 111, "好", reply_to="m2", card="路人甲"),
    ]

    examples = _build_examples(
        rows,
        user_id=222,
        bot_qqs={900001},
        bot_text_names={"测试小町"},
    )

    assert len(examples) == 1
    example = examples[0]
    assert example["text"] == "老哥我在"
    assert example["reply_target"] == "路人甲: 在吗"
    assert [item["speaker"] for item in example["context_before"]] == ["路人甲"]


def test_sync_service_collects_and_deduplicates(sqlite_engine) -> None:
    settings = _fake_settings()
    personas = {
        "default": {"name": "测试小町"},
        "test_self": {
            "name": "测试君",
            "identity": "group member",
            "source_user_id": 222,
            "source_group_id": 10001,
            "example_bank": ["我玩", "上号"],
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=manager,
    )

    with session_scope(sqlite_engine) as session:
        messages = MessageRepository(session)
        users = UserRepository(session)
        GroupRepository(session).upsert_group(
            group_id=10001, group_name="测试群", enabled=True, speak_enabled=True
        )
        users.upsert_user(user_id=111, nickname="路人甲", group_card="")
        users.upsert_user(user_id=222, nickname="测试君", group_card="测试君")
        messages.add_group_message(
            platform_msg_id="m1",
            group_id=10001,
            user_id=111,
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
            plain_text="在吗",
            raw_json={"sender": {"card": "路人甲"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="m2",
            group_id=10001,
            user_id=222,
            timestamp=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
            plain_text="老哥我在",
            raw_json={"sender": {"card": "测试君"}},
            msg_type="text",
            reply_to_msg_id="m1",
            mentioned_bot=False,
        )

    inserted = service._sync_examples("test_self", 222, 10001)
    assert inserted == 1
    assert service._sync_examples("test_self", 222, 10001) == 0

    with session_scope(sqlite_engine) as session:
        examples = PersonaStyleExampleRepository(session).load_active(user_id=222)
        assert {example.text for example in examples} >= {
            "我玩",
            "上号",
            "老哥我在",
        }
        state = PersonaStyleSyncStateRepository(session).get(
            group_id=10001, user_id=222
        )
        assert state is not None
        assert state.last_msg_id != ""

    manager.set_persona_key(10001, "test_self")
    bank = manager.style_bank(10001)
    assert any(entry["text"] == "老哥我在" for entry in bank)


def test_refresh_gate_skips_below_threshold(sqlite_engine) -> None:
    settings = _fake_settings()
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
    )
    manager.load_state()
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=manager,
    )
    with session_scope(sqlite_engine) as session:
        PersonaStyleSyncStateRepository(session).set_watermark(
            group_id=10001, user_id=222, last_msg_id="9", new_count=5
        )

    service._maybe_refresh_profile("test_self", 222, 10001)

    assert not (settings.data_dir / "personas" / "test_self.live.yaml").exists()


def test_tick_only_refreshes_personas_with_live_refresh_flag(
    sqlite_engine, monkeypatch
) -> None:
    settings = _fake_settings()
    personas = {
        "default": {"name": "测试小町"},
        "live_self": {
            "name": "直播君",
            "live_refresh": True,
            "source_user_id": 222,
            "source_group_id": 10001,
        },
        "member_self": {
            "name": "成员君",
            "source_user_id": 333,
            "source_group_id": 10001,
        },
    }
    manager = PersonaManager(
        engine=sqlite_engine,
        personas=personas,
        default_persona=personas["default"],
    )
    manager.load_state()
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=manager,
    )
    synced = []
    monkeypatch.setattr(service, "_sync_examples", lambda key, uid, gid: synced.append(key))
    monkeypatch.setattr(service, "_maybe_refresh_profile", lambda key, uid, gid: None)

    service._tick()

    assert synced == ["live_self"]


def test_refresh_triggers_on_threshold_and_overdue(
    sqlite_engine, monkeypatch
) -> None:
    from datetime import timedelta
    from pathlib import Path

    settings = _fake_settings()
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
    )
    manager.load_state()
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=manager,
    )
    calls = []

    def fake_write(**kwargs):
        del kwargs
        calls.append(1)
        return Path("unused.live.yaml")

    monkeypatch.setattr(service, "_write_refreshed_profile", fake_write)

    with session_scope(sqlite_engine) as session:
        PersonaStyleSyncStateRepository(session).set_watermark(
            group_id=10001, user_id=222, last_msg_id="9", new_count=50
        )
    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == [1]

    calls.clear()
    with session_scope(sqlite_engine) as session:
        state = PersonaStyleSyncStateRepository(session).get(
            group_id=10001, user_id=222
        )
        state.last_refresh_at = datetime.now(UTC) - timedelta(hours=25)
        session.add(state)
    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == [1]


def test_load_runtime_config_merges_live_persona(tmp_path) -> None:
    from app.config import AppSettings, load_runtime_config

    config_dir = tmp_path / "configs"
    (config_dir / "personas").mkdir(parents=True)
    (config_dir / "persona.yaml").write_text(
        "name: 测试小町\nidentity: AI\n", encoding="utf-8"
    )
    (config_dir / "groups.yaml").write_text("{}\n", encoding="utf-8")
    (config_dir / "safety.yaml").write_text("{}\n", encoding="utf-8")
    (config_dir / "personas" / "azha.yaml").write_text(
        "name: 阿渣\ncore_traits:\n- A\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    (data_dir / "personas").mkdir(parents=True)
    (data_dir / "personas" / "azha.live.yaml").write_text(
        "core_traits:\n- A\n- B\nspeech_habits:\n- 短句\n", encoding="utf-8"
    )
    settings = AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:1",
        llm_base_url="http://unused",
        llm_api_key="key",
        bot_qq=1,
        owner_qq=2,
        config_dir=config_dir,
        data_dir=data_dir,
    )

    runtime = load_runtime_config(settings)

    assert runtime.personas["azha"]["core_traits"] == ["A", "B"]
    assert runtime.personas["azha"]["speech_habits"] == ["短句"]


def test_merge_profile_replaces_lists_and_merges_mappings() -> None:
    merged = _merge_profile(
        {
            "name": "阿渣",
            "core_traits": ["A"],
            "speaking_style": {"tone": "casual", "sentence_length": "short"},
        },
        {
            "core_traits": ["A", "B"],
            "speaking_style": {"tone": "blunt"},
        },
    )

    assert merged["core_traits"] == ["A", "B"]
    assert merged["speaking_style"] == {"tone": "blunt", "sentence_length": "short"}


def test_merge_profile_unions_facts_and_external_relations() -> None:
    merged = _merge_profile(
        {
            "facts": [{"category": "游戏", "fact": "玩lolm"}],
            "external_relations": [{"name": "灰泽满", "relation": "铁粉"}],
        },
        {
            "facts": [
                {"category": "游戏", "fact": "玩lolm"},
                {"category": "体育", "fact": "看阿森纳"},
            ],
            "external_relations": [{"name": "灰泽满", "relation": "铁粉"}],
        },
    )

    assert [item.get("fact") for item in merged["facts"]] == ["玩lolm", "看阿森纳"]
    assert [item.get("name") for item in merged["external_relations"]] == ["灰泽满"]


class _fake_settings:
    from pathlib import Path

    data_dir = Path("data")
    bot_qq = 900001
    llm_base_url = "http://unused"
    llm_api_key = ""
    llm_model = "unused"
    llm_fallback_model = ""
    llm_reasoning_effort = ""
