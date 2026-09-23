from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.core.persona_live_sync import (
    PersonaLiveSyncService,
    _build_examples,
    _merge_profile,
    _normalize_live_profile_contract,
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


def test_build_examples_drops_bot_lines_from_context() -> None:
    rows = [
        _row(1, "m1", 111, "在吗", card="路人甲"),
        _row(2, "m2", 900001, "在的", card="测试小町"),
        _row(3, "m3", 222, "老哥我在", reply_to="m2", card="测试君"),
    ]

    examples = _build_examples(
        rows,
        user_id=222,
        bot_qqs={900001},
        bot_text_names={"测试小町"},
    )

    assert len(examples) == 1
    example = examples[0]
    assert example["reply_target"] is None
    assert [item["speaker"] for item in example["context_before"]] == ["路人甲"]


def test_build_examples_keeps_context_after() -> None:
    rows = [
        _row(1, "m1", 111, "在吗", card="路人甲"),
        _row(2, "m2", 222, "老哥我在", card="测试君"),
        _row(3, "m3", 111, "好", card="路人甲"),
        _row(4, "m4", 333, "那就这么定了", card="路人乙"),
    ]

    examples = _build_examples(
        rows,
        user_id=222,
        bot_qqs={900001},
        bot_text_names={"测试小町"},
    )

    assert len(examples) == 1
    assert [item["text"] for item in examples[0]["context_after"]] == [
        "好",
        "那就这么定了",
    ]


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


def test_refresh_triggers_on_threshold_or_cooldown(
    sqlite_engine, monkeypatch
) -> None:
    from datetime import timedelta

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
            group_id=10001, user_id=222, last_msg_id="9", new_count=100
        )
        state = PersonaStyleSyncStateRepository(session).get(
            group_id=10001, user_id=222
        )
    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == [1]

    # Below threshold: refresh once the 24-hour fallback is due.
    with session_scope(sqlite_engine) as session:
        repo = PersonaStyleSyncStateRepository(session)
        state = repo.get(group_id=10001, user_id=222)
        state.last_refresh_at = datetime.now(UTC) - timedelta(hours=25)
        state.new_since_refresh = 50
        session.add(state)
        session.commit()
    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == [1, 1]


def test_refresh_cooldown_requires_new_messages_and_elapsed_day(
    sqlite_engine, monkeypatch
) -> None:
    from datetime import timedelta

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
    calls: list[int] = []
    monkeypatch.setattr(
        service,
        "_write_refreshed_profile",
        lambda **_kwargs: calls.append(1) or Path("unused.live.yaml"),
    )

    with session_scope(sqlite_engine) as session:
        repo = PersonaStyleSyncStateRepository(session)
        repo.set_watermark(
            group_id=10001, user_id=222, last_msg_id="9", new_count=50
        )
        state = repo.get(group_id=10001, user_id=222)
        state.last_refresh_at = datetime.now(UTC) - timedelta(hours=23)
        session.add(state)

    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == []

    with session_scope(sqlite_engine) as session:
        state = PersonaStyleSyncStateRepository(session).get(
            group_id=10001, user_id=222
        )
        state.last_refresh_at = datetime.now(UTC) - timedelta(hours=25)
        state.new_since_refresh = 0
        session.add(state)

    service._maybe_refresh_profile("test_self", 222, 10001)
    assert calls == []


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


def _valid_live_profile() -> dict:
    return {
        "name": "测试君",
        "identity": "群成员",
        "core_traits": ["直接"],
        "speaking_style": {"tone": "casual"},
        "self_concept": "普通群友",
        "speech_habits": ["短句"],
        "style_avoid": ["客服腔"],
        "relationships": [],
        "address_rules": [],
        "facts": [],
        "external_relations": [],
    }


def test_live_profile_contract_normalizes_speaking_habits_alias() -> None:
    profile = _valid_live_profile()
    profile["speaking_habits"] = profile.pop("speech_habits")

    normalized = _normalize_live_profile_contract(profile)

    assert normalized["speech_habits"] == ["短句"]
    assert "speaking_habits" not in normalized


def test_live_profile_contract_rejects_missing_required_field() -> None:
    profile = _valid_live_profile()
    del profile["address_rules"]

    with pytest.raises(ValueError, match="missing required fields: address_rules"):
        _normalize_live_profile_contract(profile)


def test_live_profile_contract_inherits_missing_field_from_current_profile() -> None:
    current = _valid_live_profile()
    profile = _valid_live_profile()
    del profile["address_rules"]

    normalized = _normalize_live_profile_contract(
        profile,
        fallback_profile=current,
    )

    assert normalized["address_rules"] == current["address_rules"]


def test_live_profile_contract_rejects_wrong_top_level_type() -> None:
    profile = _valid_live_profile()
    profile["speech_habits"] = "短句"

    with pytest.raises(ValueError, match="invalid field types: speech_habits"):
        _normalize_live_profile_contract(profile)


def test_live_profile_contract_rejects_unknown_field() -> None:
    profile = _valid_live_profile()
    profile["speech_pattern"] = ["未知字段"]

    with pytest.raises(ValueError, match="unknown fields: speech_pattern"):
        _normalize_live_profile_contract(profile)


def test_live_profile_contract_accepts_valid_profile() -> None:
    profile = _valid_live_profile()

    assert _normalize_live_profile_contract(profile) == profile


def test_write_refreshed_profile_rejects_malformed_output_before_write(
    tmp_path, monkeypatch
) -> None:
    from app.providers.llm_client import LlmClient

    current = {**_valid_live_profile(), "source_user_id": 222}
    del current["address_rules"]
    personas = {"default": {"name": "测试小町"}, "test_self": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=None,
        settings=settings,
        personas=personas,
        manager=SimpleNamespace(personas=personas),
    )
    service._window_transcript_block = lambda **kwargs: "测试君: 短句"
    malformed = _valid_live_profile()
    del malformed["address_rules"]
    monkeypatch.setattr(
        LlmClient,
        "generate_text",
        lambda self, prompt: yaml.safe_dump(malformed, allow_unicode=True),
    )

    with pytest.raises(ValueError, match="missing required fields: address_rules"):
        service._write_refreshed_profile(
            persona_key="test_self",
            current_profile=current,
            examples=[],
            user_id=222,
            group_id=10001,
        )

    assert not (tmp_path / "personas" / "test_self.live.yaml").exists()
    assert personas["test_self"] is current


def test_write_refreshed_profile_writes_valid_output_and_preserves_current_fields(
    tmp_path, monkeypatch
) -> None:
    from app.providers.llm_client import LlmClient

    current = {**_valid_live_profile(), "source_user_id": 222}
    personas = {"default": {"name": "测试小町"}, "test_self": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=None,
        settings=settings,
        personas=personas,
        manager=SimpleNamespace(personas=personas),
    )
    service._window_transcript_block = lambda **kwargs: "测试君: 确实"
    refreshed = _valid_live_profile()
    refreshed["speaking_habits"] = ["确实"]
    del refreshed["speech_habits"]

    def generate_valid(self, prompt):
        assert self.reasoning_effort == "medium"
        return yaml.safe_dump(refreshed, allow_unicode=True)

    monkeypatch.setattr(
        LlmClient,
        "generate_text",
        generate_valid,
    )

    live_path = service._write_refreshed_profile(
        persona_key="test_self",
        current_profile=current,
        examples=[],
        user_id=222,
        group_id=10001,
    )

    written = yaml.safe_load(live_path.read_text(encoding="utf-8"))
    assert written["speech_habits"] == ["确实"]
    assert "speaking_habits" not in written
    assert personas["test_self"]["source_user_id"] == 222
    assert personas["test_self"]["speech_habits"] == ["短句", "确实"]


def test_window_transcript_block_renders_flow_with_image_placeholder(sqlite_engine) -> None:
    from datetime import timedelta

    from app.storage.repositories import MessageRepository

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
    base = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001, group_name="test", enabled=True, speak_enabled=True
        )
        UserRepository(session).upsert_user(
            user_id=111, nickname="路人甲", group_card=""
        )
        UserRepository(session).upsert_user(
            user_id=222, nickname="测试君", group_card=""
        )
        UserRepository(session).upsert_user(
            user_id=900001, nickname="测试小町", group_card=""
        )
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="ctx-1",
            group_id=10001,
            user_id=111,
            timestamp=base - timedelta(minutes=1),
            plain_text="发张图看看",
            raw_json={"sender": {"nickname": "路人甲", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="img-1",
            group_id=10001,
            user_id=222,
            timestamp=base,
            plain_text="",
            raw_json={
                "sender": {"nickname": "测试君", "card": ""},
                "message": [{"type": "image", "data": {"url": "http://img/x.png"}}],
            },
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="ctx-2",
            group_id=10001,
            user_id=111,
            timestamp=base + timedelta(minutes=1),
            plain_text="好看",
            raw_json={"sender": {"nickname": "路人甲", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.commit()
    examples = [
        {
            "msg_id": "ctx-1",
            "text": "发张图看看",
            "timestamp": base - timedelta(minutes=1),
        },
        {
            "msg_id": "ctx-2",
            "text": "好看",
            "timestamp": base + timedelta(minutes=1),
        },
    ]
    transcript = service._window_transcript_block(
        user_id=222,
        group_id=10001,
        examples=examples,
    )
    assert "[图片]" in transcript
    assert "路人甲: 发张图看看" in transcript
    assert "路人甲: 好看" in transcript


def test_window_transcript_skips_bot_lines(sqlite_engine) -> None:
    from datetime import timedelta

    from app.storage.repositories import MessageRepository

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
    base = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001, group_name="test", enabled=True, speak_enabled=True
        )
        UserRepository(session).upsert_user(
            user_id=111, nickname="路人甲", group_card=""
        )
        UserRepository(session).upsert_user(
            user_id=222, nickname="测试君", group_card=""
        )
        UserRepository(session).upsert_user(
            user_id=900001, nickname="测试小町", group_card=""
        )
        messages = MessageRepository(session)
        messages.add_group_message(
            platform_msg_id="img-2",
            group_id=10001,
            user_id=222,
            timestamp=base,
            plain_text="",
            raw_json={
                "sender": {"nickname": "测试君", "card": ""},
                "message": [{"type": "image", "data": {"url": "http://img/y.png"}}],
            },
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="react-1",
            group_id=10001,
            user_id=111,
            timestamp=base + timedelta(seconds=10),
            plain_text="这图真不错",
            raw_json={"sender": {"nickname": "路人甲", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="bot-1",
            group_id=10001,
            user_id=900001,
            timestamp=base + timedelta(seconds=20),
            plain_text="机器人发言不应出现",
            raw_json={"sender": {"nickname": "测试小町", "card": ""}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        session.commit()
    transcript = service._window_transcript_block(
        user_id=222,
        group_id=10001,
        examples=[
            {
                "msg_id": "img-2",
                "text": "",
                "timestamp": base,
            },
            {
                "msg_id": "react-1",
                "text": "这图真不错",
                "timestamp": base + timedelta(seconds=10),
            }
        ],
    )
    assert "测试君: [图片]" in transcript
    assert "路人甲: 这图真不错" in transcript
    assert "机器人发言不应出现" not in transcript


class _fake_settings:
    from pathlib import Path

    data_dir = Path("data")
    bot_qq = 900001
    llm_base_url = "http://unused"
    llm_api_key = ""
    llm_model = "unused"
    llm_fallback_model = ""
    llm_reasoning_effort = ""
