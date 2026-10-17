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
    _PersonaTranscriptPacket,
    _PersonaTranscriptSource,
    PersonaRefreshSourcesRevokedError,
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
    assert example["context_before"][0]["source_msg_id"] == "m1"
    assert example["context_before"][0]["user_id"] == 111


def test_build_examples_drops_replies_to_bot_from_style_evidence() -> None:
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

    assert examples == []


def test_build_examples_drops_reply_to_bot_resolved_outside_sync_window() -> None:
    rows = [
        _row(30, "member-reply", 222, "知道了", card="测试君", reply_to="old-bot"),
    ]
    quoted_rows = {
        "old-bot": _row(1, "old-bot", 900001, "很久以前的机器人回复", card="测试小町")
    }

    examples = _build_examples(
        rows,
        user_id=222,
        bot_qqs={900001},
        bot_text_names={"测试小町"},
        quoted_rows=quoted_rows,
    )

    assert examples == []


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


@pytest.mark.parametrize("state", ["reserved", "blocked", "uncertain", "deleted"])
def test_build_examples_excludes_ineligible_author_quote_and_context(state):
    bad = _row(1, "bad", 111, "bad-source", card="路人甲")
    bad["raw_json"] = '{"delivery_state":"' + state + '"}'
    member = _row(2, "member", 222, "clean-reply", reply_to="bad")
    examples = _build_examples([bad, member], user_id=222, bot_qqs=set(), bot_text_names=set())
    assert len(examples) == 1
    assert examples[0]["reply_target"] is None
    assert examples[0]["context_before"] == []
    member["raw_json"] = {"delivery_state": state}
    assert _build_examples([bad, member], user_id=222, bot_qqs=set(), bot_text_names=set()) == []


def test_build_examples_still_rejects_a_reply_to_recalled_bot_content():
    bot = _row(1, "bad-bot", 900001, "recalled generated content")
    bot["raw_json"] = {"delivery_state": "deleted"}
    member = _row(2, "member", 222, "reply", reply_to="bad-bot")
    assert _build_examples([bot, member], user_id=222, bot_qqs={900001}, bot_text_names=set()) == []


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
    (config_dir / "personas" / "member_alpha.yaml").write_text(
        "name: 小甲\ncore_traits:\n- A\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    (data_dir / "personas").mkdir(parents=True)
    (data_dir / "personas" / "member_alpha.live.yaml").write_text(
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

    assert runtime.personas["member_alpha"]["core_traits"] == ["A", "B"]
    assert runtime.personas["member_alpha"]["speech_habits"] == ["短句"]


def test_merge_profile_replaces_lists_and_merges_mappings() -> None:
    merged = _merge_profile(
        {
            "name": "小甲",
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


def _persist_refresh_source(engine, *, message_id="refresh-source", user_id=222, text="独立真人发言"):
    stamp = datetime.now(UTC)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(
            group_id=10001, group_name="synthetic", enabled=True, speak_enabled=True,
        )
        UserRepository(session).upsert_user(user_id=user_id, nickname="synthetic", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id=message_id, group_id=10001, user_id=user_id,
            timestamp=stamp, plain_text=text, raw_json={"sender": {"nickname": "synthetic"}},
            msg_type="text", reply_to_msg_id=None, mentioned_bot=False,
        )
    return stamp


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


def test_live_profile_contract_accepts_existing_komachi_variant_flag() -> None:
    profile = _valid_live_profile()
    profile["komachi_variant"] = True

    assert _normalize_live_profile_contract(profile)["komachi_variant"] is True

    profile["komachi_variant"] = "true"
    with pytest.raises(ValueError, match="invalid field types: komachi_variant"):
        _normalize_live_profile_contract(profile)


def test_write_refreshed_profile_rejects_malformed_output_before_write(
    sqlite_engine, tmp_path, monkeypatch
) -> None:
    from app.providers.llm_client import LlmClient

    current = {**_valid_live_profile(), "source_user_id": 222}
    del current["address_rules"]
    personas = {"default": {"name": "测试小町"}, "test_self": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=SimpleNamespace(personas=personas),
    )
    _persist_refresh_source(sqlite_engine)
    service._window_transcript_packet = lambda **kwargs: _PersonaTranscriptPacket(
        "测试君: 短句", (_PersonaTranscriptSource("refresh-source", 222),),
    )
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
    sqlite_engine, tmp_path, monkeypatch
) -> None:
    from app.providers.llm_client import LlmClient

    current = {**_valid_live_profile(), "source_user_id": 222}
    personas = {"default": {"name": "测试小町"}, "test_self": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=sqlite_engine,
        settings=settings,
        personas=personas,
        manager=SimpleNamespace(personas=personas),
    )
    _persist_refresh_source(sqlite_engine)
    service._window_transcript_packet = lambda **kwargs: _PersonaTranscriptPacket(
        "测试君: 确实", (_PersonaTranscriptSource("refresh-source", 222),),
    )
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


@pytest.mark.parametrize("case,expected_calls", [("503", 1), ("authority_at_http", 0), ("sources_at_http", 0)])
def test_refresh_real_transport_is_single_request_and_failed_update_keeps_state(
    sqlite_engine, tmp_path, monkeypatch, case, expected_calls,
):
    import httpx
    from app.providers.llm_client import LlmClient
    stamp = _persist_refresh_source(sqlite_engine)
    current = {**_valid_live_profile(), "source_user_id": 222, "source_group_id": 10001, "live_refresh": True}
    personas = {"default": {"name": "合成小町"}, "member": current}
    settings = _fake_settings()
    settings.data_dir, settings.llm_fallback_model = tmp_path, "synthetic-fallback"
    manager = PersonaManager(engine=sqlite_engine, personas=personas, default_persona=personas["default"])
    service = PersonaLiveSyncService(engine=sqlite_engine, settings=settings, personas=personas, manager=manager,
                                     refresh_threshold=1)
    service._window_transcript_packet = lambda **kwargs: _PersonaTranscriptPacket(
        "合成成员: 合成语料", (_PersonaTranscriptSource("refresh-source", 222),),
    )
    with session_scope(sqlite_engine) as session:
        PersonaStyleSyncStateRepository(session).set_watermark(group_id=10001, user_id=222,
                                                              last_msg_id="refresh-source", new_count=1)
    calls, options, checks = [], [], []
    def guard():
        checks.append(1)
        if case == "authority_at_http" and len(checks) >= 4:
            raise RuntimeError("synthetic authority revoked")
    def respond(request):
        calls.append(request)
        return httpx.Response(503)
    actual_client = LlmClient
    def create_client(**kwargs):
        options.append(kwargs)
        if case == "sources_at_http":
            supplied = kwargs["request_guard"]
            def revoke_source_before_http():
                with session_scope(sqlite_engine) as session:
                    MessageRepository(session).mark_group_message_deleted(group_id=10001, platform_msg_id="refresh-source")
                supplied()
            kwargs["request_guard"] = revoke_source_before_http
        return actual_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("app.providers.llm_client.LlmClient", create_client)
    service._maybe_refresh_profile("member", 222, 10001, execution_guard=guard)
    assert len(calls) == expected_calls
    assert len(options) == 1
    assert options[0]["request_max_attempts"] == 1 and options[0]["fallback_model"] == ""
    assert options[0]["timeout_seconds"] == 60
    assert not (tmp_path / "personas" / "member.live.yaml").exists()
    assert personas["member"] is current
    with session_scope(sqlite_engine) as session:
        state = PersonaStyleSyncStateRepository(session).get(group_id=10001, user_id=222)
        assert state.new_since_refresh == 1 and state.last_refresh_at is None


def test_refresh_task_real_http_response_after_lease_expiry_does_not_publish_and_records_distinct_usage(
    sqlite_engine, tmp_path, monkeypatch,
):
    import httpx
    from datetime import timedelta
    from sqlalchemy import select
    from app.agent.service_tools import PersonaTaskService
    from app.providers.llm_client import LlmClient
    from app.storage.models import UsageRecord
    stamp = _persist_refresh_source(sqlite_engine)
    current = {**_valid_live_profile(), "source_user_id": 222, "source_group_id": 10001, "live_refresh": True}
    personas = {"default": {"name": "合成小町"}, "member": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    manager = PersonaManager(engine=sqlite_engine, personas=personas, default_persona=personas["default"])
    service = PersonaLiveSyncService(engine=sqlite_engine, settings=settings, personas=personas, manager=manager,
                                     refresh_threshold=1)
    service._window_transcript_packet = lambda **kwargs: _PersonaTranscriptPacket(
        "合成成员: 合成语料", (_PersonaTranscriptSource("refresh-source", 222),),
    )
    clock = [datetime.now(UTC)]
    tasks = PersonaTaskService(sqlite_engine, account_id=settings.bot_qq, clock=lambda: clock[0])
    with session_scope(sqlite_engine) as session:
        source_id = MessageRepository(session).get_by_platform_msg_id("refresh-source").id
    task = tasks.enqueue(group_id=10001, persona_key="member", source_user_id=222,
                         source_message_ids=(source_id,), request_key="synthetic-lease",
                         guard=SimpleNamespace(require_valid=lambda: None))
    calls = []
    def respond(request):
        calls.append(request)
        clock[0] += timedelta(seconds=601)
        return httpx.Response(200, json={"output_text": yaml.safe_dump(_valid_live_profile(), allow_unicode=True),
            "usage": {"input_tokens": 17, "output_tokens": 11, "input_tokens_details": {"cached_tokens": 3}}})
    actual_client = LlmClient
    monkeypatch.setattr("app.providers.llm_client.LlmClient", lambda **kwargs: actual_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    result = tasks.process_next(service, allow_paid=True)
    assert result["task_id"] == task["task_id"] and result["status"] == "failed"
    assert len(calls) == 1
    assert not (tmp_path / "personas" / "member.live.yaml").exists()
    assert personas["member"] is current
    with session_scope(sqlite_engine) as session:
        state = PersonaStyleSyncStateRepository(session).get(group_id=10001, user_id=222)
        assert state.last_refresh_at is None and state.new_since_refresh == 1
        usage = list(session.scalars(select(UsageRecord)))
        assert len(usage) == 1 and usage[0].endpoint == "persona_refresh_responses"
        assert (usage[0].input_tokens, usage[0].cached_input_tokens, usage[0].output_tokens) == (17, 3, 11)


def test_profile_refresh_without_eligible_sources_skips_model(tmp_path, monkeypatch):
    from app.providers.llm_client import LlmClient
    current = _valid_live_profile()
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(engine=None, settings=settings, personas={}, manager=SimpleNamespace())
    service._window_transcript_packet = lambda **kwargs: _PersonaTranscriptPacket()
    def forbidden(*args, **kwargs):
        pytest.fail("empty source packet must not call the model")
    monkeypatch.setattr(LlmClient, "generate_text", forbidden)
    with pytest.raises(ValueError, match="persona_refresh_no_eligible_sources"):
        service._write_refreshed_profile(persona_key="member", current_profile=current, examples=[], user_id=222, group_id=10001)


@pytest.mark.parametrize("role", ["target", "context"])
@pytest.mark.parametrize("change", ["deleted", "blocked", "uncertain", "wrong_group", "wrong_author", "outbound", "serialized_outbound"])
def test_profile_refresh_revalidates_every_source_after_model(
    sqlite_engine, tmp_path, monkeypatch, role, change,
):
    from app.providers.llm_client import LlmClient
    target_at = _persist_refresh_source(sqlite_engine, message_id="refresh-target", text="真人原话")
    _persist_refresh_source(sqlite_engine, message_id="refresh-context", user_id=333, text="他人原话")
    current = {**_valid_live_profile(), "source_user_id": 222, "source_group_id": 10001}
    personas = {"default": {"name": "测试小町"}, "member": current}
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=sqlite_engine, settings=settings, personas=personas,
        manager=SimpleNamespace(personas=personas),
    )
    live = tmp_path / "personas" / "member.live.yaml"
    live.parent.mkdir()
    live.write_text("existing profile marker\n", encoding="utf-8")
    refreshed = _valid_live_profile()
    refreshed["speech_habits"] = ["来自已失效原话的新习惯"]

    def generate(self, prompt):
        assert "真人原话" in prompt[0] and "他人原话" in prompt[0]
        with session_scope(sqlite_engine) as session:
            messages = MessageRepository(session)
            source = messages.get_by_platform_msg_id(f"refresh-{role}")
            if change in {"deleted", "blocked", "uncertain"}:
                source.raw_json = {**source.raw_json, "delivery_state": change}
            elif change == "wrong_group":
                GroupRepository(session).upsert_group(
                    group_id=10002, group_name="other", enabled=True, speak_enabled=True,
                )
                source.group_id = 10002
            elif change == "wrong_author":
                source.user_id = 333 if role == "target" else 222
            elif change == "serialized_outbound":
                import json
                source.raw_json = json.dumps({"direction": "outbound"})
            else:
                source.raw_json = {**source.raw_json, "direction": "outbound"}
        return yaml.safe_dump(refreshed, allow_unicode=True)

    monkeypatch.setattr(LlmClient, "generate_text", generate)
    with pytest.raises(PersonaRefreshSourcesRevokedError, match="sources_unavailable"):
        service._write_refreshed_profile(
            persona_key="member", current_profile=current,
            examples=[{"msg_id": "refresh-target", "timestamp": target_at}],
            user_id=222, group_id=10001,
        )
    assert live.read_text(encoding="utf-8") == "existing profile marker\n"
    assert personas["member"] is current


def test_profile_refresh_does_not_derive_target_persona_from_only_remaining_context(
    sqlite_engine, tmp_path, monkeypatch,
):
    from app.providers.llm_client import LlmClient
    stamp = _persist_refresh_source(sqlite_engine, message_id="revoked-target")
    _persist_refresh_source(sqlite_engine, message_id="remaining-context", user_id=333)
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(
            group_id=10001, platform_msg_id="revoked-target",
        )
    settings = _fake_settings()
    settings.data_dir = tmp_path
    service = PersonaLiveSyncService(
        engine=sqlite_engine, settings=settings, personas={}, manager=SimpleNamespace(),
    )
    monkeypatch.setattr(LlmClient, "generate_text", lambda *args: pytest.fail("no target source must skip model"))
    with pytest.raises(ValueError, match="no_eligible_sources"):
        service._write_refreshed_profile(
            persona_key="member", current_profile=_valid_live_profile(),
            examples=[{"msg_id": "revoked-target", "timestamp": stamp}],
            user_id=222, group_id=10001,
        )


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
