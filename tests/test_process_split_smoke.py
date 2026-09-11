from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.group_main as group_main
import app.main as app_main
import app.private_main as private_main
from app.adapters.onebot_models import resolve_message_type
from app.config import AppSettings
from app.core.router import InboundRouter
from app.private_chat.service import PrivateChatService
from app.providers.semantic_embeddings import DisabledEmbeddingProvider


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:3001",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="gpt-5.4",
        llm_fallback_model="",
        llm_text_endpoint="chat_completions",
        group_image_base_url="",
        group_image_api_key="",
        group_image_size="auto",
        bot_qq=123456789,
        owner_qq=987654321,
        search_provider="tavily",
        search_base_url="https://api.tavily.com/search",
        search_api_key="search-key",
        search_timeout_seconds=8.0,
        search_region="wt-wt",
        search_backend="auto",
        context_recent_limit=60,
        context_summary_limit=3,
        context_history_limit=8,
        config_dir=tmp_path / "configs",
        data_dir=tmp_path / "data",
    )


class FakeGateway:
    instances: list["FakeGateway"] = []

    def __init__(
        self,
        *,
        ws_url: str,
        reconnect_forever: bool = False,
        reconnect_delay_seconds: float = 0.0,
    ) -> None:
        self.ws_url = ws_url
        self.reconnect_forever = reconnect_forever
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.websocket = None
        self.__class__.instances.append(self)

    async def connect_and_consume(self, handler, on_connect=None) -> None:
        self.websocket = object()
        if on_connect is not None:
            await on_connect()
        await handler({})


@pytest.mark.asyncio
async def test_group_main_builds_router_without_private_chat_service(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}
    built_memory_orchestrator = object()
    memory_lifecycle_events: list[str] = []
    FakeGateway.instances.clear()

    def fake_llm_client(**kwargs):
        captured["llm_kwargs"] = kwargs
        return object()

    class FakeMemoryCompactionService:
        async def start(self) -> None:
            memory_lifecycle_events.append("start")

        async def stop(self) -> None:
            memory_lifecycle_events.append("stop")

    monkeypatch.setattr(group_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        group_main,
        "load_runtime_config",
        lambda provided_settings: SimpleNamespace(
            persona={"name": "bot"},
            group_policy={},
            safety={},
            settings=provided_settings,
        ),
    )
    monkeypatch.setattr(group_main, "build_engine", lambda _path: object())
    monkeypatch.setattr(group_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(group_main, "sync_history_archives", lambda engine, runtime: None)
    monkeypatch.setattr(group_main, "NapCatGateway", FakeGateway)
    monkeypatch.setattr(group_main, "Sender", lambda _gateway: object())
    monkeypatch.setattr(app_main, "LlmClient", fake_llm_client)
    monkeypatch.setattr(group_main, "ReplyPolicy", lambda: object())
    monkeypatch.setattr(group_main, "ContextBuilder", lambda: object())
    monkeypatch.setattr(group_main, "build_web_search_client", lambda _settings: object())
    monkeypatch.setattr(group_main, "build_group_image_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        group_main,
        "build_memory_runtime",
        lambda **_kwargs: SimpleNamespace(
            memory_compaction_service=FakeMemoryCompactionService(),
            memory_orchestrator=built_memory_orchestrator,
            embedding_provider=DisabledEmbeddingProvider(),
        ),
    )
    monkeypatch.setattr(group_main, "InboundRouter", lambda **kwargs: captured.update(kwargs) or object())
    monkeypatch.setattr(
        group_main,
        "PersonaManager",
        lambda **kwargs: SimpleNamespace(load_state=lambda: None),
    )
    monkeypatch.setattr(group_main, "PersonaSwitchService", lambda **kwargs: object())
    monkeypatch.setattr(group_main, "_max_message_id", lambda _engine: 0)

    async def _noop_replay(**kwargs) -> None:
        del kwargs

    monkeypatch.setattr(
        group_main,
        "_replay_startup_window_mentions",
        _noop_replay,
    )

    await group_main.run()

    assert captured["private_chat_service"] is None
    assert captured["memory_orchestrator"] is built_memory_orchestrator
    assert captured["llm_kwargs"]["responses_model"] == "gpt-5.4"
    assert captured["llm_kwargs"]["responses_only"] is True
    assert memory_lifecycle_events == ["start", "stop"]
    assert len(FakeGateway.instances) == 1
    assert FakeGateway.instances[0].reconnect_forever is True


async def test_private_main_composes_chat_only_private_service(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}
    search_client = object()
    image_client = object()
    reminder_events: list[str] = []
    FakeGateway.instances.clear()

    def fake_llm_client(**kwargs):
        captured["llm_kwargs"] = kwargs
        return object()

    class FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeReminderScheduler:
        def __init__(self, **kwargs) -> None:
            captured["reminder_scheduler"] = kwargs

        async def start(self) -> None:
            reminder_events.append("start")

        async def stop(self) -> None:
            reminder_events.append("stop")

    monkeypatch.setattr(private_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        private_main,
        "load_runtime_config",
        lambda provided_settings: SimpleNamespace(
            persona={"name": "bot"},
            group_policy={},
            safety={},
            settings=provided_settings,
        ),
    )
    monkeypatch.setattr(private_main, "build_engine", lambda _path: object())
    monkeypatch.setattr(private_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(private_main, "NapCatGateway", FakeGateway)
    monkeypatch.setattr(private_main, "Sender", lambda _gateway: object())
    monkeypatch.setattr(app_main, "LlmClient", fake_llm_client)
    monkeypatch.setattr(private_main, "ReplyPolicy", lambda: object())
    monkeypatch.setattr(private_main, "ContextBuilder", lambda: object())
    monkeypatch.setattr(private_main, "build_web_search_client", lambda _settings: search_client)
    monkeypatch.setattr(
        private_main,
        "build_group_image_llm_client",
        lambda **_kwargs: image_client,
    )
    monkeypatch.setattr(private_main, "load_private_reminders", lambda *, config_dir: ["reminder"])
    monkeypatch.setattr(private_main, "PrivateReminderScheduler", FakeReminderScheduler)
    monkeypatch.setattr(private_main, "PrivateChatService", FakeService)
    monkeypatch.setattr(private_main, "InboundRouter", lambda **_kwargs: object())

    await private_main.run()

    assert captured["web_search_client"] is search_client
    # Private drawings must use the pinned image transport, not the chat model
    # that serves text replies (the split container previously fell back to it
    # and every private draw answered with the failure notice).
    assert captured["image_llm_client"] is image_client
    assert captured["image_llm_client"] is not captured["llm_client"]
    assert captured["image_model"] == "gpt-image-2"
    assert captured["image_size"] == "auto"
    assert captured["image_max_attempts"] == 1
    assert captured["reminder_scheduler"]["reminders"] == ["reminder"]
    assert captured["llm_kwargs"]["responses_model"] == "gpt-5.4"
    assert captured["llm_kwargs"]["responses_only"] is True
    assert reminder_events == ["start", "stop"]
    assert len(FakeGateway.instances) == 1
    assert FakeGateway.instances[0].reconnect_forever is True


@pytest.mark.asyncio
async def test_private_main_waits_for_gateway_before_starting_services(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)
    events: list[str] = []
    search_client = object()
    FakeGateway.instances.clear()

    class OrderedGateway(FakeGateway):
        async def connect_and_consume(self, handler, on_connect=None) -> None:
            events.append("gateway-connect")
            await asyncio.sleep(0)
            self.websocket = object()
            events.append("gateway-ready")
            if on_connect is not None:
                await on_connect()
            await handler({})

    class FakeService:
        def __init__(self, **_kwargs) -> None:
            return None

        async def start(self) -> None:
            assert OrderedGateway.instances[0].websocket is not None
            events.append("service-start")

        async def stop(self) -> None:
            events.append("service-stop")

    class FakeReminderScheduler:
        def __init__(self, **_kwargs) -> None:
            return None

        async def start(self) -> None:
            assert OrderedGateway.instances[0].websocket is not None
            events.append("reminder-start")

        async def stop(self) -> None:
            events.append("reminder-stop")

    monkeypatch.setattr(private_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        private_main,
        "load_runtime_config",
        lambda provided_settings: SimpleNamespace(
            persona={"name": "bot"},
            group_policy={},
            safety={},
            settings=provided_settings,
        ),
    )
    monkeypatch.setattr(private_main, "build_engine", lambda _path: object())
    monkeypatch.setattr(private_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(private_main, "NapCatGateway", OrderedGateway)
    monkeypatch.setattr(private_main, "Sender", lambda _gateway: object())
    monkeypatch.setattr(app_main, "LlmClient", lambda **_kwargs: object())
    monkeypatch.setattr(private_main, "ReplyPolicy", lambda: object())
    monkeypatch.setattr(private_main, "ContextBuilder", lambda: object())
    monkeypatch.setattr(private_main, "build_web_search_client", lambda _settings: search_client)
    monkeypatch.setattr(
        private_main,
        "build_group_image_llm_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(private_main, "load_private_reminders", lambda *, config_dir: ["reminder"])
    monkeypatch.setattr(private_main, "PrivateReminderScheduler", FakeReminderScheduler)
    monkeypatch.setattr(private_main, "PrivateChatService", FakeService)
    monkeypatch.setattr(private_main, "InboundRouter", lambda **_kwargs: object())

    await private_main.run()

    assert events[:4] == [
        "gateway-connect",
        "gateway-ready",
        "service-start",
        "reminder-start",
    ]
    assert events[-2:] == ["reminder-stop", "service-stop"]


# Placeholder QQ IDs only: real account numbers never enter tracked files.
PRIVATE_BOT_QQ = 900000103
PRIVATE_OWNER_QQ = 900000101
PRIVATE_ALLOWLIST_QQ = 900000102


def _snowluma_private_payload(*, user_id: int, text: str) -> dict[str, object]:
    """One direct-chat event exactly as SnowLuma's OneBot server emits it.

    Verified against the deployed bridge on 2026-09-11: ``message_type`` is
    ``private`` with ``sub_type: "friend"``, the ``message`` field is the
    OneBot array format and message IDs may be negative.
    """

    return {
        "time": 1789137600,
        "self_id": PRIVATE_BOT_QQ,
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": -1747119563,
        "user_id": user_id,
        "message": [{"type": "text", "data": {"text": text}}],
        "raw_message": text,
        "font": 0,
        "sender": {"user_id": user_id, "nickname": "placeholder", "card": ""},
        "anonymous": None,
    }


class RecordingPrivateSender:
    def __init__(self) -> None:
        self.private_sent = []

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)


class RecordingPrivateLlmClient:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.prompts: list[list[str]] = []

    def generate_text(self, prompt_lines, *, images=None, conversation_key=None, temperature=None):
        del images, conversation_key, temperature
        self.prompts.append(list(prompt_lines))
        return self.reply_text


async def _drive_private_process_with_payload(
    monkeypatch,
    *,
    sqlite_engine,
    settings: AppSettings,
    payload: dict,
    private_chat_qqs: set[int],
):
    """Run the real private-process composition with one recorded OneBot event."""

    sender = RecordingPrivateSender()
    llm_client = RecordingPrivateLlmClient(reply_text="我在的，怎么啦")
    private_chat_service = PrivateChatService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=settings.owner_qq,
        bot_qq=settings.bot_qq,
        private_chat_qqs=private_chat_qqs,
        data_dir=settings.data_dir,
        assistant_name="小町",
        persona={"name": "小町"},
        safety={},
    )
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        private_chat_service=private_chat_service,
    )

    class SnowLumaGateway:
        def __init__(
            self,
            *,
            ws_url: str,
            reconnect_forever: bool = False,
            reconnect_delay_seconds: float = 0.0,
        ) -> None:
            self.ws_url = ws_url
            self.reconnect_forever = reconnect_forever
            self.reconnect_delay_seconds = reconnect_delay_seconds
            self.websocket = None

        async def connect_and_consume(self, handler, on_connect=None) -> None:
            self.websocket = object()
            if on_connect is not None:
                await on_connect()
            await handler(payload)

    class FakeReminderScheduler:
        def __init__(self, **_kwargs) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(private_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        private_main,
        "load_runtime_config",
        lambda provided_settings: SimpleNamespace(
            persona={"name": "小町"},
            group_policy={},
            safety={},
            settings=provided_settings,
        ),
    )
    monkeypatch.setattr(private_main, "build_engine", lambda _path: sqlite_engine)
    monkeypatch.setattr(private_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(private_main, "NapCatGateway", SnowLumaGateway)
    monkeypatch.setattr(private_main, "Sender", lambda _gateway: sender)
    monkeypatch.setattr(private_main, "build_llm_client", lambda **_kwargs: llm_client)
    monkeypatch.setattr(private_main, "build_web_search_client", lambda _settings: None)
    # The real builder needs a concrete chat transport; this test drives the
    # event path, so the pinned image client is only stubbed out.
    monkeypatch.setattr(
        private_main,
        "build_group_image_llm_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(private_main, "load_private_reminders", lambda *, config_dir: [])
    monkeypatch.setattr(private_main, "PrivateReminderScheduler", FakeReminderScheduler)
    monkeypatch.setattr(private_main, "PrivateChatService", lambda **_kwargs: private_chat_service)
    monkeypatch.setattr(private_main, "InboundRouter", lambda **_kwargs: router)

    await private_main.run()
    return sender, llm_client


def _private_settings(tmp_path) -> AppSettings:
    settings = _settings(tmp_path)
    settings.bot_qq = PRIVATE_BOT_QQ
    settings.owner_qq = PRIVATE_OWNER_QQ
    settings.private_chat_qqs = ""
    return settings


def test_legacy_combined_entry_imports_the_message_type_helper() -> None:
    # app/main.py called resolve_message_type without importing it, so every
    # inbound payload raised NameError once the runtime was upgraded.
    assert app_main.resolve_message_type is resolve_message_type


@pytest.mark.asyncio
async def test_private_process_answers_snowluma_owner_dm(
    monkeypatch, tmp_path, sqlite_engine
) -> None:
    settings = _private_settings(tmp_path)

    sender, llm_client = await _drive_private_process_with_payload(
        monkeypatch,
        sqlite_engine=sqlite_engine,
        settings=settings,
        payload=_snowluma_private_payload(user_id=PRIVATE_OWNER_QQ, text="你在吗"),
        private_chat_qqs=set(),
    )

    assert [outbound.user_id for outbound in sender.private_sent] == [PRIVATE_OWNER_QQ]
    assert sender.private_sent[0].text == "我在的，怎么啦"
    assert llm_client.prompts


@pytest.mark.asyncio
async def test_private_process_answers_snowluma_allowlisted_dm(
    monkeypatch, tmp_path, sqlite_engine
) -> None:
    settings = _private_settings(tmp_path)

    sender, _llm_client = await _drive_private_process_with_payload(
        monkeypatch,
        sqlite_engine=sqlite_engine,
        settings=settings,
        payload=_snowluma_private_payload(user_id=PRIVATE_ALLOWLIST_QQ, text="在吗"),
        private_chat_qqs={PRIVATE_ALLOWLIST_QQ},
    )

    assert [outbound.user_id for outbound in sender.private_sent] == [PRIVATE_ALLOWLIST_QQ]


@pytest.mark.asyncio
async def test_private_process_ignores_group_payloads(
    monkeypatch, tmp_path, sqlite_engine
) -> None:
    settings = _private_settings(tmp_path)
    group_payload = {
        "time": 1789137600,
        "self_id": PRIVATE_BOT_QQ,
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": -997668361,
        "group_id": 900000001,
        "user_id": PRIVATE_ALLOWLIST_QQ,
        "message": [{"type": "text", "data": {"text": "群里说话"}}],
        "raw_message": "群里说话",
        "sender": {"nickname": "placeholder", "card": ""},
    }

    sender, _llm_client = await _drive_private_process_with_payload(
        monkeypatch,
        sqlite_engine=sqlite_engine,
        settings=settings,
        payload=group_payload,
        private_chat_qqs={PRIVATE_ALLOWLIST_QQ},
    )

    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_private_process_warns_on_unknown_message_type(
    monkeypatch, tmp_path, sqlite_engine, caplog
) -> None:
    settings = _private_settings(tmp_path)
    payload = _snowluma_private_payload(user_id=PRIVATE_OWNER_QQ, text="你在吗")
    payload["message_type"] = "guild"

    sender, _llm_client = await _drive_private_process_with_payload(
        monkeypatch,
        sqlite_engine=sqlite_engine,
        settings=settings,
        payload=payload,
        private_chat_qqs=set(),
    )

    assert sender.private_sent == []
    assert any(
        "inbound_message_unhandled process=private" in record.message
        for record in caplog.records
    )
