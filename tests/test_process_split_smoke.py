from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.dev_worker_main as dev_worker_main
import app.group_main as group_main
import app.main as app_main
import app.private_main as private_main
from app.config import AppSettings


def _settings() -> AppSettings:
    return AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:3001",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="gpt-5.4",
        llm_fallback_model="",
        group_image_base_url="",
        group_image_api_key="",
        group_image_size="auto",
        bot_qq=123456789,
        owner_qq=987654321,
        admin_qqs="",
        search_provider="tavily",
        search_base_url="https://api.tavily.com/search",
        search_api_key="search-key",
        search_timeout_seconds=8.0,
        search_region="wt-wt",
        search_backend="auto",
        context_recent_limit=60,
        context_summary_limit=3,
        context_history_limit=8,
        config_dir=Path("configs"),
        data_dir=Path("data"),
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

    async def connect_and_consume(self, handler) -> None:
        self.websocket = object()
        await handler({})


@pytest.mark.asyncio
async def test_group_main_builds_router_without_dev_control(monkeypatch) -> None:
    settings = _settings()
    captured: dict[str, object] = {}
    FakeGateway.instances.clear()

    def fake_llm_client(**kwargs):
        captured["llm_kwargs"] = kwargs
        return object()

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
    monkeypatch.setattr(group_main, "AdminCommandParser", lambda **_kwargs: object())
    monkeypatch.setattr(group_main, "build_web_search_client", lambda _settings: object())
    monkeypatch.setattr(group_main, "InboundRouter", lambda **kwargs: captured.update(kwargs) or object())

    await group_main.run()

    assert captured["dev_control_service"] is None
    assert captured["llm_kwargs"]["responses_model"] == "gpt-5.4"
    assert captured["llm_kwargs"]["compat_model"] == "gpt-5.4"
    assert len(FakeGateway.instances) == 1
    assert FakeGateway.instances[0].reconnect_forever is True


async def test_private_main_disables_local_worker(monkeypatch) -> None:
    settings = _settings()
    captured: dict[str, object] = {}
    search_client = object()
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
    monkeypatch.setattr(private_main, "AdminCommandParser", lambda **_kwargs: object())
    monkeypatch.setattr(private_main, "build_web_search_client", lambda _settings: search_client)
    monkeypatch.setattr(private_main, "load_private_reminders", lambda *, config_dir: ["reminder"])
    monkeypatch.setattr(private_main, "PrivateReminderScheduler", FakeReminderScheduler)
    monkeypatch.setattr(private_main, "DevControlService", FakeService)
    monkeypatch.setattr(private_main, "InboundRouter", lambda **_kwargs: object())

    await private_main.run()

    assert captured["enable_local_worker"] is False
    assert captured["web_search_client"] is search_client
    assert captured["reminder_scheduler"]["reminders"] == ["reminder"]
    assert captured["llm_kwargs"]["responses_model"] == "gpt-5.4"
    assert captured["llm_kwargs"]["compat_model"] == "gpt-5.4"
    assert reminder_events == ["start", "stop"]
    assert len(FakeGateway.instances) == 1
    assert FakeGateway.instances[0].reconnect_forever is True


@pytest.mark.asyncio
async def test_private_main_waits_for_gateway_before_starting_services(monkeypatch) -> None:
    settings = _settings()
    events: list[str] = []
    search_client = object()
    FakeGateway.instances.clear()

    class OrderedGateway(FakeGateway):
        async def connect_and_consume(self, handler) -> None:
            events.append("gateway-connect")
            await asyncio.sleep(0)
            self.websocket = object()
            events.append("gateway-ready")
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
    monkeypatch.setattr(private_main, "AdminCommandParser", lambda **_kwargs: object())
    monkeypatch.setattr(private_main, "build_web_search_client", lambda _settings: search_client)
    monkeypatch.setattr(private_main, "load_private_reminders", lambda *, config_dir: ["reminder"])
    monkeypatch.setattr(private_main, "PrivateReminderScheduler", FakeReminderScheduler)
    monkeypatch.setattr(private_main, "DevControlService", FakeService)
    monkeypatch.setattr(private_main, "InboundRouter", lambda **_kwargs: object())

    await private_main.run()

    assert events[:4] == [
        "gateway-connect",
        "gateway-ready",
        "service-start",
        "reminder-start",
    ]
    assert events[-2:] == ["reminder-stop", "service-stop"]


@pytest.mark.asyncio
async def test_dev_worker_main_enables_local_worker(monkeypatch) -> None:
    settings = _settings()
    captured: dict[str, object] = {}

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

    monkeypatch.setattr(dev_worker_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(dev_worker_main, "build_engine", lambda _path: object())
    monkeypatch.setattr(dev_worker_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(dev_worker_main, "NapCatGateway", FakeGateway)
    monkeypatch.setattr(dev_worker_main, "Sender", lambda _gateway: object())
    monkeypatch.setattr(app_main, "LlmClient", fake_llm_client)
    monkeypatch.setattr(
        dev_worker_main,
        "load_runtime_config",
        lambda _settings: SimpleNamespace(
            persona={"name": "比企谷小町"},
            safety={"deny_prompt_leak": True},
        ),
    )
    monkeypatch.setattr(dev_worker_main, "DevControlService", FakeService)

    await dev_worker_main.run()

    assert captured["llm_kwargs"]["responses_model"] == "gpt-5.4"
    assert captured["llm_kwargs"]["compat_model"] == "gpt-5.4"
    assert captured["enable_local_worker"] is True
    assert captured["assistant_name"] == "比企谷小町"
