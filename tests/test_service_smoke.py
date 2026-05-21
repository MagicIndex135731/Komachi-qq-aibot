from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import app.main as app_main
import pytest

from app.config import AppSettings
from app.main import (
    build_web_search_client,
    create_runtime_banner,
    sync_history_archives,
    should_ingest_group_message,
    should_speak_in_group,
)
from app.providers.web_search import WebSearchClient
from scripts.backfill_summaries import backfill_lines
from scripts.import_history import load_export


def _settings_for_search(*, provider: str, search_api_key: str) -> AppSettings:
    return AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:3001",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="gpt-5.4",
        llm_text_endpoint="/chat/completions",
        bot_qq=123456789,
        owner_qq=987654321,
        admin_qqs="",
        search_provider=provider,
        search_base_url="https://api.tavily.com/search",
        search_api_key=search_api_key,
        search_timeout_seconds=8.0,
        search_region="wt-wt",
        search_backend="auto",
        context_recent_limit=60,
        context_summary_limit=3,
        context_history_limit=8,
        config_dir=Path("configs"),
        data_dir=Path("data"),
    )


def test_runtime_banner_includes_model_and_bot_id() -> None:
    banner = create_runtime_banner(bot_qq=123456789, model="gpt-5.4")
    assert "123456789" in banner
    assert "gpt-5.4" in banner


def test_load_export_returns_message_list(tmp_path) -> None:
    export_path = tmp_path / "history.json"
    export_path.write_text(
        json.dumps([{"message_id": "1"}, {"message_id": "2"}], ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_export(export_path) == [{"message_id": "1"}, {"message_id": "2"}]


def test_load_export_rejects_non_list_payload(tmp_path) -> None:
    export_path = tmp_path / "history.json"
    export_path.write_text(json.dumps({"message_id": "1"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="history export must be a list of message objects"):
        load_export(export_path)


def test_backfill_lines_summarizes_fixed_windows() -> None:
    lines = [f"line {index}" for index in range(1, 6)]

    assert backfill_lines(lines, window_size=2) == [
        "Recent chat summary: line 1 | line 2",
        "Recent chat summary: line 3 | line 4",
        "Recent chat summary: line 5",
    ]


def test_group_helpers_distinguish_ingest_and_speak() -> None:
    group_policy = {
        "default_group_behavior": {"speak": False, "archive": True},
        "groups": {
            "10001": {"enabled": True, "speak": True},
            "20002": {"enabled": True, "archive": True, "speak": False},
        },
    }

    assert should_ingest_group_message(group_id=10001, group_policy=group_policy) is True
    assert should_speak_in_group(group_id=10001, group_policy=group_policy) is True

    assert should_ingest_group_message(group_id=20002, group_policy=group_policy) is False
    assert should_speak_in_group(group_id=20002, group_policy=group_policy) is False

    assert should_ingest_group_message(group_id=40004, group_policy=group_policy) is False
    assert should_speak_in_group(group_id=40004, group_policy=group_policy) is False


def test_sync_history_archives_only_targets_speak_enabled_groups(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        app_main,
        "sync_group_message_archives_from_db",
        lambda *, engine, history_dir, allowed_group_ids: captured.update(
            {"engine": engine, "history_dir": history_dir, "allowed_group_ids": allowed_group_ids}
        )
        or {},
    )

    runtime = SimpleNamespace(
        settings=SimpleNamespace(data_dir=Path("data")),
        group_policy={
            "default_group_behavior": {"enabled": False, "speak": False},
            "groups": {
                "10001": {"enabled": True, "speak": True},
                "20002": {"enabled": True, "speak": False},
                "30003": {"enabled": False, "speak": True},
            },
        },
    )

    sync_history_archives(engine="engine", runtime=runtime)

    assert captured == {
        "engine": "engine",
        "history_dir": Path("data") / "history",
        "allowed_group_ids": {10001},
    }


def test_main_runs_async_entrypoint_and_returns_zero(monkeypatch) -> None:
    state = {"called": False}

    async def fake_run() -> None:
        state["called"] = True

    monkeypatch.setattr(app_main, "run", fake_run)

    assert app_main.main() == 0
    assert state["called"] is True


def test_build_web_search_client_returns_none_without_api_key() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="   ")

    assert build_web_search_client(settings) is None


def test_build_web_search_client_builds_client_with_api_key() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")

    client = build_web_search_client(settings)

    assert isinstance(client, WebSearchClient)
    assert client.provider == "tavily"
    assert client.base_url == "https://api.tavily.com/search"


def test_build_web_search_client_supports_ddgs_without_api_key() -> None:
    settings = _settings_for_search(provider="ddgs", search_api_key="   ")

    client = build_web_search_client(settings)

    assert isinstance(client, WebSearchClient)
    assert client.provider == "ddgs"


@pytest.mark.asyncio
async def test_run_wires_web_search_client_into_router(monkeypatch) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = "gpt-5.4"
    settings.llm_text_endpoint = "/chat/completions"
    router_arguments: dict[str, object] = {}
    sync_calls: list[tuple[object, object]] = []
    llm_kwargs: dict[str, object] = {}

    class FakeGateway:
        def __init__(self, *, ws_url: str) -> None:
            self.ws_url = ws_url

        async def connect_and_consume(self, handler) -> None:
            return None

    class FakeRouter:
        def __init__(self, **kwargs) -> None:
            router_arguments.update(kwargs)

    class FakeDevControlService:
        def __init__(
            self,
            *,
            engine,
            sender,
            llm_client,
            owner_qq,
            bot_qq=None,
            private_chat_qqs=None,
            repo_root,
            data_dir,
            codex_bridge=None,
            command_runner=None,
            poll_interval_seconds=1.0,
            enable_local_worker=True,
            private_image_followup_window_seconds=1.2,
            web_search_client=None,
            assistant_name="Codex",
            persona=None,
            safety=None,
        ) -> None:
            del codex_bridge, command_runner, poll_interval_seconds, private_image_followup_window_seconds
            router_arguments["dev_control_service_init"] = {
                "engine": engine,
                "sender": sender,
                "llm_client": llm_client,
                "owner_qq": owner_qq,
                "bot_qq": bot_qq,
                "private_chat_qqs": private_chat_qqs,
                "repo_root": repo_root,
                "data_dir": data_dir,
                "enable_local_worker": enable_local_worker,
                "web_search_client": web_search_client,
                "assistant_name": assistant_name,
                "persona": persona,
                "safety": safety,
            }

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(app_main, "AppSettings", lambda: settings)
    monkeypatch.setattr(
        app_main,
        "load_runtime_config",
        lambda provided_settings: SimpleNamespace(
            persona={"name": "比企谷小町"},
            group_policy={},
            safety={},
            settings=provided_settings,
        ),
    )
    monkeypatch.setattr(app_main, "build_engine", lambda _path: object())
    monkeypatch.setattr(app_main, "create_all", lambda _engine: None)
    monkeypatch.setattr(app_main, "sync_history_archives", lambda engine, runtime: sync_calls.append((engine, runtime)))
    monkeypatch.setattr(app_main, "NapCatGateway", FakeGateway)
    monkeypatch.setattr(app_main, "Sender", lambda _gateway: object())
    monkeypatch.setattr(app_main, "LlmClient", lambda **kwargs: llm_kwargs.update(kwargs) or object())
    monkeypatch.setattr(app_main, "ReplyPolicy", lambda: object())
    monkeypatch.setattr(app_main, "ContextBuilder", lambda: object())
    monkeypatch.setattr(app_main, "AdminCommandParser", lambda **_kwargs: object())
    monkeypatch.setattr(app_main, "DevControlService", FakeDevControlService)
    monkeypatch.setattr(app_main, "InboundRouter", FakeRouter)

    await app_main.run()

    assert isinstance(router_arguments["web_search_client"], WebSearchClient)
    assert len(sync_calls) == 1
    assert llm_kwargs["model"] == "gpt-5.4"
    assert llm_kwargs["text_endpoint"] == "/chat/completions"
