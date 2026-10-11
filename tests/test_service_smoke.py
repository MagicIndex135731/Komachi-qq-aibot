from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import app.main as app_main
import pytest

from app.config import AppSettings
from app.main import (
    build_image_reference_search_client,
    build_group_image_llm_client,
    build_group_image_reference_planner_client,
    build_web_search_client,
    create_runtime_banner,
    sync_history_archives,
    should_ingest_group_message,
    should_speak_in_group,
)
from app.providers.llm_client import LlmClient
from app.providers.web_search import WebSearchClient
from scripts.backfill_summaries import backfill_lines
from scripts.import_history import load_export


def _settings_for_search(*, provider: str, search_api_key: str) -> AppSettings:
    return AppSettings.model_construct(
        napcat_ws_url="ws://127.0.0.1:3001",
        llm_base_url="https://api.example.test/v1",
        llm_api_key="test-key",
        llm_model="gpt-5.4",
        llm_fallback_model="",
        llm_text_endpoint="chat_completions",
        group_image_base_url="https://images.example.test/v1",
        group_image_api_key="image-test-key",
        group_image_model="gpt-image-2",
        group_image_generations_endpoint="/v1/images/generations",
        group_image_edits_endpoint="/v1/images/edits",
        group_image_size="auto",
        group_image_quality="high",
        group_image_output_format="png",
        group_image_queue_capacity=3,
        group_image_timeout_seconds=900.0,
        bot_qq=123456789,
        owner_qq=987654321,
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


def test_build_group_image_llm_client_uses_dedicated_direct_image_endpoint() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    primary_client = object()

    client = build_group_image_llm_client(
        settings=settings, engine=object(), llm_client=primary_client
    )

    assert client is not primary_client
    assert client.base_url == "https://images.example.test/v1"
    assert client.api_key == "image-test-key"
    assert client.model == "gpt-image-2"
    assert client.responses_model == ""
    assert client.image_responses_model == ""
    assert client.image_generations_endpoint == "/images/generations"
    assert client.image_edits_endpoint == "/images/edits"


def test_build_group_image_llm_client_reuses_chat_nova_responses_transport() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = "gpt-5.6-luna"
    settings.llm_reasoning_effort = "low"
    settings.group_image_model = ""
    primary_client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        responses_model=settings.llm_model,
        image_responses_model=settings.llm_model,
    )

    client = build_group_image_llm_client(
        settings=settings, engine=object(), llm_client=primary_client
    )

    assert client.base_url == settings.llm_base_url
    assert client.api_key == settings.llm_api_key
    assert client.model == "gpt-image-2"
    assert client.responses_model == "gpt-image-2"
    assert client.image_responses_model == "gpt-image-2"
    assert client.responses_only is True
    assert client.http_client is primary_client.http_client


def test_build_group_image_llm_client_prefers_dedicated_image_chat_transport() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_base_url = "https://api.deepseek.test"
    settings.llm_api_key = "deepseek-key"
    settings.group_image_chat_base_url = "https://image-chat.example.test"
    settings.group_image_chat_api_key = "image-chat-key"
    primary_client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="deepseek-flash",
        responses_model="deepseek-flash",
    )

    client = build_group_image_llm_client(
        settings=settings, engine=object(), llm_client=primary_client
    )

    assert client.base_url == "https://image-chat.example.test"
    assert client.api_key == "image-chat-key"
    assert client.model == "gpt-image-2"
    assert client.responses_model == "gpt-image-2"
    assert client.http_client is primary_client.http_client


def test_standard_responses_image_transport_preserves_both_selected_models() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.group_image_model = "gpt-image-2.5-flare"
    settings.group_image_responses_model = "gpt-6.1-sol"
    primary = LlmClient(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                        model="other-chat-model", responses_model="other-chat-model")
    client = build_group_image_llm_client(settings=settings, engine=object(), llm_client=primary)
    assert client.responses_model == "gpt-6.1-sol"
    assert client.image_responses_model == "gpt-6.1-sol"
    assert client.image_generation_model == "gpt-image-2.5-flare"
    assert client.http_client is primary.http_client


def test_build_group_image_llm_client_can_force_direct_images_transport() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.group_image_transport = "images"
    settings.group_image_base_url = "https://xfastapi.example.test/v1"
    settings.group_image_api_key = "image-key"
    primary_client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        responses_model=settings.llm_model,
    )

    client = build_group_image_llm_client(
        settings=settings, engine=object(), llm_client=primary_client
    )

    assert client.base_url == "https://xfastapi.example.test/v1"
    assert client.api_key == "image-key"
    assert client.responses_model == ""
    assert client.image_responses_model == ""


@pytest.mark.parametrize("model, effort", [("deepseek-flash", "medium"), ("gpt-6.1-sol", "low")])
def test_build_group_image_reference_planner_uses_configured_chat_model(model, effort) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = model
    primary_client = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        responses_model=model,
    )

    planner = build_group_image_reference_planner_client(
        settings=settings, llm_client=primary_client
    )

    assert planner is not None
    assert planner.model == model
    assert planner.responses_model == model
    assert planner.reasoning_effort == effort
    assert planner.max_output_tokens == 256


def test_build_group_image_service_uses_dedicated_image_model_and_finite_timeout(
    monkeypatch,
) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = "gpt-6.1-sol"
    settings.group_image_model = "gpt-image-2"
    captured: dict[str, object] = {}
    built_service = object()

    monkeypatch.setattr(
        app_main,
        "GroupImageGenerationService",
        lambda **kwargs: captured.update(kwargs) or built_service,
    )

    result = app_main.build_group_image_service(
        settings=settings,
        llm_client=object(),
        sender=object(),
    )

    assert result is built_service
    assert captured["model"] == "gpt-image-2"
    assert captured["size"] == "auto"
    assert captured["quality"] == "high"
    assert captured["image_max_attempts"] == 1
    assert captured["image_timeout_seconds"] == 900.0


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


def test_sync_history_archives_only_targets_archive_enabled_speaking_groups(monkeypatch) -> None:
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
                "10001": {"enabled": True, "archive": True, "speak": True},
                "10002": {"enabled": True, "archive": False, "speak": True},
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


def test_build_web_search_client_is_disabled_when_builtin_llm_web_search_is_enabled() -> None:
    settings = _settings_for_search(provider="ddgs", search_api_key="   ")
    settings.llm_builtin_web_search = True
    settings.llm_text_endpoint = "responses"

    assert build_web_search_client(settings) is None


def test_build_web_search_client_uses_wyzlab_alpha_bridge_for_shell_endpoint() -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="")
    settings.llm_base_url = "https://shell.wyzlab.ai/v1"
    settings.llm_api_key = "wyzlab-key"
    settings.llm_model = "gpt-6.1-sol"
    settings.llm_text_endpoint = "responses"
    settings.llm_builtin_web_search = True
    settings.llm_timeout_seconds = 180.0

    client = build_web_search_client(settings)

    assert isinstance(client, WebSearchClient)
    assert client.provider == "wyzlab_alpha"
    assert client.base_url == "https://shell.wyzlab.ai/v1"
    assert client.timeout_seconds == 45
    assert client.reasoning_effort == "medium"
    assert settings.agent_web_search_results == 8


def test_build_image_reference_search_client_ignores_builtin_chat_search_switch() -> None:
    settings = _settings_for_search(provider="ddgs", search_api_key="   ")
    settings.llm_builtin_web_search = True
    settings.llm_text_endpoint = "responses"

    client = build_image_reference_search_client(settings)

    assert isinstance(client, WebSearchClient)
    assert client.provider == "ddgs"


def test_build_llm_client_preserves_primary_model_and_exposes_distinct_fallback(
    monkeypatch,
) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = "gpt-5.4-mini"
    settings.llm_fallback_model = "gpt-4o-mini"
    settings.llm_vision_model = "gpt-4o"
    captured: dict[str, object] = {}
    built_client = object()

    monkeypatch.setattr(
        app_main, "LlmClient", lambda **kwargs: captured.update(kwargs) or built_client
    )

    result = app_main.build_llm_client(settings=settings, engine=object())

    assert result is built_client
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["fallback_model"] == "gpt-4o-mini"
    assert captured["vision_model"] == "gpt-4o"
    assert captured["image_responses_model"] == "gpt-5.4-mini"
    assert captured["responses_only"] is True


def test_build_llm_client_enables_responses_when_text_endpoint_requests_it(monkeypatch) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = "gpt-5.4-mini"
    settings.llm_fallback_model = "gpt-4o-mini"
    settings.llm_text_endpoint = "responses"
    captured: dict[str, object] = {}
    built_client = object()

    monkeypatch.setattr(
        app_main, "LlmClient", lambda **kwargs: captured.update(kwargs) or built_client
    )

    result = app_main.build_llm_client(settings=settings, engine=object())

    assert result is built_client
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["fallback_model"] == "gpt-4o-mini"
    assert captured["responses_model"] == "gpt-5.4-mini"
    assert captured["responses_only"] is True


def test_build_llm_client_disables_duplicate_builtin_search_for_wyzlab_alpha(monkeypatch) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="")
    settings.llm_base_url = "https://shell.wyzlab.ai/v1"
    settings.llm_api_key = "wyzlab-key"
    settings.llm_text_endpoint = "responses"
    settings.llm_builtin_web_search = True
    settings.llm_timeout_seconds = 180.0
    captured: dict[str, object] = {}

    monkeypatch.setattr(app_main, "LlmClient", lambda **kwargs: captured.update(kwargs) or object())

    app_main.build_llm_client(settings=settings, engine=object())

    assert captured["builtin_web_search"] is False


def test_build_llm_client_passes_reasoning_effort_for_responses(monkeypatch) -> None:
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_text_endpoint = "responses"
    settings.llm_reasoning_effort = "medium"
    settings.llm_web_search_model = "gpt-6.1-sol"
    captured: dict[str, object] = {}
    built_client = object()

    monkeypatch.setattr(
        app_main, "LlmClient", lambda **kwargs: captured.update(kwargs) or built_client
    )

    result = app_main.build_llm_client(settings=settings, engine=object())

    assert result is built_client
    assert captured["reasoning_effort"] == "medium"
    assert captured["web_search_model"] == "gpt-6.1-sol"


@pytest.mark.asyncio
async def test_combined_entry_delegates_to_group_composition(monkeypatch):
    import app.group_main as group_main

    calls = []

    async def fake_group_run():
        calls.append("group")

    monkeypatch.setattr(group_main, "run", fake_group_run)
    await app_main.run()
    assert calls == ["group"]


@pytest.mark.parametrize("primary_effort", ["low", "medium", "max"])
@pytest.mark.parametrize("model, effort", [("gpt-6.1-sol", "low"), ("synthetic-other-model", "medium")])
def test_query_rewrite_uses_model_reasoning_with_bounded_client(primary_effort, monkeypatch, model, effort):
    settings = _settings_for_search(provider="tavily", search_api_key="search-key")
    settings.llm_model = model
    settings.memory_query_rewrite_enabled = True
    primary = LlmClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        responses_model=settings.llm_model,
        reasoning_effort=primary_effort,
        max_output_tokens=8192,
        builtin_web_search=True,
    )
    captured = []

    def generate(client, messages):
        captured.append(client)
        return '{"resolved_query":"test"}'

    monkeypatch.setattr(LlmClient, "generate_text", generate)
    rewrite = app_main._build_query_rewrite_provider(settings=settings, llm_client=primary)
    assert rewrite is not None
    assert rewrite("test", [], 1.0) == '{"resolved_query":"test"}'
    client = captured[0]
    assert client is not primary
    assert client.model == model
    assert client.reasoning_effort == effort
    assert client.max_output_tokens == 512
    assert client.builtin_web_search is False
    assert primary.reasoning_effort == primary_effort
