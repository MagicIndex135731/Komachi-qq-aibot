from app.config import AppSettings, load_runtime_config


def test_load_runtime_config_reads_yaml_and_env(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "persona.yaml").write_text("name: 小柚\nidentity: AI assistant\n", encoding="utf-8")
    (config_dir / "groups.yaml").write_text(
        "default_group_behavior:\n  speak: false\n  archive: true\ngroups: {}\n",
        encoding="utf-8",
    )
    (config_dir / "safety.yaml").write_text(
        "must_disclose_ai_identity: true\ndeny_prompt_leak: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=config_dir, data_dir=tmp_path / "data")
    runtime = load_runtime_config(settings)

    assert runtime.persona["name"] == "小柚"
    assert runtime.group_policy["default_group_behavior"]["speak"] is False
    assert runtime.safety["must_disclose_ai_identity"] is True


def test_app_settings_exposes_search_and_context_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("SEARCH_BASE_URL", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SEARCH_REGION", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("CONTEXT_RECENT_LIMIT", raising=False)
    monkeypatch.delenv("CONTEXT_SUMMARY_LIMIT", raising=False)
    monkeypatch.delenv("CONTEXT_HISTORY_LIMIT", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.search_provider == "tavily"
    assert settings.search_base_url == "https://api.tavily.com/search"
    assert settings.search_api_key == ""
    assert settings.search_region == "wt-wt"
    assert settings.search_backend == "auto"
    assert settings.search_timeout_seconds == 8.0
    assert settings.context_recent_limit == 60
    assert settings.context_summary_limit == 3
    assert settings.context_history_limit == 8
    assert settings.llm_fallback_model == ""


def test_app_settings_exposes_private_chat_whitelist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    monkeypatch.setenv("PRIVATE_CHAT_QQS", "20002, 20002")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.private_chat_whitelist == {987654321, 20002, 20002}


def test_app_settings_reads_llm_fallback_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "cc-gpt-5.4")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.llm_fallback_model == "gpt-5.4"
