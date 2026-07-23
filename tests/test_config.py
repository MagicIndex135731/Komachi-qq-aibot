from pathlib import Path

import pytest

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
    monkeypatch.delenv("LLM_MODEL", raising=False)
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
    monkeypatch.delenv("LLM_TEXT_ENDPOINT", raising=False)

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.search_provider == "tavily"
    assert settings.search_base_url == "https://api.tavily.com/search"
    assert settings.search_api_key == ""
    assert settings.search_region == "wt-wt"
    assert settings.search_backend == "auto"
    assert settings.search_timeout_seconds == 8.0
    assert settings.context_recent_limit == 100
    assert settings.context_summary_limit == 3
    assert settings.context_history_limit == 8
    assert settings.llm_model == "gpt-5.4-mini"
    assert settings.llm_fallback_model == "gpt-5.4"
    assert settings.llm_text_endpoint == "chat_completions"


def test_app_settings_exposes_private_chat_whitelist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    monkeypatch.setenv("PRIVATE_CHAT_QQS", "10002, 20002")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.private_chat_whitelist == {987654321, 10002, 20002}


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


def test_app_settings_exposes_vision_input_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")
    monkeypatch.setenv("LLM_SUPPORTS_VISION_INPUT", "false")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.llm_supports_vision_input is False


def test_app_settings_reads_vision_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_VISION_MODEL", "gpt-4o")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.llm_vision_model == "gpt-4o"


def test_app_settings_rejects_invalid_llm_text_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_TEXT_ENDPOINT", "completions")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    with pytest.raises(ValueError, match="LLM_TEXT_ENDPOINT"):
        AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)


def test_app_settings_exposes_group_image_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.group_image_queue_capacity == 3
    assert settings.group_image_timeout_seconds == 900.0


def test_app_settings_exposes_memory_orchestration_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.memory_orchestration_v2_enabled is False
    assert settings.memory_orchestration_shadow_mode is False
    assert settings.memory_embedding_provider == "local"
    assert settings.memory_embedding_device == "cpu"
    assert settings.memory_embedding_local_files_only is False
    assert settings.memory_retrieval_channel_timeout_seconds == 2.0
    assert settings.memory_embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.memory_embedding_dimensions == 512
    assert settings.memory_embedding_cache_dir == Path("/workspace/data/models")
    assert settings.memory_embedding_base_url == ""
    assert settings.memory_embedding_api_key == ""
    assert settings.memory_embedding_version == ""
    assert settings.memory_episode_idle_minutes == 30
    assert settings.memory_episode_max_messages == 50
    assert settings.memory_episode_max_tokens == 8000
    assert settings.memory_chunk_max_tokens == 1800
    assert settings.memory_chunk_overlap_messages == 5
    assert settings.memory_query_rewrite_enabled is False
    assert settings.memory_query_rewrite_timeout_seconds == 3.0
    assert settings.memory_query_rewrite_max_output_tokens == 256
    assert settings.memory_llm_rerank_enabled is False
    assert settings.memory_normal_context_budget_tokens == 32000
    assert settings.memory_detail_context_budget_tokens == 64000
    assert settings.memory_recent_context_budget_tokens == 10000
    assert settings.memory_fts_candidate_limit == 30
    assert settings.memory_vector_candidate_limit == 30
    assert settings.memory_final_episode_limit == 6


def test_app_settings_reads_group_image_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NAPCAT_WS_URL", "ws://127.0.0.1:3001")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("GROUP_IMAGE_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("BOT_QQ", "123456789")
    monkeypatch.setenv("OWNER_QQ", "987654321")

    settings = AppSettings(config_dir=tmp_path / "configs", data_dir=tmp_path / "data", _env_file=None)

    assert settings.group_image_timeout_seconds == 600.0
