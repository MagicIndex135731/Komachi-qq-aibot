from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    napcat_ws_url: str = Field(alias="NAPCAT_WS_URL")
    llm_base_url: str = Field(alias="LLM_BASE_URL")
    llm_api_key: str = Field(alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-5.4-mini", alias="LLM_MODEL")
    llm_fallback_model: str = Field(default="gpt-5.4", alias="LLM_FALLBACK_MODEL")
    llm_text_endpoint: Literal["chat_completions", "responses"] = Field(
        default="chat_completions",
        alias="LLM_TEXT_ENDPOINT",
    )
    llm_builtin_web_search: bool = Field(default=False, alias="LLM_BUILTIN_WEB_SEARCH")
    llm_builtin_web_search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        alias="LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE",
    )
    llm_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = Field(
        default="",
        alias="LLM_REASONING_EFFORT",
    )
    llm_supports_vision_input: bool = Field(default=True, alias="LLM_SUPPORTS_VISION_INPUT")
    llm_vision_model: str = Field(default="", alias="LLM_VISION_MODEL")
    group_image_base_url: str = Field(default="", alias="GROUP_IMAGE_BASE_URL")
    group_image_api_key: str = Field(default="", alias="GROUP_IMAGE_API_KEY")
    group_image_model: str = Field(default="", alias="GROUP_IMAGE_MODEL")
    group_image_generations_endpoint: str = Field(
        default="/images/generations",
        alias="GROUP_IMAGE_GENERATIONS_ENDPOINT",
    )
    group_image_edits_endpoint: str = Field(default="/images/edits", alias="GROUP_IMAGE_EDITS_ENDPOINT")
    group_image_size: str = Field(default="auto", alias="GROUP_IMAGE_SIZE")
    group_image_quality: str = Field(default="high", alias="GROUP_IMAGE_QUALITY")
    group_image_output_format: str = Field(default="png", alias="GROUP_IMAGE_OUTPUT_FORMAT")
    group_image_queue_capacity: int = Field(default=3, alias="GROUP_IMAGE_QUEUE_CAPACITY")
    group_image_timeout_seconds: float = Field(default=900.0, alias="GROUP_IMAGE_TIMEOUT_SECONDS")
    bot_qq: int = Field(alias="BOT_QQ")
    owner_qq: int = Field(alias="OWNER_QQ")
    admin_qqs: str = Field(default="", alias="ADMIN_QQS")
    private_chat_qqs: str = Field(default="", alias="PRIVATE_CHAT_QQS")
    search_provider: str = Field(default="tavily", alias="SEARCH_PROVIDER")
    search_base_url: str = Field(default="https://api.tavily.com/search", alias="SEARCH_BASE_URL")
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")
    search_timeout_seconds: float = Field(default=8.0, alias="SEARCH_TIMEOUT_SECONDS")
    search_region: str = Field(default="wt-wt", alias="SEARCH_REGION")
    search_backend: str = Field(default="auto", alias="SEARCH_BACKEND")
    context_recent_limit: int = Field(default=100, alias="CONTEXT_RECENT_LIMIT")
    context_summary_limit: int = Field(default=3, alias="CONTEXT_SUMMARY_LIMIT")
    context_history_limit: int = Field(default=8, alias="CONTEXT_HISTORY_LIMIT")
    memory_compaction_enabled: bool = Field(default=True, alias="MEMORY_COMPACTION_ENABLED")
    memory_compaction_batch_size: int = Field(default=50, alias="MEMORY_COMPACTION_BATCH_SIZE")
    memory_compaction_max_facts: int = Field(default=24, alias="MEMORY_COMPACTION_MAX_FACTS")
    memory_compaction_retry_limit: int = Field(default=3, alias="MEMORY_COMPACTION_RETRY_LIMIT")
    memory_compaction_backfill_windows: int = Field(default=24, alias="MEMORY_COMPACTION_BACKFILL_WINDOWS")
    memory_orchestration_v2_enabled: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_V2_ENABLED")
    memory_orchestration_shadow_mode: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_SHADOW_MODE")
    memory_embedding_provider: Literal["local", "openai_compatible", "disabled"] = Field(
        default="local", alias="MEMORY_EMBEDDING_PROVIDER"
    )
    memory_embedding_device: Literal["auto", "cuda", "cpu"] = Field(
        default="cpu", alias="MEMORY_EMBEDDING_DEVICE"
    )
    memory_embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", alias="MEMORY_EMBEDDING_MODEL")
    memory_embedding_dimensions: int = Field(default=512, alias="MEMORY_EMBEDDING_DIMENSIONS")
    memory_embedding_cache_dir: Path = Field(
        default=Path("/workspace/data/models"), alias="MEMORY_EMBEDDING_CACHE_DIR"
    )
    memory_embedding_local_files_only: bool = Field(
        default=False, alias="MEMORY_EMBEDDING_LOCAL_FILES_ONLY"
    )
    memory_embedding_base_url: str = Field(default="", alias="MEMORY_EMBEDDING_BASE_URL")
    memory_embedding_api_key: str = Field(default="", alias="MEMORY_EMBEDDING_API_KEY")
    memory_embedding_version: str = Field(default="", alias="MEMORY_EMBEDDING_VERSION")
    memory_embedding_timeout_seconds: float = Field(default=10.0, alias="MEMORY_EMBEDDING_TIMEOUT_SECONDS")
    memory_retrieval_channel_timeout_seconds: float = Field(
        default=2.0, alias="MEMORY_RETRIEVAL_CHANNEL_TIMEOUT_SECONDS"
    )
    memory_episode_idle_minutes: int = Field(default=30, alias="MEMORY_EPISODE_IDLE_MINUTES")
    memory_episode_max_messages: int = Field(default=50, alias="MEMORY_EPISODE_MAX_MESSAGES")
    memory_episode_max_tokens: int = Field(default=8000, alias="MEMORY_EPISODE_MAX_TOKENS")
    memory_chunk_max_tokens: int = Field(default=1800, alias="MEMORY_CHUNK_MAX_TOKENS")
    memory_chunk_overlap_messages: int = Field(default=5, alias="MEMORY_CHUNK_OVERLAP_MESSAGES")
    memory_query_rewrite_enabled: bool = Field(default=False, alias="MEMORY_QUERY_REWRITE_ENABLED")
    memory_query_rewrite_timeout_seconds: float = Field(default=3.0, alias="MEMORY_QUERY_REWRITE_TIMEOUT_SECONDS")
    memory_query_rewrite_max_output_tokens: int = Field(
        default=256, alias="MEMORY_QUERY_REWRITE_MAX_OUTPUT_TOKENS"
    )
    memory_llm_rerank_enabled: bool = Field(default=False, alias="MEMORY_LLM_RERANK_ENABLED")
    memory_normal_context_budget_tokens: int = Field(default=32000, alias="MEMORY_NORMAL_CONTEXT_BUDGET_TOKENS")
    memory_detail_context_budget_tokens: int = Field(default=64000, alias="MEMORY_DETAIL_CONTEXT_BUDGET_TOKENS")
    memory_recent_context_budget_tokens: int = Field(default=10000, alias="MEMORY_RECENT_CONTEXT_BUDGET_TOKENS")
    memory_fts_candidate_limit: int = Field(default=30, alias="MEMORY_FTS_CANDIDATE_LIMIT")
    memory_vector_candidate_limit: int = Field(default=30, alias="MEMORY_VECTOR_CANDIDATE_LIMIT")
    memory_final_episode_limit: int = Field(default=6, alias="MEMORY_FINAL_EPISODE_LIMIT")
    llm_context_window_tokens: int = Field(default=258000, alias="LLM_CONTEXT_WINDOW_TOKENS")
    llm_max_output_tokens: int = Field(default=8192, alias="LLM_MAX_OUTPUT_TOKENS")
    llm_context_safety_margin_tokens: int = Field(default=32768, alias="LLM_CONTEXT_SAFETY_MARGIN_TOKENS")
    llm_tool_context_reserve_tokens: int = Field(default=32768, alias="LLM_TOOL_CONTEXT_RESERVE_TOKENS")
    config_dir: Path = Path("configs")
    data_dir: Path = Path("data")

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def admin_whitelist(self) -> set[int]:
        values = {self.owner_qq}
        if self.admin_qqs:
            values.update(int(item.strip()) for item in self.admin_qqs.split(",") if item.strip())
        return values

    @property
    def private_chat_whitelist(self) -> set[int]:
        values = {self.owner_qq}
        if self.private_chat_qqs:
            values.update(int(item.strip()) for item in self.private_chat_qqs.split(",") if item.strip())
        return values


@dataclass(slots=True)
class RuntimeConfig:
    settings: AppSettings
    persona: dict[str, Any]
    group_policy: dict[str, Any]
    safety: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_runtime_config(settings: AppSettings) -> RuntimeConfig:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    persona = _read_yaml(settings.config_dir / "persona.yaml")
    group_policy = _read_yaml(settings.config_dir / "groups.yaml")
    safety = _read_yaml(settings.config_dir / "safety.yaml")
    return RuntimeConfig(settings=settings, persona=persona, group_policy=group_policy, safety=safety)
