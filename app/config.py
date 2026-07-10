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
    context_recent_limit: int = Field(default=60, alias="CONTEXT_RECENT_LIMIT")
    context_summary_limit: int = Field(default=3, alias="CONTEXT_SUMMARY_LIMIT")
    context_history_limit: int = Field(default=8, alias="CONTEXT_HISTORY_LIMIT")
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
