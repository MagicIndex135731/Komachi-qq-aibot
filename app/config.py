from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.style_distill import merge_persona_lists


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
    llm_timeout_seconds: float = Field(
        default=120.0,
        gt=10,
        alias="LLM_TIMEOUT_SECONDS",
    )
    llm_supports_vision_input: bool = Field(default=True, alias="LLM_SUPPORTS_VISION_INPUT")
    llm_vision_model: str = Field(default="", alias="LLM_VISION_MODEL")
    group_image_base_url: str = Field(default="", alias="GROUP_IMAGE_BASE_URL")
    group_image_api_key: str = Field(default="", alias="GROUP_IMAGE_API_KEY")
    group_image_model: str = Field(default="gpt-image-2", alias="GROUP_IMAGE_MODEL")
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
    context_recent_limit: int = Field(default=60, alias="CONTEXT_RECENT_LIMIT")
    context_summary_limit: int = Field(default=3, alias="CONTEXT_SUMMARY_LIMIT")
    context_history_limit: int = Field(default=8, alias="CONTEXT_HISTORY_LIMIT")
    memory_compaction_enabled: bool = Field(default=True, alias="MEMORY_COMPACTION_ENABLED")
    memory_compaction_batch_size: int = Field(default=50, alias="MEMORY_COMPACTION_BATCH_SIZE")
    memory_compaction_max_facts: int = Field(default=24, alias="MEMORY_COMPACTION_MAX_FACTS")
    memory_compaction_retry_limit: int = Field(default=3, alias="MEMORY_COMPACTION_RETRY_LIMIT")
    memory_compaction_backfill_windows: int = Field(default=24, alias="MEMORY_COMPACTION_BACKFILL_WINDOWS")
    memory_compaction_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = Field(
        default="low",
        alias="MEMORY_COMPACTION_REASONING_EFFORT",
    )
    memory_compaction_max_output_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        alias="MEMORY_COMPACTION_MAX_OUTPUT_TOKENS",
    )
    memory_orchestration_v2_enabled: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_V2_ENABLED")
    memory_orchestration_shadow_mode: bool = Field(default=False, alias="MEMORY_ORCHESTRATION_SHADOW_MODE")
    memory_raw_v3_enabled: bool = Field(default=False, alias="MEMORY_RAW_V3_ENABLED")
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
        default=4.0, alias="MEMORY_RETRIEVAL_CHANNEL_TIMEOUT_SECONDS"
    )
    memory_episode_idle_minutes: int = Field(default=10, alias="MEMORY_EPISODE_IDLE_MINUTES")
    memory_episode_max_messages: int = Field(default=70, alias="MEMORY_EPISODE_MAX_MESSAGES")
    memory_episode_max_tokens: int = Field(default=8000, alias="MEMORY_EPISODE_MAX_TOKENS")
    memory_episode_topic_judge_enabled: bool = Field(
        default=False,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_ENABLED",
    )
    memory_episode_topic_judge_context_messages: int = Field(
        default=20,
        ge=2,
        le=30,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_CONTEXT_MESSAGES",
    )
    memory_episode_topic_judge_start_messages: int = Field(
        default=50,
        ge=1,
        le=100,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_START_MESSAGES",
    )
    memory_episode_topic_judge_interval: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_INTERVAL",
    )
    memory_episode_topic_judge_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = Field(
        default="low",
        alias="MEMORY_EPISODE_TOPIC_JUDGE_REASONING_EFFORT",
    )
    memory_episode_topic_judge_max_output_tokens: int = Field(
        default=128,
        ge=32,
        le=512,
        alias="MEMORY_EPISODE_TOPIC_JUDGE_MAX_OUTPUT_TOKENS",
    )
    memory_episode_post_segment_enabled: bool = Field(
        default=False,
        alias="MEMORY_EPISODE_POST_SEGMENT_ENABLED",
    )
    memory_episode_post_segment_min_messages: int = Field(
        default=25,
        ge=5,
        le=120,
        alias="MEMORY_EPISODE_POST_SEGMENT_MIN_MESSAGES",
    )
    memory_episode_post_segment_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = Field(
        default="low",
        alias="MEMORY_EPISODE_POST_SEGMENT_REASONING_EFFORT",
    )
    memory_episode_post_segment_max_output_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        alias="MEMORY_EPISODE_POST_SEGMENT_MAX_OUTPUT_TOKENS",
    )
    memory_current_default_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        alias="MEMORY_CURRENT_DEFAULT_TTL_HOURS",
    )
    memory_chunk_max_tokens: int = Field(default=2400, alias="MEMORY_CHUNK_MAX_TOKENS")
    memory_chunk_overlap_messages: int = Field(default=8, alias="MEMORY_CHUNK_OVERLAP_MESSAGES")
    memory_chunk_max_messages: int = Field(default=40, ge=1, alias="MEMORY_CHUNK_MAX_MESSAGES")
    memory_query_rewrite_enabled: bool = Field(default=False, alias="MEMORY_QUERY_REWRITE_ENABLED")
    memory_query_rewrite_timeout_seconds: float = Field(default=3.0, alias="MEMORY_QUERY_REWRITE_TIMEOUT_SECONDS")
    memory_query_rewrite_max_output_tokens: int = Field(
        default=256, alias="MEMORY_QUERY_REWRITE_MAX_OUTPUT_TOKENS"
    )
    memory_llm_rerank_enabled: bool = Field(default=False, alias="MEMORY_LLM_RERANK_ENABLED")
    memory_normal_context_budget_tokens: int = Field(default=48000, alias="MEMORY_NORMAL_CONTEXT_BUDGET_TOKENS")
    memory_detail_context_budget_tokens: int = Field(default=64000, alias="MEMORY_DETAIL_CONTEXT_BUDGET_TOKENS")
    memory_recent_context_budget_tokens: int = Field(default=10000, alias="MEMORY_RECENT_CONTEXT_BUDGET_TOKENS")
    memory_history_context_budget_tokens: int = Field(
        default=24000,
        alias="MEMORY_HISTORY_CONTEXT_BUDGET_TOKENS",
    )
    memory_context_budget_chars: int = Field(
        default=24000,
        alias="MEMORY_CONTEXT_BUDGET_CHARS",
    )
    memory_adaptive_context_enabled: bool = Field(
        default=False,
        alias="MEMORY_ADAPTIVE_CONTEXT_ENABLED",
    )
    memory_adaptive_context_budget_chars: int = Field(
        default=48000,
        gt=0,
        alias="MEMORY_ADAPTIVE_CONTEXT_BUDGET_CHARS",
    )
    memory_recent_protected_min_tokens: int = Field(
        default=1200,
        ge=0,
        alias="MEMORY_RECENT_PROTECTED_MIN_TOKENS",
    )
    memory_history_protected_min_tokens: int = Field(
        default=2400,
        ge=0,
        alias="MEMORY_HISTORY_PROTECTED_MIN_TOKENS",
    )
    memory_recent_protected_min_messages: int = Field(
        default=1,
        ge=0,
        alias="MEMORY_RECENT_PROTECTED_MIN_MESSAGES",
    )
    memory_history_protected_min_messages: int = Field(
        default=1,
        ge=0,
        alias="MEMORY_HISTORY_PROTECTED_MIN_MESSAGES",
    )
    memory_adaptive_max_recent_messages: int = Field(
        default=60,
        gt=0,
        alias="MEMORY_ADAPTIVE_MAX_RECENT_MESSAGES",
    )
    memory_adaptive_max_history_messages: int = Field(
        default=300,
        gt=0,
        alias="MEMORY_ADAPTIVE_MAX_HISTORY_MESSAGES",
    )
    memory_layered_memory_enabled: bool = Field(
        default=False,
        alias="MEMORY_LAYERED_MEMORY_ENABLED",
    )
    memory_memory_tools_enabled: bool = Field(
        default=False,
        alias="MEMORY_MEMORY_TOOLS_ENABLED",
    )
    proactive_model_judge_enabled: bool = Field(
        default=False,
        alias="PROACTIVE_MODEL_JUDGE_ENABLED",
    )
    proactive_judge_model: str = Field(
        default="",
        alias="PROACTIVE_JUDGE_MODEL",
    )
    proactive_judge_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = Field(
        default="low",
        alias="PROACTIVE_JUDGE_REASONING_EFFORT",
    )
    proactive_judge_max_output_tokens: int = Field(
        default=256,
        ge=64,
        le=2048,
        alias="PROACTIVE_JUDGE_MAX_OUTPUT_TOKENS",
    )
    proactive_judge_context_messages: int = Field(
        default=20,
        ge=1,
        le=20,
        alias="PROACTIVE_JUDGE_CONTEXT_MESSAGES",
    )
    proactive_judge_max_chars_per_message: int = Field(
        default=120,
        ge=20,
        le=1000,
        alias="PROACTIVE_JUDGE_MAX_CHARS_PER_MESSAGE",
    )
    proactive_recent_messages_limit: int = Field(
        default=60,
        ge=1,
        le=200,
        alias="PROACTIVE_RECENT_MESSAGES_LIMIT",
    )
    proactive_image_max_count: int = Field(
        default=2,
        ge=1,
        le=10,
        alias="PROACTIVE_IMAGE_MAX_COUNT",
    )
    memory_memory_tool_max_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        alias="MEMORY_MEMORY_TOOL_MAX_ROUNDS",
    )
    memory_decision_envelope_enabled: bool = Field(
        default=True,
        alias="MEMORY_DECISION_ENVELOPE_ENABLED",
    )
    memory_memory_tool_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="MEMORY_MEMORY_TOOL_MAX_RESULTS",
    )
    memory_memory_tool_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        alias="MEMORY_MEMORY_TOOL_TIMEOUT_SECONDS",
    )
    memory_member_fact_supplement_limit: int = Field(
        default=20,
        ge=1,
        le=30,
        alias="MEMORY_MEMBER_FACT_SUPPLEMENT_LIMIT",
    )
    memory_fact_semantic_ranking_enabled: bool = Field(
        default=False,
        alias="MEMORY_FACT_SEMANTIC_RANKING_ENABLED",
    )
    memory_fact_semantic_candidates: int = Field(
        default=60,
        ge=1,
        le=200,
        alias="MEMORY_FACT_SEMANTIC_CANDIDATES",
    )
    memory_search_compact_budget_tokens: int = Field(
        default=6000,
        ge=1024,
        alias="MEMORY_SEARCH_COMPACT_BUDGET_TOKENS",
    )
    memory_search_auto_budget_tokens: int = Field(
        default=12000,
        ge=2048,
        alias="MEMORY_SEARCH_AUTO_BUDGET_TOKENS",
    )
    memory_max_evidence_messages: int = Field(
        default=300,
        alias="MEMORY_MAX_EVIDENCE_MESSAGES",
    )
    memory_recent_intent_candidate_limit: int = Field(
        default=48,
        ge=1,
        le=300,
        alias="MEMORY_RECENT_INTENT_CANDIDATE_LIMIT",
    )
    memory_fts_candidate_limit: int = Field(default=30, alias="MEMORY_FTS_CANDIDATE_LIMIT")
    memory_vector_candidate_limit: int = Field(default=30, alias="MEMORY_VECTOR_CANDIDATE_LIMIT")
    memory_final_episode_limit: int = Field(default=12, alias="MEMORY_FINAL_EPISODE_LIMIT")
    llm_context_window_tokens: int = Field(default=258000, alias="LLM_CONTEXT_WINDOW_TOKENS")
    llm_max_output_tokens: int = Field(default=8192, alias="LLM_MAX_OUTPUT_TOKENS")
    llm_context_safety_margin_tokens: int = Field(default=32768, alias="LLM_CONTEXT_SAFETY_MARGIN_TOKENS")
    llm_tool_context_reserve_tokens: int = Field(default=32768, alias="LLM_TOOL_CONTEXT_RESERVE_TOKENS")
    config_dir: Path = Path("configs")
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def validate_adaptive_memory_budget(self) -> AppSettings:
        if not self.memory_adaptive_context_enabled:
            return self
        if (
            self.memory_recent_protected_min_messages
            > self.memory_adaptive_max_recent_messages
        ):
            raise ValueError("adaptive recent protected minimum exceeds message safety cap")
        if (
            self.memory_history_protected_min_messages
            > self.memory_adaptive_max_history_messages
        ):
            raise ValueError("adaptive history protected minimum exceeds message safety cap")
        if (
            self.memory_recent_protected_min_tokens
            + self.memory_history_protected_min_tokens
            > self.memory_normal_context_budget_tokens
        ):
            raise ValueError("adaptive protected token minima exceed normal memory budget")
        return self

    @property
    def memory_recent_snapshot_limit(self) -> int:
        if not self.memory_adaptive_context_enabled:
            return int(self.context_recent_limit)
        return max(
            int(self.context_recent_limit),
            int(self.memory_adaptive_max_recent_messages),
        )

    @property
    def memory_effective_context_budget_chars(self) -> int:
        if self.memory_adaptive_context_enabled:
            return int(self.memory_adaptive_context_budget_chars)
        return int(self.memory_context_budget_chars)

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
    personas: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    personas: dict[str, dict[str, Any]] = {"default": dict(persona)}
    personas_dir = settings.config_dir / "personas"
    if personas_dir.is_dir():
        for persona_path in sorted(personas_dir.glob("*.yaml")):
            try:
                persona_profile = _read_yaml(persona_path)
            except ValueError as exc:
                raise ValueError(f"invalid persona profile {persona_path.name}: {exc}") from exc
            personas[persona_path.stem] = persona_profile
    live_personas_dir = settings.data_dir / "personas"
    if live_personas_dir.is_dir():
        for persona_path in sorted(live_personas_dir.glob("*.yaml")):
            if persona_path.name.endswith(".live.yaml"):
                continue
            persona_key = persona_path.stem
            persona_profile = _read_yaml(persona_path)
            personas[persona_key] = persona_profile
        for live_path in sorted(live_personas_dir.glob("*.live.yaml")):
            persona_key = live_path.name.removesuffix(".live.yaml")
            if persona_key not in personas:
                continue
            live_profile = _read_yaml(live_path)
            personas[persona_key] = _deep_merge_mapping(
                personas[persona_key], live_profile
            )
    group_policy = _read_yaml(settings.config_dir / "groups.yaml")
    local_groups_path = settings.config_dir / "groups.local.yaml"
    if local_groups_path.exists():
        local_policy = _read_yaml(local_groups_path)
        if isinstance(local_policy, dict):
            group_policy = _deep_merge_mapping(group_policy, local_policy)
    safety = _read_yaml(settings.config_dir / "safety.yaml")
    return RuntimeConfig(
        settings=settings,
        persona=persona,
        group_policy=group_policy,
        safety=safety,
        personas=personas,
    )


def _deep_merge_mapping(base: dict, overlay: dict) -> dict:
    """Merge overlay over base, recursing into nested mappings."""
    merged = dict(base)
    for key, value in overlay.items():
        if key == "groups" and isinstance(value, dict):
            # 生产覆盖文件提供完整、真实的群表，整体替换占位群表。
            merged[key] = value
        elif key in {"facts", "external_relations"} and isinstance(value, list):
            merged[key] = merge_persona_lists(merged.get(key), value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged
