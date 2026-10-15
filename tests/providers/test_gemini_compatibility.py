"""Relay aliases encode the actual Gemini budget, independent of request fields."""
from types import SimpleNamespace

import httpx
import pytest

from app.agent.harness_runtime import model_context_format_version
from app.config import AppSettings
from app.providers.llm_client import LlmClient, default_auxiliary_reasoning_effort
from app.runtime.providers import (
    build_episode_post_segment_client,
    build_episode_topic_judge_client,
    build_group_image_reference_planner_client,
    build_group_image_llm_client,
    build_memory_compaction_client,
    build_proactive_judge_client,
)


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_gemini_request_models_encode_explicit_effort_for_every_text_route(effort):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as http:
        client = LlmClient(base_url="https://synthetic.invalid/v1", api_key="synthetic",
            model="gemini-3.8-flash-high", responses_model="gemini-3.8-flash-high",
            fallback_model="gemini-3.8-flash", vision_model="gemini-3.8-flash-high",
            image_responses_model="gemini-3.8-flash-high", compat_model="gemini-3.8-flash-high",
            reasoning_effort=effort, http_client=http)
        for field in ("model", "responses_model", "fallback_model", "vision_model", "image_responses_model", "compat_model"):
            assert getattr(client, field) == "gemini-3.8-flash-" + effort


@pytest.mark.parametrize("effort", ["minimal", "xhigh", "max"])
def test_gemini_unsupported_explicit_effort_fails_before_network(effort):
    with pytest.raises(ValueError, match="Gemini 3.8 Flash reasoning effort"):
        LlmClient(base_url="https://synthetic.invalid/v1", api_key="synthetic",
            model="gemini-3.8-flash-low", reasoning_effort=effort)


def test_tiered_or_explicit_alias_is_preserved_when_no_effort_was_selected():
    with httpx.Client() as http:
        for model in ("gemini-3.8-flash-tiered", "gemini-3.8-flash-high"):
            client = LlmClient(base_url="https://synthetic.invalid/v1", api_key="synthetic",
                model=model, reasoning_effort="", http_client=http)
            assert client.model == model
    assert default_auxiliary_reasoning_effort("gemini-3.8-flash-high") == "low"
    assert default_auxiliary_reasoning_effort("gemini-3-pro-image") == "medium"


@pytest.mark.parametrize("builder", [build_proactive_judge_client, build_episode_topic_judge_client,
    build_episode_post_segment_client, build_memory_compaction_client, build_group_image_reference_planner_client])
def test_auxiliary_clients_use_their_own_low_alias_when_primary_is_high(builder):
    settings = AppSettings(_env_file=None, BOT_QQ=900000101, OWNER_QQ=900000102,
        NAPCAT_WS_URL="ws://synthetic.invalid/ws", LLM_BASE_URL="https://synthetic.invalid/v1",
        LLM_API_KEY="synthetic", LLM_MODEL="gemini-3.8-flash-high", LLM_FALLBACK_MODEL="",
        LLM_REASONING_EFFORT="high", PROACTIVE_MODEL_JUDGE_ENABLED=True,
        PROACTIVE_JUDGE_MODEL="", PROACTIVE_JUDGE_REASONING_EFFORT="low",
        MEMORY_EPISODE_TOPIC_JUDGE_ENABLED=True, MEMORY_EPISODE_TOPIC_JUDGE_REASONING_EFFORT="low",
        MEMORY_EPISODE_POST_SEGMENT_ENABLED=True, MEMORY_EPISODE_POST_SEGMENT_REASONING_EFFORT="low",
        MEMORY_COMPACTION_REASONING_EFFORT="low")
    with httpx.Client() as http:
        primary = LlmClient(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
            model=settings.llm_model, reasoning_effort="high", http_client=http)
        child = builder(settings=settings, llm_client=primary)
        try:
            assert primary.model == "gemini-3.8-flash-high"
            assert child.model == child.responses_model == "gemini-3.8-flash-low"
        finally:
            if child.http_client is not http:
                child.http_client.close()


def test_gemini_session_format_rotates_between_models_protocols_and_provider_addresses():
    def identity(model, *, responses=True, base="https://synthetic.invalid/v1"):
        return model_context_format_version(SimpleNamespace(model=model,
            responses_model=model if responses else "", base_url=base))
    sol = identity("gpt-6.1-sol")
    low = identity("gemini-3.8-flash-low")
    assert low == identity("gemini-3.8-flash-low")
    assert len({sol, low, identity("gemini-3.8-flash-high"),
        identity("gemini-3.8-flash-low", responses=False),
        identity("gemini-3.8-flash-low", base="https://another.invalid/v1")}) == 5


def test_dedicated_gemini_images_factory_uses_the_image_timeout_budget():
    settings = AppSettings(_env_file=None, BOT_QQ=900000101, OWNER_QQ=900000102,
        NAPCAT_WS_URL="ws://synthetic.invalid/ws", LLM_BASE_URL="https://chat.example/v1",
        LLM_API_KEY="synthetic-chat", GROUP_IMAGE_BASE_URL="https://images.example/v1",
        GROUP_IMAGE_API_KEY="synthetic-image", GROUP_IMAGE_TRANSPORT="images",
        GROUP_IMAGE_MODEL="gemini-3.1-flash-image", GROUP_IMAGE_EDIT_MODEL="",
        LLM_TIMEOUT_SECONDS=30, GROUP_IMAGE_TIMEOUT_SECONDS=300)
    observed = {}
    build_group_image_llm_client(settings=settings, engine=None, llm_client=object(),
        client_factory=lambda **options: observed.update(options))
    assert observed["timeout_seconds"] == 300
    assert observed["base_url"] == "https://images.example/v1"
    assert observed["api_key"] == "synthetic-image"
    assert observed["image_responses_model"] == ""


def test_gemini_malformed_hidden_tool_call_does_not_become_a_blank_answer():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "MALFORMED_FUNCTION_CALL",
            "message": {"role": "assistant", "content": "", "reasoning_content": "Synthetic hidden tool unavailable."}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = LlmClient(base_url="https://synthetic.invalid/v1", api_key="synthetic",
            model="gemini-3.8-flash-low", reasoning_effort="low", fallback_model="", http_client=http)
        with pytest.raises(ValueError, match="did not include output text"):
            client.generate_text(["Synthetic calculation without tools."])
    assert len(requests) == 1
