from types import SimpleNamespace

import pytest

from app.runtime.providers import (
    build_episode_post_segment_client,
    build_episode_topic_judge_client,
    build_memory_compaction_client,
    build_proactive_judge_client,
)


@pytest.mark.parametrize(
    "builder, expected_identity",
    [
        (build_proactive_judge_client, False),
        (build_episode_topic_judge_client, False),
        (build_episode_post_segment_client, False),
        (build_memory_compaction_client, True),
    ],
)
def test_callable_factory_keeps_injected_fake_client_without_live_provider(
    builder, expected_identity
):
    settings = SimpleNamespace(
        proactive_model_judge_enabled=True,
        memory_episode_topic_judge_enabled=True,
        memory_episode_post_segment_enabled=True,
    )
    fake_client = object()

    def reject_live_provider(**_kwargs):
        raise AssertionError("a test fake must not construct a live provider")

    result = builder(
        settings=settings,
        llm_client=fake_client,
        client_factory=reject_live_provider,
    )
    assert result is fake_client if expected_identity else result is None
