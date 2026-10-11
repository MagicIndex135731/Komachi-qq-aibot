from datetime import UTC, datetime

import pytest

from app.agent.social_policy import configured_social_jobs, in_quiet_hours
from app.config import AppSettings, RuntimeConfig


def runtime(policy, groups=None):
    settings = AppSettings(_env_file=None, NAPCAT_WS_URL="ws://localhost:3001",
                           LLM_BASE_URL="http://localhost", LLM_API_KEY="test",
                           BOT_QQ=900000101, OWNER_QQ=900000102, SOCIAL_AGENT_ENABLED=True, AGENT_RUNTIME="legacy")
    return RuntimeConfig(settings, {"name": "小町"}, {"groups": groups or {}}, {}, social_policy=policy)


def test_social_policy_separates_explicit_groups_and_space_jobs():
    config = runtime({"enabled": True, "groups": {"900000001": {"enabled": True}},
                      "qzone": {"publish": {"enabled": True, "topics": ["秋天"]},
                                "comment": {"enabled": True, "friend_ids": [900000103]}}},
                     {"900000001": {"enabled": True, "speak": True, "proactive_reply": True}})
    jobs = configured_social_jobs(config)
    assert [job.kind for job in jobs] == ["group", "qzone_publish", "qzone_comment"]
    assert jobs[1].public_topics == ("秋天",)
    assert jobs[2].friend_ids == (900000103,)


def test_social_policy_requires_both_explicit_enable_and_existing_group_permission():
    config = runtime({"enabled": True, "groups": {"900000001": {"enabled": True}}},
                     {"900000001": {"enabled": True, "speak": False}})
    assert configured_social_jobs(config) == ()
    config.settings.social_agent_enabled = False
    assert configured_social_jobs(config) == ()


@pytest.mark.parametrize("policy", [
    {"enabled": "yes"},
    {"enabled": True, "qzone": {"publish": {"enabled": True, "interval_seconds": 0}}},
    {"enabled": True, "qzone": {"comment": {"enabled": True, "friend_ids": [True]}}},
    {"enabled": True, "qzone": {"publish": {"enabled": True, "visibility": 1}}},
    {"enabled": True, "qzone": {"publish": {"enabled": True, "topics": ["音乐"],
                                            "quiet_hours": ["23:00+08:00", "08:00+08:00"]}}},
])
def test_social_policy_rejects_malformed_or_broad_defaults(policy):
    with pytest.raises(ValueError):
        configured_social_jobs(runtime(policy))


def test_quiet_hours_use_shanghai_and_cross_midnight():
    assert in_quiet_hours(("23:00", "08:00"), datetime(2026, 10, 9, 16, tzinfo=UTC))
    assert not in_quiet_hours(("23:00", "08:00"), datetime(2026, 10, 9, 8, tzinfo=UTC))
    assert not in_quiet_hours(None, datetime(2026, 10, 9, 16, tzinfo=UTC))
    assert not in_quiet_hours(("08:00", "08:00"), datetime(2026, 10, 9, 16, tzinfo=UTC))
