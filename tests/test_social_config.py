from pathlib import Path

from app.agent.social_policy import configured_social_jobs
from app.config import AppSettings, load_runtime_config


def test_public_social_defaults_and_private_overlay_load_into_explicit_jobs(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name, contents in {
        "persona.yaml": "name: 小町\n",
        "groups.yaml": "groups: {}\n",
        "safety.yaml": "deny_prompt_leak: true\n",
    }.items():
        (config_dir / name).write_text(contents, encoding="utf-8")
    sample = Path(__file__).resolve().parents[1] / "configs/social.yaml"
    (config_dir / "social.yaml").write_bytes(sample.read_bytes())
    settings = AppSettings(_env_file=None, config_dir=config_dir, data_dir=tmp_path / "data",
                           NAPCAT_WS_URL="ws://localhost:3001", LLM_BASE_URL="http://localhost",
                           LLM_API_KEY="test", BOT_QQ=900000101, OWNER_QQ=900000102,
                           SOCIAL_AGENT_ENABLED=True)
    assert configured_social_jobs(load_runtime_config(settings)) == ()

    (config_dir / "social.local.yaml").write_text(
        "enabled: true\nqzone:\n  publish:\n    enabled: true\n    topics: [音乐]\n"
        "  comment:\n    enabled: true\n    friend_ids: [900000103]\n", encoding="utf-8",
    )
    jobs = configured_social_jobs(load_runtime_config(settings))
    assert [job.kind for job in jobs] == ["qzone_publish", "qzone_comment"]
    assert jobs[0].visibility == 4 and jobs[0].public_topics == ("音乐",)
    assert jobs[1].friend_ids == (900000103,) and jobs[1].quiet_hours == ("23:00", "08:00")
    settings.social_agent_enabled = False
    assert configured_social_jobs(load_runtime_config(settings)) == ()
