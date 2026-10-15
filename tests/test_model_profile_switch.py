"""Model switches use synthetic settings and fake runtime commands only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_profile_switch", ROOT / "infra/wsl/scripts/switch_model_profile.py"
)
switch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(switch)


def profile():
    return {
        **switch.DEFAULTS,
        "LLM_BASE_URL": "https://synthetic.invalid/v1",
        "LLM_API_KEY": "synthetic-token",
        "LLM_MODEL": "gemini-3.8-flash-medium",
        "LLM_FALLBACK_MODEL": "gemini-3.8-flash-medium",
        "LLM_VISION_MODEL": "gemini-3.8-flash-medium",
        "PROACTIVE_JUDGE_MODEL": "gemini-3.8-flash-medium",
        "GROUP_IMAGE_MODEL": "synthetic-image",
    }


def test_overlay_preserves_unrelated_settings_and_removes_duplicate_model_keys():
    original = "# private\r\nBOT_QQ=900000101\r\nLLM_MODEL=old\r\nSOCIAL_AGENT_ENABLED=true\r\nLLM_MODEL=other\r\n"
    result = switch.overlay_env(original, {"LLM_MODEL": "new", "LLM_API_KEY": "abc$#'xyz"})
    assert "BOT_QQ=900000101\r\n" in result
    assert "SOCIAL_AGENT_ENABLED=true\r\n" in result
    assert result.count("LLM_MODEL=") == 1
    assert switch.parse_env(result)["LLM_API_KEY"] == "abc$#'xyz"
    assert "\n" not in result.replace("\r\n", "")


@pytest.mark.parametrize("patch", [
    {"BOT_QQ": "900000102"},
    {"LLM_BASE_URL": "https://user:secret@synthetic.invalid"},
    {"LLM_MODEL": "model\nBOT_QQ=900000102"},
    {"LLM_REASONING_EFFORT": "unknown"},
    {"GROUP_IMAGE_TRANSPORT": "unknown"},
])
def test_invalid_or_non_model_profile_is_rejected(patch):
    with pytest.raises(switch.SwitchError):
        switch.validate_profile({**profile(), **patch})


def test_verified_reasoning_alias_updates_every_matching_text_slot(tmp_path):
    sidecar = tmp_path / "gemini.reasoning.json"
    sidecar.write_text(json.dumps({"low": "gemini-3.8-flash-low", "medium": "gemini-3.8-flash-medium"}))
    settings = profile()
    settings["LLM_WEB_SEARCH_MODEL"] = "synthetic-search"
    result = switch.with_reasoning(settings, "low", sidecar)
    for key in ("LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_VISION_MODEL", "PROACTIVE_JUDGE_MODEL"):
        assert result[key] == "gemini-3.8-flash-low"
    assert result["LLM_WEB_SEARCH_MODEL"] == "synthetic-search"
    assert result["MEMORY_COMPACTION_REASONING_EFFORT"] == "low"
    assert result["WEB_SEARCH_REASONING_EFFORT"] == settings["WEB_SEARCH_REASONING_EFFORT"]
    with pytest.raises(switch.SwitchError):
        switch.with_reasoning(settings, "minimal", sidecar)


def test_search_effort_stays_independent_when_chat_reasoning_changes(tmp_path):
    sidecar=tmp_path/'gemini.reasoning.json'
    sidecar.write_text(json.dumps({'low':'gemini-3.8-flash-low','medium':'gemini-3.8-flash-medium','high':'gemini-3.8-flash-high'}))
    settings={**profile(),'SEARCH_PROVIDER':'gemini','LLM_WEB_SEARCH_MODEL':'gemini-3.8-flash-medium','WEB_SEARCH_REASONING_EFFORT':'medium'}
    changed=switch.with_reasoning(settings,'high',sidecar)
    assert changed['LLM_MODEL']=='gemini-3.8-flash-high'
    assert changed['LLM_WEB_SEARCH_MODEL']=='gemini-3.8-flash-medium'
    assert changed['WEB_SEARCH_REASONING_EFFORT']=='medium'


def test_tiered_preserves_alias_but_keeps_auxiliary_calls_low(tmp_path):
    sidecar = tmp_path / "gemini.reasoning.json"
    sidecar.write_text(json.dumps({"low": "gemini-3.8-flash-low", "medium": "gemini-3.8-flash-medium", "high": "gemini-3.8-flash-high", "tiered": "gemini-3.8-flash-tiered"}))
    result = switch.with_reasoning(profile(), "tiered", sidecar)
    # A profile must itself point to a verified member of its sidecar map.
    assert result["LLM_MODEL"] == "gemini-3.8-flash-tiered"
    assert result["LLM_REASONING_EFFORT"] == ""
    assert result["PROACTIVE_JUDGE_MODEL"] == "gemini-3.8-flash-low"
    assert result["GROUP_AGENT_DECISION_REASONING_EFFORT"] == "low"
    assert result["MEMORY_COMPACTION_REASONING_EFFORT"] == "low"


@pytest.mark.parametrize("platform,compose", [("llbot", "docker-compose.llbot.yml"), ("snowluma", "docker-compose.snowluma.yml"), ("napcat", "docker-compose.yml")])
def test_recreate_targets_only_apps_and_preserves_gpu(tmp_path, platform, compose):
    runtime = switch.Runtime(tmp_path, tmp_path / "local.env", tmp_path / "profiles")
    command = runtime.compose_command({"QQ_PLATFORM": platform, "ENABLE_GPU": "1"})
    assert command[-7:] == ["up", "-d", "--no-deps", "--force-recreate", "--no-build", "xiaomachi", "xiaomachi-private"]
    assert str(tmp_path / "current/infra/wsl" / compose) in command
    assert str(tmp_path / "current/infra/wsl/docker-compose.gpu.yml") in command
    assert "xiaomachi-sandbox" not in command


class FakeRuntime(switch.Runtime):
    def __init__(self, root, local, profiles, fail_stage=""):
        super().__init__(root, local, profiles)
        self.fail_stage = fail_stage
        self.calls = []
        self.recreated = False
        self.image_ids = {"xiaomachi-bot": "sha256:" + "a" * 64, "xiaomachi-private": "sha256:" + "b" * 64}

    def execute(self, command, *, input_text=None, timeout=120):
        self.calls.append(command)
        if command[:4] == ["docker", "inspect", "--format", "{{.Image}}"]:
            return self.image_ids[command[-1]]
        if command[:3] == ["systemctl", "is-active", "--quiet"]:
            return ""
        if "up" in command:
            self.recreated = True
            if self.fail_stage == "recreate":
                self.fail_stage = ""
                raise switch.SwitchError("synthetic recreate failure")
        if command[0] == "bash" and self.fail_stage == "health":
            self.fail_stage = ""
            raise switch.SwitchError("synthetic health failure")
        if command[:2] == ["docker", "exec"] and input_text is not None:
            if self.fail_stage == "probe" and not self.recreated:
                raise switch.SwitchError("synthetic probe failure")
        return ""


def prepare(tmp_path, fail_stage=""):
    local = tmp_path / "workspace/infra/wsl/.env"
    shared = tmp_path / "shared/.env"
    local.parent.mkdir(parents=True)
    shared.parent.mkdir(parents=True)
    original = "QQ_PLATFORM=snowluma\nENABLE_GPU=1\nBOT_QQ=900000101\nLLM_MODEL=old\nLLM_API_KEY=old-token\n"
    local.write_text(original + "SOURCE_ONLY=true\n")
    shared.write_text(original + "RUNTIME_ONLY=true\n")
    profiles = local.parent / "runtime/model-profiles"
    profiles.mkdir(parents=True)
    runtime = FakeRuntime(tmp_path, local, profiles, fail_stage)
    return runtime, local.read_bytes(), shared.read_bytes()


def test_success_updates_source_and_shared_but_not_non_model_keys(tmp_path):
    runtime, _, _ = prepare(tmp_path)
    runtime.activate("gemini", profile())
    assert switch.parse_env(runtime.local_env.read_text())["LLM_MODEL"] == profile()["LLM_MODEL"]
    assert "SOURCE_ONLY=true" in runtime.local_env.read_text()
    assert "RUNTIME_ONLY=true" in runtime.shared_env.read_text()
    assert (runtime.profiles / "current.json").is_file()
    assert switch.parse_env((runtime.profiles / "gemini.env").read_text())["LLM_MODEL"] == profile()["LLM_MODEL"]
    recreate = [command for command in runtime.calls if "up" in command]
    assert len(recreate) == 1
    assert ["systemctl", "start", "xiaomachi-watchdog.service"] in runtime.calls


@pytest.mark.parametrize("stage", ["health", "recreate"])
def test_failed_activation_restores_both_original_files_and_apps(tmp_path, stage):
    runtime, local, shared = prepare(tmp_path, stage)
    with pytest.raises(switch.SwitchError, match="rolled back"):
        runtime.activate("gemini", profile())
    assert runtime.local_env.read_bytes() == local
    assert runtime.shared_env.read_bytes() == shared
    assert len([command for command in runtime.calls if "up" in command]) == 2
    assert not (runtime.profiles / "current.json").exists()
    assert runtime.calls[-1] == ["systemctl", "start", "xiaomachi-watchdog.service"]


def test_failed_preflight_never_changes_env_or_recreates_apps(tmp_path):
    runtime, local, shared = prepare(tmp_path, "probe")
    with pytest.raises(switch.SwitchError):
        runtime.activate("gemini", profile())
    assert runtime.local_env.read_bytes() == local
    assert runtime.shared_env.read_bytes() == shared
    assert not any("up" in command for command in runtime.calls)


def test_subprocess_error_output_cannot_leak_provider_credentials(tmp_path, monkeypatch):
    runtime = switch.Runtime(tmp_path, tmp_path / ".env", tmp_path / "profiles")
    monkeypatch.setattr(switch.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "synthetic-secret-body", "synthetic-secret-key"))
    with pytest.raises(switch.SwitchError) as failure:
        runtime.execute(["docker", "exec", "synthetic-container"])
    assert "synthetic-secret" not in str(failure.value)


def test_partial_file_update_failure_restores_source_and_shared(tmp_path, monkeypatch):
    runtime, local, shared = prepare(tmp_path)
    original_write = switch.atomic_write
    failed = False

    def failing_write(path, content):
        nonlocal failed
        if path == runtime.shared_env and not failed:
            failed = True
            raise OSError("synthetic write failure")
        return original_write(path, content)

    monkeypatch.setattr(switch, "atomic_write", failing_write)
    with pytest.raises(switch.SwitchError, match="rolled back"):
        runtime.activate("gemini", profile())
    assert runtime.local_env.read_bytes() == local
    assert runtime.shared_env.read_bytes() == shared


def test_capture_never_overwrites_saved_profile_credentials(tmp_path):
    runtime, _, _ = prepare(tmp_path)
    # Capture reads the deployed snapshot; the private token stays in its file.
    runtime.shared_env.write_text(switch.overlay_env("QQ_PLATFORM=snowluma\n", profile()))
    runtime.capture("sol")
    saved = (runtime.profiles / "sol.env").read_bytes()
    with pytest.raises(switch.SwitchError, match="refusing to overwrite"):
        runtime.capture("sol")
    assert (runtime.profiles / "sol.env").read_bytes() == saved


def test_environment_optional_overrides_are_cleared_between_profiles(tmp_path):
    runtime, _, _ = prepare(tmp_path)
    runtime.local_env.write_text(runtime.local_env.read_text() + "LLM_WEB_SEARCH_API_KEY=previous-token\nGROUP_IMAGE_CHAT_API_KEY=previous-image-token\n")
    runtime.activate("gemini", profile())
    settings = switch.parse_env(runtime.local_env.read_text())
    assert settings["LLM_WEB_SEARCH_API_KEY"] == ""
    assert settings["GROUP_IMAGE_CHAT_API_KEY"] == ""


def test_recreate_and_rollback_pin_both_exact_running_images(tmp_path):
    runtime, _, _ = prepare(tmp_path, "health")
    with pytest.raises(switch.SwitchError, match="rolled back"):
        runtime.activate("gemini", profile())
    recreates = [command for command in runtime.calls if "up" in command]
    assert len(recreates) == 2
    assert recreates[0] == recreates[1]
    override_path = Path(recreates[0][recreates[0].index("up") - 1])
    payload = json.loads(override_path.read_text())
    assert payload["services"] == {
        "xiaomachi": {"image": runtime.image_ids["xiaomachi-bot"], "pull_policy": "never"},
        "xiaomachi-private": {"image": runtime.image_ids["xiaomachi-private"], "pull_policy": "never"},
    }
    assert "xiaomachi-bot:local" not in override_path.read_text()
