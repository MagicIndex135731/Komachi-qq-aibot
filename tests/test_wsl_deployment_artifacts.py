from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wsl_required_files_exist() -> None:
    required = [
        "infra/wsl/README.md",
        "infra/wsl/.env.example",
        "infra/wsl/docker-compose.yml",
        "infra/wsl/scripts/bootstrap_wsl.sh",
        "infra/wsl/scripts/start.sh",
        "infra/wsl/scripts/stop.sh",
        "infra/wsl/scripts/status.sh",
        "infra/wsl/scripts/onebot_probe.py",
        "infra/wsl/scripts/onebot_watchdog.py",
        "infra/wsl/scripts/notify_windows.ps1",
        "infra/wsl/scripts/keepalive.sh",
        "infra/wsl/scripts/run_entry.ps1",
        "infra/wsl/scripts/xiaomachi-wsl-entry.sh",
        "infra/wsl/scripts/sync_from_windows.ps1",
        "infra/wsl/scripts/redact_env.ps1",
        "start-xiaomachi-wsl.bat",
        "stop-xiaomachi-wsl.bat",
        "status-xiaomachi-wsl.bat",
        "open-napcat-webui.bat",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_windows_bat_entries_use_ascii_only_wsl_repo_discovery() -> None:
    bat_files = [
        REPO_ROOT / "start-xiaomachi-wsl.bat",
        REPO_ROOT / "stop-xiaomachi-wsl.bat",
        REPO_ROOT / "status-xiaomachi-wsl.bat",
    ]
    assert all(path.exists() for path in bat_files)
    assert all(path.name.isascii() for path in REPO_ROOT.glob("*-wsl.bat"))
    for bat_file in bat_files:
        content = bat_file.read_text(encoding="utf-8")
        assert content.isascii()
        assert "wsl.exe bash /mnt/d/xiaomachi-wsl-entry.sh" in content
        assert "wsl.exe bash -lc" not in content
        assert "%~dp0" not in content
        assert "powershell" not in content.lower()
        assert "run_entry.ps1" not in content

    entry = (REPO_ROOT / "infra/wsl/scripts/xiaomachi-wsl-entry.sh").read_text(encoding="utf-8")
    assert "for base in /mnt/d /mnt/e /mnt/c" in entry
    assert "find \"${base}\"" in entry
    assert "pyproject.toml" in entry
    assert "infra/wsl/scripts/${ACTION}.sh" in entry


def test_open_napcat_webui_shortcut_is_ascii_and_never_starts_wsl_or_docker() -> None:
    shortcut = REPO_ROOT / "open-napcat-webui.bat"
    launcher = REPO_ROOT / "infra/wsl/scripts/open_napcat_webui.ps1"
    content = shortcut.read_text(encoding="utf-8")
    launcher_content = launcher.read_text(encoding="utf-8")

    assert content.isascii()
    assert "curl.exe" in content
    assert "http://127.0.0.1:6099/" in content
    assert "open_napcat_webui.ps1" in content
    assert "start-xiaomachi-wsl.bat" in content
    assert "wsl.exe" not in content.lower()
    assert "docker" not in content.lower()
    assert launcher_content.isascii()
    assert "infra/wsl/runtime/napcat/config/webui.json" in launcher_content.replace("\\", "/")
    assert "ConvertFrom-Json" in launcher_content
    assert "EscapeDataString" in launcher_content
    assert "http://127.0.0.1:6099/webui/qq_login?token=" in launcher_content
    assert "Start-Process" in launcher_content
    assert "local-test-token" not in launcher_content


def test_wsl_env_example_has_no_real_secrets() -> None:
    env_example = (REPO_ROOT / "infra/wsl/.env.example").read_text(encoding="utf-8")
    bot_account = "398" + "301" + "0865"
    personal_account = "180" + "753" + "3371"
    forbidden = ["sk-", "Bearer ", "OPENAI_API_KEY=", bot_account, personal_account]
    assert not any(token in env_example for token in forbidden)
    assert "NAPCAT_WS_URL=ws://napcat:3001" in env_example
    assert "NAPCAT_QUICK_PASSWORD=" in env_example
    assert "NAPCAT_QUICK_PASSWORD_MD5=" in env_example
    assert "GROUP_STREAM_WATCH_GROUP_ID=" in env_example


def test_gitignore_excludes_wsl_runtime_state() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "infra/wsl/runtime/",
        "infra/wsl/.env",
        "data/napcat/",
        "data/logs/",
        ".venv-wsl/",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore


def test_bootstrap_allows_probe_venv_on_ubuntu_2204_python() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/bootstrap_wsl.sh").read_text(encoding="utf-8")
    assert "sys.version_info >= (3, 12)" in script
    assert "./.venv-wsl/bin/python -m pip install websockets" in script
    assert "python -m pip install -e ." in script


def test_bootstrap_preconfigures_napcat_onebot_websocket_server() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/bootstrap_wsl.sh").read_text(encoding="utf-8")
    assert "runtime/napcat/config/onebot11.json" in script
    assert '"websocketServers"' in script
    assert '"host": "0.0.0.0"' in script
    assert '"port": 3001' in script
    assert '"enable": true' in script


def test_compose_uses_docker_safe_napcat_login_and_healthcheck() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")
    assert "image: mlikiowa/napcat-docker:latest" in compose
    assert "ACCOUNT=${BOT_QQ:-}" in compose
    assert "NAPCAT_QUICK_PASSWORD=${NAPCAT_QUICK_PASSWORD:-}" in compose
    assert "NAPCAT_QUICK_PASSWORD_MD5=${NAPCAT_QUICK_PASSWORD_MD5:-}" in compose
    assert "curl -fsS http://127.0.0.1:6099/" in compose
    assert "node -e" not in compose


def test_napcat_mounts_generated_images_at_the_sender_file_uri_path() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")

    assert "../../data/generated_images:/workspace/data/generated_images:ro" in compose


def test_xiaomachi_container_uses_host_network_and_optional_proxy_for_dependencies() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")
    assert "network_mode: host" in compose
    assert "NAPCAT_WS_URL=ws://127.0.0.1:3001" in compose
    assert "HTTP_PROXY=${DOCKER_HTTP_PROXY:-}" in compose
    assert "HTTPS_PROXY=${DOCKER_HTTPS_PROXY:-}" in compose
    assert "PIP_INDEX_URL=${PIP_INDEX_URL:-}" in compose


def test_xiaomachi_startup_installs_dependencies_with_proxy_friendly_timeouts() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")
    assert "pip setuptools wheel" in compose
    assert "--timeout ${PIP_DEFAULT_TIMEOUT:-120}" in compose
    assert "--retries ${PIP_RETRIES:-10}" in compose
    assert "python -m pip install --timeout ${PIP_DEFAULT_TIMEOUT:-120} --retries ${PIP_RETRIES:-10} --no-build-isolation -e ." in compose
    assert "./runtime/pip-cache:/root/.cache/pip" in compose


def test_status_script_waits_for_health_and_uses_probe_before_logs() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/status.sh").read_text(encoding="utf-8")
    assert "Waiting for NapCat healthcheck" in script
    assert "Waiting for OneBot websocket..." in script
    assert "Waiting for xiaomachi bot heartbeat..." in script
    assert "group.heartbeat.json" in script
    assert "heartbeat_age_seconds" in script
    assert "timezone.utc" in script
    assert "from datetime import UTC" not in script
    assert 'probe_output="$(mktemp)"' in script
    assert "docker inspect" in script
    assert "onebot_probe.py" in script
    assert "--ws-url ws://127.0.0.1:3001" in script
    assert "docker compose logs --tail=80 napcat" in script
    assert "docker compose logs --tail=80 xiaomachi" in script


def test_start_script_waits_for_status_readiness() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    assert 'bash "${WSL_DIR}/scripts/status.sh"' in script


def test_start_and_stop_manage_wsl_keepalive_anchor() -> None:
    start_script = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    stop_script = (REPO_ROOT / "infra/wsl/scripts/stop.sh").read_text(encoding="utf-8")
    keepalive_script = (REPO_ROOT / "infra/wsl/scripts/keepalive.sh").read_text(encoding="utf-8")

    assert "keepalive.enabled" in start_script
    assert "xiaomachi-wsl-keepalive" in start_script
    assert 'nohup setsid bash -c' in start_script
    assert 'bash "${WSL_DIR}/scripts/keepalive.sh"' not in start_script
    assert "keepalive.enabled" in stop_script
    assert "keepalive.pid" in stop_script
    assert "xiaomachi-wsl-keepalive" in stop_script
    assert 'while [[ -f "${flag_file}" ]]' in keepalive_script
    assert 'echo "$$" >"${pid_file}"' in keepalive_script


def test_stop_terminates_the_keepalive_process_group_before_state_cleanup() -> None:
    stop_script = (REPO_ROOT / "infra/wsl/scripts/stop.sh").read_text(encoding="utf-8")

    remove_flag = stop_script.index('rm -f "${flag_file}"')
    kill_group = stop_script.index('kill -- "-${existing_pid}"')
    remove_state = stop_script.index('"${runtime_dir}/onebot-watchdog.json"')
    assert remove_flag < kill_group < remove_state
