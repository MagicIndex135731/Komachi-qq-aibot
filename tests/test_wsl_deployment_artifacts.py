from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wsl_required_files_exist() -> None:
    required = [
        "infra/wsl/README.md",
        "infra/wsl/.env.example",
        "infra/wsl/docker-compose.yml",
        "infra/wsl/Dockerfile.xiaomachi",
        "infra/wsl/requirements.xiaomachi.txt",
        "infra/wsl/scripts/bootstrap_wsl.sh",
        "infra/wsl/scripts/start.sh",
        "infra/wsl/scripts/stop.sh",
        "infra/wsl/scripts/status.sh",
        "infra/wsl/scripts/onebot_probe.py",
        "infra/wsl/scripts/onebot_watchdog.py",
        "infra/wsl/scripts/notify_windows.ps1",
        "infra/wsl/scripts/keepalive.sh",
        "infra/wsl/scripts/anchor.sh",
        "infra/wsl/scripts/xiaomachi-wsl-entry.sh",
        "infra/wsl/scripts/install_linux_runtime.sh",
        "infra/wsl/systemd/xiaomachi-stack.service",
        "infra/wsl/systemd/xiaomachi-watchdog.service",
        "start-xiaomachi-wsl.bat",
        "stop-xiaomachi-wsl.bat",
        "status-xiaomachi-wsl.bat",
        "open-napcat-webui.bat",
        "open-llbot-webui.bat",
        "infra/wsl/docker-compose.llbot.yml",
        "infra/wsl/scripts/bootstrap_llbot_runtime.py",
        "infra/wsl/scripts/migrate_xiaomachi_data_volume.sh",
        "infra/wsl/scripts/open_llbot_webui.ps1",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_windows_bat_entries_prefer_fixed_linux_runtime() -> None:
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
        assert "/usr/local/bin/xiaomachi-wsl-entry" in content
        assert 'wsl.exe --user root --exec "%ENTRY%"' in content
        assert "wsl.exe bash -lc" not in content
        if bat_file.name == "start-xiaomachi-wsl.bat":
            assert '--user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh install' in content
            assert "Start-Process" in content
            assert "'/usr/local/bin/xiaomachi-wsl-entry','anchor'" in content
            assert "Xiaomachi started successfully." in content
        assert "schtasks.exe" not in content

    status_entry = (REPO_ROOT / "status-xiaomachi-wsl.bat").read_text(encoding="utf-8")
    assert 'set "STATUS_EXIT_CODE=%ERRORLEVEL%"' in status_entry
    assert "exit /b %STATUS_EXIT_CODE%" in status_entry

    entry = (REPO_ROOT / "infra/wsl/scripts/xiaomachi-wsl-entry.sh").read_text(encoding="utf-8")
    assert 'install_root="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"' in entry
    assert 'runtime_entry="${install_root}/current/infra/wsl/scripts/${ACTION}.sh"' in entry
    assert "systemctl start xiaomachi-watchdog.service" in entry
    assert "run_systemd_with_output start xiaomachi-stack.service" in entry
    assert "journalctl --no-pager --follow --output=cat" in entry
    assert "run_systemd_with_output stop xiaomachi-stack.service" in entry
    assert "for base in /mnt/d /mnt/e /mnt/c" in entry
    assert "find \"${base}\"" in entry
    assert "pyproject.toml" in entry
    assert "install_linux_runtime.sh" in entry
    assert "start|stop|status|anchor|install" in entry


def test_linux_runtime_installer_copies_allowlist_to_ext4_release_tree() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/install_linux_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert 'XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi' in script
    assert '"${INSTALL_ROOT}/current"' in script
    assert '"${shared_dir}/.env"' in script
    assert '"${shared_runtime}"' in script
    assert "app configs infra/wsl .dockerignore pyproject.toml README.md LICENSE" in script
    source_copy = script.split('if [[ ! -f "${shared_dir}/.env" ]]', 1)[0]
    assert "-cf - ." not in source_copy
    assert "--exclude='infra/wsl/runtime'" in script
    assert "systemctl daemon-reload" in script
    assert "install_windows_autostart.ps1" not in script
    assert 'install -m 0600 "${SOURCE_ROOT}/infra/wsl/.env" "${shared_dir}/.env.next"' in script
    assert 'docker rename "${original_name}" "${legacy_name}"' in script
    assert 'docker rename "${legacy_names[$index]}" "${legacy_original_names[$index]}"' in script
    assert 'docker compose -f "${release_dir}/infra/wsl/docker-compose.llbot.yml" build xiaomachi' in script


def test_systemd_uses_persistent_supervision_without_windows_autostart() -> None:
    stack = (REPO_ROOT / "infra/wsl/systemd/xiaomachi-stack.service").read_text(
        encoding="utf-8"
    )
    watchdog = (REPO_ROOT / "infra/wsl/systemd/xiaomachi-watchdog.service").read_text(
        encoding="utf-8"
    )
    entry = (REPO_ROOT / "infra/wsl/scripts/xiaomachi-wsl-entry.sh").read_text(
        encoding="utf-8"
    )
    anchor = (REPO_ROOT / "infra/wsl/scripts/anchor.sh").read_text(encoding="utf-8")

    assert "WorkingDirectory=/opt/xiaomachi/current/infra/wsl" in stack
    assert "RemainAfterExit=yes" in stack
    assert "Description=Xiaomachi persistent OneBot watchdog" in watchdog
    assert "Restart=on-failure" in watchdog
    assert "anchor.sh watchdog" in watchdog
    assert "WantedBy=" not in stack
    assert "WantedBy=" not in watchdog
    assert "systemctl enable" not in anchor
    assert "systemctl start xiaomachi-stack.service xiaomachi-watchdog.service" in anchor
    assert "while systemctl is-active --quiet xiaomachi-stack.service" in anchor
    assert "systemctl is-active --quiet xiaomachi-stack.service" in entry
    assert "Xiaomachi systemd supervision is not active." in entry


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


def test_wsl_start_opens_selected_platform_login_before_status_probe() -> None:
    start_script = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "infra/wsl/scripts/open_napcat_webui.ps1").read_text(
        encoding="utf-8"
    )

    compose_up = start_script.index('docker compose -f "${compose_file}" up -d "${service_name}"')
    image_build = start_script.index('docker compose -f "${compose_file}" build xiaomachi')
    conditional_open = start_script.index("\nopen_login_page\n")
    bot_up = start_script.index('docker compose -f "${compose_file}" up -d --no-deps xiaomachi')
    status_probe = start_script.index('bash "${SCRIPT_DIR}/status.sh"')
    assert image_build < compose_up < conditional_open < bot_up < status_probe
    assert 'docker compose -f "${compose_file}" up -d --no-deps xiaomachi' in start_script
    assert "webui_port=6099" in start_script
    assert "webui_port=3080" in start_script
    assert "wslpath -w" in start_script
    assert "powershell.exe" in start_script
    assert "-OnlyWhenLoginRequired" in start_script
    assert "|| true" in start_script
    assert "Waiting for ${service_name} WebUI (${attempt}/10)" in start_script
    assert "continuing to status diagnostics" in start_script

    assert "param(" in launcher
    assert "OnlyWhenLoginRequired" in launcher
    assert "/api/auth/login" in launcher
    assert "/api/QQLogin/CheckLoginStatus" in launcher
    assert "AllowAutoRedirect = $false" in launcher
    assert "isLogin" in launcher


def test_wsl_env_example_has_no_real_secrets() -> None:
    env_example = (REPO_ROOT / "infra/wsl/.env.example").read_text(encoding="utf-8")
    bot_account = "398" + "301" + "0865"
    personal_account = "180" + "753" + "3371"
    forbidden = ["sk-", "Bearer ", "OPENAI_API_KEY=", bot_account, personal_account]
    assert not any(token in env_example for token in forbidden)
    assert "NAPCAT_WS_URL=ws://napcat:3001" in env_example
    assert "QQ_PLATFORM=llbot" in env_example
    assert "LLBOT_WS_PORT=3002" in env_example
    assert "NAPCAT_QUICK_PASSWORD=" in env_example
    assert "NAPCAT_QUICK_PASSWORD_MD5=" in env_example
    assert "GROUP_STREAM_WATCH_GROUP_ID=" in env_example


def test_memory_orchestration_env_and_docs_define_a_safe_bot_only_rollout() -> None:
    env_example = (REPO_ROOT / "infra/wsl/.env.example").read_text(encoding="utf-8")
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    wsl_readme = (REPO_ROOT / "infra/wsl/README.md").read_text(encoding="utf-8")
    required_settings = [
        "MEMORY_ORCHESTRATION_V2_ENABLED=true",
        "MEMORY_ORCHESTRATION_SHADOW_MODE=true",
        "MEMORY_EMBEDDING_PROVIDER=local",
        "MEMORY_EMBEDDING_DEVICE=auto",
        "MEMORY_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5",
        "MEMORY_EMBEDDING_DIMENSIONS=512",
        "MEMORY_EMBEDDING_CACHE_DIR=/workspace/data/models",
        "MEMORY_EMBEDDING_BASE_URL=",
        "MEMORY_EMBEDDING_API_KEY=",
        "MEMORY_EMBEDDING_VERSION=",
        "MEMORY_EPISODE_IDLE_MINUTES=30",
        "MEMORY_EPISODE_MAX_MESSAGES=50",
        "MEMORY_EPISODE_MAX_TOKENS=8000",
        "MEMORY_CHUNK_MAX_TOKENS=1800",
        "MEMORY_CHUNK_OVERLAP_MESSAGES=5",
        "MEMORY_QUERY_REWRITE_ENABLED=false",
        "MEMORY_QUERY_REWRITE_TIMEOUT_SECONDS=3",
        "MEMORY_QUERY_REWRITE_MAX_OUTPUT_TOKENS=256",
        "MEMORY_LLM_RERANK_ENABLED=false",
        "MEMORY_NORMAL_CONTEXT_BUDGET_TOKENS=32000",
        "MEMORY_DETAIL_CONTEXT_BUDGET_TOKENS=64000",
        "MEMORY_RECENT_CONTEXT_BUDGET_TOKENS=10000",
        "MEMORY_FTS_CANDIDATE_LIMIT=30",
        "MEMORY_VECTOR_CANDIDATE_LIMIT=30",
        "MEMORY_FINAL_EPISODE_LIMIT=6",
    ]
    for setting in required_settings:
        assert setting in env_example

    for documentation in (root_readme, wsl_readme):
        assert "shadow -> backfill -> evaluate -> active" in documentation
        assert "/workspace/data/models" in documentation
        assert "MEMORY_ORCHESTRATION_V2_ENABLED=false" in documentation
        assert "MEMORY_ORCHESTRATION_V2_ENABLED=true" in documentation
        assert "MEMORY_EMBEDDING_PROVIDER=disabled" in documentation
        assert "MEMORY_EMBEDDING_DEVICE=auto" in documentation
        assert "nvidia.com/gpu=all" in documentation
        assert "docker compose build xiaomachi" in documentation
        assert "docker compose up -d --no-deps --force-recreate xiaomachi" in documentation
        assert "xiaomachi-llbot" in documentation
        assert "must not restart xiaomachi-llbot" in documentation


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


def test_bootstrap_installs_watchdog_venv_on_linux_ext4() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/bootstrap_wsl.sh").read_text(encoding="utf-8")
    assert "/opt/xiaomachi/shared/venv" in script
    assert '"${watchdog_venv}/bin/python" -m pip install websockets' in script
    assert "pip install -e ." not in script


def test_bootstrap_preconfigures_napcat_onebot_websocket_server() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/bootstrap_wsl.sh").read_text(encoding="utf-8")
    assert "runtime/napcat/config/onebot11.json" in script
    assert '"websocketServers"' in script
    assert '"host": "0.0.0.0"' in script
    assert '"port": 3001' in script
    assert '"enable": true' in script


def test_compose_uses_linux_volumes_and_no_periodic_napcat_health_process() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")
    assert "image: mlikiowa/napcat-docker:latest" in compose
    assert "ACCOUNT=${BOT_QQ:-}" in compose
    assert "NAPCAT_QUICK_PASSWORD=${NAPCAT_QUICK_PASSWORD:-}" in compose
    assert "NAPCAT_QUICK_PASSWORD_MD5=${NAPCAT_QUICK_PASSWORD_MD5:-}" in compose
    assert "healthcheck:" not in compose
    assert "./runtime/napcat" not in compose
    assert "napcat_config:/app/napcat/config" in compose
    assert "napcat_ntqq:/app/.config/QQ" in compose


def test_napcat_mounts_shared_runtime_data_at_the_sender_file_uri_path() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")

    assert "xiaomachi_data:/workspace/data:ro" in compose


def test_xiaomachi_container_uses_prebuilt_local_image() -> None:
    compose = (REPO_ROOT / "infra/wsl/docker-compose.yml").read_text(encoding="utf-8")
    assert "network_mode: host" in compose
    assert "NAPCAT_WS_URL=ws://127.0.0.1:3001" in compose
    assert "HTTP_PROXY=${DOCKER_HTTP_PROXY:-}" in compose
    assert "HTTPS_PROXY=${DOCKER_HTTPS_PROXY:-}" in compose
    assert "image: xiaomachi-bot:local" in compose
    assert "context: ../.." in compose
    assert "dockerfile: infra/wsl/Dockerfile.xiaomachi" in compose
    assert "network: host" in compose
    assert "command: [\"python\", \"-m\", \"app.group_main\"]" in compose
    assert "python -m pip install" not in compose
    assert "./runtime/pip-cache:/root/.cache/pip" not in compose


def test_xiaomachi_dockerfile_installs_dependencies_at_build_time() -> None:
    dockerfile = (REPO_ROOT / "infra/wsl/Dockerfile.xiaomachi").read_text(encoding="utf-8")
    assert "FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "COPY infra/wsl/requirements.xiaomachi.txt" in dockerfile
    assert "COPY infra/wsl/requirements.xiaomachi-gpu.txt" in dockerfile
    assert "python -m pip uninstall -y fastembed onnxruntime" in dockerfile
    gpu_requirements = (
        REPO_ROOT / "infra/wsl/requirements.xiaomachi-gpu.txt"
    ).read_text(encoding="utf-8")
    assert "fastembed-gpu>=0.6.0,<0.7.0" in gpu_requirements
    assert "onnxruntime-gpu==1.22.0" in gpu_requirements
    assert "COPY app ./app" in dockerfile
    assert "COPY configs ./configs" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert "python -m pip install" in dockerfile
    assert "ARG HTTP_PROXY" in dockerfile
    assert "ARG HTTPS_PROXY" in dockerfile
    assert "PIP_INDEX_URL=${PIP_INDEX_URL}" not in dockerfile
    assert 'CMD ["python", "-m", "app.group_main"]' in dockerfile


def test_xiaomachi_image_requirements_match_pyproject() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = set(project["project"]["dependencies"])
    expected.update(project["project"]["optional-dependencies"]["dev"])
    actual = {
        line.strip()
        for line in (REPO_ROOT / "infra/wsl/requirements.xiaomachi.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert actual == expected
    assert "fastembed>=0.6.0,<0.7.0" in actual


def test_xiaomachi_compose_requests_only_the_nvidia_cdi_device_for_the_bot() -> None:
    for name in ("docker-compose.yml", "docker-compose.llbot.yml"):
        compose = yaml.safe_load((REPO_ROOT / "infra/wsl" / name).read_text(encoding="utf-8"))
        assert compose["services"]["xiaomachi"]["devices"] == ["nvidia.com/gpu=all"]
        platform_service = "llbot" if "llbot" in compose["services"] else "napcat"
        assert "devices" not in compose["services"][platform_service]


def test_status_script_uses_on_demand_probes_before_logs() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/status.sh").read_text(encoding="utf-8")
    assert 'Waiting for ${service_name} container' in script
    assert "LLBot WebUI probe:" in script
    assert "curl -fsS --max-time 8 http://127.0.0.1:3080/ >/dev/null" in script
    assert "waiting for LLBot WebUI" in script
    assert "OneBot probe (${platform})" in script
    assert "Waiting for xiaomachi bot heartbeat..." in script
    assert "group.heartbeat.json" in script
    assert "heartbeat_age_seconds" in script
    assert "timezone.utc" in script
    assert "from datetime import UTC" not in script
    assert 'probe_output="$(mktemp)"' in script
    assert "docker inspect" in script
    assert 'bot_container_name="xiaomachi-bot"' in script
    assert 'bot_status="$(docker inspect' in script
    assert '"${bot_status}" != "running"' in script
    assert "onebot_probe.py" in script
    assert 'onebot_ws_url="ws://127.0.0.1:${llbot_ws_port}"' in script
    assert '--ws-url "${onebot_ws_url}" --request-timeout 8' in script
    assert "probe_ok=false" in script
    assert "waiting for OneBot" in script
    assert "replay protection unavailable" in script
    assert "quick login and QR login cannot proceed yet" in script
    assert 'docker compose -f "${compose_file}" logs --tail=80 "${service_name}"' in script
    assert 'docker compose -f "${compose_file}" logs --tail=80 xiaomachi' in script


def test_start_script_waits_for_status_readiness() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    assert 'bash "${SCRIPT_DIR}/status.sh"' in script


def test_start_and_stop_manage_wsl_keepalive_anchor() -> None:
    start_script = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    stop_script = (REPO_ROOT / "infra/wsl/scripts/stop.sh").read_text(encoding="utf-8")
    keepalive_script = (REPO_ROOT / "infra/wsl/scripts/keepalive.sh").read_text(encoding="utf-8")
    anchor_script = (REPO_ROOT / "infra/wsl/scripts/anchor.sh").read_text(encoding="utf-8")

    assert "keepalive.enabled" in start_script
    assert 'touch "${flag_file}"' in start_script
    assert 'nohup setsid bash -c' not in start_script
    assert "keepalive.enabled" in stop_script
    assert "keepalive.pid" in stop_script
    assert "xiaomachi-wsl-keepalive" in stop_script
    assert 'while [[ -f "${flag_file}" ]]' in keepalive_script
    assert 'echo "$$" >"${pid_file}"' in keepalive_script
    assert 'exec -a xiaomachi-wsl-keepalive' in anchor_script
    assert 'keepalive.sh' in anchor_script
    assert 'keepalive.enabled' in anchor_script
    assert 'flock -n 9' in anchor_script
    assert 'trap cleanup_failed_start EXIT' in start_script
    assert 'pkill -f xiaomachi-wsl-keepalive' in start_script
    assert 'flock -n 8' in start_script


def test_stop_terminates_the_keepalive_process_group_before_state_cleanup() -> None:
    stop_script = (REPO_ROOT / "infra/wsl/scripts/stop.sh").read_text(encoding="utf-8")

    remove_flag = stop_script.index('rm -f "${flag_file}"')
    kill_group = stop_script.index('kill -- "-${existing_pid}"')
    remove_state = stop_script.index('"${runtime_dir}"/onebot-watchdog-*.json')
    assert remove_flag < kill_group < remove_state
