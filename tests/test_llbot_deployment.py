from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WSL_DIR = REPO_ROOT / "infra/wsl"
SCRIPTS_DIR = WSL_DIR / "scripts"


def read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def test_llbot_compose_uses_pinned_image_host_network_and_persistent_data() -> None:
    compose_path = WSL_DIR / "docker-compose.llbot.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    llbot = compose["services"]["llbot"]

    assert llbot["image"] == "linyuchen/llbot:8.0.14"
    assert llbot["restart"] == "unless-stopped"
    assert llbot["network_mode"] == "host"
    assert "ports" not in llbot
    assert "llbot_data:/app/llbot/data" in llbot["volumes"]
    assert "xiaomachi_data:/workspace/data:ro" in llbot["volumes"]
    assert "HTTP_PROXY=${DOCKER_HTTP_PROXY:-}" in llbot["environment"]
    assert "HTTPS_PROXY=${DOCKER_HTTPS_PROXY:-}" in llbot["environment"]
    assert "healthcheck" not in llbot


def test_llbot_compose_keeps_xiaomachi_business_mounts_and_uses_onebot() -> None:
    compose = yaml.safe_load(
        (WSL_DIR / "docker-compose.llbot.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    xiaomachi = services["xiaomachi"]

    assert xiaomachi["image"] == "xiaomachi-bot:local"
    assert xiaomachi["build"]["context"] == "../.."
    assert xiaomachi["build"]["dockerfile"] == "infra/wsl/Dockerfile.xiaomachi"
    assert xiaomachi["build"]["network"] == "host"
    assert xiaomachi["build"]["args"]["HTTP_PROXY"] == "${DOCKER_HTTP_PROXY:-}"
    assert xiaomachi["build"]["args"]["HTTPS_PROXY"] == "${DOCKER_HTTPS_PROXY:-}"
    assert "../../:/workspace" not in xiaomachi["volumes"]
    assert "xiaomachi_data:/workspace/data" in xiaomachi["volumes"]
    assert all("/mnt/" not in volume for volume in xiaomachi["volumes"])
    assert "./runtime/pip-cache:/root/.cache/pip" not in xiaomachi["volumes"]
    assert xiaomachi["command"] == ["python", "-m", "app.group_main"]
    assert "NAPCAT_WS_URL=ws://127.0.0.1:${LLBOT_WS_PORT:-3002}" in xiaomachi[
        "environment"
    ]
    assert xiaomachi["depends_on"]["llbot"]["condition"] == "service_started"
    assert compose["volumes"]["xiaomachi_data"]["external"] is True
    assert compose["volumes"]["xiaomachi_data"]["name"] == "xiaomachi-bot-data"
    assert compose["volumes"]["llbot_data"]["external"] is True
    assert compose["volumes"]["llbot_data"]["name"] == "xiaomachi-llbot-data"


def test_llbot_data_volume_migration_stops_writer_copies_and_checks_database() -> None:
    script = read_script("migrate_xiaomachi_data_volume.sh")

    assert "xiaomachi-bot-data" in script
    assert 'docker compose -f "${compose_file}" stop xiaomachi' in script
    assert '"${repo_root}/data:/source:ro"' in script
    assert "cp -a /source/. /target/" in script
    assert "PRAGMA integrity_check" in script
    assert ".migration-complete" in script


def test_llbot_runtime_bootstrap_configures_onebot_and_webui() -> None:
    script = read_script("bootstrap_llbot_runtime.py")

    assert '"ob11"' in script
    assert '"type": "ws"' in script
    assert '"port": onebot_port' in script
    assert 'env.get("LLBOT_WS_PORT", "3002")' in script
    assert "connections = payload.get" in script
    assert '"webui"' in script
    assert '"port": 3080' in script
    assert "webui_token.txt" in script
    assert 'parser.add_argument("--data-dir", type=Path)' in script
    assert 'parser.add_argument("--env-file", type=Path)' in script


def test_llbot_runtime_bootstrap_migrates_existing_onebot_port(
    tmp_path: Path, monkeypatch
) -> None:
    wsl_dir = tmp_path
    (wsl_dir / ".env").write_text("BOT_QQ=123456\n", encoding="utf-8")
    data_dir = wsl_dir / "runtime/llbot/data"
    data_dir.mkdir(parents=True)
    config_path = data_dir / "config_123456.json"
    config_path.write_text(
        json.dumps(
            {
                "ob11": {"connect": [{"type": "ws", "enable": True, "port": 3001}]},
                "webui": {"enable": True, "port": 3087},
            }
        ),
        encoding="utf-8",
    )
    module_path = SCRIPTS_DIR / "bootstrap_llbot_runtime.py"
    spec = importlib.util.spec_from_file_location("bootstrap_llbot_runtime", module_path)
    assert spec and spec.loader
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(sys, "argv", [str(module_path), "--wsl-dir", str(wsl_dir)])

    assert bootstrap.main() == 0

    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["ob11"]["connect"][0]["port"] == 3002
    assert migrated["webui"]["port"] == 3080
    assert migrated["webui"]["host"] == ""
    assert migrated["webui"]["enable"] is True


def test_open_llbot_webui_shortcut_checks_local_webui_without_starting_stack() -> None:
    shortcut = (REPO_ROOT / "open-llbot-webui.bat").read_text(encoding="utf-8")
    launcher = read_script("open_llbot_webui.ps1")

    assert shortcut.isascii()
    assert "curl.exe" in shortcut
    assert "http://127.0.0.1:3080/" in shortcut
    assert "open_llbot_webui.ps1" in shortcut
    assert "start-xiaomachi-wsl.bat" in shortcut
    assert "pause" in shortcut
    assert "wsl.exe" not in shortcut.lower()
    assert "docker" not in shortcut.lower()
    assert '"http://127.0.0.1:3080/"' in launcher
    assert "webui_token.txt" in launcher
    assert "Could not copy the LLBot WebUI password" in launcher
    assert "Start-Process" in launcher


def test_start_script_selects_llbot_compose_and_preserves_napcat_default() -> None:
    script = read_script("start.sh")

    assert "QQ_PLATFORM" in script
    assert 'platform="${platform:-napcat}"' in script
    assert '"${platform}" != "napcat" && "${platform}" != "llbot"' in script
    assert 'compose_file="docker-compose.llbot.yml"' in script
    assert 'other_compose_file="docker-compose.yml"' in script
    assert 'compose_file="docker-compose.yml"' in script
    assert 'other_compose_file="docker-compose.llbot.yml"' in script
    assert 'launcher="open_llbot_webui.ps1"' in script
    assert 'launcher="open_napcat_webui.ps1"' in script
    assert 'service_name="llbot"' in script
    assert 'service_name="napcat"' in script
    assert 'docker compose -f "${compose_file}" build xiaomachi' in script
    assert 'migrate_xiaomachi_data_volume.sh' in script
    assert "cleanup_failed_start" in script
    assert "pkill -f xiaomachi-wsl-keepalive" in script
    assert 'flock -n 8' in script
    assert "startup is already in progress" in script
    assert 'up -d "${service_name}"' in script
    assert 'up -d --no-deps xiaomachi' in script
    assert script.index('compose_file="docker-compose.llbot.yml"') < script.index(
        'compose_file="docker-compose.yml"'
    )


def test_stop_status_keepalive_and_watchdog_are_platform_aware() -> None:
    stop_script = read_script("stop.sh")
    status_script = read_script("status.sh")
    keepalive_script = read_script("keepalive.sh")
    watchdog = read_script("onebot_watchdog.py")

    assert 'docker compose -f docker-compose.yml down --remove-orphans' in stop_script
    assert 'docker compose -f docker-compose.llbot.yml down --remove-orphans' in stop_script
    assert 'onebot-watchdog-*.json' in stop_script

    assert "QQ_PLATFORM" in status_script
    assert 'compose_file="docker-compose.llbot.yml"' in status_script
    assert 'service_name="llbot"' in status_script
    assert 'container_name="xiaomachi-llbot"' in status_script
    assert 'compose_file="docker-compose.yml"' in status_script
    assert 'service_name="napcat"' in status_script

    assert 'platform="${3:-napcat}"' in keepalive_script
    assert 'compose_file="${wsl_dir}/docker-compose.llbot.yml"' in keepalive_script
    assert 'service_name="llbot"' in keepalive_script
    assert 'onebot_ws_url="ws://127.0.0.1:${llbot_ws_port}"' in keepalive_script
    assert 'compose_file="${wsl_dir}/docker-compose.yml"' in keepalive_script
    assert 'service_name="napcat"' in keepalive_script
    assert 'onebot-watchdog-${platform}.json' in keepalive_script
    assert "--daemon" in keepalive_script
    assert "--once" not in keepalive_script

    assert 'choices=("napcat", "llbot")' in watchdog
    assert 'service_name: str = "napcat"' in watchdog
    assert 'platform: str = "napcat"' in watchdog
    assert "restart_service(compose_file, service_name)" in watchdog
