from __future__ import annotations

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

    assert llbot["image"] == "linyuchen/llbot:8.0.8"
    assert llbot["network_mode"] == "host"
    assert "ports" not in llbot
    assert "./runtime/llbot/data:/app/llbot/data" in llbot["volumes"]
    assert "HTTP_PROXY=${DOCKER_HTTP_PROXY:-}" in llbot["environment"]
    assert "HTTPS_PROXY=${DOCKER_HTTPS_PROXY:-}" in llbot["environment"]


def test_llbot_compose_keeps_xiaomachi_business_mounts_and_uses_onebot() -> None:
    compose = yaml.safe_load(
        (WSL_DIR / "docker-compose.llbot.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    xiaomachi = services["xiaomachi"]

    assert "../../:/workspace" in xiaomachi["volumes"]
    assert "./runtime/logs:/workspace/data/logs" in xiaomachi["volumes"]
    assert "./runtime/cache:/workspace/data/cache" in xiaomachi["volumes"]
    assert "./runtime/pip-cache:/root/.cache/pip" in xiaomachi["volumes"]
    assert "NAPCAT_WS_URL=ws://127.0.0.1:${LLBOT_WS_PORT:-3001}" in xiaomachi[
        "environment"
    ]
    assert xiaomachi["depends_on"]["llbot"]["condition"] == "service_healthy"


def test_llbot_runtime_bootstrap_configures_onebot_and_webui() -> None:
    script = read_script("bootstrap_llbot_runtime.py")

    assert '"ob11"' in script
    assert '"type": "ws"' in script
    assert '"port": 3001' in script
    assert '"webui"' in script
    assert '"port": 3080' in script
    assert "webui_token.txt" in script


def test_open_llbot_webui_shortcut_checks_local_webui_without_starting_stack() -> None:
    shortcut = (REPO_ROOT / "open-llbot-webui.bat").read_text(encoding="utf-8")
    launcher = read_script("open_llbot_webui.ps1")

    assert shortcut.isascii()
    assert "curl.exe" in shortcut
    assert "http://127.0.0.1:3080/" in shortcut
    assert "open_llbot_webui.ps1" in shortcut
    assert "start-xiaomachi-wsl.bat" in shortcut
    assert "wsl.exe" not in shortcut.lower()
    assert "docker" not in shortcut.lower()
    assert '"http://127.0.0.1:3080/"' in launcher
    assert "webui_token.txt" in launcher
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
    assert 'compose_file="${wsl_dir}/docker-compose.yml"' in keepalive_script
    assert 'service_name="napcat"' in keepalive_script
    assert 'onebot-watchdog-${platform}.json' in keepalive_script

    assert 'choices=("napcat", "llbot")' in watchdog
    assert 'service_name: str = "napcat"' in watchdog
    assert 'platform: str = "napcat"' in watchdog
    assert "restart_service(compose_file, service_name)" in watchdog
