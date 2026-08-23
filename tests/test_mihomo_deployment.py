from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "infra/wsl/scripts/render_mihomo_config.py"
SPEC = importlib.util.spec_from_file_location("render_mihomo_config", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source_config() -> dict[str, object]:
    return {
        "mixed-port": 7897,
        "allow-lan": True,
        "tun": {"enable": True},
        "proxies": [
            {"name": "🇭🇰 香港Y01", "type": "ss", "server": "example.invalid"},
            {"name": "🇯🇵 日本Y01", "type": "ss", "server": "example.invalid"},
        ],
        "proxy-groups": [{"name": "选择节点", "type": "select", "proxies": ["🇭🇰 香港Y01"]}],
        "rules": ["DOMAIN-SUFFIX,qq.com,选择节点", "MATCH,选择节点"],
    }


def test_rendered_profile_routes_nova_to_hong_kong_and_qq_direct() -> None:
    rendered = MODULE.render_config(source_config())

    assert rendered["mode"] == "rule"
    assert rendered["mixed-port"] == 7897
    assert rendered["bind-address"] == "127.0.0.1"
    assert rendered["allow-lan"] is False
    assert rendered["tun"]["enable"] is False
    group = rendered["proxy-groups"][0]
    assert group["name"] == MODULE.NOVA_GROUP
    assert group["type"] == "url-test"
    assert group["proxies"] == ["🇭🇰 香港Y01"]
    assert rendered["rules"][0] == f"DOMAIN,{MODULE.NOVA_HOST},{MODULE.NOVA_GROUP}"
    assert rendered["rules"].count("DOMAIN-SUFFIX,qq.com,DIRECT") == 1
    assert "DOMAIN-SUFFIX,qq.com,选择节点" not in rendered["rules"]
    assert rendered["rules"][-1] == "MATCH,选择节点"


def test_rendered_profile_fails_closed_without_hong_kong_nodes() -> None:
    source = source_config()
    source["proxies"] = [{"name": "日本Y01", "type": "ss"}]

    with pytest.raises(ValueError, match="no Hong Kong"):
        MODULE.render_config(source)


def test_mihomo_service_is_local_only_and_precedes_stack() -> None:
    unit = (REPO_ROOT / "infra/wsl/systemd/xiaomachi-mihomo.service").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "infra/wsl/scripts/install_mihomo.sh").read_text(encoding="utf-8")
    release_installer = (REPO_ROOT / "infra/wsl/scripts/install_linux_runtime.sh").read_text(encoding="utf-8")
    sync = (REPO_ROOT / "infra/wsl/scripts/sync_mihomo_from_clash_verge.ps1").read_text(encoding="utf-8")
    stack = (REPO_ROOT / "infra/wsl/systemd/xiaomachi-stack.service").read_text(encoding="utf-8")
    start = (REPO_ROOT / "infra/wsl/scripts/start.sh").read_text(encoding="utf-8")
    status = (REPO_ROOT / "infra/wsl/scripts/status.sh").read_text(encoding="utf-8")

    assert "Before=xiaomachi-stack.service" in unit
    assert "ExecStartPost=" in unit
    assert "/dev/tcp/127.0.0.1/7897" in unit
    assert "Restart=on-failure" in unit
    assert "ProtectSystem=strict" in unit
    assert "MIHOMO_SHA256=" in installer
    assert "sha256sum -c -" in installer
    assert 'CONFIG_CANDIDATE="${CONFIG_PATH}.next"' in installer
    assert '"${tester}" -t' in installer
    assert 'mv -f "${CONFIG_CANDIDATE}" "${CONFIG_PATH}"' in installer
    assert "systemctl restart xiaomachi-mihomo.service" in installer
    assert "clash-verge.yaml" in sync
    assert "config.yaml.next" in sync
    assert '"Country.mmdb" = "Country.mmdb"' in sync
    assert '"geosite.dat" = "GeoSite.dat"' in sync
    assert "ConvertTo-WslMountPath" in sync
    assert "Remove-Item -LiteralPath $rendered" in sync
    assert "xiaomachi-mihomo.service" in release_installer
    assert 'mihomo -t -d "${shared_dir}/mihomo"' in release_installer
    assert "Wants=network-online.target xiaomachi-mihomo.service" in stack
    assert "After=network-online.target docker.service xiaomachi-mihomo.service" in stack
    assert 'xiaomachi_proxy}' in start
    assert "systemctl is-active --quiet xiaomachi-mihomo.service" in start
    assert "Mihomo provider proxy probe:" in status
    assert "https://ai.novacode.top/" in status


def test_linux_runtime_exports_force_lf_line_endings() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "*.sh text eol=lf" in attributes
    assert "*.service text eol=lf" in attributes


def test_windows_runtime_task_owns_a_persistent_wsl_anchor() -> None:
    script = (REPO_ROOT / "infra/wsl/scripts/install_windows_runtime_task.ps1").read_text(
        encoding="utf-8"
    )

    assert 'ValidateSet("Install", "Remove")' in script
    assert 'New-ScheduledTaskTrigger -AtLogOn' in script
    assert "-RunLevel Limited" in script
    assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in script
    assert "-MultipleInstances IgnoreNew" in script
    runner = (REPO_ROOT / "infra/wsl/scripts/run_windows_runtime_task.vbs").read_text(
        encoding="utf-8"
    )
    assert "run_windows_runtime_task.vbs" in script
    assert "System32\\wscript.exe" in script
    assert "//B //Nologo" in script
    assert "WScript.Shell" in runner
    assert "xiaomachi-wsl-entry anchor" in runner
    assert "wsl-runtime-task.log" in runner
    assert "shell.Run(command, 0, True)" in runner
    assert 'ExpandEnvironmentStrings("%ComSpec%")' not in runner
    assert "wscript-direct" in runner
    assert "Do" in runner
    assert "Loop" in runner
    assert "anchor_stopped exit_code=" in runner
    assert 'DateDiff("s", startedAt, Now)' in runner
    assert "quick_exit_count=" in runner
    assert "restartDelayMilliseconds = 15000" in runner
    assert "WScript.Sleep restartDelayMilliseconds" in runner
    assert "/bin/bash -lc" not in script
    assert "Start-ScheduledTask -TaskName $TaskName" in script
