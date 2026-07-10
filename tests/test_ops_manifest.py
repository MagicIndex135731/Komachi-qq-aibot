from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_desktop_launcher(*, script_name: str) -> Path:
    for path in sorted(REPO_ROOT.glob("*.bat")):
        content = path.read_text(encoding="utf-8")
        if script_name in content:
            return path
    raise AssertionError(f"missing desktop launcher referencing {script_name}")


def test_ops_files_exist() -> None:
    required = [
        "scripts/run_service.ps1",
        "scripts/install_service.ps1",
        "scripts/xiaomachi_watchdog.ps1",
        "start_xiaomachi.ps1",
        "stop_xiaomachi.ps1",
        "start_xiaomachi_bots.ps1",
        "stop_xiaomachi_bots.ps1",
        "start_xiaomachi_runtime.ps1",
        "stop_xiaomachi_runtime.ps1",
        "restart_xiaomachi_runtime.ps1",
        "app/main.py",
        "app/group_main.py",
        "app/private_main.py",
        "app/dev_worker_main.py",
        "README.md",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert not missing, f"missing operations files: {missing}"

    assert _find_desktop_launcher(script_name="start_xiaomachi.ps1")
    assert _find_desktop_launcher(script_name="stop_xiaomachi.ps1")


def test_readme_mentions_search_and_context_limits() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "SEARCH_API_KEY" in readme
    assert "CONTEXT_RECENT_LIMIT" in readme
    assert "只有同时设置 `enabled: true` 和 `speak: true` 的群" in readme
    assert "enabled: true" in readme
    assert "speak: true" in readme


def test_groups_manifest_enables_target_group() -> None:
    groups = yaml.safe_load((REPO_ROOT / "configs/groups.yaml").read_text(encoding="utf-8"))
    primary = groups["groups"]["515267906"]
    limited = groups["groups"]["300520120"]

    assert primary["enabled"] is True
    assert primary["speak"] is True
    assert primary["archive"] is True
    assert primary["proactive_reply"] is True
    assert primary["image_generation"] is True
    assert limited["enabled"] is True
    assert limited["speak"] is True
    assert limited["archive"] is False
    assert limited["proactive_reply"] is False
    assert limited["image_generation"] is False


def test_install_service_script_handles_existing_service() -> None:
    script = (REPO_ROOT / "scripts/install_service.ps1").read_text(encoding="utf-8")

    assert "nssm status" in script
    assert "nssm install" in script
    assert "if (-not $serviceExists)" in script
    assert "nssm set $serviceName Application" in script
    assert "nssm set $serviceName AppParameters" in script
    assert "nssm start" in script


def test_desktop_start_and_stop_scripts_target_the_current_worktree() -> None:
    start_script = _find_desktop_launcher(script_name="start_xiaomachi.ps1").read_text(encoding="utf-8")
    stop_script = _find_desktop_launcher(script_name="stop_xiaomachi.ps1").read_text(encoding="utf-8")
    start_ps1 = (REPO_ROOT / "start_xiaomachi.ps1").read_text(encoding="utf-8")
    stop_ps1 = (REPO_ROOT / "stop_xiaomachi.ps1").read_text(encoding="utf-8")
    start_bots_ps1 = (REPO_ROOT / "start_xiaomachi_bots.ps1").read_text(encoding="utf-8")
    stop_bots_ps1 = (REPO_ROOT / "stop_xiaomachi_bots.ps1").read_text(encoding="utf-8")
    watchdog_ps1 = (REPO_ROOT / "scripts/xiaomachi_watchdog.ps1").read_text(encoding="utf-8")

    assert "%~dp0" in start_script
    assert "%~dp0" in stop_script
    assert "start_xiaomachi.ps1" in start_script
    assert "stop_xiaomachi.ps1" in stop_script
    assert "start_xiaomachi_bots.ps1" in start_ps1
    assert "stop_xiaomachi_bots.ps1" in stop_ps1
    assert "xiaomachi_watchdog.ps1" in start_bots_ps1
    assert "app.group_main" in watchdog_ps1
    assert "app.private_main" in watchdog_ps1
    assert "app.dev_worker_main" in watchdog_ps1
    assert "group.stderr.log" in watchdog_ps1
    assert "private.stderr.log" in watchdog_ps1
    assert "worker.stderr.log" in watchdog_ps1
    assert "group.pid" in watchdog_ps1
    assert "private.pid" in watchdog_ps1
    assert "worker.pid" in watchdog_ps1
    assert "xiaomachi_watchdog.ps1" in stop_bots_ps1
    assert "group.pid" in watchdog_ps1
    assert "private.pid" in watchdog_ps1
    assert "worker.pid" in watchdog_ps1
    assert "Where-Object" in stop_ps1
    assert "Stop-Process" in stop_ps1


def test_runtime_watchdog_uses_longer_heartbeat_timeout_for_slow_text_models() -> None:
    start_bots_ps1 = (REPO_ROOT / "start_xiaomachi_bots.ps1").read_text(encoding="utf-8")

    assert "-Scope runtime" in start_bots_ps1
    assert "-HeartbeatTimeoutSeconds 180" in start_bots_ps1


def test_start_script_orchestrates_napcat_and_qq_before_bots() -> None:
    start_ps1 = (REPO_ROOT / "start_xiaomachi.ps1").read_text(encoding="utf-8")
    runtime_start_ps1 = (REPO_ROOT / "start_xiaomachi_runtime.ps1").read_text(encoding="utf-8")

    assert "NAPCAT_SHELL_DIR" in start_ps1
    assert "QQ_EXE_PATH" in start_ps1
    assert "NapCatWinBootMain.exe" in start_ps1
    assert "launcher_state.json" in start_ps1
    assert "ConvertTo-Json" in start_ps1
    assert "Test-TcpEndpoint" in start_ps1
    assert "qrcode.png" in start_ps1
    assert "Waiting for NapCat websocket" in start_ps1
    assert "Start-Process -FilePath $QrCodePath" in start_ps1
    assert "Remove-Item $paths.QrCodePath" in start_ps1
    assert "Get-Item $QrCodePath" in start_ps1
    assert "start_xiaomachi_bots.ps1" in start_ps1
    assert "xiaomachi_watchdog.ps1" in runtime_start_ps1
    assert "QQ.exe" not in runtime_start_ps1
    assert "NapCatWinBootMain.exe" not in runtime_start_ps1


def test_stop_script_can_close_launcher_managed_stack() -> None:
    stop_ps1 = (REPO_ROOT / "stop_xiaomachi.ps1").read_text(encoding="utf-8")
    runtime_stop_ps1 = (REPO_ROOT / "stop_xiaomachi_runtime.ps1").read_text(encoding="utf-8")

    assert "launcher_state.json" in stop_ps1
    assert "NapCatWinBootMain" in stop_ps1
    assert "QQ.exe" in stop_ps1
    assert "ConvertFrom-Json" in stop_ps1
    assert "QQ.exe" not in runtime_stop_ps1
    assert "NapCatWinBootMain" not in runtime_stop_ps1


def test_readme_mentions_one_click_launcher_dependencies() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "一起拉起 QQ、NapCat 和小町的 Python 进程" in readme
    assert "QQ_EXE_PATH" in readme
    assert "NAPCAT_SHELL_DIR" in readme


def test_pyproject_excludes_public_release_and_tmp_pytest_from_collection() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for expected in ('".pytest_tmp"', '".tmp_pytest"', '".tmp_pytest*"', '"release"', '"scripts/public_release_assets"'):
        assert expected in pyproject


def test_public_release_sync_scripts_resolve_source_root_in_script_body() -> None:
    for relative_path in (
        "scripts/watch_public_release.ps1",
        "scripts/start_public_release_sync.ps1",
        "scripts/stop_public_release_sync.ps1",
    ):
        script = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        source_root_lines = [line for line in script.splitlines() if '[string]$SourceRoot' in line]

        assert '[string]$SourceRoot' in script
        assert source_root_lines
        assert all('(Resolve-Path (Join-Path $PSScriptRoot "..")).Path' not in line for line in source_root_lines)
        assert 'if (-not $SourceRoot)' in script


def test_watch_public_release_script_avoids_powershell_5_only_missing_getrelativepath() -> None:
    script = (REPO_ROOT / "scripts/watch_public_release.ps1").read_text(encoding="utf-8")

    assert "GetRelativePath" not in script
