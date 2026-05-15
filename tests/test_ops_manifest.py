from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_desktop_launcher(*, script_name: str) -> Path:
    for path in sorted(REPO_ROOT.glob("*.bat")):
        content = path.read_text(encoding="utf-8")
        if script_name in content:
            return path
    raise AssertionError(f"missing desktop launcher referencing {script_name}")


def test_ops_files_exist() -> None:
    required = [
        ".env.example",
        "LICENSE",
        "scripts/run_service.ps1",
        "scripts/install_service.ps1",
        "start_xiaomachi.ps1",
        "stop_xiaomachi.ps1",
        "start_xiaomachi_bots.ps1",
        "stop_xiaomachi_bots.ps1",
        "start_xiaomachi_runtime.ps1",
        "stop_xiaomachi_runtime.ps1",
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
    assert "only groups with both `enabled: true` and `speak: true` are ingested" in readme
    assert "enabled: true" in readme
    assert "speak: true" in readme


def test_groups_manifest_enables_target_group() -> None:
    groups = (REPO_ROOT / "configs/groups.yaml").read_text(encoding="utf-8")

    assert "10001" in groups
    assert "enabled: true" in groups
    assert "speak: true" in groups


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

    assert "%~dp0" in start_script
    assert "%~dp0" in stop_script
    assert "start_xiaomachi.ps1" in start_script
    assert "stop_xiaomachi.ps1" in stop_script
    assert "start_xiaomachi_bots.ps1" in start_ps1
    assert "stop_xiaomachi_bots.ps1" in stop_ps1
    assert "app.group_main" in start_bots_ps1
    assert "app.private_main" in start_bots_ps1
    assert "app.dev_worker_main" in start_bots_ps1
    assert "group.stderr.log" in start_bots_ps1
    assert "private.stderr.log" in start_bots_ps1
    assert "worker.stderr.log" in start_bots_ps1
    assert "group.pid" in start_bots_ps1
    assert "private.pid" in start_bots_ps1
    assert "worker.pid" in start_bots_ps1
    assert "group.pid" in stop_bots_ps1
    assert "private.pid" in stop_bots_ps1
    assert "worker.pid" in stop_bots_ps1
    assert "Where-Object" in stop_ps1
    assert "Stop-Process" in stop_ps1


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
    assert "app.group_main" in runtime_start_ps1
    assert "app.private_main" in runtime_start_ps1
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

    assert "starts QQ, NapCat, and the Python bot together" in readme
    assert "QQ_EXE_PATH" in readme
    assert "NAPCAT_SHELL_DIR" in readme


def test_public_release_has_license_env_example_and_reminder_state_ignore() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "LLM_MODEL" in env_example
    assert "OWNER_QQ" in env_example
    assert "data/private_reminders_state.json" in gitignore
    assert "data/generated_images/" in gitignore
    assert "data/generated_private_images/" in gitignore
    assert ".tmp_pytest*/" in gitignore
    assert "dbg_service_*/" in gitignore
    assert "MIT License" in license_text
