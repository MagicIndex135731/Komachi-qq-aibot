from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_supported_wsl_operations_files_exist() -> None:
    required = [
        "start-xiaomachi-wsl.bat",
        "stop-xiaomachi-wsl.bat",
        "status-xiaomachi-wsl.bat",
        "open-napcat-webui.bat",
        "open-llbot-webui.bat",
        "infra/wsl/docker-compose.yml",
        "infra/wsl/docker-compose.llbot.yml",
        "infra/wsl/scripts/start.sh",
        "infra/wsl/scripts/stop.sh",
        "infra/wsl/scripts/status.sh",
        "infra/wsl/scripts/keepalive.sh",
        "infra/wsl/scripts/onebot_watchdog.py",
        "app/group_main.py",
        "README.md",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert not missing, f"missing WSL operations files: {missing}"


def test_readme_documents_only_supported_wsl_launchers() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for launcher in (
        "start-xiaomachi-wsl.bat",
        "stop-xiaomachi-wsl.bat",
        "status-xiaomachi-wsl.bat",
        "open-napcat-webui.bat",
    ):
        assert launcher in readme
    assert "infra/wsl/.env" in readme
    assert "SEARCH_API_KEY" in readme
    assert "CONTEXT_RECENT_LIMIT" in readme
    assert "start_xiaomachi.ps1" not in readme
    assert "NAPCAT_SHELL_DIR" not in readme
    assert "QQ_EXE_PATH" not in readme


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


def test_pyproject_excludes_generated_release_and_test_artifacts() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for expected in (
        '".pytest_tmp"',
        '".tmp_pytest"',
        '".tmp_pytest*"',
        '"release"',
        '"scripts/public_release_assets"',
    ):
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


def test_watch_public_release_script_supports_windows_powershell_5() -> None:
    script = (REPO_ROOT / "scripts/watch_public_release.ps1").read_text(encoding="utf-8")

    assert "GetRelativePath" not in script
