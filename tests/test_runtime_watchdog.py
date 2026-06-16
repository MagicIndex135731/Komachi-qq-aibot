from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "scripts" / "xiaomachi_process_helpers.ps1"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_powershell_json(script: str, *, cwd: Path) -> dict:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _run_powershell(script: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _read_status(*, module_name: str, pid_file: Path, heartbeat_file: Path, timeout_seconds: int, grace_seconds: int) -> dict:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $spec = @{{
            Name = 'test'
            Module = {_ps_quote(module_name)}
            PidFile = {_ps_quote(str(pid_file))}
            HeartbeatFile = {_ps_quote(str(heartbeat_file))}
        }}
        $status = Get-BotSpecStatus -Spec $spec -HeartbeatTimeoutSeconds {timeout_seconds} -StartupGraceSeconds {grace_seconds}
        $status | ConvertTo-Json -Compress
        """
    ).strip()
    return _run_powershell_json(script, cwd=REPO_ROOT)


def _write_heartbeat(path: Path, *, pid: int, updated_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "updated_at": updated_at.astimezone(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def test_watchdog_script_parses_in_powershell() -> None:
    watchdog_path = REPO_ROOT / "scripts" / "xiaomachi_watchdog.ps1"
    result = _run_powershell(
        f"[void][scriptblock]::Create((Get-Content -Raw {_ps_quote(str(watchdog_path))}))",
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_get_bot_spec_status_marks_missing_process_as_stale_pid(tmp_path) -> None:
    pid_file = tmp_path / "group.pid"
    heartbeat_file = tmp_path / "group.heartbeat.json"
    pid_file.write_text("999999", encoding="ascii")

    status = _read_status(
        module_name="fakeapp.group_main",
        pid_file=pid_file,
        heartbeat_file=heartbeat_file,
        timeout_seconds=30,
        grace_seconds=5,
    )

    assert status["is_running"] is False
    assert status["needs_restart"] is True
    assert status["restart_reason"] == "process_missing"
    assert status["pid_file_stale"] is True


def test_get_bot_spec_status_marks_running_process_with_stale_heartbeat_for_restart(tmp_path) -> None:
    package_name = f"fakeapp_group_{tmp_path.name.replace('-', '_')}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "group_main.py").write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    process = subprocess.Popen([sys.executable, "-m", f"{package_name}.group_main"], cwd=tmp_path)
    try:
        time.sleep(1.0)
        assert process.poll() is None

        pid_file = tmp_path / "group.pid"
        heartbeat_file = tmp_path / "group.heartbeat.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC) - timedelta(seconds=120),
        )

        status = _read_status(
            module_name=f"{package_name}.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            timeout_seconds=30,
            grace_seconds=1,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is True
        assert status["restart_reason"] == "stale_heartbeat"
        assert status["pid_file_stale"] is False
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_get_bot_spec_status_allows_recent_startup_before_first_heartbeat(tmp_path) -> None:
    package_name = f"fakeapp_private_{tmp_path.name.replace('-', '_')}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "private_main.py").write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    process = subprocess.Popen([sys.executable, "-m", f"{package_name}.private_main"], cwd=tmp_path)
    try:
        time.sleep(1.0)
        assert process.poll() is None

        pid_file = tmp_path / "private.pid"
        heartbeat_file = tmp_path / "private.heartbeat.json"
        pid_file.write_text(str(process.pid), encoding="ascii")

        status = _read_status(
            module_name=f"{package_name}.private_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            timeout_seconds=30,
            grace_seconds=120,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is False
        assert status["restart_reason"] == ""
        assert status["pid_file_stale"] is False
    finally:
        process.terminate()
        process.wait(timeout=10)
