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


def _read_status_with_onebot_health(
    *,
    module_name: str,
    pid_file: Path,
    heartbeat_file: Path,
    onebot_health_file: Path,
    timeout_seconds: int,
    grace_seconds: int,
    offline_threshold: int,
    onebot_probe_grace_seconds: int = 0,
) -> dict:
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
            OneBotHealthFile = {_ps_quote(str(onebot_health_file))}
            OneBotOfflineRestartThreshold = {offline_threshold}
            OneBotProbeStartupGraceSeconds = {onebot_probe_grace_seconds}
        }}
        $status = Get-BotSpecStatus -Spec $spec -HeartbeatTimeoutSeconds {timeout_seconds} -StartupGraceSeconds {grace_seconds}
        $status | ConvertTo-Json -Compress
        """
    ).strip()
    return _run_powershell_json(script, cwd=REPO_ROOT)


def _read_status_with_group_stream_health(
    *,
    module_name: str,
    pid_file: Path,
    heartbeat_file: Path,
    stream_health_file: Path,
    timeout_seconds: int,
    grace_seconds: int,
    stale_threshold: int,
    onebot_probe_grace_seconds: int = 0,
) -> dict:
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
            OneBotGroupStreamHealthFile = {_ps_quote(str(stream_health_file))}
            OneBotGroupStreamStaleRestartThreshold = {stale_threshold}
            OneBotProbeStartupGraceSeconds = {onebot_probe_grace_seconds}
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


def _write_onebot_health(path: Path, *, online: bool, offline_count: int, updated_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "online": online,
                "offline_count": offline_count,
                "updated_at": updated_at.astimezone(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _write_group_stream_health(path: Path, *, stale: bool, stale_count: int, updated_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "stale": stale,
                "stale_count": stale_count,
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


def test_start_bot_spec_repairs_start_process_environment(tmp_path) -> None:
    package_name = f"fakeapp_start_env_{tmp_path.name.replace('-', '_')}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "group_main.py").write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    pid_file = tmp_path / "group.pid"
    stdout_file = tmp_path / "group.stdout.log"
    stderr_file = tmp_path / "group.stderr.log"
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $spec = @{{
            Name = 'test'
            Module = {_ps_quote(f'{package_name}.group_main')}
            PidFile = {_ps_quote(str(pid_file))}
            Stdout = {_ps_quote(str(stdout_file))}
            Stderr = {_ps_quote(str(stderr_file))}
        }}
        try {{
            Start-BotSpec -Workdir {_ps_quote(str(tmp_path))} -PythonExe {_ps_quote(sys.executable)} -Spec $spec
            $pidValue = Get-Content -LiteralPath $spec.PidFile -ErrorAction Stop | Select-Object -First 1
            [pscustomobject]@{{
                started = $true
                pid = [int]$pidValue
            }} | ConvertTo-Json -Compress
        }} finally {{
            Stop-BotSpec -Spec $spec
        }}
        """
    ).strip()

    result = _run_powershell(script, cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["started"] is True
    assert payload["pid"] > 0


def test_start_bot_spec_waits_for_matching_heartbeat_before_returning(tmp_path) -> None:
    package_name = f"fakeapp_start_heartbeat_{tmp_path.name.replace('-', '_')}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    heartbeat_file = tmp_path / "group.heartbeat.json"
    heartbeat_literal = str(heartbeat_file).replace("\\", "\\\\")
    (package_dir / "group_main.py").write_text(
        "import json\n"
        "import os\n"
        "import time\n"
        "from datetime import UTC, datetime\n"
        "from pathlib import Path\n"
        "time.sleep(4)\n"
        f"Path(r'{heartbeat_literal}').write_text(json.dumps({{'pid': os.getpid(), 'updated_at': datetime.now(UTC).isoformat()}}), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    pid_file = tmp_path / "group.pid"
    stdout_file = tmp_path / "group.stdout.log"
    stderr_file = tmp_path / "group.stderr.log"
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $spec = @{{
            Name = 'test'
            Module = {_ps_quote(f'{package_name}.group_main')}
            PidFile = {_ps_quote(str(pid_file))}
            Stdout = {_ps_quote(str(stdout_file))}
            Stderr = {_ps_quote(str(stderr_file))}
            HeartbeatFile = {_ps_quote(str(heartbeat_file))}
        }}
        try {{
            $startedAt = Get-Date
            Start-BotSpec -Workdir {_ps_quote(str(tmp_path))} -PythonExe {_ps_quote(sys.executable)} -Spec $spec
            $elapsedSeconds = ((Get-Date) - $startedAt).TotalSeconds
            $heartbeat = Get-Content -Raw -LiteralPath $spec.HeartbeatFile | ConvertFrom-Json
            [pscustomobject]@{{
                elapsed_seconds = $elapsedSeconds
                heartbeat_pid = [int]$heartbeat.pid
                pid_file = [int](Get-Content -LiteralPath $spec.PidFile | Select-Object -First 1)
            }} | ConvertTo-Json -Compress
        }} finally {{
            Stop-BotSpec -Spec $spec
        }}
        """
    ).strip()

    result = _run_powershell(script, cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["elapsed_seconds"] >= 3.5
    assert payload["heartbeat_pid"] == payload["pid_file"]


def test_stop_bot_spec_keeps_tracking_files_when_process_survives_stop(tmp_path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    try:
        time.sleep(1.0)
        assert process.poll() is None

        pid_file = tmp_path / "group.pid"
        heartbeat_file = tmp_path / "group.heartbeat.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )

        script = textwrap.dedent(
            f"""
            [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
            $OutputEncoding = [Console]::OutputEncoding
            . {_ps_quote(str(HELPERS_PATH))}
            function Stop-Process {{ param([int]$Id, [switch]$Force, [object]$ErrorAction) }}
            $spec = @{{
                Name = 'test'
                Module = 'fakeapp.group_main'
                PidFile = {_ps_quote(str(pid_file))}
                HeartbeatFile = {_ps_quote(str(heartbeat_file))}
            }}
            $result = Stop-BotSpec -Spec $spec -PassThru
            [pscustomobject]@{{
                stopped = [bool]$result.stopped
                remaining = @($result.remaining_ids).Count
                pid_file_exists = Test-Path -LiteralPath $spec.PidFile
                heartbeat_exists = Test-Path -LiteralPath $spec.HeartbeatFile
            }} | ConvertTo-Json -Compress
            """
        ).strip()

        result = _run_powershell_json(script, cwd=REPO_ROOT)

        assert result["stopped"] is False
        assert result["remaining"] == 1
        assert result["pid_file_exists"] is True
        assert result["heartbeat_exists"] is True
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_restart_bot_spec_safely_reports_start_failure_without_throwing(tmp_path) -> None:
    package_name = f"fakeapp_restart_fail_{tmp_path.name.replace('-', '_')}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "group_main.py").write_text("raise SystemExit(3)\n", encoding="utf-8")

    pid_file = tmp_path / "group.pid"
    stdout_file = tmp_path / "group.stdout.log"
    stderr_file = tmp_path / "group.stderr.log"
    heartbeat_file = tmp_path / "group.heartbeat.json"
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $spec = @{{
            Name = 'test'
            Module = {_ps_quote(f'{package_name}.group_main')}
            PidFile = {_ps_quote(str(pid_file))}
            Stdout = {_ps_quote(str(stdout_file))}
            Stderr = {_ps_quote(str(stderr_file))}
            HeartbeatFile = {_ps_quote(str(heartbeat_file))}
        }}
        Restart-BotSpecSafely -Workdir {_ps_quote(str(tmp_path))} -PythonExe {_ps_quote(sys.executable)} -Spec $spec | ConvertTo-Json -Compress
        """
    ).strip()

    result = _run_powershell(script, cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["restarted"] is False
    assert "failed to start" in payload["error"]


def test_restart_budget_suppresses_repeated_restarts_inside_window() -> None:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $state = @{{}}
        $first = Test-BotSpecRestartBudget -StateByName $state -Name 'group' -MaxAttempts 2 -WindowSeconds 600 -SuppressSeconds 300
        Register-BotSpecRestartAttempt -StateByName $state -Name 'group'
        $second = Test-BotSpecRestartBudget -StateByName $state -Name 'group' -MaxAttempts 2 -WindowSeconds 600 -SuppressSeconds 300
        Register-BotSpecRestartAttempt -StateByName $state -Name 'group'
        $third = Test-BotSpecRestartBudget -StateByName $state -Name 'group' -MaxAttempts 2 -WindowSeconds 600 -SuppressSeconds 300
        [pscustomobject]@{{
            first_allowed = [bool]$first.allowed
            second_allowed = [bool]$second.allowed
            third_allowed = [bool]$third.allowed
            third_attempts = [int]$third.attempts
            third_suppressed_seconds = [int]$third.suppressed_seconds
        }} | ConvertTo-Json -Compress
        """
    ).strip()

    result = _run_powershell_json(script, cwd=REPO_ROOT)

    assert result == {
        "first_allowed": True,
        "second_allowed": True,
        "third_allowed": False,
        "third_attempts": 2,
        "third_suppressed_seconds": 300,
    }


def test_watchdog_probe_interval_state_tracks_each_spec_independently() -> None:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $state = @{{}}
        $now = [datetimeoffset]::FromUnixTimeSeconds(1000).UtcDateTime
        $firstGroup = Test-WatchdogProbeDue -LastProbeByName $state -Name 'group' -IntervalSeconds 15 -NowUtc $now
        Register-WatchdogProbeRun -LastProbeByName $state -Name 'group' -NowUtc $now
        $earlyGroup = Test-WatchdogProbeDue -LastProbeByName $state -Name 'group' -IntervalSeconds 15 -NowUtc ($now.AddSeconds(14))
        $dueGroup = Test-WatchdogProbeDue -LastProbeByName $state -Name 'group' -IntervalSeconds 15 -NowUtc ($now.AddSeconds(15))
        $firstPrivate = Test-WatchdogProbeDue -LastProbeByName $state -Name 'private' -IntervalSeconds 15 -NowUtc ($now.AddSeconds(14))
        [pscustomobject]@{{
            first_group = [bool]$firstGroup
            early_group = [bool]$earlyGroup
            due_group = [bool]$dueGroup
            first_private = [bool]$firstPrivate
        }} | ConvertTo-Json -Compress
        """
    ).strip()

    result = _run_powershell_json(script, cwd=REPO_ROOT)

    assert result == {
        "first_group": True,
        "early_group": False,
        "due_group": True,
        "first_private": True,
    }


def test_onebot_status_payload_online_requires_account_online() -> None:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $onlinePayload = '{{"status":"ok","retcode":0,"data":{{"online":true,"good":true}}}}' | ConvertFrom-Json
        $degradedPayload = '{{"status":"ok","retcode":0,"data":{{"online":false,"good":true}}}}' | ConvertFrom-Json
        $offlinePayload = '{{"status":"ok","retcode":0,"data":{{"online":false,"good":false}}}}' | ConvertFrom-Json
        @{{
            online = Test-OneBotStatusPayloadOnline $onlinePayload
            degraded = Test-OneBotStatusPayloadOnline $degradedPayload
            offline = Test-OneBotStatusPayloadOnline $offlinePayload
        }} | ConvertTo-Json -Compress
        """
    ).strip()

    result = _run_powershell_json(script, cwd=REPO_ROOT)

    assert result == {"online": True, "degraded": False, "offline": False}


def test_onebot_python_probes_emit_ascii_json_for_windows_stdout() -> None:
    helpers = HELPERS_PATH.read_text(encoding="utf-8")

    assert "ensure_ascii=False" not in helpers


def test_group_history_payload_staleness_uses_latest_message_time() -> None:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $now = [datetimeoffset]::FromUnixTimeSeconds(2000).UtcDateTime
        $freshPayload = '{{"status":"ok","retcode":0,"data":{{"messages":[{{"time":1500}},{{"time":1980}}]}}}}' | ConvertFrom-Json
        $stalePayload = '{{"status":"ok","retcode":0,"data":{{"messages":[{{"time":1000}},{{"time":1100}}]}}}}' | ConvertFrom-Json
        @{{
            fresh = Test-OneBotGroupHistoryPayloadFresh -Payload $freshPayload -MaxLagSeconds 60 -NowUtc $now
            stale = Test-OneBotGroupHistoryPayloadFresh -Payload $stalePayload -MaxLagSeconds 60 -NowUtc $now
        }} | ConvertTo-Json -Compress
        """
    ).strip()

    result = _run_powershell_json(script, cwd=REPO_ROOT)

    assert result == {"fresh": True, "stale": False}


def test_group_history_ok_payload_with_old_messages_does_not_mark_stream_stale() -> None:
    script = textwrap.dedent(
        f"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
        $OutputEncoding = [Console]::OutputEncoding
        . {_ps_quote(str(HELPERS_PATH))}
        $payload = '{{"status":"ok","retcode":0,"data":{{"messages":[{{"time":1000}},{{"time":1100}}]}}}}' | ConvertFrom-Json
        $healthPath = Join-Path $env:TEMP ('group-stream-health-' + [guid]::NewGuid().ToString() + '.json')
        try {{
            Write-OneBotGroupStreamHealthFromPayload `
                -Path $healthPath `
                -Payload $payload `
                -PreviousStaleCount 2 `
                -MaxLagSeconds 60 `
                -NowUtc ([datetimeoffset]::FromUnixTimeSeconds(2000).UtcDateTime)
            Get-Content -Raw -LiteralPath $healthPath
        }} finally {{
            Remove-Item $healthPath -Force -ErrorAction SilentlyContinue
        }}
        """
    ).strip()

    result = _run_powershell_json(script, cwd=REPO_ROOT)

    assert result["stale"] is False
    assert result["stale_count"] == 0
    assert result["latest_message_time"] == 1100
    assert result["lag_seconds"] == 900


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


def test_get_bot_spec_status_recovers_running_process_from_heartbeat_when_pidfile_is_missing(tmp_path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    try:
        time.sleep(1.0)
        assert process.poll() is None

        pid_file = tmp_path / "group.pid"
        heartbeat_file = tmp_path / "group.heartbeat.json"
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )

        status = _read_status(
            module_name="fakeapp.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            timeout_seconds=30,
            grace_seconds=1,
        )

        assert status["is_running"] is True
        assert status["pid"] == process.pid
        assert status["needs_restart"] is False
        assert status["pid_file_stale"] is True
    finally:
        process.terminate()
        process.wait(timeout=10)


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


def test_get_bot_spec_status_marks_running_process_with_repeated_onebot_offline_for_restart(tmp_path) -> None:
    package_name = f"fakeapp_onebot_{tmp_path.name.replace('-', '_')}"
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
        onebot_health_file = tmp_path / "group.onebot_health.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )
        _write_onebot_health(
            onebot_health_file,
            online=False,
            offline_count=3,
            updated_at=datetime.now(UTC),
        )

        status = _read_status_with_onebot_health(
            module_name=f"{package_name}.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            onebot_health_file=onebot_health_file,
            timeout_seconds=30,
            grace_seconds=1,
            offline_threshold=3,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is True
        assert status["restart_reason"] == "onebot_offline"
        assert status["pid_file_stale"] is False
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_get_bot_spec_status_marks_running_process_with_repeated_group_stream_stale_for_restart(tmp_path) -> None:
    package_name = f"fakeapp_stream_{tmp_path.name.replace('-', '_')}"
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
        stream_health_file = tmp_path / "group.stream_health.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )
        _write_group_stream_health(
            stream_health_file,
            stale=True,
            stale_count=3,
            updated_at=datetime.now(UTC),
        )

        status = _read_status_with_group_stream_health(
            module_name=f"{package_name}.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            stream_health_file=stream_health_file,
            timeout_seconds=30,
            grace_seconds=1,
            stale_threshold=3,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is True
        assert status["restart_reason"] == "onebot_group_stream_stale"
        assert status["pid_file_stale"] is False
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_get_bot_spec_status_defers_onebot_offline_during_probe_startup_grace(tmp_path) -> None:
    package_name = f"fakeapp_onebot_grace_{tmp_path.name.replace('-', '_')}"
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
        onebot_health_file = tmp_path / "group.onebot_health.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )
        _write_onebot_health(
            onebot_health_file,
            online=False,
            offline_count=3,
            updated_at=datetime.now(UTC),
        )

        status = _read_status_with_onebot_health(
            module_name=f"{package_name}.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            onebot_health_file=onebot_health_file,
            timeout_seconds=30,
            grace_seconds=1,
            offline_threshold=3,
            onebot_probe_grace_seconds=120,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is False
        assert status["restart_reason"] == ""
        assert status["pid_file_stale"] is False
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_get_bot_spec_status_defers_group_stream_stale_during_probe_startup_grace(tmp_path) -> None:
    package_name = f"fakeapp_stream_grace_{tmp_path.name.replace('-', '_')}"
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
        stream_health_file = tmp_path / "group.stream_health.json"
        pid_file.write_text(str(process.pid), encoding="ascii")
        _write_heartbeat(
            heartbeat_file,
            pid=process.pid,
            updated_at=datetime.now(UTC),
        )
        _write_group_stream_health(
            stream_health_file,
            stale=True,
            stale_count=3,
            updated_at=datetime.now(UTC),
        )

        status = _read_status_with_group_stream_health(
            module_name=f"{package_name}.group_main",
            pid_file=pid_file,
            heartbeat_file=heartbeat_file,
            stream_health_file=stream_health_file,
            timeout_seconds=30,
            grace_seconds=1,
            stale_threshold=3,
            onebot_probe_grace_seconds=120,
        )

        assert status["is_running"] is True
        assert status["needs_restart"] is False
        assert status["restart_reason"] == ""
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
