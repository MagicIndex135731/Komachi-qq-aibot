$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

function Resolve-PythonExecutable([string]$Workdir) {
    $candidates = @(
        (Join-Path $Workdir ".venv\Scripts\python.exe"),
        "C:\work\anaconda\python.exe"
    )

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Python executable not found. Install the venv or update the launcher scripts."
}

function Get-BotProcessesForModule([string]$ModuleName) {
    $processes = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*" -and $_.CommandLine -like "* -m $ModuleName*"
    })

    $processes | Where-Object {
        $process = $_
        -not ($processes | Where-Object { $_.ProcessId -eq $process.ParentProcessId })
    }
}

function Stop-BotSpec([hashtable]$Spec) {
    $targets = @()
    $pidValue = ""
    if (Test-Path $Spec.PidFile) {
        $pidValue = (Get-Content $Spec.PidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
        if ($pidValue -match "^\d+$") {
            $target = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$pidValue) -ErrorAction SilentlyContinue
            if ($target -and $target.Name -like "python*" -and $target.CommandLine -like "*-m $($Spec.Module)*") {
                $targets += $target
            }
        }
    }

    foreach ($process in @(Get-BotProcessesForModule $Spec.Module)) {
        if (-not ($targets | Where-Object { $_.ProcessId -eq $process.ProcessId })) {
            $targets += $process
        }
    }

    foreach ($process in $targets) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
    if ($Spec.ContainsKey("HeartbeatFile") -and $Spec.HeartbeatFile) {
        Remove-Item $Spec.HeartbeatFile -Force -ErrorAction SilentlyContinue
    }
}

function Start-BotSpec([string]$Workdir, [string]$PythonExe, [hashtable]$Spec) {
    $running = @(Get-BotProcessesForModule $Spec.Module)
    if ($running.Count -gt 1) {
        foreach ($process in $running) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 1
        $running = @()
    }

    if ($running.Count -eq 1) {
        Set-Content -Path $Spec.PidFile -Value $running[0].ProcessId -Encoding ascii
        Write-Host "$($Spec.Name) already running. PID: $($running[0].ProcessId)"
        return
    }

    if ($Spec.ContainsKey("HeartbeatFile") -and $Spec.HeartbeatFile) {
        Remove-Item $Spec.HeartbeatFile -Force -ErrorAction SilentlyContinue
    }

    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList "-m", $Spec.Module `
        -WorkingDirectory $Workdir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Spec.Stdout `
        -RedirectStandardError $Spec.Stderr `
        -PassThru

    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
        throw "$($Spec.Name) failed to start. Check log: $($Spec.Stderr)"
    }

    Set-Content -Path $Spec.PidFile -Value $proc.Id -Encoding ascii
    Write-Host "$($Spec.Name) started. PID: $($proc.Id)"
}

function Get-ProcessStartTime([object]$Process) {
    if (-not $Process) {
        return $null
    }

    try {
        return (Get-Process -Id $Process.ProcessId -ErrorAction Stop).StartTime
    } catch {
    }
    if (-not $Process.CreationDate) {
        return $null
    }
    try {
        return [System.Management.ManagementDateTimeConverter]::ToDateTime([string]$Process.CreationDate)
    } catch {
        return $null
    }
}

function Read-HeartbeatState([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path)) {
        return $null
    }

    try {
        $raw = Get-Content -Path $Path -Raw -ErrorAction Stop
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            return ($raw | ConvertFrom-Json)
        }
    } catch {
        return $null
    }

    return $null
}

function Get-BotSpecStatus(
    [hashtable]$Spec,
    [int]$HeartbeatTimeoutSeconds = 45,
    [int]$StartupGraceSeconds = 20
) {
    $running = @(Get-BotProcessesForModule $Spec.Module)
    $runningProcess = $null
    $restartReason = ""
    $needsRestart = $false
    $heartbeatAgeSeconds = -1

    if ($running.Count -gt 1) {
        $runningProcess = $running[0]
        $needsRestart = $true
        $restartReason = "duplicate_processes"
    } elseif ($running.Count -eq 1) {
        $runningProcess = $running[0]
    }

    $pidValue = ""
    if (Test-Path $Spec.PidFile) {
        $pidValue = (Get-Content $Spec.PidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
    }

    $runningPid = if ($runningProcess) { [int]$runningProcess.ProcessId } else { 0 }
    $pidFileStale = $false
    if ($pidValue -match "^\d+$") {
        $pidFileStale = (-not $runningProcess) -or ([int]$pidValue -ne $runningPid)
    }

    if (-not $runningProcess -and -not $needsRestart) {
        $needsRestart = $true
        $restartReason = "process_missing"
    }

    if ($runningProcess -and -not $needsRestart) {
        $heartbeatState = $null
        if ($Spec.ContainsKey("HeartbeatFile")) {
            $heartbeatState = Read-HeartbeatState $Spec.HeartbeatFile
        }

        $startTime = Get-ProcessStartTime $runningProcess
        $uptimeSeconds = 0
        if ($startTime) {
            $uptimeSeconds = [int][Math]::Max(0, ((Get-Date) - $startTime).TotalSeconds)
        }

        $hasMatchingHeartbeat = $false
        if ($heartbeatState -and $heartbeatState.updated_at) {
            try {
                $heartbeatAt = [datetimeoffset]::Parse([string]$heartbeatState.updated_at)
                $heartbeatAgeSeconds = [int][Math]::Max(0, ((Get-Date).ToUniversalTime() - $heartbeatAt.UtcDateTime).TotalSeconds)
                $hasMatchingHeartbeat = ([int]$heartbeatState.pid -eq $runningPid)
            } catch {
                $heartbeatAgeSeconds = -1
                $hasMatchingHeartbeat = $false
            }
        }

        if (-not $heartbeatState -or -not $heartbeatState.updated_at) {
            if ($uptimeSeconds -gt $StartupGraceSeconds) {
                $needsRestart = $true
                $restartReason = "missing_heartbeat"
            }
        } elseif (-not $hasMatchingHeartbeat) {
            if ($uptimeSeconds -gt $StartupGraceSeconds) {
                $needsRestart = $true
                $restartReason = "heartbeat_pid_mismatch"
            }
        } elseif ($heartbeatAgeSeconds -gt $HeartbeatTimeoutSeconds) {
            $needsRestart = $true
            $restartReason = "stale_heartbeat"
        }
    }

    return [pscustomobject]@{
        name = [string]$Spec.Name
        module = [string]$Spec.Module
        pid = $runningPid
        is_running = [bool]$runningProcess
        needs_restart = $needsRestart
        restart_reason = $restartReason
        pid_file_stale = $pidFileStale
        heartbeat_age_seconds = $heartbeatAgeSeconds
    }
}

function Repair-BotSpecPidFile([hashtable]$Spec, [int]$Pid) {
    if ($Pid -gt 0) {
        Set-Content -Path $Spec.PidFile -Value $Pid -Encoding ascii
    }
}
