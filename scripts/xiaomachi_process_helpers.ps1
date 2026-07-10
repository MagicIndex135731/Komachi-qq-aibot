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

function Repair-StartProcessEnvironment {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if ([string]::IsNullOrEmpty($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    if (-not [string]::IsNullOrEmpty($pathValue)) {
        if ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT) {
            [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
            [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
            return
        }
        [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
}

function Get-BotProcessesForModule([string]$ModuleName) {
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -like "python*" -and $_.CommandLine -like "* -m $ModuleName*"
        })
    } catch {
        return @()
    }

    $leafProcesses = @($processes | Where-Object {
        $process = $_
        -not ($processes | Where-Object { [int]$_.ParentProcessId -eq [int]$process.ProcessId })
    })

    if ($leafProcesses.Count -gt 0) {
        return $leafProcesses
    }

    return $processes
}

function Get-BotProcessFromPidFile([string]$PidFile) {
    if (-not $PidFile -or -not (Test-Path $PidFile)) {
        return $null
    }

    $pidValue = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
    if (-not ($pidValue -match "^\d+$")) {
        return $null
    }

    try {
        $process = Get-Process -Id ([int]$pidValue) -ErrorAction Stop
        if ($process.ProcessName -notlike "python*") {
            return $null
        }
        return [pscustomobject]@{
            ProcessId = [int]$process.Id
            ParentProcessId = 0
            CreationDate = $null
        }
    } catch {
        return $null
    }
}

function Get-BotProcessFromHeartbeat([string]$HeartbeatFile) {
    $heartbeatState = Read-HeartbeatState $HeartbeatFile
    if (-not $heartbeatState -or -not $heartbeatState.pid) {
        return $null
    }

    try {
        $process = Get-Process -Id ([int]$heartbeatState.pid) -ErrorAction Stop
        if ($process.ProcessName -notlike "python*") {
            return $null
        }
        return [pscustomobject]@{
            ProcessId = [int]$process.Id
            ParentProcessId = 0
            CreationDate = $null
        }
    } catch {
        return $null
    }
}

function Get-BotProcessesForSpec([hashtable]$Spec) {
    if ($Spec.ContainsKey("HeartbeatFile") -and $Spec.HeartbeatFile) {
        $heartbeatProcess = Get-BotProcessFromHeartbeat $Spec.HeartbeatFile
        if ($heartbeatProcess) {
            return @($heartbeatProcess)
        }
    }

    $pidProcess = Get-BotProcessFromPidFile $Spec.PidFile
    if ($pidProcess) {
        return @($pidProcess)
    }

    $running = @(Get-BotProcessesForModule $Spec.Module)
    if ($running.Count -gt 0) {
        return $running
    }

    return @()
}

function Stop-BotSpec([hashtable]$Spec, [switch]$PassThru) {
    $targets = @(Get-BotProcessesForSpec $Spec)

    foreach ($process in $targets) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Start-Sleep -Milliseconds 500
    $remaining = @()
    foreach ($process in $targets) {
        $stillRunning = Get-Process -Id ([int]$process.ProcessId) -ErrorAction SilentlyContinue
        if ($stillRunning) {
            $remaining += [int]$process.ProcessId
        }
    }

    if ($remaining.Count -gt 0) {
        if ($PassThru) {
            return [pscustomobject]@{
                stopped = $false
                stopped_ids = @($targets | ForEach-Object { [int]$_.ProcessId } | Where-Object { $remaining -notcontains $_ })
                remaining_ids = $remaining
            }
        }
        return
    }

    Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
    if ($Spec.ContainsKey("HeartbeatFile") -and $Spec.HeartbeatFile) {
        Remove-Item $Spec.HeartbeatFile -Force -ErrorAction SilentlyContinue
    }
    if ($Spec.ContainsKey("OneBotHealthFile") -and $Spec.OneBotHealthFile) {
        Remove-Item $Spec.OneBotHealthFile -Force -ErrorAction SilentlyContinue
    }
    if ($Spec.ContainsKey("OneBotGroupStreamHealthFile") -and $Spec.OneBotGroupStreamHealthFile) {
        Remove-Item $Spec.OneBotGroupStreamHealthFile -Force -ErrorAction SilentlyContinue
    }

    if ($PassThru) {
        return [pscustomobject]@{
            stopped = $true
            stopped_ids = @($targets | ForEach-Object { [int]$_.ProcessId })
            remaining_ids = @()
        }
    }
}

function Wait-BotSpecHeartbeat(
    [hashtable]$Spec,
    [int]$ProcessId,
    [int]$TimeoutSeconds = 30
) {
    if (-not $Spec.ContainsKey("HeartbeatFile") -or -not $Spec.HeartbeatFile) {
        return [pscustomobject]@{
            ready = $true
            error = ""
        }
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $launcherProcess = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue

        $heartbeatState = Read-HeartbeatState $Spec.HeartbeatFile
        if ($heartbeatState -and $heartbeatState.updated_at) {
            try {
                $heartbeatPid = [int]$heartbeatState.pid
                $heartbeatProcess = Get-Process -Id $heartbeatPid -ErrorAction SilentlyContinue
                if ($heartbeatProcess -and $heartbeatProcess.ProcessName -like "python*") {
                    return [pscustomobject]@{
                        ready = $true
                        error = ""
                        process_id = $heartbeatPid
                    }
                }
            } catch {
            }
        }

        if (-not $launcherProcess) {
            return [pscustomobject]@{
                ready = $false
                error = ("{0} exited before publishing heartbeat. Check log: {1}" -f $Spec.Name, $Spec.Stderr)
                process_id = 0
            }
        }

        Start-Sleep -Milliseconds 500
    }

    return [pscustomobject]@{
        ready = $false
        error = ("{0} started but did not publish matching heartbeat within {1} seconds. Check log: {2}" -f $Spec.Name, $TimeoutSeconds, $Spec.Stderr)
        process_id = 0
    }
}

function Start-BotSpec(
    [string]$Workdir,
    [string]$PythonExe,
    [hashtable]$Spec,
    [int]$HeartbeatReadyTimeoutSeconds = 30
) {
    $running = @(Get-BotProcessesForSpec $Spec)
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

    Repair-StartProcessEnvironment
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
    $heartbeatReady = Wait-BotSpecHeartbeat -Spec $Spec -ProcessId ([int]$proc.Id) -TimeoutSeconds $HeartbeatReadyTimeoutSeconds
    if (-not $heartbeatReady.ready) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
        throw $heartbeatReady.error
    }

    $trackedPid = [int]$proc.Id
    if ($heartbeatReady.process_id) {
        try {
            $trackedPid = [int]$heartbeatReady.process_id
        } catch {
            $trackedPid = [int]$proc.Id
        }
    }
    if ($trackedPid -ne [int]$proc.Id) {
        Set-Content -Path $Spec.PidFile -Value $trackedPid -Encoding ascii
    }
    Write-Host "$($Spec.Name) started. PID: $trackedPid"
}

function Restart-BotSpecSafely(
    [string]$Workdir,
    [string]$PythonExe,
    [hashtable]$Spec,
    [int]$HeartbeatReadyTimeoutSeconds = 30
) {
    try {
        Stop-BotSpec -Spec $Spec
        Start-BotSpec -Workdir $Workdir -PythonExe $PythonExe -Spec $Spec -HeartbeatReadyTimeoutSeconds $HeartbeatReadyTimeoutSeconds
        return [pscustomobject]@{
            restarted = $true
            error = ""
        }
    } catch {
        $errorText = ([string]$_.Exception.Message).Trim()
        if ([string]::IsNullOrWhiteSpace($errorText)) {
            $errorText = ([string]$_).Trim()
        }
        return [pscustomobject]@{
            restarted = $false
            error = ($errorText -replace "\s+", " ")
        }
    }
}

function Get-BotSpecRestartState([hashtable]$StateByName, [string]$Name) {
    if (-not $StateByName.ContainsKey($Name)) {
        $StateByName[$Name] = @{
            attempts = @()
            suppressed_until = $null
        }
    }

    return $StateByName[$Name]
}

function Test-BotSpecRestartBudget(
    [hashtable]$StateByName,
    [string]$Name,
    [int]$MaxAttempts = 5,
    [int]$WindowSeconds = 600,
    [int]$SuppressSeconds = 300
) {
    $state = Get-BotSpecRestartState -StateByName $StateByName -Name $Name
    $now = Get-Date
    if ($state.suppressed_until -and $now -lt $state.suppressed_until) {
        return [pscustomobject]@{
            allowed = $false
            attempts = @($state.attempts).Count
            suppressed_seconds = [int][Math]::Max(0, ($state.suppressed_until - $now).TotalSeconds)
        }
    }

    $cutoff = $now.AddSeconds(-1 * $WindowSeconds)
    $state.attempts = @($state.attempts | Where-Object { $_ -gt $cutoff })
    if (@($state.attempts).Count -ge $MaxAttempts) {
        $state.suppressed_until = $now.AddSeconds($SuppressSeconds)
        return [pscustomobject]@{
            allowed = $false
            attempts = @($state.attempts).Count
            suppressed_seconds = $SuppressSeconds
        }
    }

    return [pscustomobject]@{
        allowed = $true
        attempts = @($state.attempts).Count
        suppressed_seconds = 0
    }
}

function Register-BotSpecRestartAttempt([hashtable]$StateByName, [string]$Name) {
    $state = Get-BotSpecRestartState -StateByName $StateByName -Name $Name
    $state.attempts = @($state.attempts) + @(Get-Date)
}

function Test-WatchdogProbeDue(
    [hashtable]$LastProbeByName,
    [string]$Name,
    [int]$IntervalSeconds,
    [datetime]$NowUtc = (Get-Date).ToUniversalTime()
) {
    if ($IntervalSeconds -le 0) {
        return $true
    }

    if (-not $LastProbeByName.ContainsKey($Name)) {
        return $true
    }

    try {
        $lastProbeAt = [datetime]$LastProbeByName[$Name]
        return (($NowUtc - $lastProbeAt).TotalSeconds -ge $IntervalSeconds)
    } catch {
        return $true
    }
}

function Register-WatchdogProbeRun(
    [hashtable]$LastProbeByName,
    [string]$Name,
    [datetime]$NowUtc = (Get-Date).ToUniversalTime()
) {
    $LastProbeByName[$Name] = $NowUtc
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

function Test-OneBotStatusPayloadOnline([object]$Payload) {
    if (-not $Payload) {
        return $false
    }

    try {
        if ([string]$Payload.status -ne "ok") {
            return $false
        }
        if ([int]$Payload.retcode -ne 0) {
            return $false
        }
        if (-not $Payload.data) {
            return $false
        }
        return ([bool]$Payload.data.online)
    } catch {
        return $false
    }
}

function Read-OneBotHealthState([string]$Path) {
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

function Write-OneBotHealthState(
    [string]$Path,
    [bool]$Online,
    [int]$OfflineCount,
    [string]$ErrorMessage = ""
) {
    if (-not $Path) {
        return
    }

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    @{
        online = $Online
        offline_count = $OfflineCount
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Compress | Set-Content -Path $Path -Encoding utf8
}

function Read-OneBotGroupStreamHealthState([string]$Path) {
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

function Write-OneBotGroupStreamHealthState(
    [string]$Path,
    [bool]$Stale,
    [int]$StaleCount,
    [int]$LatestMessageTime = 0,
    [int]$LagSeconds = -1,
    [string]$ErrorMessage = ""
) {
    if (-not $Path) {
        return
    }

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    @{
        stale = $Stale
        stale_count = $StaleCount
        latest_message_time = $LatestMessageTime
        lag_seconds = $LagSeconds
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        error = $ErrorMessage
    } | ConvertTo-Json -Compress | Set-Content -Path $Path -Encoding utf8
}

function Get-OneBotHistoryMessages([object]$Payload) {
    if (-not $Payload -or [string]$Payload.status -ne "ok") {
        return @()
    }
    try {
        if ([int]$Payload.retcode -ne 0 -or -not $Payload.data) {
            return @()
        }
    } catch {
        return @()
    }

    if ($Payload.data -is [array]) {
        return @($Payload.data)
    }
    if ($Payload.data.messages -is [array]) {
        return @($Payload.data.messages)
    }
    if ($Payload.data.data -is [array]) {
        return @($Payload.data.data)
    }
    if ($Payload.data.list -is [array]) {
        return @($Payload.data.list)
    }
    return @()
}

function Get-OneBotGroupHistoryLatestUnixTime([object]$Payload) {
    $latest = 0
    foreach ($message in @(Get-OneBotHistoryMessages $Payload)) {
        try {
            $candidate = [int64]$message.time
            if ($candidate -gt $latest) {
                $latest = $candidate
            }
        } catch {
        }
    }
    return $latest
}

function Test-OneBotGroupHistoryPayloadFresh(
    [object]$Payload,
    [int]$MaxLagSeconds,
    [datetime]$NowUtc = (Get-Date).ToUniversalTime()
) {
    $latest = Get-OneBotGroupHistoryLatestUnixTime $Payload
    if ($latest -le 0) {
        return $false
    }

    $latestAt = [datetimeoffset]::FromUnixTimeSeconds($latest).UtcDateTime
    $lagSeconds = [int][Math]::Max(0, ($NowUtc - $latestAt).TotalSeconds)
    return $lagSeconds -le $MaxLagSeconds
}

function Write-OneBotGroupStreamHealthFromPayload(
    [string]$Path,
    [object]$Payload,
    [int]$PreviousStaleCount,
    [int]$MaxLagSeconds,
    [datetime]$NowUtc = (Get-Date).ToUniversalTime()
) {
    $latest = Get-OneBotGroupHistoryLatestUnixTime $Payload
    $lagSeconds = -1
    if ($latest -gt 0) {
        $latestAt = [datetimeoffset]::FromUnixTimeSeconds($latest).UtcDateTime
        $lagSeconds = [int][Math]::Max(0, ($NowUtc - $latestAt).TotalSeconds)
    }

    if ($Payload -and [string]$Payload.status -eq "ok") {
        if (Test-OneBotGroupHistoryPayloadFresh -Payload $Payload -MaxLagSeconds $MaxLagSeconds -NowUtc $NowUtc) {
            Write-OneBotGroupStreamHealthState -Path $Path -Stale $false -StaleCount 0 -LatestMessageTime $latest -LagSeconds $lagSeconds
            return
        }

        $detail = "group_history_stale latest_message_time={0} lag_seconds={1} max_lag_seconds={2}" -f $latest, $lagSeconds, $MaxLagSeconds
        Write-OneBotGroupStreamHealthState -Path $Path -Stale $true -StaleCount ($PreviousStaleCount + 1) -LatestMessageTime $latest -LagSeconds $lagSeconds -ErrorMessage $detail
        return
    }

    $payloadText = $Payload | ConvertTo-Json -Compress -Depth 8
    Write-OneBotGroupStreamHealthState -Path $Path -Stale $true -StaleCount ($PreviousStaleCount + 1) -LatestMessageTime $latest -LagSeconds $lagSeconds -ErrorMessage $payloadText
}

function Invoke-OneBotStatusProbe([string]$PythonExe, [string]$WsUrl) {
    if (-not $PythonExe -or -not $WsUrl) {
        return $null
    }

    $probeScript = @"
import asyncio
import json
import sys
from uuid import uuid4

import websockets

async def main():
    echo = "health-" + uuid4().hex
    async with websockets.connect(sys.argv[1]) as ws:
        await ws.send(json.dumps({"action": "get_status", "params": {}, "echo": echo}))
        for _ in range(20):
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if payload.get("echo") == echo:
                print(json.dumps(payload))
                return
        raise TimeoutError("get_status response not received")

asyncio.run(main())
"@

    try {
        $output = $probeScript | & $PythonExe - $WsUrl 2>&1
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{
                ok = $false
                payload = $null
                error = ($output | Out-String).Trim()
            }
        }

        return [pscustomobject]@{
            ok = $true
            payload = (($output | Out-String).Trim() | ConvertFrom-Json)
            error = ""
        }
    } catch {
        return [pscustomobject]@{
            ok = $false
            payload = $null
            error = [string]$_.Exception.Message
        }
    }
}

function Invoke-OneBotGroupHistoryProbe([string]$PythonExe, [string]$WsUrl, [int]$GroupId, [int]$Count = 10) {
    if (-not $PythonExe -or -not $WsUrl -or $GroupId -le 0) {
        return $null
    }

    $probeScript = @"
import asyncio
import json
import sys
from uuid import uuid4

import websockets

async def main():
    echo = "group-history-" + uuid4().hex
    ws_url = sys.argv[1]
    group_id = int(sys.argv[2])
    count = int(sys.argv[3])
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"action": "get_group_msg_history", "params": {"group_id": group_id, "count": count}, "echo": echo}))
        for _ in range(20):
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if payload.get("echo") == echo:
                print(json.dumps(payload))
                return
        raise TimeoutError("get_group_msg_history response not received")

asyncio.run(main())
"@

    try {
        $output = $probeScript | & $PythonExe - $WsUrl $GroupId $Count 2>&1
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{
                ok = $false
                payload = $null
                error = ($output | Out-String).Trim()
            }
        }

        return [pscustomobject]@{
            ok = $true
            payload = (($output | Out-String).Trim() | ConvertFrom-Json)
            error = ""
        }
    } catch {
        return [pscustomobject]@{
            ok = $false
            payload = $null
            error = [string]$_.Exception.Message
        }
    }
}

function Update-OneBotHealthStateForSpec([hashtable]$Spec, [string]$PythonExe) {
    if (
        -not $Spec.ContainsKey("OneBotHealthFile") -or
        -not $Spec.OneBotHealthFile -or
        -not $Spec.ContainsKey("OneBotWsUrl") -or
        -not $Spec.OneBotWsUrl
    ) {
        return
    }

    $previous = Read-OneBotHealthState $Spec.OneBotHealthFile
    $previousOfflineCount = 0
    if ($previous -and $previous.offline_count) {
        try {
            $previousOfflineCount = [int]$previous.offline_count
        } catch {
            $previousOfflineCount = 0
        }
    }

    $probe = Invoke-OneBotStatusProbe -PythonExe $PythonExe -WsUrl ([string]$Spec.OneBotWsUrl)
    if ($probe -and $probe.ok -and (Test-OneBotStatusPayloadOnline $probe.payload)) {
        Write-OneBotHealthState -Path $Spec.OneBotHealthFile -Online $true -OfflineCount 0
        return
    }

    $errorMessage = ""
    if ($probe -and $probe.error) {
        $errorMessage = [string]$probe.error
    } elseif ($probe -and $probe.payload) {
        $errorMessage = ($probe.payload | ConvertTo-Json -Compress -Depth 8)
    } else {
        $errorMessage = "onebot status probe failed"
    }
    Write-OneBotHealthState -Path $Spec.OneBotHealthFile -Online $false -OfflineCount ($previousOfflineCount + 1) -ErrorMessage $errorMessage
}

function Update-OneBotGroupStreamHealthStateForSpec([hashtable]$Spec, [string]$PythonExe) {
    if (
        -not $Spec.ContainsKey("OneBotGroupStreamHealthFile") -or
        -not $Spec.OneBotGroupStreamHealthFile -or
        -not $Spec.ContainsKey("OneBotWsUrl") -or
        -not $Spec.OneBotWsUrl -or
        -not $Spec.ContainsKey("OneBotGroupId") -or
        -not $Spec.OneBotGroupId
    ) {
        return
    }

    $previous = Read-OneBotGroupStreamHealthState $Spec.OneBotGroupStreamHealthFile
    $previousStaleCount = 0
    if ($previous -and $previous.stale_count) {
        try {
            $previousStaleCount = [int]$previous.stale_count
        } catch {
            $previousStaleCount = 0
        }
    }

    $maxLagSeconds = 1800
    if ($Spec.ContainsKey("OneBotGroupStreamMaxLagSeconds") -and $Spec.OneBotGroupStreamMaxLagSeconds) {
        $maxLagSeconds = [int]$Spec.OneBotGroupStreamMaxLagSeconds
    }

    $probe = Invoke-OneBotGroupHistoryProbe -PythonExe $PythonExe -WsUrl ([string]$Spec.OneBotWsUrl) -GroupId ([int]$Spec.OneBotGroupId)
    if ($probe -and $probe.ok) {
        Write-OneBotGroupStreamHealthFromPayload `
            -Path $Spec.OneBotGroupStreamHealthFile `
            -Payload $probe.payload `
            -PreviousStaleCount $previousStaleCount `
            -MaxLagSeconds $maxLagSeconds
        return
    }

    $errorMessage = if ($probe -and $probe.error) { [string]$probe.error } else { "group history probe failed" }
    Write-OneBotGroupStreamHealthState -Path $Spec.OneBotGroupStreamHealthFile -Stale $true -StaleCount ($previousStaleCount + 1) -ErrorMessage $errorMessage
}

function Get-BotSpecStatus(
    [hashtable]$Spec,
    [int]$HeartbeatTimeoutSeconds = 45,
    [int]$StartupGraceSeconds = 20
) {
    $running = @(Get-BotProcessesForSpec $Spec)
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
    } elseif ($runningProcess) {
        $pidFileStale = $true
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
        $oneBotProbeGraceSeconds = 0
        if ($Spec.ContainsKey("OneBotProbeStartupGraceSeconds") -and $Spec.OneBotProbeStartupGraceSeconds) {
            try {
                $oneBotProbeGraceSeconds = [int]$Spec.OneBotProbeStartupGraceSeconds
            } catch {
                $oneBotProbeGraceSeconds = 0
            }
        }
        $withinOneBotProbeGrace = ($oneBotProbeGraceSeconds -gt 0 -and $uptimeSeconds -lt $oneBotProbeGraceSeconds)

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

        if (-not $needsRestart -and -not $withinOneBotProbeGrace -and $Spec.ContainsKey("OneBotHealthFile") -and $Spec.OneBotHealthFile) {
            $onebotState = Read-OneBotHealthState $Spec.OneBotHealthFile
            $offlineThreshold = 3
            if ($Spec.ContainsKey("OneBotOfflineRestartThreshold") -and $Spec.OneBotOfflineRestartThreshold) {
                $offlineThreshold = [int]$Spec.OneBotOfflineRestartThreshold
            }

            if ($onebotState -and $onebotState.online -eq $false) {
                try {
                    if ([int]$onebotState.offline_count -ge $offlineThreshold) {
                        $needsRestart = $true
                        $restartReason = "onebot_offline"
                    }
                } catch {
                }
            }
        }

        if (-not $needsRestart -and -not $withinOneBotProbeGrace -and $Spec.ContainsKey("OneBotGroupStreamHealthFile") -and $Spec.OneBotGroupStreamHealthFile) {
            $streamState = Read-OneBotGroupStreamHealthState $Spec.OneBotGroupStreamHealthFile
            $staleThreshold = 3
            if ($Spec.ContainsKey("OneBotGroupStreamStaleRestartThreshold") -and $Spec.OneBotGroupStreamStaleRestartThreshold) {
                $staleThreshold = [int]$Spec.OneBotGroupStreamStaleRestartThreshold
            }

            if ($streamState -and $streamState.stale -eq $true) {
                try {
                    if ([int]$streamState.stale_count -ge $staleThreshold) {
                        $needsRestart = $true
                        $restartReason = "onebot_group_stream_stale"
                    }
                } catch {
                }
            }
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

function Repair-BotSpecPidFile([hashtable]$Spec, [int]$ProcessId) {
    if ($ProcessId -gt 0) {
        Set-Content -Path $Spec.PidFile -Value $ProcessId -Encoding ascii
    }
}
