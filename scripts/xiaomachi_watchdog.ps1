param(
    [ValidateSet("start", "run", "stop")]
    [string]$Action = "start",

    [ValidateSet("runtime", "worker", "all")]
    [string]$Scope = "runtime",

    [int]$HeartbeatTimeoutSeconds = 45,
    [int]$StartupGraceSeconds = 20,
    [int]$CheckIntervalSeconds = 5,
    [int]$HeartbeatReadyTimeoutSeconds = 30,
    [int]$RestartMaxAttempts = 5,
    [int]$RestartWindowSeconds = 600,
    [int]$RestartSuppressSeconds = 300,
    [int]$OneBotStatusProbeIntervalSeconds = 15,
    [int]$OneBotGroupStreamProbeIntervalSeconds = 60
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $workdir "data\logs"
$stateFile = Join-Path $logDir "launcher_state.json"
$envFile = Join-Path $workdir ".env"
$alertFile = Join-Path $logDir "xiaomachi.alert.json"
$stopRequestFile = Join-Path $logDir "xiaomachi.stop_requested.json"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $workdir "scripts\xiaomachi_process_helpers.ps1")

function Read-DotEnvFile([string]$Path) {
    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content -Path $Path -Encoding utf8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $separatorIndex = $trimmed.IndexOf("=")
        if ($separatorIndex -lt 1) {
            continue
        }

        $key = $trimmed.Substring(0, $separatorIndex).Trim()
        $value = $trimmed.Substring($separatorIndex + 1).Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        $values[$key] = $value
    }

    return $values
}

$dotenv = Read-DotEnvFile $envFile

function Get-ConfigValue([string]$Key, [string]$Default = "") {
    $runtimeValue = [Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($runtimeValue)) {
        return $runtimeValue
    }

    if ($dotenv.ContainsKey($Key)) {
        $fileValue = [string]$dotenv[$Key]
        if (-not [string]::IsNullOrWhiteSpace($fileValue)) {
            return $fileValue
        }
    }

    return $Default
}

$napcatWsUrl = Get-ConfigValue "NAPCAT_WS_URL" "ws://127.0.0.1:3001"
$groupStreamWatchGroupId = [int](Get-ConfigValue "GROUP_STREAM_WATCH_GROUP_ID" "0")
$groupStreamMaxLagSeconds = [int](Get-ConfigValue "GROUP_STREAM_MAX_LAG_SECONDS" "1800")
$fullRestartCooldownSeconds = [int](Get-ConfigValue "XIAOMACHI_FULL_RESTART_COOLDOWN_SECONDS" "600")
$oneBotProbeStartupGraceSeconds = [int](Get-ConfigValue "ONEBOT_PROBE_STARTUP_GRACE_SECONDS" "180")

function Get-WatchdogPidFile([string]$CurrentScope) {
    return (Join-Path $logDir ("{0}.watchdog.pid" -f $CurrentScope))
}

function Get-WatchdogLogFile([string]$CurrentScope) {
    return (Join-Path $logDir ("{0}.watchdog.log" -f $CurrentScope))
}

function Write-WatchdogLog([string]$CurrentScope, [string]$Message) {
    $timestamp = (Get-Date).ToString("s")
    Add-Content -Path (Get-WatchdogLogFile $CurrentScope) -Value ("[{0}] {1}" -f $timestamp, $Message) -Encoding utf8
}

function Write-XiaomachiAlert([string]$Reason, [string]$Detail = "") {
    @{
        reason = $Reason
        detail = $Detail
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Compress | Set-Content -Path $alertFile -Encoding utf8
}

function Test-StopRequested {
    return (Test-Path -LiteralPath $stopRequestFile)
}

function Show-XiaomachiDesktopAlert([string]$CurrentScope, [string]$Title, [string]$Message) {
    $escapedTitle = $Title.Replace("'", "''")
    $escapedMessage = $Message.Replace("'", "''")
$script = @"
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.MessageBox]::Show(
    '$escapedMessage',
    '$escapedTitle',
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Warning,
    [System.Windows.Forms.MessageBoxDefaultButton]::Button1,
    [System.Windows.Forms.MessageBoxOptions]::ServiceNotification
) | Out-Null
"@

    try {
        Repair-StartProcessEnvironment
        Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", $script) | Out-Null
    } catch {
        Write-WatchdogLog $CurrentScope ("desktop_alert_failed error={0}" -f (([string]$_.Exception.Message) -replace "\s+", " "))
    }
}

function Test-WatchdogProcessMatchesScope([object]$Process, [string]$CurrentScope) {
    if (-not $Process) {
        return $false
    }
    if ($Process.Name -notlike "powershell*") {
        return $false
    }
    $commandLine = [string]$Process.CommandLine
    return (
        $commandLine -like "*xiaomachi_watchdog.ps1*" -and
        $commandLine -like "*-Action run*" -and
        $commandLine -like "*-Scope $CurrentScope*"
    )
}

function Get-WatchdogProcessesForScope([string]$CurrentScope) {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            Test-WatchdogProcessMatchesScope -Process $_ -CurrentScope $CurrentScope
        })
    } catch {
        $pidFile = Get-WatchdogPidFile $CurrentScope
        if (-not (Test-Path $pidFile)) {
            return @()
        }
        $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
        if (-not ($pidValue -match "^\d+$")) {
            return @()
        }
        $fallback = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if ($fallback -and $fallback.ProcessName -like "powershell*") {
            return @([pscustomobject]@{
                ProcessId = [int]$fallback.Id
                Name = $fallback.ProcessName
                CommandLine = ""
            })
        }
    }

    return @()
}

function Stop-ExtraWatchdogProcesses([string]$CurrentScope, [int]$KeepPid = 0) {
    $watchdogs = @(Get-WatchdogProcessesForScope $CurrentScope)
    if ($watchdogs.Count -le 0) {
        return
    }

    $keepers = @()
    if ($KeepPid -gt 0) {
        $keepers = @($watchdogs | Where-Object { [int]$_.ProcessId -eq $KeepPid })
    }
    if ($keepers.Count -eq 0) {
        $keepers = @($watchdogs | Sort-Object ProcessId | Select-Object -First 1)
        $KeepPid = [int]$keepers[0].ProcessId
    }

    foreach ($process in @($watchdogs | Where-Object { [int]$_.ProcessId -ne $KeepPid })) {
        Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
        Write-WatchdogLog $CurrentScope ("watchdog_duplicate_stopped scope={0} pid={1} keep_pid={2}" -f $CurrentScope, $process.ProcessId, $KeepPid)
    }
}

function Stop-ChildPythonProcessesForParents([int[]]$ParentProcessIds) {
    $parentIds = @($ParentProcessIds | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($parentIds.Count -le 0) {
        return
    }

    try {
        $children = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -like "python*" -and $parentIds -contains [int]$_.ParentProcessId
        })
    } catch {
        return
    }

    foreach ($child in $children) {
        Stop-Process -Id ([int]$child.ProcessId) -Force -ErrorAction SilentlyContinue
    }
}

function Get-WatchdogProcess([string]$CurrentScope) {
    $pidFile = Get-WatchdogPidFile $CurrentScope
    $runningWatchdogs = @(Get-WatchdogProcessesForScope $CurrentScope)
    if (-not (Test-Path $pidFile)) {
        if ($runningWatchdogs.Count -gt 0) {
            $selected = @($runningWatchdogs | Sort-Object ProcessId | Select-Object -First 1)[0]
            Set-Content -Path $pidFile -Value ([int]$selected.ProcessId) -Encoding ascii
            Stop-ExtraWatchdogProcesses -CurrentScope $CurrentScope -KeepPid ([int]$selected.ProcessId)
            return $selected
        }
        return $null
    }

    $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
    if (-not ($pidValue -match "^\d+$")) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        if ($runningWatchdogs.Count -gt 0) {
            $selected = @($runningWatchdogs | Sort-Object ProcessId | Select-Object -First 1)[0]
            Set-Content -Path $pidFile -Value ([int]$selected.ProcessId) -Encoding ascii
            Stop-ExtraWatchdogProcesses -CurrentScope $CurrentScope -KeepPid ([int]$selected.ProcessId)
            return $selected
        }
        return $null
    }

    $process = $null
    try {
        $process = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$pidValue) -ErrorAction Stop
    } catch {
        $fallback = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if ($fallback -and $fallback.ProcessName -like "powershell*") {
            return [pscustomobject]@{
                ProcessId = [int]$fallback.Id
            }
        }
    }

    if ($process -and (Test-WatchdogProcessMatchesScope -Process $process -CurrentScope $CurrentScope)) {
        Stop-ExtraWatchdogProcesses -CurrentScope $CurrentScope -KeepPid ([int]$process.ProcessId)
        return $process
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    if ($runningWatchdogs.Count -gt 0) {
        $selected = @($runningWatchdogs | Sort-Object ProcessId | Select-Object -First 1)[0]
        Set-Content -Path $pidFile -Value ([int]$selected.ProcessId) -Encoding ascii
        Stop-ExtraWatchdogProcesses -CurrentScope $CurrentScope -KeepPid ([int]$selected.ProcessId)
        return $selected
    }
    return $null
}

function Get-ScopeProcessSpecs([string]$CurrentScope) {
    $runtimeSpecs = @(
        @{
            Name = "group"
            Module = "app.group_main"
            PidFile = Join-Path $logDir "group.pid"
            Stdout = Join-Path $logDir "group.stdout.log"
            Stderr = Join-Path $logDir "group.stderr.log"
            HeartbeatFile = Join-Path $logDir "group.heartbeat.json"
            OneBotHealthFile = Join-Path $logDir "group.onebot_health.json"
            OneBotGroupStreamHealthFile = Join-Path $logDir "group.stream_health.json"
            OneBotWsUrl = $napcatWsUrl
            OneBotOfflineRestartThreshold = 3
            OneBotProbeStartupGraceSeconds = $oneBotProbeStartupGraceSeconds
            OneBotGroupId = $groupStreamWatchGroupId
            OneBotGroupStreamMaxLagSeconds = $groupStreamMaxLagSeconds
            OneBotGroupStreamStaleRestartThreshold = 3
        },
        @{
            Name = "private"
            Module = "app.private_main"
            PidFile = Join-Path $logDir "private.pid"
            Stdout = Join-Path $logDir "private.stdout.log"
            Stderr = Join-Path $logDir "private.stderr.log"
            HeartbeatFile = Join-Path $logDir "private.heartbeat.json"
        }
    )

    $workerSpecs = @(
        @{
            Name = "worker"
            Module = "app.dev_worker_main"
            PidFile = Join-Path $logDir "worker.pid"
            Stdout = Join-Path $logDir "worker.stdout.log"
            Stderr = Join-Path $logDir "worker.stderr.log"
            HeartbeatFile = Join-Path $logDir "worker.heartbeat.json"
        }
    )

    switch ($CurrentScope) {
        "runtime" { return $runtimeSpecs }
        "worker" { return $workerSpecs }
        "all" { return @($runtimeSpecs + $workerSpecs) }
        default { throw "Unsupported watchdog scope: $CurrentScope" }
    }
}

function Read-LauncherState {
    if (-not (Test-Path $stateFile)) {
        return $null
    }

    try {
        return (Get-Content -Path $stateFile -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Stop-TrackedProcesses([int[]]$Ids) {
    foreach ($id in ($Ids | Where-Object { $_ -gt 0 } | Select-Object -Unique)) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
}

function Get-ProcessDescendantIds([int[]]$ParentIds) {
    $allProcesses = @()
    try {
        $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        return @()
    }

    $pending = @($ParentIds | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    $descendants = @()
    while ($pending.Count -gt 0) {
        $parentId = [int]$pending[0]
        $pending = @($pending | Select-Object -Skip 1)
        $children = @($allProcesses | Where-Object { [int]$_.ParentProcessId -eq $parentId })
        foreach ($child in $children) {
            $childId = [int]$child.ProcessId
            if ($descendants -notcontains $childId) {
                $descendants += $childId
                $pending += $childId
            }
        }
    }

    return $descendants
}

function Get-NapCatBootProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -eq "NapCatWinBootMain.exe"
        })
    } catch {
        return @(Get-Process -Name "NapCatWinBootMain" -ErrorAction SilentlyContinue | ForEach-Object {
            [pscustomobject]@{
                ProcessId = [int]$_.Id
            }
        })
    }
}

function Get-XiaomachiQQProcesses {
    $qqPath = Get-ConfigValue "QQ_EXE_PATH" ""
    $workdirPrefix = ($workdir.TrimEnd("\") + "\")
    @(Get-QQProcesses) | Where-Object {
        $exePath = [string]$_.ExecutablePath
        $commandLine = [string]$_.CommandLine
        (
            (-not [string]::IsNullOrWhiteSpace($qqPath) -and $exePath -ieq $qqPath) -or
            $exePath.StartsWith($workdirPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
            $commandLine.IndexOf($workdir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $commandLine.IndexOf("xiaomachi-qq-", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        )
    }
}

function Stop-OrphanedXiaomachiQQProcesses {
    $qqTargets = @(Get-XiaomachiQQProcesses)
    $qqIds = @($qqTargets | Where-Object { $_ } | ForEach-Object { [int]$_.ProcessId } | Select-Object -Unique)
    if ($qqIds.Count -le 0) {
        return @()
    }

    Stop-TrackedProcesses -Ids $qqIds
    return @("orphaned_xiaomachi_qq=" + ($qqIds -join ","))
}

function Get-QQProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -eq "QQ.exe" })
    } catch {
        return @(Get-Process -Name QQ -ErrorAction SilentlyContinue | ForEach-Object {
            [pscustomobject]@{
                ProcessId = [int]$_.Id
            }
        })
    }
}

function Get-LastFullRestartAt {
    $path = Join-Path $logDir "full_restart.last.json"
    if (-not (Test-Path $path)) {
        return $null
    }
    try {
        $raw = Get-Content -Path $path -Raw | ConvertFrom-Json
        if ($raw.updated_at) {
            return [datetimeoffset]::Parse([string]$raw.updated_at).UtcDateTime
        }
    } catch {
    }
    return $null
}

function Set-LastFullRestartAt([string]$Reason) {
    $path = Join-Path $logDir "full_restart.last.json"
    @{
        reason = $Reason
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Compress | Set-Content -Path $path -Encoding utf8
}

function Restart-LauncherManagedNapCat([string]$CurrentScope) {
    $lastRestartAt = Get-LastFullRestartAt
    if ($lastRestartAt) {
        $ageSeconds = [int][Math]::Max(0, ((Get-Date).ToUniversalTime() - $lastRestartAt).TotalSeconds)
        if ($ageSeconds -lt $fullRestartCooldownSeconds) {
            Write-WatchdogLog $CurrentScope ("full_restart_skipped reason=cooldown age_seconds={0}" -f $ageSeconds)
            Write-XiaomachiAlert -Reason "full_restart_cooldown" -Detail ("age_seconds={0}" -f $ageSeconds)
            return
        }
    }

    $state = Read-LauncherState
    if (-not $state) {
        Write-WatchdogLog $CurrentScope "onebot_offline_full_restart_skipped reason=missing_launcher_state"
        Write-XiaomachiAlert -Reason "full_restart_skipped" -Detail "missing_launcher_state"
        return
    }

    $stopped = @()
    if ($state.boot_started_by_launcher) {
        $bootTargets = @(Get-NapCatBootProcesses)
        if ($state.boot_pid) {
            try {
                $bootTargets += @(Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$state.boot_pid) -ErrorAction Stop)
            } catch {
                $fallbackBoot = Get-Process -Id ([int]$state.boot_pid) -ErrorAction SilentlyContinue
                if ($fallbackBoot -and $fallbackBoot.ProcessName -eq "NapCatWinBootMain") {
                    $bootTargets += [pscustomobject]@{
                        ProcessId = [int]$fallbackBoot.Id
                    }
                }
            }
        }
        $bootIds = @($bootTargets | Where-Object { $_ } | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($bootIds.Count -gt 0) {
            $targetIds = @($bootIds + @(Get-ProcessDescendantIds -ParentIds $bootIds) | Select-Object -Unique)
            Stop-TrackedProcesses -Ids $targetIds
            $stopped += ("NapCatWinBootMain=" + ($targetIds -join ","))
        }
    }

    if ($state.qq_started_by_launcher) {
        $qqTargets = @(Get-XiaomachiQQProcesses)
        $qqIds = @($qqTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($qqIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $qqIds
            $stopped += ("QQ.exe=" + ($qqIds -join ","))
        }
    }

    Start-Sleep -Seconds 2
    $startScript = Join-Path $workdir "start_xiaomachi.ps1"
    Repair-StartProcessEnvironment
    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $startScript) `
        -WorkingDirectory $workdir `
        -WindowStyle Hidden | Out-Null
    Write-WatchdogLog $CurrentScope ("onebot_offline_full_restart_requested stopped={0}" -f (($stopped -join ";") -replace "\s+", "_"))
    Set-LastFullRestartAt -Reason "onebot_health_failure"
    Write-XiaomachiAlert -Reason "full_restart_requested" -Detail ("stopped={0}" -f (($stopped -join ";") -replace "\s+", "_"))
}

function Stop-LauncherManagedNapCatStack([string]$CurrentScope, [switch]$FullTree) {
    $state = Read-LauncherState
    if (-not $state) {
        Write-WatchdogLog $CurrentScope "launcher_stack_stop_skipped reason=missing_launcher_state"
        return @()
    }

    $stopped = @()
    if ($state.boot_started_by_launcher) {
        $bootTargets = @(Get-NapCatBootProcesses)
        if ($state.boot_pid) {
            try {
                $bootTargets += @(Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$state.boot_pid) -ErrorAction Stop)
            } catch {
                $fallbackBoot = Get-Process -Id ([int]$state.boot_pid) -ErrorAction SilentlyContinue
                if ($fallbackBoot -and $fallbackBoot.ProcessName -eq "NapCatWinBootMain") {
                    $bootTargets += [pscustomobject]@{
                        ProcessId = [int]$fallbackBoot.Id
                    }
                }
            }
        }
        $bootIds = @($bootTargets | Where-Object { $_ } | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($bootIds.Count -gt 0) {
            $targetIds = $bootIds
            if ($FullTree) {
                $targetIds = @($bootIds + @(Get-ProcessDescendantIds -ParentIds $bootIds) | Select-Object -Unique)
            }
            Stop-TrackedProcesses -Ids $targetIds
            $stopped += ("NapCatWinBootMain=" + ($targetIds -join ","))
        }
    }

    if ($state.qq_started_by_launcher) {
        $qqTargets = @(Get-XiaomachiQQProcesses)
        $qqIds = @($qqTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($qqIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $qqIds
            $stopped += ("QQ.exe=" + ($qqIds -join ","))
        }
    }

    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
    return $stopped
}

function Enter-OneBotManualLoginMode(
    [string]$CurrentScope,
    [array]$ProcessSpecs,
    [string]$Title = "Xiaomachi needs QQ login",
    [string]$Message = "",
    [string]$AlertReason = "manual_login_required",
    [string]$AlertDetail = "QQ account is offline. Use the Xiaomachi startup BAT to start again."
) {
    if (-not $Message) {
        $Message = "QQ account is offline. Xiaomachi has stopped its runtime and QQ/NapCat stack. Start Xiaomachi again with the startup BAT, then finish QQ login if prompted."
    }
    Show-XiaomachiDesktopAlert -CurrentScope $CurrentScope -Title $Title -Message $Message

    $stopped = @(Stop-LauncherManagedNapCatStack -CurrentScope $CurrentScope -FullTree)
    $stopped += @(Stop-OrphanedXiaomachiQQProcesses)
    foreach ($spec in $ProcessSpecs) {
        Stop-BotSpec -Spec $spec
    }
    if ($CurrentScope -ne "worker") {
        Stop-Watchdog -CurrentScope worker
    }

    Write-WatchdogLog $CurrentScope ("onebot_offline_manual_reset_requested stopped={0}" -f (($stopped -join ";") -replace "\s+", "_"))
    Write-XiaomachiAlert -Reason $AlertReason -Detail $AlertDetail
}

function Start-Watchdog([string]$CurrentScope) {
    $existing = Get-WatchdogProcess $CurrentScope
    if ($existing) {
        Write-Host "Xiaomachi $CurrentScope watchdog already running. PID: $($existing.ProcessId)"
        return
    }

    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-Action", "run",
        "-Scope", $CurrentScope,
        "-HeartbeatTimeoutSeconds", [string]$HeartbeatTimeoutSeconds,
        "-StartupGraceSeconds", [string]$StartupGraceSeconds,
        "-CheckIntervalSeconds", [string]$CheckIntervalSeconds,
        "-HeartbeatReadyTimeoutSeconds", [string]$HeartbeatReadyTimeoutSeconds,
        "-RestartMaxAttempts", [string]$RestartMaxAttempts,
        "-RestartWindowSeconds", [string]$RestartWindowSeconds,
        "-RestartSuppressSeconds", [string]$RestartSuppressSeconds,
        "-OneBotStatusProbeIntervalSeconds", [string]$OneBotStatusProbeIntervalSeconds,
        "-OneBotGroupStreamProbeIntervalSeconds", [string]$OneBotGroupStreamProbeIntervalSeconds
    )

    Repair-StartProcessEnvironment
    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $workdir `
        -WindowStyle Hidden `
        -PassThru

    $pidFile = Get-WatchdogPidFile $CurrentScope
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        $spawned = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if (-not $spawned) {
            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
            Write-WatchdogLog $CurrentScope ("watchdog_start_failed reason=process_exited pid={0}" -f $process.Id)
            break
        }

        if (Test-Path $pidFile) {
            $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
            if ($pidValue -eq [string]$process.Id) {
                Write-Host "Xiaomachi $CurrentScope watchdog started. PID: $($process.Id)"
                return
            }
        }
        Start-Sleep -Milliseconds 250
    }

    throw "Failed to start Xiaomachi $CurrentScope watchdog."
}

function Run-Watchdog([string]$CurrentScope) {
    $pidFile = Get-WatchdogPidFile $CurrentScope
    $pythonExe = Resolve-PythonExecutable $workdir
    $processSpecs = @(Get-ScopeProcessSpecs $CurrentScope)
    $restartStateByName = @{}
    $lastOneBotStatusProbeByName = @{}
    $lastOneBotGroupStreamProbeByName = @{}

    Set-Content -Path $pidFile -Value $PID -Encoding ascii
    Write-WatchdogLog $CurrentScope ("watchdog_started scope={0} pid={1}" -f $CurrentScope, $PID)

    try {
        while ($true) {
            if (Test-StopRequested) {
                Write-WatchdogLog $CurrentScope ("watchdog_stop_marker_seen scope={0}" -f $CurrentScope)
                foreach ($spec in $processSpecs) {
                    Stop-BotSpec -Spec $spec
                }
                return
            }

            foreach ($spec in $processSpecs) {
                $status = Get-BotSpecStatus `
                    -Spec $spec `
                    -HeartbeatTimeoutSeconds $HeartbeatTimeoutSeconds `
                    -StartupGraceSeconds $StartupGraceSeconds

                if ($status.needs_restart) {
                    Write-WatchdogLog $CurrentScope ("restart_requested name={0} reason={1} pid={2}" -f $spec.Name, $status.restart_reason, $status.pid)
                    Write-XiaomachiAlert -Reason $status.restart_reason -Detail ("name={0} pid={1}" -f $spec.Name, $status.pid)
                    $restartBudget = Test-BotSpecRestartBudget `
                        -StateByName $restartStateByName `
                        -Name ([string]$spec.Name) `
                        -MaxAttempts $RestartMaxAttempts `
                        -WindowSeconds $RestartWindowSeconds `
                        -SuppressSeconds $RestartSuppressSeconds
                    if (-not $restartBudget.allowed) {
                        Write-WatchdogLog $CurrentScope ("restart_suppressed name={0} reason={1} attempts={2} suppressed_seconds={3}" -f $spec.Name, $status.restart_reason, $restartBudget.attempts, $restartBudget.suppressed_seconds)
                        Write-XiaomachiAlert -Reason "restart_suppressed" -Detail ("name={0} reason={1} attempts={2}" -f $spec.Name, $status.restart_reason, $restartBudget.attempts)
                        continue
                    }

                    Register-BotSpecRestartAttempt -StateByName $restartStateByName -Name ([string]$spec.Name)
                    if ($status.restart_reason -eq "onebot_offline") {
                        Enter-OneBotManualLoginMode $CurrentScope $processSpecs
                        return
                    }
                    if ($status.restart_reason -eq "onebot_group_stream_stale") {
                        Restart-LauncherManagedNapCat $CurrentScope
                        return
                    }
                    $restartResult = Restart-BotSpecSafely `
                        -Workdir $workdir `
                        -PythonExe $pythonExe `
                        -Spec $spec `
                        -HeartbeatReadyTimeoutSeconds $HeartbeatReadyTimeoutSeconds
                    if ($restartResult.restarted) {
                        Write-WatchdogLog $CurrentScope ("restart_completed name={0}" -f $spec.Name)
                    } else {
                        Write-WatchdogLog $CurrentScope ("restart_failed name={0} reason={1} error={2}" -f $spec.Name, $status.restart_reason, $restartResult.error)
                        Write-XiaomachiAlert -Reason "restart_failed" -Detail ("name={0} reason={1} error={2}" -f $spec.Name, $status.restart_reason, $restartResult.error)
                    }
                    continue
                }

                if ($status.is_running -and $status.pid_file_stale) {
                    Repair-BotSpecPidFile -Spec $spec -ProcessId ([int]$status.pid)
                    Write-WatchdogLog $CurrentScope ("pidfile_repaired name={0} pid={1}" -f $spec.Name, $status.pid)
                }
            }

            foreach ($spec in $processSpecs) {
                $nowUtc = (Get-Date).ToUniversalTime()
                if (Test-WatchdogProbeDue -LastProbeByName $lastOneBotStatusProbeByName -Name ([string]$spec.Name) -IntervalSeconds $OneBotStatusProbeIntervalSeconds -NowUtc $nowUtc) {
                    Update-OneBotHealthStateForSpec -Spec $spec -PythonExe $pythonExe
                    Register-WatchdogProbeRun -LastProbeByName $lastOneBotStatusProbeByName -Name ([string]$spec.Name) -NowUtc $nowUtc
                }
                if (Test-WatchdogProbeDue -LastProbeByName $lastOneBotGroupStreamProbeByName -Name ([string]$spec.Name) -IntervalSeconds $OneBotGroupStreamProbeIntervalSeconds -NowUtc $nowUtc) {
                    Update-OneBotGroupStreamHealthStateForSpec -Spec $spec -PythonExe $pythonExe
                    Register-WatchdogProbeRun -LastProbeByName $lastOneBotGroupStreamProbeByName -Name ([string]$spec.Name) -NowUtc $nowUtc
                }
            }

            Start-Sleep -Seconds $CheckIntervalSeconds
        }
    } catch {
        $errorText = ([string]$_.Exception.Message).Trim()
        if ([string]::IsNullOrWhiteSpace($errorText)) {
            $errorText = ([string]$_).Trim()
        }
        $errorText = $errorText -replace "\s+", " "
        Write-WatchdogLog $CurrentScope ("watchdog_error scope={0} error={1}" -f $CurrentScope, $errorText)
        Write-XiaomachiAlert -Reason "watchdog_error" -Detail ("scope={0} error={1}" -f $CurrentScope, $errorText)
        throw
    } finally {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        Write-WatchdogLog $CurrentScope ("watchdog_stopped scope={0}" -f $CurrentScope)
    }
}

function Stop-Watchdog([string]$CurrentScope) {
    $processSpecs = @(Get-ScopeProcessSpecs $CurrentScope)
    $watchdogs = @(Get-WatchdogProcessesForScope $CurrentScope)
    $stopFailures = @()
    if ($watchdogs.Count -le 0) {
        $watchdog = Get-WatchdogProcess $CurrentScope
        if ($watchdog) {
            $watchdogs = @($watchdog)
        }
    }

    $watchdogIds = @($watchdogs | ForEach-Object { [int]$_.ProcessId } | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    foreach ($extra in $watchdogs) {
        Stop-Process -Id ([int]$extra.ProcessId) -Force -ErrorAction SilentlyContinue
    }
    if ($watchdogIds.Count -gt 0) {
        Start-Sleep -Milliseconds 500
        Stop-ChildPythonProcessesForParents -ParentProcessIds $watchdogIds
        Start-Sleep -Milliseconds 500
    }

    foreach ($spec in $processSpecs) {
        $stopResult = Stop-BotSpec -Spec $spec -PassThru
        if ($stopResult -and -not $stopResult.stopped) {
            $remaining = @($stopResult.remaining_ids) -join ","
            $stopFailures += ("{0}={1}" -f $spec.Name, $remaining)
            Write-WatchdogLog $CurrentScope ("bot_stop_failed name={0} remaining_pids={1}" -f $spec.Name, $remaining)
        }
    }

    Remove-Item (Get-WatchdogPidFile $CurrentScope) -Force -ErrorAction SilentlyContinue
    Write-WatchdogLog $CurrentScope ("watchdog_stop_requested scope={0}" -f $CurrentScope)
    if ($stopFailures.Count -gt 0) {
        $detail = "scope={0} remaining={1}" -f $CurrentScope, ($stopFailures -join ";")
        Write-XiaomachiAlert -Reason "bot_stop_failed" -Detail $detail
        Write-Host ("Xiaomachi {0} watchdog stopped, but some bot processes are still running: {1}" -f $CurrentScope, ($stopFailures -join "; "))
        exit 2
    }
    Write-Host "Xiaomachi $CurrentScope watchdog stopped."
}

switch ($Action) {
    "start" { Start-Watchdog $Scope }
    "run" { Run-Watchdog $Scope }
    "stop" { Stop-Watchdog $Scope }
    default { throw "Unsupported watchdog action: $Action" }
}

exit 0
