param(
    [ValidateSet("start", "run", "stop")]
    [string]$Action = "start",

    [ValidateSet("runtime", "worker", "all")]
    [string]$Scope = "runtime",

    [int]$HeartbeatTimeoutSeconds = 45,
    [int]$StartupGraceSeconds = 20,
    [int]$CheckIntervalSeconds = 5
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $workdir "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $workdir "scripts\xiaomachi_process_helpers.ps1")

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

function Get-WatchdogProcess([string]$CurrentScope) {
    $pidFile = Get-WatchdogPidFile $CurrentScope
    if (-not (Test-Path $pidFile)) {
        return $null
    }

    $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
    if (-not ($pidValue -match "^\d+$")) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        return $null
    }

    $process = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$pidValue) -ErrorAction SilentlyContinue
    if (
        $process -and
        $process.Name -like "powershell*" -and
        $process.CommandLine -like "*xiaomachi_watchdog.ps1*" -and
        $process.CommandLine -like "*-Action run*" -and
        $process.CommandLine -like "*-Scope $CurrentScope*"
    ) {
        return $process
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
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
        "-CheckIntervalSeconds", [string]$CheckIntervalSeconds
    )

    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $workdir `
        -WindowStyle Hidden `
        -PassThru

    $pidFile = Get-WatchdogPidFile $CurrentScope
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
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

    Set-Content -Path $pidFile -Value $PID -Encoding ascii
    Write-WatchdogLog $CurrentScope ("watchdog_started scope={0} pid={1}" -f $CurrentScope, $PID)

    try {
        while ($true) {
            foreach ($spec in $processSpecs) {
                $status = Get-BotSpecStatus `
                    -Spec $spec `
                    -HeartbeatTimeoutSeconds $HeartbeatTimeoutSeconds `
                    -StartupGraceSeconds $StartupGraceSeconds

                if ($status.needs_restart) {
                    Write-WatchdogLog $CurrentScope ("restart_requested name={0} reason={1} pid={2}" -f $spec.Name, $status.restart_reason, $status.pid)
                    Stop-BotSpec -Spec $spec
                    Start-BotSpec -Workdir $workdir -PythonExe $pythonExe -Spec $spec
                    Write-WatchdogLog $CurrentScope ("restart_completed name={0}" -f $spec.Name)
                    continue
                }

                if ($status.is_running -and $status.pid_file_stale) {
                    Repair-BotSpecPidFile -Spec $spec -Pid ([int]$status.pid)
                    Write-WatchdogLog $CurrentScope ("pidfile_repaired name={0} pid={1}" -f $spec.Name, $status.pid)
                }
            }

            Start-Sleep -Seconds $CheckIntervalSeconds
        }
    } finally {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        Write-WatchdogLog $CurrentScope ("watchdog_stopped scope={0}" -f $CurrentScope)
    }
}

function Stop-Watchdog([string]$CurrentScope) {
    $processSpecs = @(Get-ScopeProcessSpecs $CurrentScope)
    $watchdog = Get-WatchdogProcess $CurrentScope
    if ($watchdog) {
        Stop-Process -Id $watchdog.ProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    foreach ($spec in $processSpecs) {
        Stop-BotSpec -Spec $spec
    }

    Remove-Item (Get-WatchdogPidFile $CurrentScope) -Force -ErrorAction SilentlyContinue
    Write-WatchdogLog $CurrentScope ("watchdog_stop_requested scope={0}" -f $CurrentScope)
    Write-Host "Xiaomachi $CurrentScope watchdog stopped."
}

switch ($Action) {
    "start" { Start-Watchdog $Scope }
    "run" { Run-Watchdog $Scope }
    "stop" { Stop-Watchdog $Scope }
    default { throw "Unsupported watchdog action: $Action" }
}

exit 0
