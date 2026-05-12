$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $workdir "data\logs"
$pidFile = Join-Path $logDir "xiaomachi.pid"
$stateFile = Join-Path $logDir "launcher_state.json"

function Get-BotProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*" -and $_.CommandLine -like "*-m app.main*"
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

function Get-NapCatBatchProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "cmd.exe" -and $_.CommandLine -like "*launcher-user.bat*"
    }
}

$botStopScript = Join-Path $workdir "stop_xiaomachi_bots.ps1"
& powershell -ExecutionPolicy Bypass -File $botStopScript
$botStopped = $LASTEXITCODE -eq 0

$state = Read-LauncherState
if ($state) {
    $extraStopped = @()

    if ($state.boot_started_by_launcher) {
        $bootTargets = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "NapCatWinBootMain.exe" })
        if ($state.boot_pid) {
            $bootTargets += @(Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$state.boot_pid) -ErrorAction SilentlyContinue)
        }
        $bootIds = @($bootTargets | Where-Object { $_ } | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($bootIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $bootIds
            $extraStopped += ("NapCatWinBootMain=" + ($bootIds -join ","))
        }
    }

    if ($state.qq_started_by_launcher) {
        $qqTargets = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "QQ.exe" })
        $qqIds = @($qqTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($qqIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $qqIds
            $extraStopped += ("QQ.exe=" + ($qqIds -join ","))
        }
    }

    $batchTargets = @(Get-NapCatBatchProcesses)
    $batchIds = @($batchTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
    if ($batchIds.Count -gt 0) {
        Stop-TrackedProcesses -Ids $batchIds
        $extraStopped += ("launcher-user.bat=" + ($batchIds -join ","))
    }

    if ($extraStopped.Count -gt 0) {
        Write-Host ("Launcher-managed components stopped: " + ($extraStopped -join "; "))
    }
}

Start-Sleep -Seconds 1
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Remove-Item $stateFile -Force -ErrorAction SilentlyContinue

if (-not $botStopped -and -not $state) {
    exit 0
}

exit 0
