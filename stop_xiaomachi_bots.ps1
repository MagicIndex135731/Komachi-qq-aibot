$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchdogScript = Join-Path $workdir "scripts\xiaomachi_watchdog.ps1"

& powershell -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -Action stop -Scope runtime
$runtimeStopExitCode = $LASTEXITCODE
& powershell -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -Action stop -Scope worker
$workerStopExitCode = $LASTEXITCODE

if ($runtimeStopExitCode -ne 0 -or $workerStopExitCode -ne 0) {
    Write-Host ("Failed to stop all Xiaomachi bot processes. runtime_exit={0} worker_exit={1}" -f $runtimeStopExitCode, $workerStopExitCode)
    exit 2
}

Write-Host "All Xiaomachi bot processes stopped."
exit 0
