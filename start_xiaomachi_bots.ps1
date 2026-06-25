$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchdogScript = Join-Path $workdir "scripts\xiaomachi_watchdog.ps1"

& powershell -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -Action start -Scope runtime -HeartbeatTimeoutSeconds 180 -OneBotStatusProbeIntervalSeconds 15 -OneBotGroupStreamProbeIntervalSeconds 60
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& powershell -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -Action start -Scope worker
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

exit 0
