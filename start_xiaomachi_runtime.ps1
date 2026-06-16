$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$watchdogScript = Join-Path $workdir "scripts\xiaomachi_watchdog.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -Action start -Scope runtime

exit 0
