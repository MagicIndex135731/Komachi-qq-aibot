$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $workdir "stop_xiaomachi_runtime.ps1")
Start-Sleep -Milliseconds 800
& (Join-Path $workdir "start_xiaomachi_runtime.ps1")

Write-Host "Xiaomachi runtime restart handoff finished."
exit 0
