$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $workdir "data\logs"

. (Join-Path $workdir "scripts\xiaomachi_process_helpers.ps1")

$processSpecs = @(
    @{
        Name = "group"
        Module = "app.group_main"
        PidFile = Join-Path $logDir "group.pid"
    },
    @{
        Name = "private"
        Module = "app.private_main"
        PidFile = Join-Path $logDir "private.pid"
    },
    @{
        Name = "worker"
        Module = "app.dev_worker_main"
        PidFile = Join-Path $logDir "worker.pid"
    }
)

foreach ($spec in $processSpecs) {
    Stop-BotSpec -Spec $spec
}

Write-Host "All Xiaomachi bot processes stopped."
exit 0
