$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $workdir "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $workdir "scripts\xiaomachi_process_helpers.ps1")

$pythonExe = Resolve-PythonExecutable $workdir
$processSpecs = @(
    @{
        Name = "group"
        Module = "app.group_main"
        PidFile = Join-Path $logDir "group.pid"
        Stdout = Join-Path $logDir "group.stdout.log"
        Stderr = Join-Path $logDir "group.stderr.log"
    },
    @{
        Name = "private"
        Module = "app.private_main"
        PidFile = Join-Path $logDir "private.pid"
        Stdout = Join-Path $logDir "private.stdout.log"
        Stderr = Join-Path $logDir "private.stderr.log"
    }
)

foreach ($spec in $processSpecs) {
    Start-BotSpec -Workdir $workdir -PythonExe $pythonExe -Spec $spec
}

exit 0
