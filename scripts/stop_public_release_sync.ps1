[CmdletBinding()]
param(
    [string]$SourceRoot = ""
)

$ErrorActionPreference = "Stop"

$scriptRepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $SourceRoot) {
    $SourceRoot = $scriptRepoRoot
}

$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
$StateDir = Join-Path $SourceRoot "data\dev_control"
$PidFile = Join-Path $StateDir "public_release_sync.pid"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Output "Public release sync watcher is not running."
    exit 0
}

$existingPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
if ($existingPid) {
    $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existingProcess) {
        Stop-Process -Id $existingPid -Force
        Write-Output "Stopped public release sync watcher (PID $existingPid)."
    }
    else {
        Write-Output "Removed stale public release sync PID file ($existingPid)."
    }
}

Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
