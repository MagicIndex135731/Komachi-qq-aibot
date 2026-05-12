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
$StdOutLog = Join-Path $StateDir "public_release_sync.out.log"
$StdErrLog = Join-Path $StateDir "public_release_sync.err.log"
$WatchScript = Join-Path $PSScriptRoot "watch_public_release.ps1"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

if (Test-Path -LiteralPath $PidFile) {
    $existingPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($existingPid) {
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) {
            Write-Output "Public release sync watcher already running (PID $existingPid)."
            exit 0
        }
    }
    Remove-Item -LiteralPath $PidFile -Force
}

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $WatchScript,
        "-SourceRoot", $SourceRoot
    ) `
    -WorkingDirectory $SourceRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $StdOutLog `
    -RedirectStandardError $StdErrLog

Set-Content -LiteralPath $PidFile -Value $process.Id -NoNewline
Write-Output "Started public release sync watcher (PID $($process.Id))."
