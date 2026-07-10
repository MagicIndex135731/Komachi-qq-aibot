$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslDir = Split-Path -Parent $scriptDir
$workdir = Split-Path -Parent (Split-Path -Parent $wslDir)
$source = Join-Path $workdir ".env"
$target = Join-Path $wslDir "runtime\redacted-env.snapshot"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null

if (-not (Test-Path $source)) {
    throw "Missing .env: $source"
}

$secretPattern = "(?i)(KEY|TOKEN|SECRET|PASSWORD|BASE_URL|API)"
$lines = foreach ($line in Get-Content -Path $source -Encoding utf8) {
    if ($line -match "^\s*#" -or $line -notmatch "=") {
        $line
        continue
    }
    $key = $line.Substring(0, $line.IndexOf("=")).Trim()
    if ($key -match $secretPattern) {
        "$key=<redacted>"
    } else {
        $line
    }
}

$lines | Set-Content -Path $target -Encoding utf8
Write-Host "Wrote redacted env snapshot: $target"
