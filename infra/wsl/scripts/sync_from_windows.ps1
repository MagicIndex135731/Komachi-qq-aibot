$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslDir = Split-Path -Parent $scriptDir
$workdir = Split-Path -Parent (Split-Path -Parent $wslDir)
$sourceEnv = Join-Path $workdir ".env"
$targetEnv = Join-Path $wslDir ".env"

if (-not (Test-Path $sourceEnv)) {
    throw "Missing Windows .env: $sourceEnv"
}

New-Item -ItemType Directory -Force -Path $wslDir | Out-Null
Copy-Item -LiteralPath $sourceEnv -Destination $targetEnv -Force

$content = Get-Content -Path $targetEnv -Raw -Encoding utf8
$content = $content -replace "(?m)^NAPCAT_WS_URL=.*$", "NAPCAT_WS_URL=ws://napcat:3001"
$content = $content -replace "(?m)^QQ_EXE_PATH=.*$", "# QQ_EXE_PATH is intentionally unused in WSL deployment"
$content = $content -replace "(?m)^QQ_EXTRA_ARGS=.*$", "# QQ_EXTRA_ARGS is intentionally unused in WSL deployment"
$content = $content -replace "(?m)^NAPCAT_SHELL_DIR=.*$", "# NAPCAT_SHELL_DIR is intentionally unused in WSL deployment"
$content = $content -replace "(?m)^NAPCAT_BOOT_PATH=.*$", "# NAPCAT_BOOT_PATH is intentionally unused in WSL deployment"
$content = $content -replace "(?m)^NAPCAT_INJECT_DLL_PATH=.*$", "# NAPCAT_INJECT_DLL_PATH is intentionally unused in WSL deployment"

if ($content -notmatch "(?m)^NAPCAT_WS_URL=") {
    $content = $content.TrimEnd() + "`r`nNAPCAT_WS_URL=ws://napcat:3001`r`n"
}

$content | Set-Content -Path $targetEnv -Encoding utf8

Write-Host "Copied sanitized runtime env to: $targetEnv"
Write-Host "Do not commit infra/wsl/.env."
