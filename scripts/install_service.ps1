$ErrorActionPreference = "Stop"

$serviceName = "QQAIBot"
$root = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $root "scripts\run_service.ps1"

nssm status $serviceName 2>$null
$serviceExists = $LASTEXITCODE -eq 0

if (-not $serviceExists) {
    nssm install $serviceName "powershell.exe" "-ExecutionPolicy Bypass -File `"$runner`""
} else {
    nssm stop $serviceName 2>$null
    nssm set $serviceName Application "powershell.exe"
    nssm set $serviceName AppParameters "-ExecutionPolicy Bypass -File `"$runner`""
}

nssm set $serviceName Start SERVICE_AUTO_START
nssm start $serviceName
