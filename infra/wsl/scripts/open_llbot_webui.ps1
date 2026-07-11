param(
    [switch]$OnlyWhenLoginRequired
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$tokenPath = Join-Path $repoRoot "infra\wsl\runtime\llbot\data\webui_token.txt"
$url = "http://127.0.0.1:3080/"

try {
    Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3 | Out-Null
} catch {
    exit 1
}

if (Test-Path -LiteralPath $tokenPath) {
    $token = (Get-Content -Raw -LiteralPath $tokenPath).Trim()
    if ($token) {
        Set-Clipboard -Value $token
        Write-Host "LLBot WebUI password copied to the clipboard."
    }
}

Start-Process $url
