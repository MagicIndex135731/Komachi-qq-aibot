param(
    [switch]$OnlyWhenLoginRequired,
    [string]$WebUiUrl = "http://127.0.0.1:3080/"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$tokenPath = Join-Path $repoRoot "infra\wsl\runtime\llbot\data\webui_token.txt"
$url = ""

try {
    $uri = [Uri]$WebUiUrl
    if ($uri.Scheme -ne "http" -or $uri.Port -ne 3080 -or [string]::IsNullOrWhiteSpace($uri.Host)) {
        throw "Invalid LLBot WebUI URL."
    }
    $url = $uri.AbsoluteUri
    Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3 | Out-Null
} catch {
    exit 1
}

if (Test-Path -LiteralPath $tokenPath) {
    $token = (Get-Content -Raw -LiteralPath $tokenPath).Trim()
    if ($token) {
        try {
            Set-Clipboard -Value $token
            Write-Host "LLBot WebUI password copied to the clipboard."
        } catch {
            Write-Warning "Could not copy the LLBot WebUI password to the clipboard."
        }
    }
}

Start-Process $url
