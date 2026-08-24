param(
    [switch]$OnlyWhenLoginRequired,
    [string]$WebUiUrl = "http://127.0.0.1:3080/"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$tokenPath = Join-Path $repoRoot "infra\wsl\runtime\llbot\data\webui_token.txt"
$url = ""

function Get-LLBotWebUiToken {
    param([Parameter(Mandatory = $true)][string]$SourceTokenPath)

    if (Test-Path -LiteralPath $SourceTokenPath) {
        return (Get-Content -Raw -LiteralPath $SourceTokenPath).Trim()
    }

    # Immutable WSL releases keep the login state in /opt/xiaomachi/shared,
    # not beside the Windows source checkout that launched this script.
    try {
        $token = wsl.exe --user root --exec bash -lc 'cat /opt/xiaomachi/shared/runtime/llbot/data/webui_token.txt' 2>$null
        return $token.Trim()
    }
    catch {
        return ""
    }
}

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

$token = Get-LLBotWebUiToken -SourceTokenPath $tokenPath
if ($token) {
    try {
        Set-Clipboard -Value $token
        Write-Host "LLBot WebUI password copied to the clipboard."
    } catch {
        Write-Warning "Could not copy the LLBot WebUI password to the clipboard."
    }
}

Start-Process $url
