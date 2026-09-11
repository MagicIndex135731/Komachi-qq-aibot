param(
    [switch]$OnlyWhenLoginRequired,
    [string]$WebUiUrl = "http://127.0.0.1:3080/"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$legacyTokenPath = Join-Path $repoRoot "infra\wsl\runtime\llbot\data\webui_token.txt"
$url = ""

function Get-LLBotWebUiToken {
    param([Parameter(Mandatory = $true)][string]$SourceTokenPath)

    # The live password lives in the Linux runtime data volume; the
    # /opt/xiaomachi/shared copy covers older layouts and the Windows-side file
    # is the last fallback.
    foreach ($command in @(
            'cat /var/lib/docker/volumes/xiaomachi-llbot-data/_data/webui_token.txt',
            'cat /opt/xiaomachi/shared/runtime/llbot/data/webui_token.txt'
        )) {
        try {
            $token = (wsl.exe --user root --exec bash -lc $command 2>$null) -join ""
            if ($token -and $token.Trim()) {
                return $token.Trim()
            }
        } catch {
        }
    }
    if (Test-Path -LiteralPath $SourceTokenPath) {
        return (Get-Content -Raw -LiteralPath $SourceTokenPath).Trim()
    }
    return ""
}

try {
    $uri = [Uri]$WebUiUrl
    if ($uri.Scheme -ne "http" -or $uri.Port -ne 3080 -or [string]::IsNullOrWhiteSpace($uri.Host)) {
        throw "Invalid LLBot WebUI URL."
    }
    $url = $uri.AbsoluteUri
} catch {
    Write-Host "Invalid LLBot WebUI URL: $WebUiUrl"
    exit 1
}

# Reachability is already proven by the launcher (curl.exe against the WSL
# address).  Probing again from PowerShell would go through the WinHTTP/IE
# proxy, which rejects the private WSL address on some machines and used to
# abort this script silently.
$token = Get-LLBotWebUiToken -SourceTokenPath $legacyTokenPath
if ($token) {
    try {
        Set-Clipboard -Value $token
        Write-Host "LLBot WebUI password copied to the clipboard."
    } catch {
        Write-Warning "Could not copy the LLBot WebUI password to the clipboard."
    }
}

try {
    Start-Process $url
} catch {
    Write-Host "Could not open a browser for $url"
    exit 1
}
