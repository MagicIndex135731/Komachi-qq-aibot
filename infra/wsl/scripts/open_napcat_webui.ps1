$ErrorActionPreference = "Stop"

try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
    $configPath = Join-Path $repoRoot "infra\wsl\runtime\napcat\config\webui.json"
    $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
    $token = [string]$config.token
    if ([string]::IsNullOrWhiteSpace($token)) {
        exit 1
    }

    $encodedToken = [uri]::EscapeDataString($token)
    $url = "http://127.0.0.1:6099/webui/qq_login?token=$encodedToken"
    Start-Process -FilePath $url
}
catch {
    exit 1
}
