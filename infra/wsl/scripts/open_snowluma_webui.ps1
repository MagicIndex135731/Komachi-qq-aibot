param(
    [switch]$OnlyWhenLoginRequired,
    [string]$WebUiUrl = "http://127.0.0.1:5099/"
)

$ErrorActionPreference = "Stop"
$url = ""

try {
    $uri = [Uri]$WebUiUrl
    if ($uri.Scheme -ne "http" -or $uri.Port -ne 5099 -or [string]::IsNullOrWhiteSpace($uri.Host)) {
        throw "Invalid SnowLuma WebUI URL."
    }
    $url = $uri.AbsoluteUri
} catch {
    Write-Host "Invalid SnowLuma WebUI URL: $WebUiUrl"
    exit 1
}

function Get-SnowLumaWebUiPassword {
    # SnowLuma prints the initial admin password exactly once, on the first
    # start of a fresh /app/data volume.  Later starts reuse the stored account.
    $attempts = @(
        'docker logs xiaomachi-snowluma 2>&1 | grep -E "临时密码|initial credentials" | tail -n 1',
        'docker logs xiaomachi-snowluma 2>&1 | grep -iE "password=" | tail -n 1'
    )
    foreach ($command in $attempts) {
        try {
            $line = (wsl.exe --user root --exec bash -lc $command 2>$null) -join ""
            $line = $line.Trim()
            if (-not $line) { continue }
            Write-Host "SnowLuma login hint: $line"
            $parts = $line -split '\s+'
            if ($parts.Count -ge 1) {
                return $parts[-1].Trim()
            }
        } catch {
        }
    }
    return ""
}

# Reachability is proven by the launcher (curl.exe against the WSL address);
# probing again from PowerShell would use the WinHTTP/IE proxy.
$password = Get-SnowLumaWebUiPassword
if ($password) {
    try {
        Set-Clipboard -Value $password
        Write-Host "SnowLuma WebUI password copied to the clipboard (user: admin)."
    } catch {
        Write-Warning "Could not copy the SnowLuma WebUI password to the clipboard."
    }
} else {
    Write-Host "SnowLuma WebUI password is not in the container logs; check the WebUI account you set on first login."
}

Write-Host "SnowLuma WebUI: $url"
Write-Host "QQ client desktop (login QR fallback) via noVNC: http://${($uri.Host)}:6081/"

try {
    Start-Process $url
} catch {
    Write-Host "Could not open a browser for $url"
    exit 1
}
