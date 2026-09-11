param(
    [Parameter(Mandatory = $true)][string]$DesktopUrl
)

$ErrorActionPreference = "Stop"

try {
    $uri = [Uri]$DesktopUrl
    if ($uri.Scheme -ne "http" -or $uri.Port -ne 6081 -or [string]::IsNullOrWhiteSpace($uri.Host)) {
        throw "Invalid SnowLuma desktop URL."
    }
} catch {
    Write-Host "Invalid SnowLuma desktop URL: $DesktopUrl"
    exit 1
}

Write-Host "Opening the headless QQ desktop (noVNC): $DesktopUrl"
Write-Host "Click 刷新 on the QR window and scan with the phone QQ app."

try {
    Start-Process $DesktopUrl
} catch {
    Write-Host "Could not open a browser for $DesktopUrl"
    exit 1
}
