[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu_Migrated",
    [string]$Source = "",
    [string]$InstallRoot = "/opt/xiaomachi",
    [switch]$UnmarkedRateOne
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

function ConvertTo-WslMountPath([string]$WindowsPath) {
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only local drive paths can be synchronized into WSL: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

if (-not $Source) {
    $clashDirectory = Join-Path $env:APPDATA "io.github.clash-verge-rev.clash-verge-rev"
    $Source = & python (Join-Path $PSScriptRoot "render_mihomo_config.py") `
        --print-ikuuu-source --clash-directory $clashDirectory
    if ($LASTEXITCODE -ne 0) { throw "Cannot identify one ikuuu subscription; specify its Source explicitly." }
}
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Clash Verge merged profile not found: $Source"
}

$cacheDir = Join-Path $repoRoot "infra\wsl\.cache"
$rendered = Join-Path $cacheDir "mihomo-config.yaml"
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

try {
    $renderArguments = @('--source', $Source, '--output', $rendered, '--page-only')
    if ($UnmarkedRateOne) { $renderArguments += '--unmarked-rate-one' }
    & python (Join-Path $PSScriptRoot "render_mihomo_config.py") @renderArguments
    if ($LASTEXITCODE -ne 0) { throw "Mihomo profile rendering failed." }

    $wslRendered = ConvertTo-WslMountPath $rendered
    & wsl.exe -d $Distro -- sudo install -d -m 0700 "$InstallRoot/shared/mihomo"
    if ($LASTEXITCODE -ne 0) { throw "Cannot create the private Mihomo directory." }
    & wsl.exe -d $Distro -- sudo install -m 0600 $wslRendered "$InstallRoot/shared/mihomo/config.yaml.next"
    if ($LASTEXITCODE -ne 0) { throw "Cannot install the private Mihomo profile." }

    $assetMap = @{
        "Country.mmdb" = "Country.mmdb"
        "ASN.mmdb" = "ASN.mmdb"
        "geoip.dat" = "GeoIP.dat"
        "geosite.dat" = "GeoSite.dat"
    }
    $sourceDir = Split-Path -Parent $Source
    if ((Split-Path -Leaf $sourceDir) -eq 'profiles') {
        $sourceDir = Split-Path -Parent $sourceDir
    }
    foreach ($entry in $assetMap.GetEnumerator()) {
        $asset = Join-Path $sourceDir $entry.Key
        if (Test-Path -LiteralPath $asset -PathType Leaf) {
            $wslAsset = ConvertTo-WslMountPath $asset
            & wsl.exe -d $Distro -- sudo install -m 0644 $wslAsset "$InstallRoot/shared/mihomo/$($entry.Value)"
            if ($LASTEXITCODE -ne 0) { throw "Cannot install Mihomo rule asset: $($entry.Key)" }
        }
    }

    $installer = Join-Path $PSScriptRoot "install_mihomo.sh"
    $wslInstaller = ConvertTo-WslMountPath $installer
    & wsl.exe -d $Distro -- bash $wslInstaller
    if ($LASTEXITCODE -ne 0) { throw "Mihomo installation or validation failed." }
}
finally {
    Remove-Item -LiteralPath $rendered -Force -ErrorAction SilentlyContinue
}

Write-Host "WSL Mihomo now owns Xiaomachi provider routing; Windows Clash remains independent."
