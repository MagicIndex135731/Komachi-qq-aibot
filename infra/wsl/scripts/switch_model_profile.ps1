[CmdletBinding()]
param(
    [ValidateSet("sol", "gemini")][string]$Profile = "",
    [ValidateSet("minimal", "low", "medium", "high", "tiered")][string]$Reasoning = "",
    [ValidateSet("sol", "gemini")][string]$Capture = "",
    [switch]$List,
    [switch]$Check,
    [string]$Distro = ""
)

$ErrorActionPreference = "Stop"
$projectDirectory = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../../..")).Path
$wsl = Join-Path $env:WINDIR "System32/wsl.exe"
if (@(@($Profile, $Capture) | Where-Object { $_ }).Count -gt 1) {
    throw "Choose Profile or Capture, not both."
}
if ($List -and ($Profile -or $Capture -or $Reasoning -or $Check)) {
    throw "List is a standalone read-only action."
}
if (-not $Profile -and -not $Capture -and -not $List) {
    Write-Host "Xiaomachi model profiles"
    Write-Host "  1  Your API: Sol 6.1 low + image 2.5"
    Write-Host "  2  Antigravity: Gemini 3.8 Flash + Gemini image"
    Write-Host "  Q  Quit"
    $choice = Read-Host "Choose 1 or 2"
    switch ($choice.ToLowerInvariant()) {
        "1" { $Profile = "sol" }
        "2" { $Profile = "gemini" }
        "q" { exit 0 }
        default { throw "Invalid selection. No settings were changed." }
    }
}

$wslArguments = @()
if ($Distro) { $wslArguments += @("-d", $Distro) }
$wslArguments += @("--user", "root", "--cd", $projectDirectory, "--exec", "python3", "infra/wsl/scripts/switch_model_profile.py")
if ($List) { $wslArguments += "--list" }
elseif ($Capture) { $wslArguments += @("--capture", $Capture) }
else { $wslArguments += @("--profile", $Profile) }
if ($Reasoning) { $wslArguments += @("--reasoning", $Reasoning) }
if ($Check) { $wslArguments += "--check" }
& $wsl @wslArguments
exit $LASTEXITCODE
