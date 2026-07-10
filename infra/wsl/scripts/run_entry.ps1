param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Write-Host "Runner=$PSCommandPath"
Write-Host "RepoRoot=$repoRoot"
$escapedRepoRoot = $repoRoot.Replace("\", "\\")
$wslPathOutput = & wsl.exe wslpath -a $escapedRepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to convert repository path to a WSL path: $repoRoot"
}
$wslRepo = ($wslPathOutput | Select-Object -First 1).Trim()
if ([string]::IsNullOrWhiteSpace($wslRepo)) {
    throw "Failed to convert repository path to a WSL path: $repoRoot"
}
Write-Host "WslRepo=$wslRepo"

function Quote-BashSingle {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$command = "cd $(Quote-BashSingle $wslRepo) && bash infra/wsl/scripts/$Action.sh"
& wsl.exe bash -lc $command
exit $LASTEXITCODE
