param(
    [switch]$Force,
    [switch]$Legacy,
    [string]$ReportPath
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$workRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "data/test-work"))
$targets = @()
foreach ($parent in @((Join-Path $repoRoot "data"), $workRoot)) {
    if ((Test-Path -LiteralPath $parent) -and
        ((Get-Item -LiteralPath $parent -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing linked test workspace: $parent"
    }
}
if (Test-Path -LiteralPath $workRoot) {
    $targets += @(Get-ChildItem -LiteralPath $workRoot -Force)
}
if ($Legacy) {
    $targets += @(Get-ChildItem -LiteralPath $repoRoot -Force | Where-Object {
        $_.PSIsContainer -and $_.Name -match '^(\.test-tmp.*|\.pytest.*|\.tmp_pytest.*|\.codex_tmp.*|dbg_service_.*|\.artifacts)$'
    })
}
$rows = @()
foreach ($target in $targets) {
    $path = (Resolve-Path -LiteralPath $target.FullName).Path
    $parent = Split-Path -Parent $path
    if (($parent -ne $workRoot -and (-not $Legacy -or $parent -ne $repoRoot)) -or
        ($target.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing unexpected cleanup target: $path"
    }
    $bytes = if ($target.PSIsContainer) {
        (Get-ChildItem -LiteralPath $path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    } else { $target.Length }
    $row = [ordered]@{ path = $path; bytes = [long]$bytes; status = "preview" }
    if ($Force) {
        try {
            Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
            $row.status = "removed"
        } catch {
            $row.status = "failed"
            $row.error = $_.Exception.Message
            Write-Warning "Could not remove $path"
        }
    }
    $rows += [pscustomobject]$row
}
$report = [ordered]@{
    recorded_at = [DateTimeOffset]::Now.ToString("o")
    applied = [bool]$Force
    legacy = [bool]$Legacy
    entries = $rows
    bytes = [long](($rows | Measure-Object -Property bytes -Sum).Sum)
    failures = @($rows | Where-Object status -eq "failed").Count
}
if ($ReportPath) {
    $absoluteReport = [IO.Path]::GetFullPath($ReportPath)
    if (-not $absoluteReport.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup report must be inside the repository"
    }
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $absoluteReport -Encoding utf8
}
Write-Host ("{0}: {1} entries, {2:N1} MiB, {3} failures." -f
    $(if ($Force) { "Cleanup" } else { "Preview" }), $rows.Count, ($report.bytes / 1MB), $report.failures)
if (-not $Force) { Write-Host "Re-run with -Force to delete; -Legacy also includes old root test folders." }
if ($report.failures) { exit 1 }
