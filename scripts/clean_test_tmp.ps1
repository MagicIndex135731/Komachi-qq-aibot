param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tmpRoot = Join-Path $repoRoot ".test-tmp"

if (-not (Test-Path -LiteralPath $tmpRoot)) {
    Write-Host "No .test-tmp directory under $repoRoot; nothing to clean."
    exit 0
}

$resolved = (Resolve-Path -LiteralPath $tmpRoot).Path
if ((Split-Path -Parent $resolved) -ne $repoRoot) {
    throw "Refusing to clean unexpected path: $resolved"
}

$items = @(Get-ChildItem -LiteralPath $resolved -Force)
if ($items.Count -eq 0) {
    Write-Host ".test-tmp is already empty."
    exit 0
}

$bytes = ($items |
    ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue } |
    Measure-Object -Property Length -Sum).Sum
$sizeMb = [math]::Round(([double]$bytes) / 1MB, 1)

if (-not $Force) {
    Write-Host ("Would remove {0} entries ({1} MB) under {2}." -f $items.Count, $sizeMb, $resolved)
    Write-Host "Re-run with -Force to delete them."
    exit 0
}

$failed = 0
foreach ($item in $items) {
    try {
        Remove-Item -LiteralPath $item.FullName -Recurse -Force -ErrorAction Stop
    } catch {
        $failed++
        Write-Warning ("Could not remove {0}: {1}" -f $item.Name, $_.Exception.Message)
    }
}

Write-Host ("Removed {0} of {1} entries ({2} MB)." -f ($items.Count - $failed), $items.Count, $sizeMb)
