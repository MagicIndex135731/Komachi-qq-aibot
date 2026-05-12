[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$ReleaseRoot,
    [string]$AssetRoot,
    [int]$DebounceMilliseconds = 250,
    [int]$SettleMilliseconds = 100
)

$ErrorActionPreference = "Stop"

$scriptRepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $SourceRoot) {
    $SourceRoot = $scriptRepoRoot
}

$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
if (-not $ReleaseRoot) {
    $sourceDirectory = Get-Item -LiteralPath $SourceRoot
    if ($sourceDirectory.Parent -and $sourceDirectory.Parent.Name -eq ".worktrees" -and $sourceDirectory.Parent.Parent) {
        $ReleaseRoot = Join-Path $sourceDirectory.Parent.Parent.FullName "release\github-public"
    }
    else {
        $ReleaseRoot = Join-Path $SourceRoot "release\github-public"
    }
}
if (-not $AssetRoot) {
    $AssetRoot = Join-Path $SourceRoot "scripts\public_release_assets"
}
$ReleaseRoot = [System.IO.Path]::GetFullPath($ReleaseRoot)
$AssetRoot = [System.IO.Path]::GetFullPath($AssetRoot)
$SyncScript = Join-Path $PSScriptRoot "sync_public_release.py"
$PythonExe = (Get-Command python -ErrorAction Stop).Source
$RecentEvents = @{}
$IgnoredTopLevels = @(
    ".git",
    ".venv",
    ".pytest_cache",
    ".tmp_pytest",
    "__pycache__",
    "release"
)

function Invoke-SyncCli {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("reconcile", "sync", "delete")]
        [string]$Command,
        [string]$RelativePath
    )

    $arguments = @(
        $SyncScript,
        "--source-root", $SourceRoot,
        "--release-root", $ReleaseRoot,
        "--asset-root", $AssetRoot,
        $Command
    )
    if ($RelativePath) {
        $arguments += $RelativePath
    }

    & $PythonExe @arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Public release sync command failed: $Command $RelativePath"
    }
}

function Get-RelativePathValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $normalizedBasePath = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\")
    $normalizedTargetPath = [System.IO.Path]::GetFullPath($TargetPath)
    $baseUri = New-Object System.Uri(($normalizedBasePath + "\"))
    $targetUri = New-Object System.Uri($normalizedTargetPath)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace("/", "\")
}

function Get-RelativePathOrNull {
    param(
        [string]$FullPath
    )

    if ([string]::IsNullOrWhiteSpace($FullPath)) {
        return $null
    }

    $normalizedPath = [System.IO.Path]::GetFullPath($FullPath)
    if ($normalizedPath.StartsWith($ReleaseRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }
    if (-not $normalizedPath.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }

    $relativePath = (Get-RelativePathValue -BasePath $SourceRoot -TargetPath $normalizedPath).Replace("\", "/")
    if ([string]::IsNullOrWhiteSpace($relativePath) -or $relativePath -eq ".") {
        return $null
    }

    $topLevel = $relativePath.Split("/")[0]
    if ($IgnoredTopLevels -contains $topLevel) {
        return $null
    }

    return $relativePath
}

function Test-AndRememberEvent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Verb,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $key = "$Verb|$RelativePath"
    $now = Get-Date

    foreach ($existingKey in @($RecentEvents.Keys)) {
        if (($now - $RecentEvents[$existingKey]).TotalMilliseconds -gt ($DebounceMilliseconds * 20)) {
            $RecentEvents.Remove($existingKey)
        }
    }

    if ($RecentEvents.ContainsKey($key)) {
        $elapsed = ($now - $RecentEvents[$key]).TotalMilliseconds
        if ($elapsed -lt $DebounceMilliseconds) {
            return $false
        }
    }

    $RecentEvents[$key] = $now
    return $true
}

function Handle-FileSystemEvent {
    param(
        [Parameter(Mandatory = $true)]
        $EventRecord
    )

    $eventArgs = $EventRecord.SourceEventArgs
    if ($eventArgs -is [System.IO.RenamedEventArgs]) {
        $oldRelativePath = Get-RelativePathOrNull -FullPath $eventArgs.OldFullPath
        if ($oldRelativePath -and (Test-AndRememberEvent -Verb "delete" -RelativePath $oldRelativePath)) {
            Invoke-SyncCli -Command "delete" -RelativePath $oldRelativePath
        }

        $newRelativePath = Get-RelativePathOrNull -FullPath $eventArgs.FullPath
        if ($newRelativePath -and (Test-AndRememberEvent -Verb "sync" -RelativePath $newRelativePath)) {
            Start-Sleep -Milliseconds $SettleMilliseconds
            Invoke-SyncCli -Command "sync" -RelativePath $newRelativePath
        }
        return
    }

    $relativePath = Get-RelativePathOrNull -FullPath $eventArgs.FullPath
    if (-not $relativePath) {
        return
    }

    $changeType = [string]$eventArgs.ChangeType
    if ($changeType -eq "Deleted") {
        if (Test-AndRememberEvent -Verb "delete" -RelativePath $relativePath) {
            Invoke-SyncCli -Command "delete" -RelativePath $relativePath
        }
        return
    }

    if (Test-AndRememberEvent -Verb "sync" -RelativePath $relativePath) {
        Start-Sleep -Milliseconds $SettleMilliseconds
        Invoke-SyncCli -Command "sync" -RelativePath $relativePath
    }
}

Write-Output "Starting public release sync watcher."
Invoke-SyncCli -Command "reconcile"

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $SourceRoot
$watcher.Filter = "*"
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName `
    -bor [System.IO.NotifyFilters]::DirectoryName `
    -bor [System.IO.NotifyFilters]::LastWrite `
    -bor [System.IO.NotifyFilters]::CreationTime
$watcher.EnableRaisingEvents = $true

$sourceIdentifiers = @(
    "public-release-sync.changed",
    "public-release-sync.created",
    "public-release-sync.deleted",
    "public-release-sync.renamed"
)

Register-ObjectEvent -InputObject $watcher -EventName Changed -SourceIdentifier $sourceIdentifiers[0] | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier $sourceIdentifiers[1] | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Deleted -SourceIdentifier $sourceIdentifiers[2] | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Renamed -SourceIdentifier $sourceIdentifiers[3] | Out-Null

try {
    while ($true) {
        $eventRecord = Wait-Event
        if (-not $eventRecord) {
            continue
        }

        try {
            Handle-FileSystemEvent -EventRecord $eventRecord
        }
        catch {
            Write-Error $_
        }
        finally {
            Remove-Event -EventIdentifier $eventRecord.EventIdentifier -ErrorAction SilentlyContinue
        }
    }
}
finally {
    foreach ($sourceIdentifier in $sourceIdentifiers) {
        Unregister-Event -SourceIdentifier $sourceIdentifier -ErrorAction SilentlyContinue
        Remove-Event -SourceIdentifier $sourceIdentifier -ErrorAction SilentlyContinue
    }
    $watcher.Dispose()
}
