$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $workdir "data\logs"
$pidFile = Join-Path $logDir "xiaomachi.pid"
$stateFile = Join-Path $logDir "launcher_state.json"
$stopRequestFile = Join-Path $logDir "xiaomachi.stop_requested.json"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

@{
    requested_at = (Get-Date).ToUniversalTime().ToString("o")
    pid = $PID
} | ConvertTo-Json -Compress | Set-Content -Path $stopRequestFile -Encoding utf8

function Get-BotProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*" -and $_.CommandLine -like "*-m app.main*"
    }
}

function Read-LauncherState {
    if (-not (Test-Path $stateFile)) {
        return $null
    }

    try {
        return (Get-Content -Path $stateFile -Raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Stop-TrackedProcesses([int[]]$Ids) {
    foreach ($id in ($Ids | Where-Object { $_ -gt 0 } | Select-Object -Unique)) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
}

function Get-ProcessDescendantIds([int[]]$ParentIds) {
    $allProcesses = @()
    try {
        $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        return @()
    }

    $pending = @($ParentIds | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    $descendants = @()
    while ($pending.Count -gt 0) {
        $parentId = [int]$pending[0]
        $pending = @($pending | Select-Object -Skip 1)
        $children = @($allProcesses | Where-Object { [int]$_.ParentProcessId -eq $parentId })
        foreach ($child in $children) {
            $childId = [int]$child.ProcessId
            if ($descendants -notcontains $childId) {
                $descendants += $childId
                $pending += $childId
            }
        }
    }

    return $descendants
}

function Stop-NapCatProcessTrees {
    $bootTargets = @(Get-NapCatBootProcesses)
    $bootIds = @($bootTargets | Where-Object { $_ } | ForEach-Object { [int]$_.ProcessId } | Select-Object -Unique)
    if ($bootIds.Count -le 0) {
        return [pscustomobject]@{
            stopped = $true
            detail = ""
            remaining_ids = @()
        }
    }

    $treeIds = @($bootIds + @(Get-ProcessDescendantIds -ParentIds $bootIds) | Select-Object -Unique)
    Stop-TrackedProcesses -Ids $treeIds
    foreach ($bootId in $bootIds) {
        Start-Process `
            -FilePath "taskkill.exe" `
            -ArgumentList @("/T", "/F", "/PID", [string]$bootId) `
            -WindowStyle Hidden `
            -Wait | Out-Null
    }
    Start-Sleep -Milliseconds 800
    $remainingIds = @($treeIds | Where-Object { Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue })
    if ($remainingIds.Count -gt 0) {
        return [pscustomobject]@{
            stopped = $false
            detail = ("napcat_stop_failed remaining_pids={0}" -f ($remainingIds -join ","))
            remaining_ids = $remainingIds
        }
    }

    return [pscustomobject]@{
        stopped = $true
        detail = ("napcat_process_tree=" + ($treeIds -join ","))
        remaining_ids = @()
    }
}

function Get-NapCatBootProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -eq "NapCatWinBootMain.exe"
        })
    } catch {
        return @(Get-Process -Name "NapCatWinBootMain" -ErrorAction SilentlyContinue | ForEach-Object {
            [pscustomobject]@{
                ProcessId = [int]$_.Id
            }
        })
    }
}

function Get-QQProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -eq "QQ.exe" })
    } catch {
        return @(Get-Process -Name QQ -ErrorAction SilentlyContinue | ForEach-Object {
            [pscustomobject]@{
                ProcessId = [int]$_.Id
            }
        })
    }
}

function Get-XiaomachiQQProcesses {
    $qqPath = ""
    $envFile = Join-Path $workdir ".env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content -Path $envFile -Encoding utf8) {
            $trimmed = $line.Trim()
            if ($trimmed -match "^QQ_EXE_PATH\s*=\s*(.+)$") {
                $qqPath = $Matches[1].Trim().Trim('"').Trim("'")
                break
            }
        }
    }

    $workdirPrefix = ($workdir.TrimEnd("\") + "\")
    @(Get-QQProcesses) | Where-Object {
        $exePath = [string]$_.ExecutablePath
        $commandLine = [string]$_.CommandLine
        (
            (-not [string]::IsNullOrWhiteSpace($qqPath) -and $exePath -ieq $qqPath) -or
            $exePath.StartsWith($workdirPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
            $commandLine.IndexOf($workdir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $commandLine.IndexOf("xiaomachi-qq-", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        )
    }
}

function Stop-OrphanedXiaomachiQQProcesses {
    $qqTargets = @(Get-XiaomachiQQProcesses)
    $qqIds = @($qqTargets | Where-Object { $_ } | ForEach-Object { [int]$_.ProcessId } | Select-Object -Unique)
    if ($qqIds.Count -le 0) {
        return [pscustomobject]@{
            stopped = $true
            detail = ""
            remaining_ids = @()
        }
    }

    Stop-TrackedProcesses -Ids $qqIds
    Start-Sleep -Milliseconds 500
    $remainingIds = @($qqIds | Where-Object { Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue })
    if ($remainingIds.Count -gt 0) {
        return [pscustomobject]@{
            stopped = $false
            detail = ("orphaned_xiaomachi_qq_stop_failed remaining_pids={0}" -f ($remainingIds -join ","))
            remaining_ids = $remainingIds
        }
    }

    return [pscustomobject]@{
        stopped = $true
        detail = ("orphaned_xiaomachi_qq=" + ($qqIds -join ","))
        remaining_ids = @()
    }
}

function Get-NapCatBatchProcesses {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -eq "cmd.exe" -and $_.CommandLine -like "*launcher-user.bat*"
        })
    } catch {
        return @()
    }
}

$botStopScript = Join-Path $workdir "stop_xiaomachi_bots.ps1"
& powershell -ExecutionPolicy Bypass -File $botStopScript
$botStopExitCode = $LASTEXITCODE
$botStopped = $LASTEXITCODE -eq 0

$extraStopped = @()
$stopFailures = @()
$napcatStopResult = Stop-NapCatProcessTrees
if ($napcatStopResult.detail) {
    if ($napcatStopResult.stopped) {
        $extraStopped += $napcatStopResult.detail
    } else {
        $stopFailures += $napcatStopResult.detail
    }
}
$orphanedQQStopResult = Stop-OrphanedXiaomachiQQProcesses
if ($orphanedQQStopResult.detail) {
    if ($orphanedQQStopResult.stopped) {
        $extraStopped += $orphanedQQStopResult.detail
    } else {
        $stopFailures += $orphanedQQStopResult.detail
    }
}

$state = Read-LauncherState
if ($state) {
    if ($state.boot_started_by_launcher) {
        $bootTargets = @(Get-NapCatBootProcesses)
        if ($state.boot_pid) {
            try {
                $bootTargets += @(Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$state.boot_pid) -ErrorAction Stop)
            } catch {
                $fallbackBoot = Get-Process -Id ([int]$state.boot_pid) -ErrorAction SilentlyContinue
                if ($fallbackBoot -and $fallbackBoot.ProcessName -eq "NapCatWinBootMain") {
                    $bootTargets += [pscustomobject]@{
                        ProcessId = [int]$fallbackBoot.Id
                    }
                }
            }
        }
        $bootIds = @($bootTargets | Where-Object { $_ } | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($bootIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $bootIds
            $extraStopped += ("NapCatWinBootMain=" + ($bootIds -join ","))
        }
    }

    if ($state.qq_started_by_launcher) {
        $qqTargets = @(Get-QQProcesses)
        $qqIds = @($qqTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($qqIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $qqIds
            $extraStopped += ("QQ.exe=" + ($qqIds -join ","))
        }
    }

    $batchTargets = @(Get-NapCatBatchProcesses)
    $batchIds = @($batchTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
    if ($batchIds.Count -gt 0) {
        Stop-TrackedProcesses -Ids $batchIds
        $extraStopped += ("launcher-user.bat=" + ($batchIds -join ","))
    }

    if ($extraStopped.Count -gt 0) {
        Write-Host ("Launcher-managed components stopped: " + ($extraStopped -join "; "))
    }
} elseif ($extraStopped.Count -gt 0) {
    Write-Host ("Launcher-managed components stopped: " + ($extraStopped -join "; "))
}

Start-Sleep -Seconds 1
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
Remove-Item $stopRequestFile -Force -ErrorAction SilentlyContinue

if (-not $botStopped -and -not $state) {
    exit $botStopExitCode
}

if ($botStopExitCode -ne 0) {
    exit $botStopExitCode
}

if ($stopFailures.Count -gt 0) {
    Write-Host ("Failed to stop launcher-managed components: " + ($stopFailures -join "; "))
    exit 2
}

exit 0
