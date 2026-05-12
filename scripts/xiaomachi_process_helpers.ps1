$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

function Resolve-PythonExecutable([string]$Workdir) {
    $candidates = @(
        (Join-Path $Workdir ".venv\Scripts\python.exe"),
        "C:\work\anaconda\python.exe"
    )

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Python executable not found. Install the venv or update the launcher scripts."
}

function Get-BotProcessesForModule([string]$ModuleName) {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*" -and $_.CommandLine -like "*-m $ModuleName*"
    }
}

function Stop-BotSpec([hashtable]$Spec) {
    $targets = @()
    $pidValue = ""
    if (Test-Path $Spec.PidFile) {
        $pidValue = (Get-Content $Spec.PidFile -ErrorAction SilentlyContinue | Select-Object -First 1 | Out-String).Trim()
        if ($pidValue -match "^\d+$") {
            $target = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$pidValue) -ErrorAction SilentlyContinue
            if ($target -and $target.Name -like "python*" -and $target.CommandLine -like "*-m $($Spec.Module)*") {
                $targets += $target
            }
        }
    }

    foreach ($process in @(Get-BotProcessesForModule $Spec.Module)) {
        if (-not ($targets | Where-Object { $_.ProcessId -eq $process.ProcessId })) {
            $targets += $process
        }
    }

    foreach ($process in $targets) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
}

function Start-BotSpec([string]$Workdir, [string]$PythonExe, [hashtable]$Spec) {
    $running = @(Get-BotProcessesForModule $Spec.Module)
    if ($running.Count -gt 1) {
        foreach ($process in $running) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 1
        $running = @()
    }

    if ($running.Count -eq 1) {
        Set-Content -Path $Spec.PidFile -Value $running[0].ProcessId -Encoding ascii
        Write-Host "$($Spec.Name) already running. PID: $($running[0].ProcessId)"
        return
    }

    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList "-m", $Spec.Module `
        -WorkingDirectory $Workdir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Spec.Stdout `
        -RedirectStandardError $Spec.Stderr `
        -PassThru

    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Remove-Item $Spec.PidFile -Force -ErrorAction SilentlyContinue
        throw "$($Spec.Name) failed to start. Check log: $($Spec.Stderr)"
    }

    Set-Content -Path $Spec.PidFile -Value $proc.Id -Encoding ascii
    Write-Host "$($Spec.Name) started. PID: $($proc.Id)"
}
