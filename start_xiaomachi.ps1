$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $workdir "data\logs"
$pidFile = Join-Path $logDir "xiaomachi.pid"
$stateFile = Join-Path $logDir "launcher_state.json"
$stdout = Join-Path $logDir "runtime.stdout.log"
$stderr = Join-Path $logDir "runtime.stderr.log"
$envFile = Join-Path $workdir ".env"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Read-DotEnvFile([string]$Path) {
    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content -Path $Path -Encoding utf8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $separatorIndex = $trimmed.IndexOf("=")
        if ($separatorIndex -lt 1) {
            continue
        }

        $key = $trimmed.Substring(0, $separatorIndex).Trim()
        $value = $trimmed.Substring($separatorIndex + 1).Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        $values[$key] = $value
    }

    return $values
}

$dotenv = Read-DotEnvFile $envFile

function Get-ConfigValue([string]$Key, [string]$Default = "") {
    $runtimeValue = [Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($runtimeValue)) {
        return $runtimeValue
    }

    if ($dotenv.ContainsKey($Key)) {
        $fileValue = [string]$dotenv[$Key]
        if (-not [string]::IsNullOrWhiteSpace($fileValue)) {
            return $fileValue
        }
    }

    return $Default
}

function Test-ConfigFlag([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    return @("1", "true", "yes", "on") -contains $Value.Trim().ToLowerInvariant()
}

function Save-LauncherState([hashtable]$State) {
    $State | ConvertTo-Json -Depth 4 | Set-Content -Path $stateFile -Encoding utf8
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

function Remove-LauncherState {
    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
}

function Get-BotProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like "python*" -and $_.CommandLine -like "*-m app.main*"
    }
}

function Stop-BotProcesses([System.Collections.IEnumerable]$Processes) {
    $ids = @($Processes | ForEach-Object { $_.ProcessId })
    foreach ($id in $ids) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    if ($ids.Count -gt 0) {
        Start-Sleep -Seconds 1
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
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

function Test-LauncherStateManagedNapCat([object]$State) {
    if (-not $State -or -not $State.boot_started_by_launcher -or -not $State.boot_pid) {
        return $false
    }

    try {
        $bootProcess = Get-Process -Id ([int]$State.boot_pid) -ErrorAction Stop
        return ($bootProcess.ProcessName -eq "NapCatWinBootMain")
    } catch {
        return $false
    }
}

function Stop-UnmanagedNapCatStack {
    $napcatBoots = @(Get-NapCatBootProcesses)
    $napcatIds = @($napcatBoots | ForEach-Object { [int]$_.ProcessId } | Select-Object -Unique)
    if ($napcatIds.Count -le 0) {
        return [pscustomobject]@{
            stopped = $true
            attempted = $false
            remaining_ids = @()
        }
    }

    $napcatTreeIds = @($napcatIds + @(Get-ProcessDescendantIds -ParentIds $napcatIds) | Select-Object -Unique)
    Write-Host "Detected an unmanaged NapCat websocket. Relaunching Xiaomachi using the isolated QQ user-data directory."
    Stop-TrackedProcesses -Ids $napcatTreeIds
    Start-Sleep -Seconds 2
    $remainingIds = @($napcatTreeIds | Where-Object { Get-Process -Id ([int]$_) -ErrorAction SilentlyContinue })
    if ($remainingIds.Count -gt 0) {
        Write-Host ("unmanaged_napcat_stop_failed remaining_pids={0}" -f ($remainingIds -join ","))
        return [pscustomobject]@{
            stopped = $false
            attempted = $true
            remaining_ids = $remainingIds
        }
    }

    Remove-LauncherState
    return [pscustomobject]@{
        stopped = $true
        attempted = $true
        remaining_ids = @()
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

function Stop-LauncherManagedNapCatStack {
    $state = Read-LauncherState
    if (-not $state) {
        return $false
    }

    $stoppedAny = $false
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
            $stoppedAny = $true
        }
    }

    if ($state.qq_started_by_launcher) {
        $qqTargets = @(Get-QQProcesses)
        $qqIds = @($qqTargets | ForEach-Object { $_.ProcessId } | Select-Object -Unique)
        if ($qqIds.Count -gt 0) {
            Stop-TrackedProcesses -Ids $qqIds
            $stoppedAny = $true
        }
    }

    if ($stoppedAny) {
        Start-Sleep -Seconds 2
    }
    return $stoppedAny
}

function Test-TcpEndpoint([string]$TargetHost, [int]$Port, [int]$TimeoutMs = 800) {
    $client = New-Object System.Net.Sockets.TcpClient
    $asyncResult = $null
    try {
        $asyncResult = $client.BeginConnect($TargetHost, $Port, $null, $null)
        if (-not $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $client.EndConnect($asyncResult)
        return $true
    } catch {
        return $false
    } finally {
        if ($asyncResult) {
            $asyncResult.AsyncWaitHandle.Close()
        }
        $client.Close()
    }
}

function Get-WsEndpoint([string]$WsUrl) {
    $uri = [System.Uri]$WsUrl
    return @{
        Host = $uri.Host
        Port = $uri.Port
    }
}

function Resolve-PythonExecutable {
    $candidates = @(
        (Join-Path $workdir ".venv\Scripts\python.exe"),
        "C:\work\anaconda\python.exe"
    )

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Python executable not found. Install the venv or update start_xiaomachi.ps1."
}

function Repair-StartProcessEnvironment {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if ([string]::IsNullOrEmpty($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    if (-not [string]::IsNullOrEmpty($pathValue)) {
        if ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT) {
            [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
            [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
            return
        }
        [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-NapCatStartupAdministrator {
    $startAsAdmin = Get-ConfigValue "NAPCAT_START_AS_ADMIN" "false"
    if (-not (Test-ConfigFlag $startAsAdmin)) {
        return
    }

    if (Test-IsAdministrator) {
        return
    }

    Write-Host "NapCat/QQ startup is configured to use administrator privileges. Relaunching this launcher with UAC..."
    Repair-StartProcessEnvironment
    try {
        Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath) `
            -WorkingDirectory $workdir `
            -Verb RunAs | Out-Null
        exit 0
    } catch {
        Write-Host "Failed to relaunch with configured administrator privileges: $($_.Exception.Message)"
        Write-Host "Set NAPCAT_START_AS_ADMIN=false, or run the launcher from an administrator PowerShell and approve the UAC prompt."
        exit 1
    }
}

function Test-OneBotOnline([string]$WsUrl) {
    $pythonExe = Resolve-PythonExecutable
    $probeScript = @"
import asyncio
import json
import sys
from uuid import uuid4

import websockets

async def main():
    echo = "start-health-" + uuid4().hex
    async with websockets.connect(sys.argv[1]) as ws:
        await ws.send(json.dumps({"action": "get_status", "params": {}, "echo": echo}))
        for _ in range(20):
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if payload.get("echo") == echo:
                data = payload.get("data") or {}
                ok = payload.get("status") == "ok" and payload.get("retcode") == 0 and data.get("online") is True
                print("online" if ok else "offline")
                return 0 if ok else 1
        print("offline")
        return 1

raise SystemExit(asyncio.run(main()))
"@

    try {
        $output = $probeScript | & $pythonExe - $WsUrl 2>&1
        return ($LASTEXITCODE -eq 0 -and (($output | Out-String).Trim()) -eq "online")
    } catch {
        return $false
    }
}

function Resolve-QQPath {
    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($candidate in @(
        (Get-ConfigValue "QQ_EXE_PATH"),
        "C:\app\QQ\QQ.exe",
        (Join-Path ${env:ProgramFiles} "Tencent\QQ\QQ.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Tencent\QQ\QQ.exe")
    )) {
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            [void]$candidates.Add($candidate)
        }
    }

    foreach ($registryPath in @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\QQ",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\QQ"
    )) {
        try {
            $uninstallString = (Get-ItemProperty -Path $registryPath -Name UninstallString -ErrorAction Stop).UninstallString
            if ($uninstallString) {
                $installDir = Split-Path -Parent $uninstallString.Trim('"')
                $qqPath = Join-Path $installDir "QQ.exe"
                [void]$candidates.Add($qqPath)
            }
        } catch {
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "QQ executable not found. Set QQ_EXE_PATH in .env to your QQ.exe path."
}

function Resolve-QQExtraArg([string]$Arg) {
    $prefix = "--user-data-dir="
    if (-not $Arg.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $Arg
    }

    $userDataDir = $Arg.Substring($prefix.Length).Trim('"')
    if ([string]::IsNullOrWhiteSpace($userDataDir)) {
        return $Arg
    }

    if (-not [System.IO.Path]::IsPathRooted($userDataDir)) {
        $userDataDir = Join-Path $workdir $userDataDir
    }
    New-Item -ItemType Directory -Force -Path $userDataDir | Out-Null

    return "$prefix$userDataDir"
}

function Get-QQExtraArgs {
    $extra = Get-ConfigValue "QQ_EXTRA_ARGS"
    if ([string]::IsNullOrWhiteSpace($extra)) {
        return @()
    }
    return @(
        $extra -split "\s+" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { Resolve-QQExtraArg $_ }
    )
}

function Resolve-NapCatShellDir {
    $candidates = New-Object System.Collections.Generic.List[string]
    $overrideShell = Get-ConfigValue "NAPCAT_SHELL_DIR"
    $overrideBoot = Get-ConfigValue "NAPCAT_BOOT_PATH"

    foreach ($candidate in @(
        $overrideShell,
        (Join-Path $env:USERPROFILE "Downloads\NapCatQQ\NapCat.Shell"),
        (Join-Path $env:USERPROFILE "Downloads\NapCat.Shell")
    )) {
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            [void]$candidates.Add($candidate)
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($overrideBoot)) {
        if (Test-Path $overrideBoot) {
            $bootItem = Get-Item $overrideBoot
            if ($bootItem.PSIsContainer) {
                [void]$candidates.Add($bootItem.FullName)
            } else {
                [void]$candidates.Add($bootItem.DirectoryName)
            }
        } else {
            [void]$candidates.Add((Split-Path -Parent $overrideBoot))
        }
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        $bootPath = Join-Path $candidate "NapCatWinBootMain.exe"
        if (Test-Path $bootPath) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "NapCat shell directory not found. Set NAPCAT_SHELL_DIR in .env to the NapCat.Shell folder."
}

function Resolve-ConfiguredPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $Path
    }

    if (-not [System.IO.Path]::IsPathRooted($Path)) {
        $Path = Join-Path $workdir $Path
    }

    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-NapCatLauncherPathMatchesShell([string]$Path, [string]$ShellDir, [string]$ConfigKey) {
    $fullPath = Resolve-ConfiguredPath $Path
    $fullShellDir = (Resolve-ConfiguredPath $ShellDir).TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar

    if (-not $fullPath.StartsWith($fullShellDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        if ($ConfigKey -eq "NAPCAT_BOOT_PATH") {
            throw "NAPCAT_BOOT_PATH must be inside NAPCAT_SHELL_DIR. OneKey boot/hook launchers are not compatible with this launcher."
        }
        if ($ConfigKey -eq "NAPCAT_INJECT_DLL_PATH") {
            throw "NAPCAT_INJECT_DLL_PATH must be inside NAPCAT_SHELL_DIR. OneKey boot/hook launchers are not compatible with this launcher."
        }
        throw "$ConfigKey must be inside NAPCAT_SHELL_DIR. OneKey boot/hook launchers are not compatible with this launcher."
    }
}

function Resolve-NapCatPaths {
    $shellDir = Resolve-NapCatShellDir
    if ($shellDir -match "(?i)onekey" -or (Split-Path -Leaf $shellDir) -eq "bootmain") {
        throw "OneKey boot/hook launchers are not compatible with this launcher. Use the regular NapCat.Shell directory in NAPCAT_SHELL_DIR."
    }

    $bootPath = Resolve-ConfiguredPath (Get-ConfigValue "NAPCAT_BOOT_PATH" (Join-Path $shellDir "NapCatWinBootMain.exe"))
    $injectPath = Resolve-ConfiguredPath (Get-ConfigValue "NAPCAT_INJECT_DLL_PATH" (Join-Path $shellDir "NapCatWinBootHook.dll"))
    Assert-NapCatLauncherPathMatchesShell -Path $bootPath -ShellDir $shellDir -ConfigKey "NAPCAT_BOOT_PATH"
    Assert-NapCatLauncherPathMatchesShell -Path $injectPath -ShellDir $shellDir -ConfigKey "NAPCAT_INJECT_DLL_PATH"

    $patchPath = Resolve-ConfiguredPath (Get-ConfigValue "NAPCAT_PATCH_PACKAGE" (Join-Path $logDir "napcat.qqnt.runtime.json"))
    $paths = @{
        ShellDir = $shellDir
        BootPath = $bootPath
        InjectPath = $injectPath
        PatchPath = $patchPath
        MainPath = Join-Path $shellDir "napcat.mjs"
        LoadPath = Join-Path $shellDir "loadNapCat.js"
        BundledPatchPath = Join-Path $shellDir "qqnt.json"
        QrCodePath = Join-Path $shellDir "cache\qrcode.png"
    }

    foreach ($requiredKey in @("BootPath", "InjectPath", "MainPath")) {
        $requiredPath = [string]$paths[$requiredKey]
        if (-not (Test-Path $requiredPath)) {
            throw "$requiredKey not found: $requiredPath"
        }
    }

    return $paths
}

function Get-QQCurrentPackagePath([string]$QQPath) {
    $qqDir = Split-Path -Parent $QQPath
    $versionsDir = Join-Path $qqDir "versions"
    $configPath = Join-Path $versionsDir "config.json"
    if (-not (Test-Path $configPath)) {
        return $null
    }

    try {
        $config = Get-Content -Path $configPath -Raw -ErrorAction Stop | ConvertFrom-Json
        $curVersion = [string]$config.curVersion
        if ([string]::IsNullOrWhiteSpace($curVersion)) {
            return $null
        }

        $packagePath = Join-Path $versionsDir (Join-Path $curVersion "resources\app\package.json")
        if (Test-Path $packagePath) {
            return (Resolve-Path $packagePath).Path
        }
    } catch {
        return $null
    }

    return $null
}

function Write-NapCatPatchPackage([hashtable]$Paths, [string]$QQPath) {
    $sourcePackagePath = Get-QQCurrentPackagePath -QQPath $QQPath
    if (-not $sourcePackagePath) {
        $sourcePackagePath = [string]$Paths.BundledPatchPath
    }
    if (-not (Test-Path $sourcePackagePath)) {
        throw "NapCat patch package source not found: $sourcePackagePath"
    }

    $package = Get-Content -Path $sourcePackagePath -Raw -ErrorAction Stop | ConvertFrom-Json
    $package.main = "./loadNapCat.js"
    $json = $package | ConvertTo-Json -Depth 8

    $parent = Split-Path -Parent ([string]$Paths.PatchPath)
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $writeNeeded = $true
    if (Test-Path $Paths.PatchPath) {
        try {
            $current = (Get-Content -Path $Paths.PatchPath -Raw -ErrorAction Stop).Trim()
            $writeNeeded = ($current -ne $json.Trim())
        } catch {
            $writeNeeded = $true
        }
    }

    if ($writeNeeded) {
        Set-Content -Path $Paths.PatchPath -Value $json -Encoding utf8
    }
}

function Write-NapCatLoadScript([hashtable]$Paths) {
    $mainImport = $Paths.MainPath.Replace("\", "/")
    $script = '(async () => {await import("file:///' + $mainImport + '")})()'
    if (Test-Path $Paths.LoadPath) {
        try {
            $current = (Get-Content -Path $Paths.LoadPath -Raw -ErrorAction Stop).Trim()
            if ($current -eq $script) {
                return
            }
        } catch {
        }
    }

    try {
        Set-Content -Path $Paths.LoadPath -Value $script -Encoding utf8
    } catch {
        if (Test-Path $Paths.LoadPath) {
            try {
                $current = (Get-Content -Path $Paths.LoadPath -Raw -ErrorAction Stop).Trim()
                if ($current -eq $script) {
                    return
                }
            } catch {
            }
        }
        throw
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

function Wait-ForNapCatEndpoint(
    [string]$TargetHost,
    [int]$Port,
    [int]$TimeoutSeconds,
    [string]$QrCodePath,
    [datetime]$LaunchStartedAt,
    [int]$BootProcessId = 0
) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $qrCodeOpened = $false

    Write-Host "Waiting for NapCat websocket at ws://$TargetHost`:$Port ..."

    while ((Get-Date) -lt $deadline) {
        if (Test-TcpEndpoint -TargetHost $TargetHost -Port $Port) {
            return $true
        }

        $elapsedSeconds = ((Get-Date) - $LaunchStartedAt).TotalSeconds
        if ($BootProcessId -gt 0 -and $elapsedSeconds -ge 8) {
            $bootRunning = $true
            try {
                [void](Get-Process -Id $BootProcessId -ErrorAction Stop)
            } catch {
                $bootRunning = $false
            }

            $qqRunning = @(Get-Process -Name QQ -ErrorAction SilentlyContinue).Count -gt 0
            $qrCodeExists = Test-Path $QrCodePath
            if (-not $bootRunning -and -not $qqRunning -and -not $qrCodeExists) {
                Write-Host "NapCat boot process exited before websocket was ready. QQ/NapCat likely crashed during startup."
                Write-Host "Close any QQ.exe error dialog, then rerun the launcher after checking QQ_EXE_PATH and the QQ user-data directory."
                return $false
            }
        }

        if ((-not $qrCodeOpened) -and (Test-Path $QrCodePath)) {
            $qrFile = Get-Item $QrCodePath
            if ($qrFile.LastWriteTime -ge $LaunchStartedAt) {
                Write-Host "Detected QQ login QR code. Opening: $QrCodePath"
                $qrCodeOpened = $true
                Repair-StartProcessEnvironment
                try {
                    Start-Process -FilePath $QrCodePath -ErrorAction Stop | Out-Null
                } catch {
                    Write-Host "Could not open QR code image automatically: $QrCodePath"
                }
            } else {
                Write-Host "Ignoring old QQ login QR code: $QrCodePath"
                $qrCodeOpened = $true
            }
        }
        Start-Sleep -Seconds 2
    }

    if (Test-Path $QrCodePath) {
        Write-Host "QQ was opened, but NapCat is still waiting for QQ login."
        Write-Host "If QQ shows a QR code or login prompt, finish the login and run the launcher again."
    } else {
        Write-Host "NapCat did not open the websocket endpoint in time."
    }

    return $false
}

function Ensure-NapCatReady([string]$TargetHost, [int]$Port, [int]$WaitSeconds) {
    $launcherState = Read-LauncherState
    if (Test-TcpEndpoint -TargetHost $TargetHost -Port $Port) {
        if (Test-OneBotOnline -WsUrl $napcatWsUrl) {
            if (Test-LauncherStateManagedNapCat $launcherState) {
                return @{
                    PortReady = $true
                    Started = $false
                    QqStarted = $false
                    BootPid = [int]$launcherState.boot_pid
                    ShellDir = [string]$launcherState.napcat_shell_dir
                    QrCodePath = ""
                }
            }

            Write-Host "unmanaged_napcat_relaunch reason=online_but_not_launcher_managed"
            $unmanagedStop = Stop-UnmanagedNapCatStack
            if (-not $unmanagedStop.stopped) {
                Write-Host "Could not stop the unmanaged NapCat/QQ process tree. Stop Xiaomachi once as administrator, then run the startup BAT normally."
                return @{
                    PortReady = $false
                    Started = $false
                    QqStarted = $false
                    BootPid = 0
                    ShellDir = ""
                    QrCodePath = ""
                }
            }
        }

        if (Test-TcpEndpoint -TargetHost $TargetHost -Port $Port) {
            Write-Host "NapCat websocket is open, but OneBot reports offline. Relaunching launcher-managed QQ/NapCat..."
            if (-not (Stop-LauncherManagedNapCatStack)) {
                $unmanagedStop = Stop-UnmanagedNapCatStack
                if (-not $unmanagedStop.stopped) {
                    Write-Host "Could not stop the unmanaged NapCat/QQ process tree. Stop Xiaomachi once as administrator, then run the startup BAT normally."
                    return @{
                        PortReady = $false
                        Started = $false
                        QqStarted = $false
                        BootPid = 0
                        ShellDir = ""
                        QrCodePath = ""
                    }
                }
            }
        }
    }

    if (Test-TcpEndpoint -TargetHost $TargetHost -Port $Port) {
        return @{
            PortReady = $true
            Started = $false
            QqStarted = $false
            BootPid = 0
            ShellDir = ""
            QrCodePath = ""
        }
    }

    $paths = Resolve-NapCatPaths
    $existingBoot = @(Get-NapCatBootProcesses)
    if ($existingBoot.Count -gt 0) {
        $launchStartedAt = (Get-Date).AddSeconds(-2)
        $ready = Wait-ForNapCatEndpoint -TargetHost $TargetHost -Port $Port -TimeoutSeconds $WaitSeconds -QrCodePath $paths.QrCodePath -LaunchStartedAt $launchStartedAt -BootProcessId ([int]$existingBoot[0].ProcessId)
        return @{
            PortReady = $ready
            Started = $false
            QqStarted = $false
            BootPid = [int]$existingBoot[0].ProcessId
            ShellDir = $paths.ShellDir
            QrCodePath = $paths.QrCodePath
        }
    }

    $qqWasRunningBefore = @(Get-Process -Name QQ -ErrorAction SilentlyContinue).Count -gt 0
    $qqPath = Resolve-QQPath
    $qqExtraArgs = @(Get-QQExtraArgs)

    Write-NapCatLoadScript -Paths $paths
    Write-NapCatPatchPackage -Paths $paths -QQPath $qqPath
    Remove-Item $paths.QrCodePath -Force -ErrorAction SilentlyContinue

    $env:NAPCAT_PATCH_PACKAGE = $paths.PatchPath
    $env:NAPCAT_LOAD_PATH = $paths.LoadPath
    $env:NAPCAT_INJECT_PATH = $paths.InjectPath
    $env:NAPCAT_LAUNCHER_PATH = $paths.BootPath
    $env:NAPCAT_MAIN_PATH = $paths.MainPath
    $launchStartedAt = Get-Date

    $bootArguments = @($qqPath, $paths.InjectPath) + $qqExtraArgs
    Ensure-NapCatStartupAdministrator
    Repair-StartProcessEnvironment
    $bootProc = Start-Process `
        -FilePath $paths.BootPath `
        -ArgumentList $bootArguments `
        -WorkingDirectory $paths.ShellDir `
        -WindowStyle Hidden `
        -PassThru

    $ready = Wait-ForNapCatEndpoint -TargetHost $TargetHost -Port $Port -TimeoutSeconds $WaitSeconds -QrCodePath $paths.QrCodePath -LaunchStartedAt $launchStartedAt -BootProcessId $bootProc.Id
    return @{
        PortReady = $ready
        Started = $true
        QqStarted = (-not $qqWasRunningBefore)
        BootPid = $bootProc.Id
        ShellDir = $paths.ShellDir
        QrCodePath = $paths.QrCodePath
    }
}

$napcatWsUrl = Get-ConfigValue "NAPCAT_WS_URL" "ws://127.0.0.1:3001"
$napcatWaitSeconds = [int](Get-ConfigValue "NAPCAT_WAIT_TIMEOUT_SECONDS" "180")
$endpoint = Get-WsEndpoint $napcatWsUrl
$previousState = Read-LauncherState

$napcatResult = Ensure-NapCatReady -TargetHost $endpoint.Host -Port $endpoint.Port -WaitSeconds $napcatWaitSeconds
$managedBoot = [bool]$napcatResult.Started
$managedQq = [bool]$napcatResult.QqStarted
$managedBootPid = [int]$napcatResult.BootPid
$managedShellDir = [string]$napcatResult.ShellDir

if (-not $managedBoot -and $previousState -and $previousState.boot_started_by_launcher) {
    $managedBoot = $true
    if (-not $managedBootPid) {
        $managedBootPid = [int]$previousState.boot_pid
    }
}

if (-not $managedQq -and $previousState -and $previousState.qq_started_by_launcher) {
    $managedQq = $true
}

if (-not $managedShellDir -and $previousState) {
    $managedShellDir = [string]$previousState.napcat_shell_dir
}

Save-LauncherState @{
    qq_started_by_launcher = $managedQq
    boot_started_by_launcher = $managedBoot
    boot_pid = $managedBootPid
    napcat_shell_dir = $managedShellDir
    napcat_ws_url = $napcatWsUrl
    managed_at = (Get-Date).ToString("o")
}

if (-not $napcatResult.PortReady) {
    Write-Host "NapCat websocket is not ready at $napcatWsUrl."
    Write-Host "The bot processes were not started."
    exit 1
}

$botStartScript = Join-Path $workdir "start_xiaomachi_bots.ps1"
& powershell -ExecutionPolicy Bypass -File $botStartScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start Xiaomachi bot processes."
    exit $LASTEXITCODE
}

Write-Host "NapCat websocket is ready at $napcatWsUrl."
Write-Host "Xiaomachi bot processes started."
exit 0
