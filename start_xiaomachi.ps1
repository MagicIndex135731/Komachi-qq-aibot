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

    foreach ($line in Get-Content -Path $Path) {
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

function Resolve-NapCatPaths {
    $shellDir = Resolve-NapCatShellDir
    $bootPath = Get-ConfigValue "NAPCAT_BOOT_PATH" (Join-Path $shellDir "NapCatWinBootMain.exe")
    $injectPath = Get-ConfigValue "NAPCAT_INJECT_DLL_PATH" (Join-Path $shellDir "NapCatWinBootHook.dll")
    $paths = @{
        ShellDir = $shellDir
        BootPath = $bootPath
        InjectPath = $injectPath
        PatchPath = Join-Path $shellDir "qqnt.json"
        MainPath = Join-Path $shellDir "napcat.mjs"
        LoadPath = Join-Path $shellDir "loadNapCat.js"
        QrCodePath = Join-Path $shellDir "cache\qrcode.png"
    }

    foreach ($requiredKey in @("BootPath", "InjectPath", "PatchPath", "MainPath")) {
        $requiredPath = [string]$paths[$requiredKey]
        if (-not (Test-Path $requiredPath)) {
            throw "$requiredKey not found: $requiredPath"
        }
    }

    return $paths
}

function Write-NapCatLoadScript([hashtable]$Paths) {
    $mainImport = $Paths.MainPath.Replace("\", "/")
    $script = '(async () => {await import("file:///' + $mainImport + '")})()'
    Set-Content -Path $Paths.LoadPath -Value $script -Encoding utf8
}

function Get-NapCatBootProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "NapCatWinBootMain.exe"
    }
}

function Wait-ForNapCatEndpoint(
    [string]$TargetHost,
    [int]$Port,
    [int]$TimeoutSeconds,
    [string]$QrCodePath,
    [datetime]$LaunchStartedAt
) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $qrCodeOpened = $false

    Write-Host "Waiting for NapCat websocket at ws://$TargetHost`:$Port ..."

    while ((Get-Date) -lt $deadline) {
        if (Test-TcpEndpoint -TargetHost $TargetHost -Port $Port) {
            return $true
        }
        if ((-not $qrCodeOpened) -and (Test-Path $QrCodePath)) {
            $qrFile = Get-Item $QrCodePath
            if ($qrFile.LastWriteTime -ge $LaunchStartedAt) {
                $qrCodeOpened = $true
                Write-Host "Detected QQ login QR code. Opening: $QrCodePath"
                Start-Process -FilePath $QrCodePath -ErrorAction SilentlyContinue | Out-Null
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
        $ready = Wait-ForNapCatEndpoint -TargetHost $TargetHost -Port $Port -TimeoutSeconds $WaitSeconds -QrCodePath $paths.QrCodePath -LaunchStartedAt $launchStartedAt
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

    Write-NapCatLoadScript -Paths $paths
    Remove-Item $paths.QrCodePath -Force -ErrorAction SilentlyContinue

    $env:NAPCAT_PATCH_PACKAGE = $paths.PatchPath
    $env:NAPCAT_LOAD_PATH = $paths.LoadPath
    $env:NAPCAT_INJECT_PATH = $paths.InjectPath
    $env:NAPCAT_LAUNCHER_PATH = $paths.BootPath
    $env:NAPCAT_MAIN_PATH = $paths.MainPath
    $launchStartedAt = Get-Date

    $bootProc = Start-Process `
        -FilePath $paths.BootPath `
        -ArgumentList @($qqPath, $paths.InjectPath) `
        -WorkingDirectory $paths.ShellDir `
        -WindowStyle Hidden `
        -PassThru

    $ready = Wait-ForNapCatEndpoint -TargetHost $TargetHost -Port $Port -TimeoutSeconds $WaitSeconds -QrCodePath $paths.QrCodePath -LaunchStartedAt $launchStartedAt
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
