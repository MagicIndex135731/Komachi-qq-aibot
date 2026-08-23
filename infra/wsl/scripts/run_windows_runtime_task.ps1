[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu_Migrated"
)

$ErrorActionPreference = "Stop"
$logDirectory = Join-Path $env:LOCALAPPDATA "Xiaomachi\logs"
$logPath = Join-Path $logDirectory "wsl-runtime-task.log"
$stdoutPath = Join-Path $logDirectory "wsl-runtime-task.stdout.log"
$stderrPath = Join-Path $logDirectory "wsl-runtime-task.stderr.log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$wsl = Join-Path $env:WINDIR "System32\wsl.exe"
$entry = "/usr/local/bin/xiaomachi-wsl-entry"
while ($true) {
    $startedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "$startedAt starting distro=$Distro"

    try {
        $process = Start-Process `
            -FilePath $wsl `
            -ArgumentList @("-d", $Distro, "--user", "root", "--exec", $entry, "anchor") `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        $exitCode = $process.ExitCode
    }
    catch {
        Add-Content -LiteralPath $logPath -Value $_.Exception.ToString()
        $exitCode = 1
    }

    $stoppedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "$stoppedAt anchor_stopped exit_code=$exitCode restarting_in_milliseconds=200"
    Start-Sleep -Milliseconds 200
}
