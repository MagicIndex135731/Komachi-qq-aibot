[CmdletBinding()]
param(
    [ValidateSet("Install", "Remove")]
    [string]$Action = "Install",
    [string]$Distro = "Ubuntu_Migrated",
    [string]$TaskName = "Xiaomachi WSL Runtime"
)

$ErrorActionPreference = "Stop"

if ($Action -eq "Remove") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

$entry = "/usr/local/bin/xiaomachi-wsl-entry"
& wsl.exe -d $Distro --user root --exec test -x $entry
if ($LASTEXITCODE -ne 0) {
    throw "Xiaomachi Linux runtime is not installed in WSL distro '$Distro'."
}

# The repository-owned runner supplies the interactive user's environment and
# records unexpected WSL exits. Task Scheduler invoking wsl.exe directly returns
# 0xFFFFFFFF on some WSL Store builds even though the same command works in a
# terminal.
$runner = (Resolve-Path (Join-Path $PSScriptRoot "run_windows_runtime_task.ps1")).Path
$taskArguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`" -Distro `"$Distro`""
$taskAction = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument $taskArguments
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$taskPrincipal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$task = New-ScheduledTask `
    -Action $taskAction `
    -Trigger $taskTrigger `
    -Principal $taskPrincipal `
    -Settings $taskSettings `
    -Description "Keeps the Xiaomachi WSL runtime alive and restores its systemd services after login."
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host "Installed and started scheduled task: $TaskName"
Write-Host "The task owns the WSL anchor; start/status/stop BAT files remain the operator controls."
