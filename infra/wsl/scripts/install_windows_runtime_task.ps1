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

# The repository-owned WScript runner launches wsl.exe directly without a cmd
# or PowerShell console host. Store WSL can delegate an intermediate console to
# Windows Terminal even when callers request a hidden window, so adding cmd.exe
# for output redirection would reintroduce a visible blank terminal. The runner
# keeps its own lifecycle log and applies backoff to short-lived anchor exits.
$runner = (Resolve-Path (Join-Path $PSScriptRoot "run_windows_runtime_task.vbs")).Path
$taskArguments = "//B //Nologo `"$runner`" `"$Distro`""
$taskAction = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wscript.exe" -Argument $taskArguments
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
