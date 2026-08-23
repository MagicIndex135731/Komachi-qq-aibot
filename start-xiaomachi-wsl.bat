@echo off
setlocal
echo WSL BAT VERSION 20260824-WSL-READY-GATE
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
set "TASK_INSTALLER=%~dp0infra\wsl\scripts\install_windows_runtime_task.ps1"
wsl.exe --user root --exec test -x "%ENTRY%"
if errorlevel 1 (
  echo First-time Linux runtime installation...
  wsl.exe --user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh install
  if errorlevel 1 goto :failed
)
echo Starting Xiaomachi. The window closes only after all readiness checks pass.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$task = Get-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' -ErrorAction SilentlyContinue; if ($task) { Start-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' } else { . $env:TASK_INSTALLER }"
if errorlevel 1 goto :failed
wsl.exe --user root --exec "%ENTRY%" start
if errorlevel 1 goto :failed
echo Xiaomachi started successfully.
echo Verifying Xiaomachi readiness...
wsl.exe --user root --exec "%ENTRY%" status
if errorlevel 1 goto :failed
timeout /t 2 /nobreak >nul
exit /b 0

:failed
echo Xiaomachi failed to start. Review the output above.
powershell.exe -NoProfile -Command "$task = Get-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' -ErrorAction SilentlyContinue; if ($task) { Stop-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' }"
wsl.exe --user root --exec "%ENTRY%" stop >nul 2>&1
pause
exit /b 1
