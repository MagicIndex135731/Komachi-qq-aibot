@echo off
setlocal
echo WSL BAT VERSION 20260823-WSL-MIHOMO
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
wsl.exe --user root --exec test -x "%ENTRY%"
if errorlevel 1 (
  echo First-time Linux runtime installation...
  wsl.exe --user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh install
  if errorlevel 1 goto :failed
)
echo Starting Xiaomachi. This may take 1-2 minutes...
powershell.exe -NoProfile -WindowStyle Hidden -Command "$task = Get-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' -ErrorAction SilentlyContinue; if ($task) { Start-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' } else { Start-Process -WindowStyle Hidden -FilePath 'wsl.exe' -ArgumentList '--user','root','--exec','/usr/local/bin/xiaomachi-wsl-entry','anchor' }"
wsl.exe --user root --exec "%ENTRY%" start
if errorlevel 1 goto :failed
echo Xiaomachi started successfully.
wsl.exe --user root --exec systemctl is-active xiaomachi-stack.service xiaomachi-watchdog.service
pause
exit /b 0

:failed
echo Xiaomachi failed to start. Review the output above.
wsl.exe --user root --exec "%ENTRY%" stop >nul 2>&1
pause
exit /b 1
