@echo off
setlocal
echo WSL BAT VERSION 20260823-WSL-MIHOMO
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
powershell.exe -NoProfile -Command "$task = Get-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' -ErrorAction SilentlyContinue; if ($task) { Stop-ScheduledTask -TaskName 'Xiaomachi WSL Runtime' }"
wsl.exe --user root --exec test -x "%ENTRY%"
if errorlevel 1 (
  wsl.exe --user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh stop
) else (
  wsl.exe --user root --exec "%ENTRY%" stop
)
set "STOP_EXIT_CODE=%ERRORLEVEL%"
if "%STOP_EXIT_CODE%"=="0" echo Xiaomachi stopped successfully.
pause
exit /b %STOP_EXIT_CODE%
