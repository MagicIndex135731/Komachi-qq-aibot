@echo off
setlocal
echo WSL BAT VERSION 20260717-LINUX-RUNTIME
set "ENTRY=/usr/local/bin/xiaomachi-wsl-entry"
wsl.exe --user root --exec test -x "%ENTRY%"
if errorlevel 1 (
  wsl.exe --user root --cd "%~dp0" --exec bash infra/wsl/scripts/xiaomachi-wsl-entry.sh status
) else (
  wsl.exe --user root --exec "%ENTRY%" status
)
set "STATUS_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %STATUS_EXIT_CODE%
