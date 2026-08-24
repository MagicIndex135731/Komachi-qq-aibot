@echo off
setlocal
set "WEBUI_PORT=3080"

rem LLBot uses WSL host networking. Windows localhost forwarding is not
rem available on every WSL network configuration, so prove readiness inside
rem WSL first and then open the current WSL IPv4 address from Windows.
wsl.exe --user root --exec bash -lc "curl -fsS --max-time 3 http://127.0.0.1:%WEBUI_PORT%/ > /dev/null"
if errorlevel 1 goto unavailable

for /f "tokens=1" %%I in ('wsl.exe --user root --exec hostname -I') do (
  if not defined WSL_WEBUI_IP set "WSL_WEBUI_IP=%%I"
)
if not defined WSL_WEBUI_IP goto forwarding_unavailable

set "WEBUI_URL=http://%WSL_WEBUI_IP%:%WEBUI_PORT%/"
curl.exe --silent --show-error --fail --max-time 3 "%WEBUI_URL%" >nul 2>nul
if errorlevel 1 goto forwarding_unavailable

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0infra\wsl\scripts\open_llbot_webui.ps1" -WebUiUrl "%WEBUI_URL%"
exit /b %errorlevel%

:unavailable
echo LLBot WebUI is not available. Run start-xiaomachi-wsl.bat first.
pause
exit /b 1

:forwarding_unavailable
echo LLBot WebUI is running inside WSL, but Windows cannot reach its current WSL address.
echo Restart WSL networking, then run this shortcut again.
pause
exit /b 1
