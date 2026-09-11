@echo off
setlocal
set "WEBUI_PORT=5099"
set "PS_SCRIPT=%~dp0infra\wsl\scripts\open_snowluma_webui.ps1"

rem SnowLuma uses WSL host networking. Windows localhost forwarding is not
rem available on every WSL network configuration, so prove readiness inside
rem WSL first and then open the current WSL IPv4 address from Windows.
set "WEBUI_CODE="
for /f "usebackq delims=" %%C in (`wsl.exe --user root --exec bash -lc "curl -s -o /dev/null -w %%{http_code} --max-time 3 http://127.0.0.1:%WEBUI_PORT%/"`) do set "WEBUI_CODE=%%C"
if not "%WEBUI_CODE%"=="200" goto unavailable

for /f "tokens=1" %%I in ('wsl.exe --user root --exec hostname -I') do (
  if not defined WSL_WEBUI_IP set "WSL_WEBUI_IP=%%I"
)
if not defined WSL_WEBUI_IP goto forwarding_unavailable

set "WEBUI_URL=http://%WSL_WEBUI_IP%:%WEBUI_PORT%/"
curl.exe --silent --show-error --fail --max-time 3 "%WEBUI_URL%" >nul 2>nul
if errorlevel 1 goto forwarding_unavailable

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -WebUiUrl "%WEBUI_URL%"
if errorlevel 1 goto open_failed
exit /b 0

:unavailable
echo SnowLuma WebUI is not answering inside WSL (http=%WEBUI_CODE%).
echo Run start-xiaomachi-wsl.bat first, or inspect: wsl.exe --user root --exec docker logs --tail 50 xiaomachi-snowluma
pause
exit /b 1

:forwarding_unavailable
echo SnowLuma WebUI is running inside WSL, but Windows cannot reach its current WSL address.
echo Restart WSL networking, then run this shortcut again.
pause
exit /b 1

:open_failed
echo The SnowLuma WebUI is reachable but the browser could not be opened.
echo Open this address manually: %WEBUI_URL%
pause
exit /b 1
