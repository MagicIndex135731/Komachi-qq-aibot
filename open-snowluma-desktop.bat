@echo off
setlocal
set "DESKTOP_PORT=6081"
set "PS_SCRIPT=%~dp0infra\wsl\scripts\open_snowluma_desktop.ps1"

rem The QQ client desktop (where the login QR lives) is reachable through
rem noVNC. Prove readiness inside WSL first, then open the current WSL address
rem from Windows and auto-fill the VNC password from infra\wsl\.env.
set "DESKTOP_CODE="
for /f "usebackq delims=" %%C in (`wsl.exe --user root --exec bash -lc "curl -s -o /dev/null -w %%{http_code} --max-time 3 http://127.0.0.1:%DESKTOP_PORT%/"`) do set "DESKTOP_CODE=%%C"
if not "%DESKTOP_CODE%"=="200" goto unavailable

for /f "tokens=1" %%I in ('wsl.exe --user root --exec hostname -I') do (
  if not defined WSL_DESKTOP_IP set "WSL_DESKTOP_IP=%%I"
)
if not defined WSL_DESKTOP_IP goto forwarding_unavailable

set "VNC_PASSWD="
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b "VNC_PASSWD=" "%~dp0infra\wsl\.env" 2^>nul`) do set "VNC_PASSWD=%%B"

set "DESKTOP_URL=http://%WSL_DESKTOP_IP%:%DESKTOP_PORT%/vnc.html?autoconnect=1&resize=scale"
if defined VNC_PASSWD set "DESKTOP_URL=%DESKTOP_URL%&password=%VNC_PASSWD%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -DesktopUrl "%DESKTOP_URL%"
if errorlevel 1 goto open_failed
exit /b 0

:unavailable
echo SnowLuma desktop (noVNC) is not answering inside WSL (http=%DESKTOP_CODE%).
echo Start the stack first: start-xiaomachi-wsl.bat
pause
exit /b 1

:forwarding_unavailable
echo The SnowLuma desktop is running inside WSL, but Windows cannot reach its current WSL address.
echo Restart WSL networking, then run this shortcut again.
pause
exit /b 1

:open_failed
echo The desktop is reachable but the browser could not be opened.
echo Open this address manually: %DESKTOP_URL%
pause
exit /b 1
