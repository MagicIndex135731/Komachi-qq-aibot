@echo off
setlocal
curl.exe --silent --show-error --fail --max-time 3 http://127.0.0.1:3080/ >nul 2>nul
if errorlevel 1 goto unavailable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0infra\wsl\scripts\open_llbot_webui.ps1"
exit /b %errorlevel%

:unavailable
echo LLBot WebUI is not available. Run start-xiaomachi-wsl.bat first.
exit /b 1
