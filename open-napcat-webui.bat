@echo off
setlocal
curl.exe --silent --show-error --fail --max-time 3 http://127.0.0.1:6099/ >nul 2>nul
if errorlevel 1 goto unavailable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0infra\wsl\scripts\open_napcat_webui.ps1"
if errorlevel 1 goto token_missing
exit /b 0

:unavailable
echo NapCat WebUI is not available. Run start-xiaomachi-wsl.bat first.
exit /b 1

:token_missing
echo NapCat WebUI token is unavailable. Check local NapCat configuration.
exit /b 1
