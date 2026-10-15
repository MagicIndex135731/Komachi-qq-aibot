@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\open_agent_monitor.ps1 %*
if errorlevel 1 pause
