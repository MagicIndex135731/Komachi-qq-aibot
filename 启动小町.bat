@echo off
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_xiaomachi.ps1"
exit /b %errorlevel%
