@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0switch_model_profile.ps1" %*
set "SWITCH_EXIT_CODE=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %SWITCH_EXIT_CODE%
