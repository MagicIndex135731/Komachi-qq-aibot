@echo off
setlocal
call "%~dp0infra\wsl\scripts\switch_model_profile.cmd" %*
exit /b %ERRORLEVEL%
