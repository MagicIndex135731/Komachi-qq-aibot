@echo off
setlocal
echo WSL BAT VERSION 20260710-ASCII
powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process -WindowStyle Hidden -FilePath 'wsl.exe' -ArgumentList 'bash','/mnt/d/xiaomachi-wsl-entry.sh','anchor'"
wsl.exe bash /mnt/d/xiaomachi-wsl-entry.sh start
pause
