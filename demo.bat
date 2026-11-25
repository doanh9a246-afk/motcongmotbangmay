@echo off
rem Double-click hoac go: demo   (them buoc Newton 3: demo -Newton3)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1" %*
pause
