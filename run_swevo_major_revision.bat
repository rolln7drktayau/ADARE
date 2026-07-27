@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_swevo_major_revision.ps1" %*
exit /b %ERRORLEVEL%
