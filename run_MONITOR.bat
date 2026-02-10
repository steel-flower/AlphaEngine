@echo off
title ALPHA ENGINE SIGMA v3.2 MONITORING...
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:LOOP
echo.
echo ========================================================
echo   ALPHA ENGINE SIGMA v3.2 REAL-TIME MONITORING
echo   System running since: %date% %time%
echo ========================================================
echo.

python alpha_monitor_v3.py

echo.
echo [!] Monitoring script crashed or stopped. Restarting in 10 seconds...
timeout /t 10
goto LOOP
