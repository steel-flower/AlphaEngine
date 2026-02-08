@echo off
REM ========================================
REM Alpha Engine Sigma - Version Restore Script
REM ========================================

echo.
echo ======================================================
echo    Alpha Engine Sigma - Version Restore Utility
echo ======================================================
echo.

set VERSION=%1

if "%VERSION%"=="" (
    echo Available versions:
    echo.
    dir /b old_versions\*.py
    echo.
    echo Usage: restore_version.bat v7.2
    echo.
    pause
    exit /b 1
)

echo [Warning] This will overwrite the current alpha_engine_sigma.py
echo [Restore] Target version: %VERSION%
echo.
set /p CONFIRM="Continue? (Y/N): "

if /i "%CONFIRM%" NEQ "Y" (
    echo [Cancelled] Restore cancelled by user
    pause
    exit /b 0
)

echo.
echo [Restore] Restoring from old_versions\alpha_engine_sigma_%VERSION%.py...
copy old_versions\alpha_engine_sigma_%VERSION%.py alpha_engine_sigma.py

if %ERRORLEVEL% EQU 0 (
    echo [Success] Restored to %VERSION%
    echo [Info] You can now run: run_SIGMA.bat
) else (
    echo [Error] Restore failed! File not found.
)

echo.
pause
