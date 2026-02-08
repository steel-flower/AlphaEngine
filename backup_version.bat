@echo off
REM ========================================
REM Alpha Engine Sigma - Version Backup Script
REM ========================================

echo.
echo ======================================================
echo    Alpha Engine Sigma - Version Backup Utility
echo ======================================================
echo.

set VERSION=%1

if "%VERSION%"=="" (
    echo [Error] Please specify version number
    echo Usage: backup_version.bat v7.2
    echo.
    pause
    exit /b 1
)

echo [Backup] Creating backup for %VERSION%...
copy alpha_engine_sigma.py old_versions\alpha_engine_sigma_%VERSION%.py

if %ERRORLEVEL% EQU 0 (
    echo [Success] Backup created: old_versions\alpha_engine_sigma_%VERSION%.py
) else (
    echo [Error] Backup failed!
)

echo.
pause
