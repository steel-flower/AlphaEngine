@echo off
title ALPHA ENGINE 6.0 MASTER - INFINITE DISCOVERY
cd /d %~dp0

echo.
echo ======================================================
echo    ALPHA ENGINE 6.0 MASTER (INFINITE DISCOVERY)
echo ======================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] Python not found. Please install Python.
    pause
    exit /b
)

:: 2. Check Libraries
echo [Info] Checking libraries...
python -c "import yfinance, pandas, numpy, torch, sklearn, tqdm" >nul 2>nul
if %errorlevel% neq 0 (
    echo [Info] Installing missing components...
    pip install yfinance pandas numpy torch scikit-learn tqdm
)

echo [Success] Launching Infinite Master Discovery Engine...
echo.

:: Set environment to UTF-8
set PYTHONIOENCODING=utf-8
python alpha_engine_sigma.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] Execution failed. Check the error message above.
)

pause
