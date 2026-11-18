@echo off
setlocal enabledelayedexpansion
title Auto Document Fill
cd /d "%~dp0"

echo ========================================================
echo   Auto Document Fill System
echo ========================================================
echo.

set "PY_EXE=C:\Users\antar\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "!PY_EXE!" (
    set "PY_EXE=python"
)

echo Checking Python environment...
"!PY_EXE!" --version
if errorlevel 1 (
    echo [ERROR] Python not found or failed to execute.
    echo Please ensure Python is installed and accessible.
    pause
    exit /b 1
)

echo Freeing port 5000 if in use...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

echo.
echo Starting Flask web server...
echo The application will automatically open in your default browser at:
echo http://127.0.0.1:5000
echo.
echo (Keep this black command window open while using the app)
echo.

:: Automatically open default browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

"!PY_EXE!" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application stopped unexpectedly.
    pause
)
