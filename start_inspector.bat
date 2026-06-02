@echo off
title LED Pin Inspection System - HMI Launcher
echo ==========================================================
echo       LED Pin Inspection System - Startup Launcher
echo ==========================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Python is not installed. Attempting one-click installation via Windows Package Manager (winget)...
    where winget >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python and winget are both missing on this system.
        echo Please download and install Python manually: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    
    echo Installing Python 3... This may take a minute...
    winget install --id Python.Python.3.11 --silent --show-progress --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [ERROR] Silent installation failed. Opening download page instead...
        start https://www.python.org/downloads/
        pause
        exit /b 1
    )
    
    echo.
    echo [INFO] Python installed successfully.
    echo Please close this window and run start_inspector.bat again to complete setup!
    echo.
    pause
    exit /b 0
)

echo [1/3] Python installation found.
echo.

:: Check and install OpenCV and Numpy if missing
echo [2/3] Checking required libraries (opencv-python, numpy)...
python -c "import cv2, numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Required dependencies not found. Installing now...
    python -m pip install opencv-python numpy
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Check your internet connection.
        pause
        exit /b 1
    )
)
echo Dependencies check completed successfully.
echo.

:: Auto-start Web HMI in default web browser
echo [3/3] Launching Industrial Web HMI in default browser...
start http://localhost:8000/
echo.

echo ==========================================================
echo   System is now starting. Keep this console window open!
echo   To stop the inspection server, close this window or
echo   press Ctrl+C in this console.
echo ==========================================================
echo.

python src/main.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] The inspection server closed with error code %errorlevel%.
    pause
)
