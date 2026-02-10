@echo off
@setlocal enabledelayedexpansion
title LOOK-DGC Launcher
echo ================================================
echo   LOOK-DGC - Digital Image Forensics Toolkit
echo   Developed by: Gopichand
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11 or later
    pause
    exit /b 1
)

REM Check if gui directory exists
if not exist "gui" (
    echo Error: GUI directory not found
    echo Please ensure you are running this from the LOOK-DGC root directory
    pause
    exit /b 1
)

REM Check and install dependencies
REM First validate dependencies, then install if needed
echo Validating dependencies...
cd gui

python ../validate_deps.py
if errorlevel 1 (
    echo Some dependencies are missing. Attempting to install...
    python ../check_deps.py
    if errorlevel 1 (
        echo Warning: Some dependencies may have failed to install
        echo Continuing with launch attempt...
        echo If you encounter issues, try installing dependencies manually:
        echo   pip install -r gui/requirements.txt
        echo.
    )
)

cd ..

REM Launch LOOK-DGC
echo Starting LOOK-DGC...
python launch_look_dgc.py

echo.
echo LOOK-DGC has closed.
pause