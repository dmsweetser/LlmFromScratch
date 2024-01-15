@echo off
setlocal enabledelayedexpansion

echo Setting up Python virtual environment...

REM Check if Python is installed
where py -V:3.9 > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python before running this script.
    exit /b 1
)

REM Create a virtual environment
py -V:3.9 -m venv venv
if %errorlevel% neq 0 (
    echo Error: Unable to create virtual environment.
    exit /b 1
)

echo Virtual environment created successfully.

REM Activate the virtual environment
call venv\Scripts\Activate.bat
if %errorlevel% neq 0 (
    echo Error: Unable to activate virtual environment.
    exit /b 1
)

echo Virtual environment activated successfully.

echo Installing required packages...

REM Install required packages
pip install -r requirements.txt
export LD_LIBRARY_PATH=/gnu/store/v8d7j5i02nfz951x1szbl9xrd873vc3l-zlib-1.2.12/lib:$LD_LIBRARY_PATH
if %errorlevel% neq 0 (
    echo Error: Unable to install required packages.
    exit /b 1
)

echo Required packages installed successfully.

echo.
echo Environment setup complete.
echo To activate the virtual environment, run: venv\Scripts\Activate
