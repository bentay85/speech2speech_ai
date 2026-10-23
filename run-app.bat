@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found in .venv - see readme.md for setup.
    pause
    exit /b 1
)
start "" ".venv\Scripts\python.exe" "src\app.py"
