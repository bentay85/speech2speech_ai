@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Virtual environment not found in .venv - see readme.md for setup.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "src\app.py"
