@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if not exist "%PROJECT_ROOT%.venv\Scripts\pythonw.exe" (
    echo Python environment not found. Run scripts\setup.ps1 first.
    pause
    exit /b 1
)
"%PROJECT_ROOT%.venv\Scripts\pythonw.exe" "%PROJECT_ROOT%scripts\audio\gui.py"
