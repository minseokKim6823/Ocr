@echo off
rem Stock code/name extractor - double-click to run, or drag an image onto it.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if "%~1"=="" (
    ".venv\Scripts\python.exe" vis_fields.py
) else (
    ".venv\Scripts\python.exe" vis_fields.py %* --open
)

echo.
echo Done. Result images saved next to originals as *_fields.jpg
pause
