
@echo off
title DocuChat Launcher

cd /d "%~dp0"

echo Starting DocuChat backend...
start "DocuChat Backend" cmd /k "cd /d "%~dp0backend" && .\venv\Scripts\python.exe -m uvicorn main:app --reload"

timeout /t 3 /nobreak >nul

echo Starting DocuChat frontend...
start "DocuChat Frontend" cmd /k "cd /d "%~dp0frontend" && python -m http.server 5500"

timeout /t 2 /nobreak >nul

start "" "http://localhost:5500"

echo DocuChat launch commands sent.
