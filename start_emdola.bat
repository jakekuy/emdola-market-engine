@echo off
title EMDOLA Server
cd /d "%~dp0"
echo Starting EMDOLA...
start "" /b powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:8000'"
.venv\Scripts\uvicorn backend.main:app
