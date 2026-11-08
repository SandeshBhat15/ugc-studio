@echo off
title UGC Studio
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem Find a Python 3 interpreter: PATH first, then the Windows launcher.
set PY=
where python >nul 2>nul && set PY=python
if not defined PY (where py >nul 2>nul && set PY=py -3)
if not defined PY (
  echo.
  echo   Python 3 was not found.
  echo   Install it from https://www.python.org/downloads/ and tick
  echo   "Add python.exe to PATH" during setup, then run this file again.
  echo.
  pause
  exit /b 1
)

echo.
echo   UGC Studio
echo   ----------
echo   Starting... your browser will open shortly.
echo   Keep this window open while you work.
echo   Press Ctrl+C here to stop the app.
echo.
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8756"
%PY% server.py
pause
