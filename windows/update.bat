@echo off
setlocal
title IOPaint - update

cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"

echo Pulling the latest changes ...
git pull
if errorlevel 1 echo (git pull failed or this is not a git checkout - continuing)

if not exist "%PY%" (
    echo IOPaint is not installed yet, run windows\install.bat first.
    pause
    exit /b 1
)

echo Updating the Python package ...
"%PY%" -m pip install -e ".[plugins]" --upgrade-strategy only-if-needed

echo.
echo Done.
pause
