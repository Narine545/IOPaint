@echo off
setlocal
title IOPaint - where are my files
cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
call "%~dp0settings.cmd"
"%PY%" -m iopaint paths --model-dir "%IOPAINT_MODEL_DIR%" --output-dir "%IOPAINT_OUTPUT_DIR%"
echo.
pause
