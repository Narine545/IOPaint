@echo off
setlocal
title IOPaint - free disk space
cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
call "%~dp0settings.cmd"
echo This deletes every model and cache IOPaint downloaded.
echo Your results in %IOPAINT_OUTPUT_DIR% are NOT touched.
echo Models are downloaded again automatically when needed.
echo.
"%PY%" -m iopaint clean --all --model-dir "%IOPAINT_MODEL_DIR%" --output-dir "%IOPAINT_OUTPUT_DIR%"
echo.
pause
