@echo off
setlocal enabledelayedexpansion
title IOPaint

cd /d "%~dp0.."
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo IOPaint is not installed yet, running the installer first...
    call "%~dp0install.bat"
    if not exist "%PY%" exit /b 1
)

call "%~dp0settings.cmd"

set "XDG_CACHE_HOME=%IOPAINT_MODEL_DIR%"
set "U2NET_HOME=%IOPAINT_MODEL_DIR%"
if not exist "%IOPAINT_MODEL_DIR%" mkdir "%IOPAINT_MODEL_DIR%"

set "EXTRA="
if /i "%IOPAINT_INBROWSER%"=="true" set "EXTRA=%EXTRA% --inbrowser"
if not "%IOPAINT_OUTPUT_DIR%"=="" (
    if not exist "%IOPAINT_OUTPUT_DIR%" mkdir "%IOPAINT_OUTPUT_DIR%"
    set "EXTRA=%EXTRA% --output-dir "%IOPAINT_OUTPUT_DIR%""
)

echo ============================================================
echo   IOPaint  -  http://127.0.0.1:%IOPAINT_PORT%
echo   erase model      : %IOPAINT_MODEL%
echo   remove bg model  : %IOPAINT_REMOVE_BG_MODEL%
echo   alpha matting    : %IOPAINT_ALPHA_MATTING%
echo   Close this window to stop the server.
echo ============================================================
echo.

"%PY%" -m iopaint start ^
    --model %IOPAINT_MODEL% ^
    --device cpu ^
    --port %IOPAINT_PORT% ^
    --host 127.0.0.1 ^
    --cpu-threads %IOPAINT_CPU_THREADS% ^
    --model-dir "%IOPAINT_MODEL_DIR%" ^
    --enable-remove-bg ^
    --remove-bg-model %IOPAINT_REMOVE_BG_MODEL% ^
    --remove-bg-device cpu ^
    --remove-bg-alpha-matting %IOPAINT_ALPHA_MATTING% ^
    --remove-bg-matting-max-size %IOPAINT_MATTING_MAX_SIZE% ^
    --remove-bg-remove-color-spill ^
    --enable-anime-seg ^
    --enable-interactive-seg ^
    --interactive-seg-model sam2_1_tiny ^
    --interactive-seg-device cpu ^
    %EXTRA%

if errorlevel 1 (
    echo.
    echo IOPaint exited with an error, see the log above.
    pause
)
