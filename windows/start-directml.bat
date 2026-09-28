@echo off
setlocal enabledelayedexpansion
title IOPaint (DirectML / Radeon iGPU - experimental)

cd /d "%~dp0.."
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo IOPaint is not installed yet, run windows\install.bat first.
    pause
    exit /b 1
)

call "%~dp0settings.cmd"

set "XDG_CACHE_HOME=%IOPAINT_MODEL_DIR%"
set "U2NET_HOME=%IOPAINT_MODEL_DIR%"
if not exist "%IOPAINT_MODEL_DIR%" mkdir "%IOPAINT_MODEL_DIR%"

rem ---- make sure the DirectML build of onnxruntime is installed -----------
"%PY%" -c "import onnxruntime,sys; sys.exit(0 if 'DmlExecutionProvider' in onnxruntime.get_available_providers() else 1)" >nul 2>&1
if errorlevel 1 (
    echo Installing onnxruntime-directml ...
    "%PY%" -m pip uninstall -y onnxruntime >nul 2>&1
    "%PY%" -m pip install onnxruntime-directml
    if errorlevel 1 (
        echo [ERROR] Could not install onnxruntime-directml.
        pause
        exit /b 1
    )
)

set "EXTRA="
if /i "%IOPAINT_INBROWSER%"=="true" set "EXTRA=%EXTRA% --inbrowser"
if not "%IOPAINT_OUTPUT_DIR%"=="" (
    if not exist "%IOPAINT_OUTPUT_DIR%" mkdir "%IOPAINT_OUTPUT_DIR%"
    set "EXTRA=%EXTRA% --output-dir "%IOPAINT_OUTPUT_DIR%""
)

echo ============================================================
echo   IOPaint (DirectML)  -  http://127.0.0.1:%IOPAINT_PORT%
echo   Background removal runs on the Radeon/Intel GPU,
echo   the erase model (%IOPAINT_MODEL%) stays on the CPU.
echo   Note: on a 2 GB iGPU big images may fall back to the CPU.
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
    --remove-bg-device directml ^
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
