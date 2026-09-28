@echo off
setlocal enabledelayedexpansion
title IOPaint - installation

cd /d "%~dp0.."
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"

echo ============================================================
echo   IOPaint installation (CPU build, optimized for AMD/Intel)
echo ============================================================
echo.

rem ---------------------------------------------------------------- python
set "BASEPY="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined BASEPY (
        py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
    )
)
if not defined BASEPY (
    python -c "import sys; assert sys.version_info[:2] >= (3,10)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
    echo [ERROR] Python 3.10 - 3.13 was not found.
    echo.
    echo Install it from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" during the installation.
    echo.
    pause
    exit /b 1
)
echo [1/6] Using interpreter: %BASEPY%
%BASEPY% --version

rem ------------------------------------------------------------------ venv
if not exist "%PY%" (
    echo [2/6] Creating virtual environment in .venv ...
    %BASEPY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/6] Virtual environment already exists, reusing it.
)

"%PY%" -m pip install --upgrade pip wheel >nul
if errorlevel 1 (
    echo [ERROR] pip upgrade failed. Check your internet connection / proxy.
    pause
    exit /b 1
)

rem ----------------------------------------------------------------- torch
echo [3/6] Installing PyTorch (CPU build, ~250 MB instead of ~2.5 GB CUDA) ...
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed.
        pause
        exit /b 1
    )
) else (
    echo      PyTorch is already installed, skipping.
)

rem --------------------------------------------------------------- iopaint
echo [4/6] Installing IOPaint + background removal plugins ...
"%PY%" -m pip install -e ".[plugins]"
if errorlevel 1 (
    echo [ERROR] IOPaint installation failed.
    pause
    exit /b 1
)

rem ------------------------------------------------------- model download
call "%~dp0settings.cmd"
set "XDG_CACHE_HOME=%IOPAINT_MODEL_DIR%"
set "U2NET_HOME=%IOPAINT_MODEL_DIR%"
if not exist "%IOPAINT_MODEL_DIR%" mkdir "%IOPAINT_MODEL_DIR%"

echo [5/6] Downloading the erase model (%IOPAINT_MODEL%) ...
"%PY%" -m iopaint download --model %IOPAINT_MODEL% --model-dir "%IOPAINT_MODEL_DIR%"

echo [6/6] Downloading the background removal model (%IOPAINT_REMOVE_BG_MODEL%) ...
"%PY%" -c "import os; os.environ.setdefault('U2NET_HOME', r'%IOPAINT_MODEL_DIR%'); from rembg import new_session; new_session('%IOPAINT_REMOVE_BG_MODEL%'); print('ok')" 2>nul
if errorlevel 1 echo      (the model will be downloaded on first use)

echo.
echo ============================================================
echo   Done! Start IOPaint with:  start.bat
echo   (or the IOPaint.bat shortcut in the project root)
echo ============================================================
echo.
pause
