@echo off
setlocal
title IOPaint - install diffusion models support

cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo IOPaint is not installed yet, run windows\install.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo  This installs diffusers/transformers/peft (~3 GB on disk)
echo  so that Stable Diffusion / PowerPaint / AnyText models can
echo  be used. On a CPU-only machine with 16 GB of RAM these
echo  models are SLOW (minutes per image) - erase models such as
echo  lama/migan are a much better fit for your hardware.
echo ============================================================
echo.
set /p ANSWER="Continue? [y/N] "
if /i not "%ANSWER%"=="y" exit /b 0

"%PY%" -m pip install -e ".[diffusion]"
if errorlevel 1 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

echo.
echo Done. Start the server with e.g.:
echo   .venv\Scripts\python -m iopaint start --model runwayml/stable-diffusion-inpainting --device cpu --low-mem
pause
