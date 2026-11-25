@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo.
echo ===== ER:LC FSD setup =====
echo Folder: %CD%
echo.
if not exist "fsd\app.py" (
  echo The FSD files aren't next to setup.bat.
  echo Download the whole repo: GitHub ^> green Code button ^> Download ZIP, extract ALL of it,
  echo then run setup.bat from inside the extracted folder.
  pause & exit /b 1
)

rem ---- pick Python: 3.12 / 3.11 / 3.13 work with every package ----
set PY=
for %%V in (3.12 3.11 3.13) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set PY=py -%%V
  )
)
if not defined PY (
  python -c "import sys" >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python isn't installed. Get Python 3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" on the first screen.
  pause & exit /b 1
)
for /f "delims=" %%v in ('%PY% -c "import sys; print('%%d.%%d' %% sys.version_info[:2])"') do set PYVER=%%v
echo Using Python %PYVER%
if "%PYVER%"=="3.14" (
  echo NOTE: Python 3.14 is very new and some parts may not install. If you see errors,
  echo install Python 3.12 from python.org, then run setup.bat again ^(it switches automatically^).
)

rem ---- recreate .venv if it was made with a different Python ----
if exist .venv\Scripts\python.exe (
  for /f "delims=" %%v in ('.venv\Scripts\python.exe -c "import sys; print('%%d.%%d' %% sys.version_info[:2])"') do set VENVVER=%%v
  if not "!VENVVER!"=="%PYVER%" (
    echo Old environment uses Python !VENVVER!, rebuilding for %PYVER%...
    rmdir /s /q .venv
  )
)
if not exist .venv (
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo [1/4] PyTorch with NVIDIA GPU support (about 2.5 GB, one time)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || (
  echo GPU build failed, installing CPU build instead.
  python -m pip install torch torchvision
)

echo.
echo [2/4] Main packages...
python -m pip install "numpy>=1.26" "opencv-python>=4.9" "pygame>=2.5" "requests>=2.31" "mss>=9.0" "ultralytics>=8.3" "pytest>=8.0" "dxcam>=0.0.5" || (
  echo Install failed. Send a screenshot of the error.
  pause & exit /b 1
)

echo.
echo [3/4] Text reader for the speedometer and speed signs...
python -m pip install "rapidocr-onnxruntime>=1.3" || (
  echo RapidOCR isn't available for this Python, using EasyOCR instead...
  python -m pip install easyocr || (echo OCR install failed. Send a screenshot. & pause & exit /b 1)
)

echo.
echo [4/4] Starter vision model...
python -c "from ultralytics import YOLO; YOLO(r'models\yolo11s.pt')"
if not exist .env copy .env.example .env >nul

echo.
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none found - update your NVIDIA driver, using CPU for now')"
python -m pytest -q tests
echo.
echo Done. Next:
echo   1. Open .env in Notepad and paste your Gemini and Groq keys (NOT .env.example)
echo   2. Run start.bat
pause
