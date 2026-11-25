@echo off
setlocal
cd /d "%~dp0"
echo.
echo ===== ER:LC FSD setup =====
echo.
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; assert sys.version_info >= (3,10), sys.version" 2>nul || (
  echo Python 3.10+ is required. Get it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause & exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing PyTorch with NVIDIA GPU support (about 2.5 GB, one time)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || (
  echo GPU build failed, installing CPU build instead.
  python -m pip install torch torchvision
)
echo.
echo Installing the rest...
python -m pip install -r requirements.txt || (echo Install failed & pause & exit /b 1)
echo.
echo Downloading the starter vision model...
python -c "from ultralytics import YOLO; YOLO(r'models\yolo11s.pt')"
if not exist .env copy .env.example .env >nul
echo.
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none found, using CPU (slower)')"
python -m pytest -q tests
echo.
echo Done. Next:
echo   1. Open .env in Notepad and paste your free Gemini and Groq keys (optional but recommended)
echo   2. Run start.bat
pause
