@echo off
cd /d "%~dp0"
if not exist .venv\setup_ok (
  echo First run: installing everything. This takes 5-15 minutes, one time only.
  call setup.bat --from-start
  if not exist .venv\setup_ok (echo Setup didn't finish. Send Claude a screenshot. & pause & exit /b 1)
)
if not exist .env copy .env.example .env >nul
call .venv\Scripts\activate.bat
python -m fsd %*
if errorlevel 1 pause
