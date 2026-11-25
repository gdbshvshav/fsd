@echo off
setlocal
cd /d "%~dp0"

rem ===== Auto-update from GitHub (runs every start) =====
rem Everything in this block is read by Windows in one go, so replacing
rem start.bat during the update is safe.
if /i not "%~1"=="--no-update" (
  where git >nul 2>nul || (
    echo Installing Git so FSD can update itself ^(one time^)...
    winget install -e --id Git.Git --silent --accept-package-agreements --accept-source-agreements >nul 2>nul
    set "PATH=%PATH%;%ProgramFiles%\Git\cmd"
  )
  where git >nul 2>nul && (
    if not exist .git (
      git init -q
      git remote add origin https://github.com/gdbshvshav/fsd.git
      echo First update: a GitHub sign-in window may pop up once. Sign in with your GitHub account.
    )
    echo Checking GitHub for updates...
    git fetch -q origin main && git reset -q --hard origin/main && (
      for /f %%h in ('git rev-parse --short HEAD') do echo Up to date ^(%%h^).
    ) || echo Couldn't reach GitHub, starting the version you have.
  ) || echo Git isn't available, skipping updates.
  call "%~f0" --no-update %*
  exit /b
)

rem ===== First-time install =====
if not exist .venv\setup_ok (
  echo First run: installing everything. This takes 5-15 minutes, one time only.
  call setup.bat --from-start
  if not exist .venv\setup_ok (echo Setup didn't finish. Send Claude a screenshot. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat

rem ===== New packages after an update =====
for /f %%h in ('certutil -hashfile requirements.txt MD5 ^| find /v ":"') do set REQHASH=%%h
set OLDHASH=
if exist .venv\req_hash set /p OLDHASH=<.venv\req_hash
if not "%REQHASH%"=="%OLDHASH%" (
  echo Installing new packages from the update...
  python -m pip install -q -r requirements.txt && (echo %REQHASH%> .venv\req_hash)
)

if not exist .env if exist .env.example copy .env.example .env >nul
shift
python -m fsd %1 %2 %3 %4 %5
if errorlevel 1 pause
