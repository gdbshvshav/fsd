@echo off
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
:menu
echo.
echo ===== ER:LC FSD tools =====
echo  1  Calibrate camera (auto, 5 s countdown)
echo  2  Calibrate camera (click the lane lines)
echo  3  Check calibration picture
echo  4  Dry run (shows what FSD would do, presses no keys)
echo  5  Training: label recorded frames with Gemini
echo  6  Training: review labels
echo  7  Training: train the ER:LC model on your GPU
echo  8  Score saved disengage scenarios
echo  9  Undo last learned fix (roll back params)
echo  0  Exit
set /p c=Pick: 
if "%c%"=="1" python -m fsd.tools.calibrate
if "%c%"=="2" python -m fsd.tools.calibrate --manual
if "%c%"=="3" python -m fsd.tools.inspect_image data\calibration_frame.png
if "%c%"=="4" python -m fsd --dry-run
if "%c%"=="5" python -m fsd.learning.training label
if "%c%"=="6" python -m fsd.learning.training review
if "%c%"=="7" python -m fsd.learning.training train
if "%c%"=="8" python -m fsd.learning.replay
if "%c%"=="9" python -c "from fsd.params import ParamStore; print('now on v', ParamStore().rollback())"
if "%c%"=="0" exit /b 0
goto menu
