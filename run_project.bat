@echo off
rem ==========================================================================
rem  MP3D Lesion Detector - one-click setup and launch for Windows 10 / 11.
rem
rem  First run (needs internet; about 3.5 GB of downloads, 15-30 minutes):
rem    the Visual C++ runtime if missing, uv, Python 3.10, PyTorch and mmcv,
rem    the trained model and the demo CT slices - all kept inside this folder.
rem  Later runs skip all of that, start the app straight away, work offline.
rem ==========================================================================
setlocal
pushd "%~dp0"
title MP3D Lesion Detector

set "HERE=%CD%"
set "TOOLS=%HERE%\.tools"
set "UV=%TOOLS%\uv.exe"
set "UV_VERSION=0.10.9"
set "PY=%HERE%\.venv\Scripts\python.exe"
set "UV_CACHE_DIR=%TOOLS%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_NO_CONFIG=1"
set "CURL=%SystemRoot%\System32\curl.exe"
set "TAR=%SystemRoot%\System32\tar.exe"

echo.
echo  MP3D Lesion Detector
echo  ====================
echo.

if not exist "%HERE%\webapp\app.py" goto :not_extracted
if not exist "%CURL%" goto :old_windows
if not exist "%TAR%" goto :old_windows

rem --- 1. Microsoft Visual C++ runtime, which PyTorch needs ------------------
if exist "%SystemRoot%\System32\msvcp140.dll" if exist "%SystemRoot%\System32\vcruntime140_1.dll" goto :vcredist_ok
echo [1/5] Installing the Microsoft Visual C++ runtime.
echo       Windows will ask for permission - please click Yes.
"%CURL%" -L --fail --retry 5 --retry-delay 3 -o "%TEMP%\mp3d_vc_redist.x64.exe" https://aka.ms/vs/17/release/vc_redist.x64.exe
if errorlevel 1 goto :fail_download
"%TEMP%\mp3d_vc_redist.x64.exe" /install /passive /norestart
set "RC=%ERRORLEVEL%"
del "%TEMP%\mp3d_vc_redist.x64.exe" >nul 2>&1
if "%RC%"=="0" goto :vcredist_ok
if "%RC%"=="1638" goto :vcredist_ok
if "%RC%"=="3010" goto :vcredist_ok
echo ERROR: the Visual C++ runtime was not installed (installer code %RC%).
goto :fail
:vcredist_ok
echo [1/5] Microsoft Visual C++ runtime: OK

rem --- 2. uv, which installs Python and the packages ------------------------
if exist "%UV%" goto :uv_ok
echo [2/5] Downloading uv %UV_VERSION%...
if not exist "%TOOLS%" mkdir "%TOOLS%"
"%CURL%" -L --fail --retry 5 --retry-delay 3 -o "%TOOLS%\uv.zip" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 goto :fail_download
"%TAR%" -xf "%TOOLS%\uv.zip" -C "%TOOLS%"
if errorlevel 1 goto :fail
del "%TOOLS%\uv.zip"
if not exist "%UV%" goto :fail
:uv_ok
echo [2/5] uv: OK

rem --- 3. Python 3.10 virtual environment in .venv -------------------------
if not exist "%PY%" goto :make_venv
"%PY%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto :venv_ok
echo The .venv folder was made on another PC and does not work here; rebuilding it.
:make_venv
echo [3/5] Creating the Python 3.10 environment...
"%UV%" venv --clear --managed-python --python 3.10 "%HERE%\.venv"
if errorlevel 1 goto :fail
:venv_ok
echo [3/5] Python environment: OK

rem --- 4. Python packages: PyTorch 1.12 (CUDA 11.3), mmcv-full 1.6, ... -----
if exist "%HERE%\.venv\mp3d_packages.ok" goto :packages_ok
"%PY%" tools\check_env.py >nul 2>&1
if not errorlevel 1 goto :packages_mark
echo [4/5] Installing Python packages - about 3 GB to download, first run only...
powershell -NoProfile -Command "if ((Get-PSDrive -Name '%HERE:~0,1%').Free -lt 8GB) { exit 2 }" >nul 2>&1
if errorlevel 2 if not errorlevel 3 goto :no_space
"%UV%" pip install --python "%PY%" -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
if errorlevel 1 goto :fail
"%UV%" pip install --python "%PY%" -e . --no-deps --no-build-isolation
if errorlevel 1 goto :fail
"%UV%" cache clean >nul 2>&1
"%PY%" tools\check_env.py
if errorlevel 1 goto :fail
:packages_mark
echo ok> "%HERE%\.venv\mp3d_packages.ok"
:packages_ok
echo [4/5] Python packages: OK

rem --- 5. Trained model and demo data from the GitHub release ---------------
echo [5/5] Checking the trained model and demo data...
"%PY%" tools\fetch_release_files.py
if errorlevel 1 goto :fail

echo.
echo  Starting the web app. Loading the model takes up to a minute, then
echo  your browser opens by itself.
echo  Keep this window open while you use the app - close it to stop the app.
echo.
if not defined MP3D_OPEN_BROWSER set "MP3D_OPEN_BROWSER=1"
"%PY%" webapp\app.py
if errorlevel 1 goto :fail
popd
exit /b 0

:not_extracted
echo ERROR: run this file from inside the project folder. If you downloaded
echo the project as a ZIP file, extract the whole ZIP first, then run it again.
goto :fail_end

:old_windows
echo ERROR: this needs Windows 10 version 1803 or newer, or Windows 11.
goto :fail_end

:no_space
echo ERROR: not enough free disk space. The first run needs about 8 GB free
echo on the drive that holds this folder.
goto :fail_end

:fail_download
echo.
echo ERROR: a download failed. Check the internet connection and run this
echo file again.
goto :fail_end

:fail
echo.
echo ERROR: the step above did not finish. Run this file again - it carries on
echo where it stopped. If it keeps failing, take a screenshot of this window.

:fail_end
echo.
pause
popd
exit /b 1
