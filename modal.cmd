@echo off
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

where modal.exe >nul 2>nul
if not errorlevel 1 goto use_modal

where py.exe >nul 2>nul
if not errorlevel 1 goto use_py

where python.exe >nul 2>nul
if not errorlevel 1 goto use_python

echo Modal CLI was not found. Install it with "python -m pip install modal" or add it to PATH.
exit /b 1

:use_modal
call modal.exe %*
exit /b %errorlevel%

:use_py
py.exe -3 -m modal %*
exit /b %errorlevel%

:use_python
python.exe -m modal %*
exit /b %errorlevel%
