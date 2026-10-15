@echo off
chcp 65001 >nul
title QC Studio 2D BDD100K - CVAT Integrated
echo ========================================================
echo   🚀 DANG KHOI DONG QC STUDIO + CVAT INTEGRATION...
echo ========================================================
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File .\run_qc_studio.ps1 %*
pause
