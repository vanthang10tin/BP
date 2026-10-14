@echo off
chcp 65001 >nul
echo ========================================================
echo   🚀 KHỞI ĐỘNG QC STUDIO 2D - BDD100K (N2-04A)
echo ========================================================
echo.
cd /d "%~dp0"

set PY_CMD=python
where python >nul 2>&1
if errorlevel 1 (
    where py >nul 2>&1
    if errorlevel 1 (
        echo [LỖI] Không tìm thấy Python hoặc py trên máy tính!
        echo Vui lòng cài đặt Python 3.8+ từ https://python.org và tích chọn Add to PATH.
        pause
        exit /b 1
    ) else (
        set PY_CMD=py
    )
)

echo [1/2] Kiểm tra thư viện phụ thuộc (%PY_CMD%)...
%PY_CMD% -m pip install -r requirements.txt --quiet

echo [2/2] Bắt đầu chạy Web Dashboard Server...
echo Mở trình duyệt tại: http://localhost:8000
echo Nhấn Ctrl+C để dừng server.
echo.
%PY_CMD% backend/app.py
pause
