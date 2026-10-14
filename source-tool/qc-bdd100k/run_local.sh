#!/usr/bin/env bash
echo "========================================================"
echo "  🚀 KHỞI ĐỘNG QC STUDIO 2D - BDD100K (N2-04A)"
echo "========================================================"
echo ""
cd "$(dirname "$0")"

# Tự động nhận diện python3 hoặc python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "❌ [LỖI] Không tìm thấy Python trên hệ thống!"
    echo "Vui lòng cài đặt Python 3.8+ (ví dụ: sudo apt install python3 python3-pip)."
    exit 1
fi

echo "[1/2] Kiểm tra thư viện phụ thuộc ($PY_CMD)..."
$PY_CMD -m pip install -r requirements.txt --quiet

echo "[2/2] Bắt đầu chạy Web Dashboard Server..."
echo "Mở trình duyệt tại: http://localhost:8000"
echo "Nhấn Ctrl+C để dừng server."
echo ""
$PY_CMD backend/app.py
