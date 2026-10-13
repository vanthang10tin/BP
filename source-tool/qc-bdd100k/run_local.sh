#!/usr/bin/env bash
echo "========================================================"
echo "  🚀 KHỞI ĐỘNG QC STUDIO 2D - BDD100K (N2-04A)"
echo "========================================================"
echo ""
cd "$(dirname "$0")"

echo "[1/2] Kiểm tra thư viện..."
pip install -r requirements.txt --quiet

echo "[2/2] Bắt đầu chạy Web Dashboard Server..."
echo "Mở trình duyệt tại: http://localhost:8000"
echo "Nhấn Ctrl+C để dừng server."
echo ""
python backend/app.py
