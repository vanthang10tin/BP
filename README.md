# BP - AI Data Annotation & Quality Control (BDD100K)

Dự án nghiên cứu, thiết kế quy trình và phát triển công cụ kiểm soát chất lượng dữ liệu gán nhãn AI (Quality Control - QC) cho tập dữ liệu BDD100K (2D Bounding Box & Polygon Segmentation), đề tài **N2-04A - Quality Control Check**.

---

## 📁 Cấu Trúc Repository

```text
BP/
├── .gitignore                     # Cấu hình bỏ qua file rác, cache, venv
├── README.md                      # Tài liệu tổng quan dự án và hướng dẫn sử dụng
├── flow/                          # Hình ảnh minh chứng quy trình thao tác và luồng nghiệp vụ
│   ├── Screenshot 2026-10-13 204421.png
│   └── ... (ảnh chụp màn hình quy trình gán nhãn & QC)
└── source-tool/
    ├── README.md                  # Hướng dẫn và danh mục các công cụ
    └── qc-bdd100k/                # Ứng dụng QC Studio 2D tự động cho BDD100K
        ├── backend/               # Server FastAPI / Python HTTP server & AI Consensus Engine
        ├── frontend/              # Web Dashboard hiện đại (Dark Modern UI, Canvas review)
        ├── sample_test_labels.json# File nhãn kiểm thử mẫu
        ├── real_val_sample_20.json# Bộ nhãn mẫu 20 frame thực tế từ BDD100K
        ├── qc_checker.py          # Công cụ dòng lệnh CLI kiểm tra nhanh
        ├── requirements.txt       # Danh sách thư viện phụ thuộc
        ├── run_local.bat          # Script 1-click chạy trên Windows
        ├── run_local.sh           # Script 1-click chạy trên Linux/macOS
        ├── Dockerfile             # Container hóa phục vụ triển khai
        └── README.md              # Tài liệu chi tiết kỹ thuật hệ thống QC 4 tầng
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Nhanh (Quick Start)

Dự án hỗ trợ chạy trên cả **Windows**, **macOS** và **Linux**.

### Cách 1: Chạy 1-Click (Khuyên dùng)

* **Trên Windows:**
  Nhấp đúp chuột vào file:
  ```text
  source-tool/qc-bdd100k/run_local.bat
  ```
  *(Script sẽ tự động cài `requirements.txt` và khởi động server)*

* **Trên Linux / macOS:**
  Mở terminal tại thư mục `source-tool/qc-bdd100k/` và chạy:
  ```bash
  chmod +x run_local.sh
  ./run_local.sh
  ```

Sau khi server khởi động, mở trình duyệt web truy cập:
👉 **`http://localhost:8000`**

---

### Cách 2: Chạy Thủ Công Bằng Python

1. **Di chuyển vào thư mục tool:**
   ```bash
   cd source-tool/qc-bdd100k
   ```

2. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Lưu ý: Hệ thống hỗ trợ chế độ Zero-Dependency Fallback với thư viện chuẩn của Python nếu chưa cài FastAPI)*

3. **Khởi chạy ứng dụng Web Dashboard:**
   ```bash
   python backend/app.py
   ```
   Truy cập: **`http://localhost:8000`**

---

### Cách 3: Chạy Qua Dòng Lệnh CLI (`qc_checker.py`)

Nếu bạn muốn kiểm tra dữ liệu nhãn trực tiếp trên Terminal:

* **Kiểm tra và xuất báo cáo 4 Tầng QA:**
  ```bash
  cd source-tool/qc-bdd100k
  python qc_checker.py --input sample_test_labels.json --verbose
  ```

* **Tự động sửa lỗi kẹp biên & xóa box rác (1-Click Auto-Heal):**
  ```bash
  python qc_checker.py --input sample_test_labels.json --auto-heal --output cleaned_labels.json
  ```

---

### Cách 4: Chạy Bằng Docker

```bash
cd source-tool/qc-bdd100k
docker build -t qc-studio-2d .
docker run -p 8000:8000 qc-studio-2d
```

---

## 🧪 Chạy Kiểm Thử Tự Động (Test Suite)

Để đảm bảo toàn bộ hệ thống hoạt động chính xác trước khi sử dụng:

```bash
cd source-tool/qc-bdd100k/backend
python test_qc_system.py
```
Toàn bộ **8/8 bài test** (Guideline Manager, Multi-Model Consensus, 4 Tầng QA, 1-Click Auto-Heal, Xuất Markdown, Chuẩn hóa tọa độ COCO/YOLO/VOC/BDD/Gemini, Honeypot, Benchmark Recall@20%) sẽ được thực thi tự động.

---

## 👥 Tác Giả & Đóng Góp

* **Đoàn Văn Thắng** ([@vanthang10tin](https://github.com/vanthang10tin))
* Đề tài: **N2-04A - QC Tự Động Cho BBox & Segmentation 2D BDD100K**