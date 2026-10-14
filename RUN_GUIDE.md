# HƯỚNG DẪN CHẠY CODE TỪ A-Z (RUN GUIDE)
### Công Cụ: QC Studio 2D - BDD100K (Đề tài N2-04A)
**Tác giả:** Đoàn Văn Thắng ([@vanthang10tin](https://github.com/vanthang10tin)) | Nhóm P-037

---

Tài liệu này cung cấp hướng dẫn chi tiết từng bước để bất kỳ ai khi clone repository này về đều có thể cài đặt và khởi chạy thành công 100% trên mọi hệ điều hành (**Windows**, **macOS**, **Linux**).

---

## 1. Yêu Cầu Môi Trường (Prerequisites)

- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux (Ubuntu, Debian, Fedora...).
- **Python:** Phiên bản `Python 3.8` trở lên (Khuyên dùng `Python 3.10` - `3.12`).
  *Kiểm tra phiên bản bằng lệnh:*
  ```bash
  python --version
  # hoặc
  python3 --version
  ```
- **Trình duyệt web:** Chrome, Edge, Firefox, Brave, Safari...

---

## 2. Cách Khởi Chạy Nhanh Nhất (1-Click Run)

### Trên Windows:
1. Mở thư mục:
   ```text
   source-tool\qc-bdd100k\
   ```
2. Nhấp đúp chuột (Double Click) vào file:
   ```text
   run_local.bat
   ```
3. Script sẽ tự động:
   - Chuyển mã hóa console sang UTF-8.
   - Cài đặt các thư viện cần thiết trong `requirements.txt`.
   - Khởi động Web Server tại cổng `8000`.
4. Mở trình duyệt truy cập: 👉 **`http://localhost:8000`**

---

### Trên Linux / macOS:
1. Mở Terminal và di chuyển vào thư mục tool:
   ```bash
   cd source-tool/qc-bdd100k
   ```
2. Cấp quyền thực thi và chạy script:
   ```bash
   chmod +x run_local.sh
   ./run_local.sh
   ```
3. Mở trình duyệt truy cập: 👉 **`http://localhost:8000`**

---

## 3. Cách Khởi Chạy Bằng Dòng Lệnh Python (Manual Setup)

Nếu bạn muốn tạo môi trường ảo độc lập (Virtualenv):

### Bước 1: Di chuyển vào thư mục tool
```bash
cd source-tool/qc-bdd100k
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyên dùng)
- **Trên Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```
> **Ghi chú về Zero-Dependency:** Nếu máy của bạn không thể cài đặt `fastapi`/`uvicorn` do giới hạn mạng, server được lập trình sẵn cơ chế tự động chuyển sang `http.server` tích hợp sẵn của Python mà không bị crash.

### Bước 4: Khởi chạy Web Server
```bash
python backend/app.py
```
Khi thấy thông báo:
```text
======================================================================
🚀 QC STUDIO 2D BACKEND SERVER ĐÃ KHỞI CHẠY THÀNH CÔNG!
👉 Mở trình duyệt web truy cập: http://localhost:8000
======================================================================
```
Hãy mở trình duyệt và truy cập **`http://localhost:8000`**.

---

## 4. Hướng Dẫn Sử Dụng Giao Diện Web Dashboard

Giao diện **QC Studio 2D** được thiết kế hiện đại, hỗ trợ các tính năng nghiệp vụ:

1. **Nạp dữ liệu kiểm tra:**
   - Nhấn nút **"📂 Tải File Nhãn BDD100K JSON"** để tải lên file nhãn xuất từ CVAT.
   - Hoặc nhấn **"🧪 Tải Dữ Liệu Mẫu (4 Frames)"** để thử nghiệm ngay lập tức.
   - Hoặc nhấn **"📊 Dữ Liệu Thật (20 Frames)"** để kiểm tra tập dữ liệu thực tế từ BDD100K Val.
2. **Theo dõi bảng Thống kê 4 Tầng QA:**
   - **Tầng 1 (Luật):** Đếm các lỗi hình học, tràn biên, chấm rác, trùng box, polygon tự cắt.
   - **Tầng 2 (Model ObjectLab):** Đếm các lỗi nghi vấn nhầm nhãn, lệch biên, sót vật thể.
   - **Tầng 3 (Hàng đợi Review có ưu tiên):** Xếp hạng danh sách từ **Rank 1 (Cực cao)** xuống **Rank 4 (Pass)**.
   - **Tầng 4 (Honeypot):** Điểm số Precision / Recall đối soát với đáp án chuẩn.
3. **Duyệt Hàng Đợi Review (Ưu tiên từ Cao xuống Thấp):**
   - Click vào từng thẻ frame ở cột bên trái để nạp lên Canvas trung tâm.
   - Các frame nguy cơ cao nhất được đánh dấu viền đỏ/cam và badge ưu tiên rõ ràng.
4. **Trực quan hóa Canvas:**
   - Canvas tự động vẽ các Bounding Box và Polygon Segmentation.
   - Các box bị vi phạm sẽ đổi sang màu cảnh báo (Đỏ/Cam) kèm nhãn lỗi cụ thể.
5. **Tính năng ⚡ 1-Click Auto-Heal:**
   - Nhấn nút **"⚡ 1-Click Auto-Heal"** ở thanh công cụ.
   - Hệ thống tự động kẹp các box tràn biên về [0, 1280]x[0, 720], xóa các chấm rác <16px², gỡ các box vẽ trùng và bổ sung thuộc tính bắt buộc còn thiếu.
   - Cho phép tải về file JSON đã được làm sạch ngay lập tức.
6. **Tính năng 📋 Xuất Phiếu Trả Job Markdown:**
   - Nhấn **"📋 Xuất Phiếu Trả Job"** ➔ Nhập tên Annotator ➔ Hệ thống tự động kết xuất phiếu review Markdown đầy đủ lý do, trích dẫn từng điều khoản trong Sổ Quyết Định để gửi trả lại cho annotator.

---

## 5. Chạy Kiểm Tra Qua Dòng Lệnh CLI (`qc_checker.py`)

Nếu bạn cần tích hợp vào script CI/CD hoặc muốn kiểm tra nhanh qua Terminal mà không cần mở trình duyệt:

```bash
cd source-tool/qc-bdd100k
```

### Các lệnh phổ biến:

- **Kiểm tra file nhãn và in tóm tắt kết quả:**
  ```bash
  python qc_checker.py --input sample_test_labels.json
  ```

- **In chi tiết từng frame, từng lỗi và điều khoản vi phạm:**
  ```bash
  python qc_checker.py --input sample_test_labels.json --verbose
  ```

- **Tự động sửa lỗi (Auto-Heal) và xuất file sạch:**
  ```bash
  python qc_checker.py --input sample_test_labels.json --auto-heal --output cleaned_labels.json
  ```

- **Kiểm tra trên file nhãn thực tế 20 frame:**
  ```bash
  python qc_checker.py --input real_val_sample_20.json
  ```

---

## 6. Chạy Bằng Docker Container

Nếu môi trường của bạn đã cài đặt Docker:

```bash
cd source-tool/qc-bdd100k

# 1. Build Docker image
docker build -t qc-studio-2d:latest .

# 2. Khởi chạy container
docker run -d -p 8000:8000 --name qc-studio-app qc-studio-2d:latest

# 3. Dừng container khi hoàn thành
docker stop qc-studio-app && docker rm qc-studio-app
```
Truy cập: **`http://localhost:8000`**

---

## 7. Chạy Kiểm Thử Tự Động (Test Suite)

Dự án có sẵn 3 kịch bản kiểm thử tự động toàn diện:

### 1. Kiểm thử 8/8 bài test logic tầng nghiệp vụ
```bash
cd source-tool/qc-bdd100k/backend
python test_qc_system.py
```
*Kết quả kỳ vọng:* Toàn bộ 8 bài test đều đạt `[PASSED]` 100% (Guideline, 4 QA Layers, Auto-Heal, Markdown, Coordinate Conventions, Honeypot, Recall@20%).

### 2. Kiểm thử trên tập dữ liệu thực tế 20 frames BDD100K
```bash
cd source-tool/qc-bdd100k/backend
python test_real_dataset.py
```
*Kết quả kỳ vọng:* Phân tích 20 frame, phát hiện 101 lỗi thực tế và xuất file báo cáo `real_data_qc_report.csv`.

### 3. Kiểm thử tích hợp Client - Server API
*(Yêu cầu backend server đang chạy ở terminal khác hoặc cổng 8000)*
```bash
cd source-tool/qc-bdd100k/backend
python verify_client.py
```

---

## 8. Xử Lý Các Vấn Đề Thường Gặp (Troubleshooting / FAQ)

| Vấn đề | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `Address already in use` hoặc `Error: Port 8000 is occupied` | Đã có một tiến trình khác đang dùng cổng 8000 | Đổi port bằng cách chạy: `python -m uvicorn app:app --port 8080` hoặc tắt tiến trình đang chiếm port 8000 |
| `ModuleNotFoundError: No module named 'shapely'` | Chưa cài đặt thư viện hình học Shapely | Chạy: `pip install shapely` |
| Font tiếng Việt trên Windows CMD bị vỡ hoặc lỗi font | Windows CMD dùng mã trang mặc định (CP437 hoặc CP1252) | Chạy lệnh `chcp 65001` trên CMD trước khi chạy python |
| Giao diện web không hiển thị Canvas | Trình duyệt cũ chặn Web Canvas hoặc lỗi cache JS | Nhấn `Ctrl + F5` (hoặc `Cmd + Shift + R`) để tải lại toàn bộ cache trang |

---

Nếu gặp bất kỳ vấn đề nào khác trong quá trình chạy, vui lòng mở Issue hoặc liên hệ [@vanthang10tin](https://github.com/vanthang10tin).
