# Hướng Dẫn Triển Khai Portable (Chạy Trên Mọi Máy Tính)

Tài liệu này hướng dẫn cách đưa toàn bộ hệ thống **CVAT 2D + QC Studio (Backend FastAPI + Frontend Blazor)** sang bất kỳ máy tính mới nào (Windows, Linux, macOS) và khởi chạy ngay lập tức mà không cần cài đặt môi trường lập trình (.NET, Python, Nginx...).

---

## 1. Yêu cầu duy nhất trên máy đích
Máy tính chỉ cần cài đặt **Docker Desktop** (trên Windows/macOS) hoặc **Docker Engine + Docker Compose** (trên Linux).
- Không cần cài Python.
- Không cần cài .NET SDK.
- Không cần cấu hình cổng hay cài Nginx trên máy host.

---

## 2. Kiến trúc All-in-One Compose Stack

Toàn bộ dịch vụ được đóng gói trong một file duy nhất [`docker-compose.yml`](docker-compose.yml):

| Dịch vụ | Cổng truy cập | Vai trò |
|---|---|---|
| **CVAT Gateway & UI** | `http://localhost:8080` | Giao diện gán nhãn và máy chủ CVAT (đã cấu hình fix lỗi CSRF). |
| **QC Studio UI** | `http://localhost:5000` | Không gian làm việc trực quan QC (Blazor UI, Review queue, Canvas, Export). |
| **QC Engine API** | `http://localhost:8000` | Bộ lõi kiểm tra quy tắc nhãn BDD100K và tích hợp đẩy Issue sang CVAT. |
| **CVAT Core Services** | Nội bộ Docker network | PostgreSQL, Redis, Kvrocks, OPA, Workers xử lý background. |

Tất cả container liên kết với nhau qua mạng nội bộ `cvat_cvat`. QC Studio tự động nhận diện và kết nối trực tiếp với CVAT thông qua tên dịch vụ nội bộ.

---

## 3. Các bước khởi chạy trên máy mới

### Bước 1: Sao chép mã nguồn sang máy mới
Clone repository hoặc copy toàn bộ thư mục `BP` sang máy mới.

### Bước 2: Khởi chạy 1-Click

- **Trên Windows:**
  Nhấp chuột phải vào file [`start_all_docker.ps1`](start_all_docker.ps1) chọn **Run with PowerShell** (hoặc mở PowerShell gõ):
  ```powershell
  .\start_all_docker.ps1
  ```
- **Trên Linux / macOS:**
  Mở Terminal và gõ:
  ```bash
  chmod +x start_all_docker.sh
  ./start_all_docker.sh
  ```
- **Hoặc dùng lệnh Docker Compose chuẩn:**
  ```bash
  docker compose up -d
  ```

Docker sẽ tự động tải các image cần thiết, build ứng dụng QC Studio và khởi động toàn bộ hệ thống.

---

## 4. Sử dụng và liên kết CVAT với QC Studio

1. **Truy cập CVAT (`http://localhost:8080`):**
   - Đăng nhập (hoặc tạo tài khoản admin lần đầu nếu là máy mới).
   - Vào góc trên bên phải: **Profile -> Personal Access Tokens -> Create** để tạo token.
2. **Nạp token vào cấu hình:**
   - Mở file `.env` ở thư mục gốc, dán token vào dòng:
     ```env
     CVAT_ACCESS_TOKEN=dán_token_vào_đây
     ```
   - Chạy lệnh cập nhật:
     ```bash
     docker compose up -d qc_api
     ```
3. **Sử dụng QC Studio (`http://localhost:5000`):**
   - Tải lên bộ nhãn (ví dụ các gói dữ liệu mẫu tại `test-data/bdd100k-walkthroughs/`).
   - Duyệt và bấm **Confirm** các lỗi cần báo cáo.
   - Sang tab **Export**, nhập **CVAT Job ID**, bấm **Load job metadata** rồi bấm **Push confirmed** để tự động tạo Issue trên đúng frame của CVAT.

---

## 5. Dừng hệ thống khi không sử dụng

Khi kết thúc phiên làm việc:
```bash
docker compose down
```
*Lưu ý: Toàn bộ dữ liệu bài gán nhãn, tài khoản, tác vụ trong CVAT và lịch sử QC đều được bảo toàn an toàn trong các Docker Volumes (`cvat_data`, `cvat_db`, `qc_data`).*
