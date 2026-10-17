# Hướng dẫn Deploy QC Studio 2D (Blazor UI + FastAPI Backend)

QC Studio 2D sử dụng **Blazor Web App (.NET 10)** làm giao diện người dùng chính, kết nối với **Python FastAPI Backend** để thực hiện bóc tách nhãn, thẩm định hình học và phân loại 4 tầng QA.

Có 3 cách triển khai và chạy thử:

---

## Cách 1: Chạy và test ngay trên máy Local (Không cần Docker, 1 lệnh là chạy)

Cách nhanh nhất để trải nghiệm và kiểm thử ngay giao diện Blazor:

```powershell
# Chạy cả Backend API (port 8000) và Blazor UI (port 5000):
.\run_qc_studio.ps1
```
*(Hoặc click đúp file `run_qc_studio.bat` từ File Explorer).*

- Script sẽ tự động restore, build và khởi chạy cả Backend lẫn Blazor.
- Trình duyệt sẽ tự động mở: **`http://127.0.0.1:5000`**
- Bấm nút **"Nạp bộ test 50 ảnh"** ở màn hình chính để nạp ngay dữ liệu BDD100K thật và thử các tính năng HUD BBox/Segmentation, lọc lỗi, LQS và Auto-Heal.
- Khi muốn tắt toàn bộ: chạy `.\stop_qc_studio.ps1`.

---

## Cách 2: Deploy lên Cloud miễn phí (Render / Railway) qua Docker (Khuyên dùng)

Hệ thống đã được đóng gói thành một Dockerfile tối ưu All-in-One: trong cùng 1 container, Python FastAPI chạy ngầm ở `127.0.0.1:8000`, còn Blazor Web App lắng nghe cổng public `$PORT` (do Render/Railway tự cấp) để người dùng truy cập trực tiếp.

### Các bước deploy lên Render:

1. **Đẩy mã nguồn lên GitHub:**
   ```bash
   git add .
   git commit -m "Deploy Blazor UI with FastAPI backend"
   git push origin main
   ```
2. **Tạo Web Service trên Render:**
   - Đăng nhập [dashboard.render.com](https://dashboard.render.com).
   - Chọn **New +** $\rightarrow$ **Web Service**.
   - Chọn kho GitHub của bạn (`BP`).
3. **Cấu hình trên Render:**
   - **Language / Runtime:** Chọn **Docker** (Render sẽ tự động dùng file `Dockerfile` ở thư mục gốc).
   - **Plan:** Chọn **Free** ($0/tháng).
   - **Advanced $\rightarrow$ Environment Variables (nếu cần):**
     - `QC_PUBLIC_DEMO`: `true` (mặc định để bảo vệ an toàn, không lộ credential).
4. **Bấm "Deploy Web Service":**
   - Render sẽ tự động kéo image .NET 10 SDK, build Blazor, cài Python packages và khởi động container. Quá trình mất khoảng 2 - 3 phút.
   - Khi hoàn tất, Render cấp một URL dạng: `https://qc-studio-2d-blazor.onrender.com`.
   - Bấm vào URL để vào thẳng giao diện Blazor trên nền web đám mây.

*(Lưu ý về gói Render Free: Web Service sẽ tạm ngủ (spin down) nếu sau 15 phút không có người truy cập; khi có người bấm lại vào link, service sẽ mất khoảng 45-60 giây để khởi động lại).*

---

## Cách 3: Chạy bằng Docker Compose (Dành cho VPS hoặc máy có Docker Desktop)

Nếu bạn có máy chủ Linux (VPS) hoặc máy cá nhân đã cài Docker Desktop:

```bash
# Khởi động dịch vụ container:
docker compose up -d --build
```

- Mở trình duyệt tại: **`http://localhost:5000`**.
- Dừng dịch vụ: `docker compose down`.

---

## Chế độ an toàn Public Demo

Khi chạy trên môi trường công khai (`QC_PUBLIC_DEMO=true`):
- Các thao tác lưu khóa API Gemini hoặc token CVAT vào file cấu hình dùng chung bị chặn (trả về mã 403).
- Bản public chỉ dùng để thẩm định và trình diễn với các tập dữ liệu mẫu không nhạy cảm.
