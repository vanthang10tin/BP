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
     - `QC_MODAL_API_URL` (tùy chọn): endpoint của deployment Modal do bạn quản lý. Để trống nếu không dùng Modal; repo không cung cấp endpoint dùng chung.
4. **Bấm "Deploy Web Service":**
   - Render sẽ tự động kéo image .NET 10 SDK, build Blazor, cài Python packages và khởi động container. Quá trình mất khoảng 2 - 3 phút.
   - Khi hoàn tất, Render cấp một URL dạng: `https://qc-studio-2d-blazor.onrender.com`.
   - Bấm vào URL để vào thẳng giao diện Blazor trên nền web đám mây.

*(Lưu ý về gói Render Free: Web Service sẽ tạm ngủ (spin down) nếu sau 15 phút không có người truy cập; khi có người bấm lại vào link, service sẽ mất khoảng 45-60 giây để khởi động lại).*

### Modal inference (tùy chọn)

`deploy_vision.py` mặc định dùng GPU **T4**. BBox (YOLO26x + D-FINE-X) và segmentation (SegFormer Cityscapes 19 lớp + YOLO26x-Seg + SAM 2.1 Large) dùng các pool container riêng. Model của từng luồng chỉ nạp khi luồng đó nhận inference. SAM xử lý tối đa 16 box mỗi batch, không bỏ box còn lại.

```powershell
python -m pip install modal
$env:QC_MODAL_GPU = "T4" # hoặc L4; chuỗi rỗng để chạy CPU
python -m modal deploy deploy_vision.py
python test_modal_endpoint.py https://<your-modal-app>.modal.run
```

Các override weights hiện dùng: `QC_MODAL_YOLO_WEIGHTS`, `QC_MODAL_DFINE_WEIGHTS`, `QC_MODAL_YOLO_SEG_WEIGHTS`, `QC_MODAL_SAM_WEIGHTS`. Giá trị phải trỏ tới file có trong container hoặc tên được loader hỗ trợ; đường dẫn Windows không tự được upload. Các weights mặc định được tải khi build image; lỗi HTTP làm build thất bại. SegFormer được tải từ Hugging Face ở lần khởi động luồng segment đầu tiên.

Đặt `QC_MODAL_API_URL` ở backend sau khi smoke test thành công. `/status` chỉ xác nhận API online và trả `model_readiness=not_checked`; cần gọi cả `/bbox` và `/segmentation` để kiểm chứng inference. Client chờ tối đa 600 giây cho mỗi lần inference, kể cả cold start. Script smoke test trả exit code 1 nếu bất kỳ endpoint nào lỗi; ảnh tổng hợp chỉ kiểm tra hợp đồng API, không đo độ chính xác.

Ảnh đầu vào tối đa 10 MiB và 25 megapixel. Request sai trả HTTP 400; lỗi inference segment trả HTTP 503 để backend không coi kết quả rỗng là đánh giá thành công. BBox trả trạng thái từng detector trong `model_runs` khi `include_metadata=true`.

GPU Modal cần tài khoản có quyền chạy GPU; xem [hướng dẫn GPU của Modal](https://modal.com/docs/guide/gpu).

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
