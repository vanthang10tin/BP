# Chạy nhanh QC Studio trên máy Windows

Hướng dẫn cài đặt đầy đủ, chọn bộ dữ liệu 50 ảnh, thao tác BBox/Segmentation và xử lý lỗi nằm trong [INSTALL_AND_USE.md](INSTALL_AND_USE.md).

## Khởi chạy

Từ thư mục `BP`, mở PowerShell và chạy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_qc_studio.ps1
```

Script kiểm tra Python, .NET, dependencies và health API; tự build frontend nếu cần; chạy API ở `http://127.0.0.1:8000`, giao diện Blazor ở `http://127.0.0.1:5000`, rồi mở trình duyệt. Script chỉ quản lý tiến trình QC Studio; không dừng CVAT, Docker hay model containers.

Dừng ứng dụng bằng:

```powershell
.\stop_qc_studio.ps1
```

Log nằm trong `source-tool/qc-bdd100k/.qc-data/runtime/`. Nếu một dịch vụ không lên, xem `backend.log`, `backend-error.log`, `frontend.log`, `frontend-server.log` và `frontend-error.log` ở đó.

Nếu máy chưa cài thư viện Python, cài một lần từ thư mục `BP`:

```powershell
python -m pip install -r source-tool/qc-bdd100k/requirements.txt
```

## Quy trình trả Job về thẳng CVAT kèm Issues

1. **Token & Cấu hình:** Đã tự động cấu hình trong file `.env` (API Token CVAT lấy từ container `cvat_server`):
   - `CVAT_URL`: `http://127.0.0.1:8080`
   - `CVAT_ACCESS_TOKEN`: set a scoped, expiring token in backend-only `.env`; do not store real tokens in docs or source.
2. **Thao tác trong QC Studio (http://127.0.0.1:5000):**
   - **Bước 1:** Trong mục CVAT Sync & Job Return, chọn đúng Job CVAT (ví dụ: `Job #56 - Xanhsm`). Hệ thống sẽ tự động map frame ID tương ứng với từng ảnh.
   - **Bước 2:** Thực hiện review, xác nhận lỗi (hoặc hệ thống gom toàn bộ lỗi QC phát hiện).
   - **Bước 3:** Bấm nút **`⚡ TRẢ JOB VỀ CVAT KÈM ISSUES`**.
   - **Kết quả:**
     - Các issue được tạo thẳng lên từng frame của Job trên CVAT kèm tọa độ bounding box khoanh vùng lỗi chính xác.
     - Trạng thái Job trên CVAT tự động chuyển sang `stage: "annotation"` và `state: "rejected"` để annotator nhận lại và sửa ngay.
     - Trên giao diện hiện thẻ thành công và đường link bấm trực tiếp: `Mở thẳng Job trên CVAT (có sẵn issue)`.

Xem [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) để biết kiến trúc mục tiêu và các giới hạn còn lại.
