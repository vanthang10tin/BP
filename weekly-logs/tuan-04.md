# BÁO CÁO TIẾN ĐỘ DỰ ÁN — TUẦN 04

**Dự án:** Hệ thống hỗ trợ kiểm soát và đánh giá chất lượng dữ liệu hình ảnh (QC Studio 2D)  
**Thời gian:** Tuần 04  
**Nhân sự phụ trách:**
- Đoàn Văn Thắng
- Nguyễn Thanh Đức

> Ghi chú kiểm chứng hiện tại: các mục dưới đây mô tả phần giao diện/endpoint trong source. Kết nối tới CVAT thật, state transition trên job và toàn bộ luồng UI chưa được xác minh end-to-end; cần thử trên một disposable job trước khi trình diễn.

---

## 1. MỤC TIÊU TUẦN 04
- Xây dựng giao diện người dùng trực quan trên nền tảng Blazor WebApp (.NET 10) chạy tại cổng `5000` kết nối với FastAPI backend tại cổng `8000`.
- Thiết kế màn hình Review 3 cột chuyên nghiệp:
  1. Cột trái: Hàng đợi ưu tiên (Priority Queue) sắp xếp theo chỉ số chất lượng nhãn LQS (từ frame lỗi nặng nhất đến sạch nhất).
  2. Cột giữa: Khung vẽ Canvas hiển thị hình ảnh kèm bounding box nhãn thực tế (xanh) đè lên dự đoán của AI (cam/đỏ) và vùng bằng chứng vi phạm.
  3. Cột phải: Bảng chi tiết vi phạm, trích dẫn điều khoản guideline, và các nút quyết định thẩm định (Confirm / Dismiss / Auto-Heal / Return to CVAT).
- Tích hợp 2 chiều với CVAT REST API:
  - Cho phép người dùng kéo Job và ảnh trực tiếp từ CVAT về QC Studio.
  - Cho phép đẩy danh sách issue đã xác nhận trực tiếp lên đúng Frame và vị trí tọa độ của Job trên CVAT, đồng thời chuyển trạng thái Job thành `rejected` để annotator sửa lại.

---

## 2. NỘI DUNG VÀ KẾT QUẢ TRIỂN KHAI

### 👤 Đoàn Văn Thắng
- **Phát triển giao diện Blazor WebApp (`Home.razor`, `App.razor`, `app.css`):**
  - Xây dựng layout responsive, hiện đại với dark mode cao cấp, hiệu ứng chuyển động mượt mà (smooth animations, badge styling, interactive SVG canvas).
  - Tích hợp điều khiển bàn phím nhanh: Phím mũi tên lên/xuống để duyệt nhanh các frame trong hàng đợi; phím tắt xác nhận nhanh quyết định.
  - Xây dựng module kết xuất báo cáo: Xuất file Excel/CSV chuẩn quốc tế với encoding `utf-8-sig` (tránh lỗi font chữ tiếng Việt khi mở bằng Microsoft Excel).
  - Tích hợp cấu hình linh hoạt cho CVAT Token và Gemini API Key ngay trên giao diện web.

### 👤 Nguyễn Thanh Đức
- **Hiện thực hóa tích hợp CVAT Handoff (`app.py`):**
  - Triển khai endpoint `/api/cvat/jobs`: Liệt kê các Job khả dụng trên CVAT Server.
  - Triển khai endpoint `/api/cvat/jobs/{job_id}/pull`: Tải siêu dữ liệu nhãn, danh mục, và kéo ảnh từng frame (tối ưu tải đa luồng ThreadPoolExecutor kèm đệm SSD cache).
  - Triển khai endpoint `/api/cvat/push-issues`:
    - Ánh xạ chính xác tên file ảnh sang Frame ID của Job trên CVAT.
    - Loại trừ trùng lặp thông minh (Deduplication) dựa trên spatial IoU $\ge 0.70$ hoặc nội dung comment để không tạo các issue rác chồng chéo.
    - Tự động gọi PATCH `/api/jobs/{job_id}` chuyển `stage: annotation` và `state: rejected`.
  - Triển khai endpoint `/api/cvat/issues/{issue_id}/resolve`: Đồng bộ trạng thái đã xử lý lỗi ngược lại CVAT.

---

## 3. KẾT LUẬN & ĐỊNH HƯỚNG TUẦN 05
- Có route cho Blazor/API và các endpoint CVAT trong source; chưa có bằng chứng kiểm thử end-to-end với CVAT Server thật.
- Lên kế hoạch Tuần 05: Kiểm thử liên thông toàn diện trên các tập dữ liệu thực tế (50 frames BDD100K thật), bổ sung bộ phân tích đa định dạng dữ liệu (Flexible Parser), đóng gói sản phẩm và hoàn thiện tài liệu nộp bài.
