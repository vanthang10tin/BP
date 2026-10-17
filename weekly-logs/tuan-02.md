# BÁO CÁO TIẾN ĐỘ DỰ ÁN — TUẦN 02

**Dự án:** Hệ thống hỗ trợ kiểm soát và đánh giá chất lượng dữ liệu hình ảnh (QC Studio 2D)  
**Thời gian:** Tuần 02  
**Nhân sự phụ trách:**
- Đoàn Văn Thắng
- Nguyễn Thanh Đức

---

## 1. MỤC TIÊU TUẦN 02
- Thiết kế chi tiết kiến trúc 4 tầng QA (4-Layer Quality Assurance Architecture).
- Xây dựng mô-đun Rule Engine (Tầng 1) kiểm tra các vi phạm xác định về hình học và thuộc tính nhãn.
- Xây dựng Guideline Manager để quản lý và trích dẫn trực tiếp các điều khoản quy chuẩn gán nhãn vào từng lỗi phát hiện.
- Thiết kế hệ thống tính điểm chất lượng nhãn Label Quality Score (LQS) và phân loại lỗi theo 3 nhóm kinh điển (ObjectLab Triad: Boundary Error, Class Confusion, Missing Object).

---

## 2. NỘI DUNG VÀ KẾT QUẢ TRIỂN KHAI

### 👤 Đoàn Văn Thắng
- **Thiết kế sơ đồ kiến trúc hệ thống (System Architecture Design):**
  - Xây dựng sơ đồ kiến trúc chi tiết dạng SVG chuẩn (`ARCHITECTURE_FLOW.svg`) mô tả luồng dữ liệu qua các khối: Input -> FastAPI Orchestrator -> Rule Engine & AI Consensus -> Blazor UI -> CVAT Handoff & Export.
  - Phác thảo giao diện quản lý Guideline: Cho phép xem, biên soạn và kích hoạt phiên bản quy định gán nhãn cho từng workspace.
  - Thiết kế cấu trúc lưu trữ cục bộ nhẹ (SQLite Workspace Store) lưu trữ phiên bản guideline, metadata và cấu hình kiểm tra.

### 👤 Nguyễn Thanh Đức
- **Phát triển Rule Engine (Tầng 1):**
  - Lập trình các thuật toán kiểm tra hình học xác định với độ phức tạp $O(N)$ hoặc $O(N \log N)$:
    - `BOX_OUT_OF_IMAGE`: Tọa độ tràn khỏi kích thước khung ảnh.
    - `BOX_TOO_SMALL`: Box quá nhỏ theo ngưỡng cấu hình. Bản code hiện tại mặc định cạnh <4px hoặc diện tích <16px²; ngưỡng có thể cấu hình.
    - `BOX_INVERTED`: Tọa độ bị lộn ngược ($x_1 > x_2$ hoặc $y_1 > y_2$).
    - `BOX_DUPLICATE_SAME_CLASS`: Box bị vẽ đè trùng lặp cùng danh mục (ngưỡng IoU $\ge 0.85$).
    - `MISSING_MANDATORY_ATTRIBUTE`: Thiếu thuộc tính bắt buộc (`occluded`, `truncated`).
    - `POLYGON_SELF_INTERSECT`: Đa giác polygon tự cắt chéo cạnh (kiểm tra qua thư viện Shapely).
    - `MASK_BOX_MISMATCH`: Mặt nạ mask và bounding box bị lệch tâm hoặc không khớp vị trí.
  - Xây dựng lớp `GuidelineManager`: Đọc cấu trúc quy chuẩn gán nhãn, tự động gắn trích dẫn điều khoản (ví dụ: `§1.2 Tọa độ biên`, `§2.1 Phân loại xe con và xe tải`) vào từng cảnh báo lỗi.

---

## 3. KẾT LUẬN & ĐỊNH HƯỚNG TUẦN 03
- Hoàn thành các rule hình học trong phạm vi cài đặt và kiểm tra bằng các ca ví dụ tự tạo. Các ca này là kiểm tra hành vi, không phải phép đo accuracy/coverage trên gold set.
- Lên kế hoạch Tuần 03: Tích hợp mô hình AI Tầng 2 (Model Consensus) để phát hiện các lỗi ngữ nghĩa (sót vật thể, phân loại nhầm danh mục).
