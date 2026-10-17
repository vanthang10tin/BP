# BÁO CÁO TIẾN ĐỘ DỰ ÁN — TUẦN 03

**Dự án:** Hệ thống hỗ trợ kiểm soát và đánh giá chất lượng dữ liệu hình ảnh (QC Studio 2D)  
**Thời gian:** Tuần 03  
**Nhân sự phụ trách:**
- Đoàn Văn Thắng
- Nguyễn Thanh Đức

---

## 1. MỤC TIÊU TUẦN 03
- Tích hợp Tầng 2: Multi-Model AI Consensus Engine hỗ trợ phát hiện các vật thể bị bỏ sót (`POTENTIAL_MISSED_OBJECT`) và phân loại sai danh mục (`CLASS_CONFUSION`).
- Xây dựng mô-đun chuyển đổi hệ tọa độ chuẩn xác (Coordinate Normalization & Conversion) hỗ trợ chuyển đổi qua lại giữa COCO, YOLO, Pascal VOC, BDD100K và Gemini VLM.
- Phát triển tính năng **1-Click Auto-Heal**: Tự động khắc phục các lỗi hình học xác định mà không làm sai lệch ý đồ người gán nhãn.
- Tích hợp nhà cung cấp mô hình ngôn ngữ thị giác Google Gemini API làm tùy chọn nâng cao (Opt-in) cho việc phân tích ngữ cảnh hình ảnh phức tạp.

---

## 2. NỘI DUNG VÀ KẾT QUẢ TRIỂN KHAI

### 👤 Đoàn Văn Thắng
- **Xây dựng module chuyển đổi không gian tọa độ (`coordinate_utils.py`):**
  - Hiện thực hóa chuẩn hóa tọa độ tuân thủ theo tiêu chuẩn học thuật (Slide 37 & Slide 22):
    - COCO format: `[x, y, w, h]` (pixel) $\leftrightarrow$ `[x1, y1, x2, y2]`.
    - YOLO format: `[cx, cy, w, h]` (normalized 0-1) $\leftrightarrow$ pixel bounds.
    - Pascal VOC: 1-indexed pixel bounds $\leftrightarrow$ 0-indexed bounds.
    - Google Gemini VLM: `[ymin, xmin, ymax, xmax]` normalized 0-1000 $\leftrightarrow$ pixel bounds.
  - Xây dựng cơ chế bộ đệm ảnh SSD Cache (`.qc-data/images`) để tối ưu tốc độ tải và xem trước ảnh tức thì cho người dùng.

### 👤 Nguyễn Thanh Đức
- **Tích hợp Model Ensemble & AI Engine (`models_ensemble.py`, `local_model_providers.py`):**
  - Tích hợp YOLO26 cục bộ khi có weights/ảnh; các provider segmentation/DINO cần endpoint và mapping riêng. Prediction đối chiếu với nhãn qua IoU chỉ là candidate, chưa phải nhãn lỗi chuẩn.
  - Phân tích lỗi ngữ nghĩa:
    - Nếu nhãn có IoU $\ge 0.50$ với dự đoán nhưng danh mục khác nhau $\rightarrow$ Cảnh báo `CLASS_CONFUSION`.
    - Nếu model phát hiện vật thể với độ tin cậy cao ($\ge 0.55$) nhưng không có nhãn nào trong bán kính IoU $0.20$ $\rightarrow$ Cảnh báo `POTENTIAL_MISSED_OBJECT`.
- **Lập trình tính năng 1-Click Auto-Heal:**
  - Tự động kẹp biên (Auto-Clamp) các box tràn khung hình về đúng $[0, W] \times [0, H]$.
  - Tự động xóa box dưới ngưỡng cấu hình (mặc định cạnh <4px hoặc diện tích <16px²).
  - Tự động sửa các box bị lộn ngược tọa độ ($x_1 \leftrightarrow x_2$).
  - Tự động gỡ trùng lặp các box vẽ đè cùng danh mục (IoU $\ge 0.85$).
  - Tự động bổ sung các trường thuộc tính mặc định còn thiếu (`occluded: false`, `truncated: false`).

---

## 3. KẾT LUẬN & ĐỊNH HƯỚNG TUẦN 04
- Hoàn thành chức năng Auto-Heal và các kiểm tra hành vi bằng dữ liệu ví dụ tự tạo. Không có phép đo accuracy 100%; candidate từ model cần người review.
- Lên kế hoạch Tuần 04: Xây dựng Dashboard Web chuyên nghiệp (Blazor Server) và tích hợp API với nền tảng CVAT để chuyển giao lỗi trực tiếp.
