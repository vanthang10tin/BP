# TÀI LIỆU ĐẶC TẢ YÊU CẦU & THIẾT KẾ HỆ THỐNG (SOFTWARE SPECIFICATION)

**Dự án:** QC Studio 2D — Hệ thống Tự động Kiểm soát & Thẩm định Chất lượng Dữ liệu Gán nhãn cho Xe tự hành  
**Đội ngũ phát triển:** Đoàn Văn Thắng — Nguyễn Thanh Đức  
**Phiên bản:** 2.0.0 (Tháng 10/2026)  

> Tài liệu này ghi yêu cầu/mục tiêu thiết kế, không phải tuyên bố mọi chức năng đã hoàn tất. Trạng thái kiểm chứng nằm trong [`TESTING.md`](TESTING.md); số liệu giảm thời gian và độ chính xác chưa được đo.

---

## 1. GIỚI THIỆU & MỤC TIÊU DỰ ÁN

### 1.1. Bối cảnh
Đề tài tập trung vào việc kiểm tra nhãn bounding box và semantic segmentation trước khi bàn giao. Tỷ lệ lỗi nhãn và chi phí QA phụ thuộc từng dự án; nhóm chưa có nguồn kiểm chứng/đo lường thực nghiệm cho các tỷ lệ đó, nên tài liệu này không gán phần trăm chung cho ngành.

### 1.2. Mục tiêu hệ thống
QC Studio 2D hỗ trợ rà soát bằng rule, model candidates khi được cấu hình, và hàng đợi review. Mục tiêu giảm thời gian review là giả thuyết cần đo bằng pilot đối chứng; hiện chưa có kết quả giảm 60–80% đã được xác minh.
- Sắp xếp candidate findings để reviewer kiểm tra; đây không phải cơ chế tự xác nhận nhãn đúng.
- Tự động sửa chữa các lỗi hình học xác định với tính năng 1-Click Auto-Heal.
- Kết nối hai chiều với nền tảng gán nhãn CVAT để phản hồi lỗi trực tiếp tới annotator.
- Bảo vệ tuyệt đối quyền riêng tư dữ liệu với kiến trúc Local-First.

---

## 2. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

| Mã yêu cầu | Tên chức năng | Mô tả chi tiết | Mức độ ưu tiên |
|:---|:---|:---|:---:|
| **FR-01** | Nạp dữ liệu đa định dạng (Flexible Ingestion) | Hệ thống phải hỗ trợ nạp tệp nhãn đơn lẻ hoặc tệp nén ZIP chứa ảnh và nhãn thuộc các định dạng: BDD100K JSON, COCO JSON, Pascal VOC XML, CVAT for images XML, YOLO TXT và CSV. | **P0 (Bắt buộc)** |
| **FR-02** | Rà soát lỗi hình học (Rule Engine) | Kiểm tra các điều kiện đang cài: box ngoài biên, box đảo/trống/quá nhỏ (mặc định width/height <4px hoặc area <16px²), box trùng cùng class (IoU ≥0.85), thuộc tính thiếu và một số lỗi polygon/mask. | **P0 (Bắt buộc)** |
| **FR-03** | Model candidates | Khi có ảnh và provider/model weights phù hợp, so sánh prediction với nhãn để tạo missed-object/class-disagreement candidates. YOLO26 local được triển khai; các endpoint provider khác cần cấu hình. Không gọi đây là model ensemble hay đã xác minh độ chính xác. | **P1 (Có điều kiện)** |
| **FR-04** | Xếp hạng hàng đợi & Tính điểm LQS Tầng 3 | Tính toán chỉ số chất lượng nhãn Label Quality Score (thang điểm 0 - 100) và sắp xếp hàng đợi frame ảnh theo độ nghiêm trọng của lỗi (frame lỗi nặng nhất lên đầu). | **P0 (Bắt buộc)** |
| **FR-05** | Tự động sửa lỗi 1-Click (Auto-Heal) | Cho phép người dùng bấm 1 nút để tự động kẹp biên (auto-clamp), xóa box rác, hoán vị tọa độ lộn ngược, gỡ box trùng và điền thuộc tính mặc định hợp lệ. | **P0 (Bắt buộc)** |
| **FR-06** | Trực quan hóa tương tác (Visual Review Dashboard) | Giao diện hiển thị trực quan frame ảnh kèm bounding box nhãn thực tế (xanh lá), dự đoán của AI (cam) và vùng bằng chứng nghi vấn (đỏ). Hỗ trợ zoom, pan và phím tắt duyệt nhanh. | **P0 (Bắt buộc)** |
| **FR-07** | Tích hợp CVAT tùy chọn | Có API pull/push khi cấu hình URL/token; cần pilot trên CVAT test job để xác nhận frame mapping và state update trước khi dùng job thật. | **P1 (Có điều kiện)** |
| **FR-08** | Kết xuất báo cáo đa dạng (Export Deliverables) | Xuất báo cáo đánh giá chất lượng ra tệp Microsoft Excel/CSV (`utf-8-sig`), báo cáo Markdown phục vụ nghiệm thu, và tệp nhãn JSON đã được làm sạch sau khi Auto-Heal. | **P0 (Bắt buộc)** |

---

## 3. ĐẶC TẢ YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)

1. **Hiệu năng (Performance):**
   - **Mục tiêu cần benchmark:** chạy rule engine trên 50 frames (khoảng 1,500 objects) dưới 1.5 giây trên máy cấu hình ghi trong báo cáo. Chưa xác nhận mục tiêu hiệu năng này.
   - Thời gian hiển thị Canvas và chuyển đổi qua lại giữa các frame ảnh trong hàng đợi < 100ms.
2. **Quyền riêng tư & Bảo mật (Privacy & Security):**
   - Các rule cục bộ không cần dịch vụ model bên ngoài; provider/Gemini/CVAT hoạt động khi cấu hình tương ứng. Gemini vision chỉ gọi khi user bật và cấu hình key; ảnh gửi tới Google API.
   - Không dùng từ “bảo mật tuyệt đối”: public deployment hiện chưa có đăng nhập và không phù hợp với dữ liệu nhạy cảm.
   - Không lưu trữ lộ lọt CVAT Token hoặc API Key trong mã nguồn hay bundle frontend.
3. **Tính khả chuyển (Portability & Compatibility):**
   - Hỗ trợ chạy mượt mà trên hệ điều hành Windows, Linux và macOS.
   - Cung cấp kịch bản khởi chạy 1-click tự động (`run_qc_studio.ps1` và `run_qc_studio.bat`).
4. **Tính trung thực khoa học (Scientific Integrity):**
   - Không tạo số liệu giả, không báo cáo chỉ số Recall/Precision nếu chưa được đối sánh với tập nhãn chuẩn độc lập (Gold Standard).

---

## 4. ĐẶC TẢ HỢP ĐỒNG DỮ LIỆU & API SCHEMAS

### 4.1. Chuẩn hóa Bounding Box Nội bộ (Internal Coordinate Format)
Mọi định dạng nạp vào đều được quy chuẩn về dạng pixel 0-indexed:
```json
{
  "category": "car",
  "box2d": {
    "x1": 150.0,
    "y1": 220.0,
    "x2": 480.0,
    "y2": 510.0
  },
  "attributes": {
    "occluded": false,
    "truncated": false
  }
}
```

### 4.2. Cấu trúc Cảnh báo Lỗi (Error Issue Schema)
```json
{
  "object_id": "CAR_01",
  "category": "car",
  "severity": "CRITICAL",
  "error_type": "BOX_OUT_OF_IMAGE",
  "qa_layer": "Tầng 1 (Luật)",
  "component": "Rule-Engine Geometry Checker",
  "explanation": "Box tràn ngoài biên ảnh [-15, 100, 1350, 500]. §1.2: Mọi tọa độ Bbox phải nằm trọn vẹn trong khung ảnh.",
  "action": "Tự động kẹp biên (Auto-Clamp)",
  "auto_healable": true,
  "suggested_box": [0.0, 100.0, 1280.0, 500.0],
  "guideline_rule": "§1.2: Mọi tọa độ Bbox phải nằm trọn vẹn trong khung ảnh."
}
```

### 4.3. Các REST API chính
- `GET /api/health`: Trạng thái hệ thống và các provider khả dụng.
- `POST /api/qc/check`: Nhận tệp nhãn (JSON, XML, TXT, CSV, ZIP), thực hiện rà soát và trả về danh sách frame đã sắp xếp kèm phân tích lỗi.
- `POST /api/auto-heal`: Thực hiện tự động sửa các lỗi hình học xác định trên dataset.
- `POST /api/cvat/push-issues`: Gửi danh sách issue đã xác nhận lên CVAT Job tương ứng.
- `POST /api/export/csv`: Xuất báo cáo kết quả kiểm tra chất lượng ra file CSV chuẩn UTF-8 BOM.
