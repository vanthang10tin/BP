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
- Cung cấp Auto-Heal cho dữ liệu/loại lỗi được hỗ trợ; người dùng cần kiểm tra kết quả trước khi sử dụng.
- Hỗ trợ trao đổi issue với CVAT khi được cấu hình; cần kiểm chứng trên test job trước khi dùng job thật.
- Ưu tiên xử lý cục bộ cho rule checks; các tích hợp tùy chọn có thể truyền dữ liệu ra ngoài theo mô tả ở mục 3.

### 1.3. Phạm vi và người dùng

- **Annotator/reviewer:** nạp dữ liệu, xem frame và candidate, xác nhận hoặc bác bỏ finding, xuất kết quả.
- **Quản trị viên dự án:** cấu hình profile/guideline và tích hợp tùy chọn.
- **Ngoài phạm vi được đảm bảo:** tự động kết luận nhãn đúng/sai; huấn luyện model; lưu trữ lâu dài trên Render Free; bảo vệ truy cập bằng tài khoản/role; tương thích mọi biến thể của định dạng đầu vào.

### 1.4. Luồng sử dụng chính

1. Chọn/tạo workspace và profile, hoặc cung cấp dataset cho luồng kiểm tra trực tiếp.
2. Nạp ảnh/nhãn theo định dạng được hỗ trợ; parser chuẩn hóa đầu vào hoặc trả lỗi nếu dữ liệu không hợp lệ/không hỗ trợ.
3. Chạy rule checks; nếu đã cấu hình provider và có ảnh thì có thể tạo thêm model candidates.
4. Xem findings, đối chiếu ảnh/guideline và ghi quyết định reviewer. Candidate không được xem là finding đã xác nhận cho đến khi người dùng duyệt.
5. Tùy nhu cầu, chạy Auto-Heal, xuất báo cáo hoặc đẩy issue đã xác nhận sang CVAT. Kiểm tra lại dữ liệu đầu ra trước khi bàn giao.

Thiết kế thành phần và luồng triển khai được mô tả trong [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 2. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

| Mã yêu cầu | Tên chức năng | Mô tả chi tiết | Mức độ ưu tiên |
|:---|:---|:---|:---:|
| **FR-01** | Nạp dữ liệu đa định dạng (Flexible Ingestion) | Hệ thống phải hỗ trợ nạp tệp nhãn đơn lẻ hoặc tệp nén ZIP chứa ảnh và nhãn thuộc các định dạng: BDD100K JSON, COCO JSON, Pascal VOC XML, CVAT for images XML, YOLO TXT và CSV. | **P0 (Bắt buộc)** |
| **FR-02** | Rà soát lỗi hình học (Rule Engine) | Kiểm tra các điều kiện đang cài: box ngoài biên, box đảo/trống/quá nhỏ (mặc định width/height <4px hoặc area <16px²), box trùng cùng class (IoU ≥0.85), thuộc tính thiếu và một số lỗi polygon/mask. | **P0 (Bắt buộc)** |
| **FR-03** | Model candidates | Khi có ảnh và provider/model weights phù hợp, so sánh prediction với nhãn để tạo missed-object/class-disagreement candidates. YOLO26 local được triển khai; các endpoint provider khác cần cấu hình. Không gọi đây là model ensemble hay đã xác minh độ chính xác. | **P1 (Có điều kiện)** |
| **FR-04** | Xếp hạng hàng đợi & Tính điểm LQS Tầng 3 | Tính toán chỉ số chất lượng nhãn Label Quality Score (thang điểm 0 - 100) và sắp xếp hàng đợi frame ảnh theo độ nghiêm trọng của lỗi (frame lỗi nặng nhất lên đầu). | **P0 (Bắt buộc)** |
| **FR-05** | Auto-Heal có kiểm soát | Cung cấp API sửa dữ liệu và trả về dữ liệu đã xử lý cùng thống kê. Chỉ áp dụng cho loại lỗi được pipeline đánh dấu có thể sửa; người dùng phải review trước khi dùng kết quả. Danh sách phép sửa thực tế cần được kiểm chứng cho từng loại dữ liệu. | **P0 (Bắt buộc)** |
| **FR-06** | Trực quan hóa tương tác (Visual Review Dashboard) | Giao diện hiển thị trực quan frame ảnh kèm bounding box nhãn thực tế (xanh lá), dự đoán của AI (cam) và vùng bằng chứng nghi vấn (đỏ). Hỗ trợ zoom, pan và phím tắt duyệt nhanh. | **P0 (Bắt buộc)** |
| **FR-07** | Tích hợp CVAT tùy chọn | Có API pull/push khi cấu hình URL/token; cần pilot trên CVAT test job để xác nhận frame mapping và state update trước khi dùng job thật. | **P1 (Có điều kiện)** |
| **FR-08** | Kết xuất báo cáo | Hỗ trợ CSV (`utf-8-sig`) cho QC và segmentation, Markdown feedback/guideline draft, và dữ liệu đã Auto-Heal trong response API. Báo cáo Markdown là draft/feedback, không phải biên bản nghiệm thu tự động; chưa cam kết workbook Excel hoặc tệp nhãn đã làm sạch có thể tải xuống dưới dạng riêng. | **P0 (Bắt buộc)** |

---

## 3. ĐẶC TẢ YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)

1. **Hiệu năng (Performance):**
   - **Mục tiêu cần benchmark:** chạy rule engine trên 50 frames (khoảng 1,500 objects) dưới 1.5 giây trên máy cấu hình ghi trong báo cáo. Chưa xác nhận mục tiêu hiệu năng này.
   - Mục tiêu trải nghiệm: hiển thị Canvas và chuyển frame dưới 100ms; hiện chưa có phép đo/chứng cứ để xác nhận.
2. **Quyền riêng tư & Bảo mật (Privacy & Security):**
   - Các rule cục bộ không cần dịch vụ model bên ngoài; provider/Gemini/CVAT hoạt động khi cấu hình tương ứng. Gemini vision chỉ gọi khi user bật và cấu hình key; ảnh gửi tới Google API.
   - Bản public hiện chưa có đăng nhập/phân quyền và không phù hợp với dữ liệu nhạy cảm. Không cam kết quyền riêng tư/bảo mật tuyệt đối.
   - Không commit CVAT Token/API Key hoặc đưa chúng vào bundle frontend. Chỉ cấu hình qua backend/environment hoặc cơ chế theo request đã triển khai.
3. **Tính khả chuyển (Portability & Compatibility):**
   - Chạy local qua backend Python; các script khởi chạy một bước hiện có dành cho Windows (`run_qc_studio.ps1`, `run_qc_studio.bat`).
   - Docker/Render là đường triển khai Linux/container. Chưa có bằng chứng kiểm thử tương đương trên mọi phiên bản Windows, Linux và macOS.
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

| API | Vai trò | Ghi chú/điều kiện |
|---|---|---|
| `GET /api/health`, `GET /api/models` | Health/capabilities và provider availability | Health không đồng nghĩa model weights đã nạp hoặc inference đã được xác minh. |
| `GET/POST /api/workspaces`, `GET/PUT /api/workspaces/{workspace_id}` | Quản lý workspace/profile | Dữ liệu thuộc workspace store; persistence phụ thuộc môi trường triển khai. |
| `POST /api/dataset/parse-flexible` | Parse JSON/XML/TXT/CSV/ZIP được hỗ trợ | Giới hạn upload hiện tại là 64 MiB; schema/biến thể không hỗ trợ cần trả lỗi. |
| `POST /api/qc/check` | Kiểm tra dataset trực tiếp/legacy | JSON hoặc multipart upload; trả summary và `sorted_frames`. |
| `POST /api/qc/run` | QC theo workspace, BBox hoặc semantic segmentation | Cần `task_type` hợp lệ; model inference phụ thuộc provider/weights. |
| `POST /api/auto-heal` | Trả dữ liệu đã xử lý và thống kê | Review kết quả trước khi bàn giao; không thay thế bản gốc nếu chưa xác nhận. |
| `POST /api/export/csv`, `POST /api/export/segmentation-csv` | Xuất báo cáo CSV | CSV dùng UTF-8 BOM. |
| `POST /api/export/feedback`, `POST /api/export/guideline-draft` | Tạo Markdown feedback/draft | Draft guideline chỉ dùng issue đã được người review xác nhận và không tự kích hoạt. |
| `GET /api/cvat/status`, `GET /api/cvat/jobs`, `POST /api/cvat/push-issues` | Tích hợp CVAT tùy chọn | Cần URL/token; xác minh mapping trên test job trước khi dùng thật. |

API docs tại `/docs` trên service đang chạy là nguồn tham chiếu cho schema request/response chi tiết.

### 4.4. Xử lý lỗi và điều kiện chấp nhận

- Đầu vào không hợp lệ/không hỗ trợ phải được báo lỗi rõ ràng; upload vượt 64 MiB trả HTTP 413. Không biểu diễn thiếu provider/model như kết quả inference thành công rỗng.
- `task_type` của workspace phải là `bbox` hoặc `semantic_segmentation`; workspace không tồn tại phải trả lỗi thay vì tự tạo thay thế.
- Kết quả cần phân biệt candidate/pending với quyết định reviewer; không gọi trạng thái không có candidate là “đã xác minh sạch”.
- Điều kiện nghiệm thu tối thiểu theo luồng: parser giữ đúng thông tin cần thiết; QC trả kết quả nhất quán với task/profile; Auto-Heal trả dữ liệu và thống kê; export giữ encoding/metadata; tích hợp ngoài chỉ được nghiệm thu sau test riêng trên môi trường thử. Trạng thái thực tế được theo dõi tại [`TESTING.md`](TESTING.md).
