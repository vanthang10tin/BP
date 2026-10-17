# NHẬT KÝ TUẦN — HÀNH TRÌNH PHÁT TRIỂN DỰ ÁN (WEEKLY LOGS)

**Dự án:** QC Studio 2D — Hệ thống Tự động Kiểm soát & Thẩm định Chất lượng Dữ liệu Gán nhãn cho Xe tự hành  
**Nhân sự phụ trách:**
- **Đoàn Văn Thắng** — Trưởng nhóm / System Architect / Frontend Engineer (Blazor & Interactive UX)
- **Nguyễn Thanh Đức** — AI & Core Logic Engineer (Rule Engine, Consensus AI, CVAT API)

> Các nhãn “Hoàn thành” dưới đây tóm tắt công việc/source theo nhật ký, không đồng nghĩa đã được kiểm thử live. Trạng thái test xác minh được, các giới hạn upload/CVAT/deploy và số liệu hiện hành xem [`docs/TESTING.md`](../docs/TESTING.md).

---

## 📅 Bảng tổng kết tiến độ 5 tuần

| Tuần | Mục tiêu chính | Trọng tâm công việc | Trạng thái |
|:---:|:---|:---|:---:|
| **[Tuần 01](tuan-01.md)** | Khảo sát bài toán & Nghiên cứu dữ liệu | Phân tích bài toán kiểm định chất lượng nhãn cho xe tự hành (BDD100K). Nghiên cứu định dạng nhãn 2D (BBox, Poly2D, Semantic Train-ID Mask). Khảo sát hệ sinh thái CVAT/Nuclio. | ✅ Hoàn thành |
| **[Tuần 02](tuan-02.md)** | Thiết kế kiến trúc 4 tầng QA & Xây dựng Rule Engine | Thiết kế kiến trúc tổng thể local-first. Xây dựng Rule Engine tầng 1 (Boundary clamping, Degenerate box, Box duplicate, Self-intersecting polygon). | ✅ Hoàn thành |
| **[Tuần 03](tuan-03.md)** | Tích hợp AI Ensemble & Tính năng Auto-Heal | Tích hợp Model Detector (YOLO26, DINOv3) và nhà cung cấp Gemini VLM. Xây dựng chuẩn hóa không gian tọa độ (Slide 37, 22) và thuật toán 1-Click Auto-Heal. | ✅ Hoàn thành |
| **[Tuần 04](tuan-04.md)** | Phát triển Web Dashboard & Đồng bộ 2 chiều CVAT | Xây dựng giao diện Blazor cao cấp kết nối FastAPI backend. Triển khai API 2 chiều với CVAT REST API (kéo Job, gắn issue chính xác frame và trả Job). | ✅ Hoàn thành |
| **[Tuần 05](tuan-05.md)** | Kiểm thử liên thông, Parser đa định dạng & Đóng gói MVP | Xây dựng Flexible Parser (VOC XML, YOLO TXT, CSV, JSON, ZIP), chạy smoke test 50 frames BDD100K thật, hoàn thiện tài liệu nộp bài và đóng gói chạy 1-click. | ✅ Hoàn thành |

---

## 🎯 Nguyên tắc làm việc & Báo cáo
1. **Trung thực trong dữ liệu (Truthful & Scientific):** Tuyệt đối không tự làm tròn hay bịa số liệu Recall/Precision khi chưa có tập Gold Standard độc lập được thẩm định bởi 2 reviewer trở lên.
2. **Reviewer là người quyết định cuối cùng (Human-in-the-loop):** Mọi cảnh báo từ Model và Rule Engine đều đóng vai trò là "Ứng viên kiểm duyệt" (Candidate Findings).
3. **Bảo mật & Tự chủ (Offline-First):** Hệ thống ưu tiên chạy cục bộ hoàn toàn, bảo mật dữ liệu khách hàng/doanh nghiệp; tùy chọn cloud VLM (Gemini) chỉ kích hoạt khi có sự đồng ý tường minh của người dùng.
