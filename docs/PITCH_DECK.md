# BẢN THUYẾT TRÌNH DỰ ÁN — PITCH DECK

**Tên dự án:** QC Studio 2D — Next-Gen AI Quality Control & Annotation Auditing Studio for Autonomous Driving  
**Đội thi:** Đoàn Văn Thắng — Nguyễn Thanh Đức  
**Mã đề tài:** N2-04A  

---

```
  ██████╗  ██████╗    ███████╗████████╗██╗   ██╗██████╗ ██╗ ██████╗     ██████╗ ██████╗ 
 ██╔═══██╗██╔════╝    ██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║██╔═══██╗    ╚════██╗██╔══██╗
 ██║   ██║██║         ███████╗   ██║   ██║   ██║██║  ██║██║██║   ██║     █████╔╝██║  ██║
 ██║▄▄ ██║██║         ╚════██║   ██║   ██║   ██║██║  ██║██║██║   ██║    ██╔═══╝ ██║  ██║
 ╚██████╔╝╚██████╗    ███████║   ██║   ╚██████╔╝██████╔╝██║╚██████╔╝    ███████╗██████╔╝
  ╚══▀▀═╝  ╚═════╝    ╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝     ╚══════╝╚═════╝ 
```

---

## 🖥️ SLIDE 1: TRANG TIÊU ĐỀ (TITLE & HOOK)

- **Tiêu đề chính:** **QC STUDIO 2D**
- **Tiêu đề phụ:** Hệ thống Tự động Kiểm soát & Thẩm định Chất lượng Dữ liệu Gán nhãn cho Xe tự hành
- **Tagline:** *"QC candidates cho annotation giao thông — phát hiện một số bất thường và hỗ trợ reviewer rà soát theo quy trình có thể kiểm chứng."*
- **Người trình bày:**
  - **Đoàn Văn Thắng** — System Architect & Frontend Engineer
  - **Nguyễn Thanh Đức** — AI & Core Logic Engineer
- **Ảnh đại diện dự án:** `thumbnail.jpg` (Hình ảnh HUD kiểm định chất lượng nhãn xe tự hành trên đường phố).

---

## ⚠️ SLIDE 2: BÀI TOÁN & NỖI ĐAU THỰC TẾ (THE PROBLEM: DATA DEBT IN AUTONOMOUS DRIVING)

- **Bài toán của đề tài:**
  - Nhãn bbox/polygon/mask có thể chứa lỗi cấu trúc hoặc hình học cần được kiểm tra trước khi bàn giao.
  - Mức tốn công và giá trị của hàng đợi QC là giả thuyết cần đo trên quy trình review thật.
- **Phạm vi:** QC Studio 2D đưa ra finding theo rule và model candidates khi có provider; con người xác nhận. Chưa có benchmark độc lập để suy rộng tỷ lệ lỗi.

---

## 🔍 SLIDE 3: GIẢ THUYẾT BÀI TOÁN & NHU CẦU REVIEW (PROBLEM HYPOTHESIS)

Các giả thuyết nhóm muốn kiểm tra:
1. **Nhu cầu rà soát:** Nhóm đặt giả thuyết rằng rule checks và model candidates có thể hỗ trợ reviewer tìm một số trường hợp cần chú ý; tỷ lệ false positive/chất lượng chưa được đo.
2. **Quy trình review:** Handoff từ finding sang người sửa dữ liệu là tình huống nhóm muốn hỗ trợ; chưa có nghiên cứu thời gian hay tỷ lệ nhầm lẫn để định lượng.
3. **Quyền riêng tư (Data Privacy):** Một số quy trình cần gửi dữ liệu lên dịch vụ ngoài; dự án cung cấp chế độ rule/model local khi provider local được cấu hình. Public demo chỉ dành cho dữ liệu không nhạy cảm.

---

## 💡 SLIDE 4: GIẢI PHÁP CỦA CHÚNG TÔI (OUR SOLUTION: QC STUDIO 2D)

**QC Studio 2D** hỗ trợ quy trình review annotation bằng:
1. Các rule hình học được cấu hình và model disagreement candidates khi provider khả dụng.
2. Hàng đợi ưu tiên theo heuristic; chưa đo mức cải thiện thời gian review.
3. Thao tác Auto-Heal cho một số trường hợp hình học xác định; reviewer cần kiểm tra thay đổi.
4. API CVAT tùy chọn; cần xác minh trên job thử trước khi dùng dữ liệu/job thật. Không coi gold evaluator là đã đánh giá độc lập nếu chưa nạp benchmark adjudicated.

---

## ⚙️ SLIDE 5: CHI TIẾT LÕI THẨM ĐỊNH 4 TẦNG QA (THE 4-LAYER ENGINE)

| Tầng QA | Bản chất công nghệ | Đối tượng phát hiện | Tỷ lệ chính xác |
|:---|:---|:---|:---:|
| **Tầng 1: Rule Engine** | Kiểm tra hình học/ontology xác định theo cấu hình | Cờ lỗi box/polygon/thuộc tính trong phạm vi rule hiện triển khai. | **Chưa đo accuracy trên gold set** |
| **Tầng 2: Model candidates** | YOLO26 cục bộ khi có ảnh + weights; provider khác cần cấu hình riêng | Nghi vấn missed object/class disagreement để reviewer xác minh. | **Chưa đo precision/recall** |
| **Tầng 3: Queue & LQS** | Điểm heuristic và sắp xếp candidate | Ưu tiên frame có finding; không tự chứng minh frame không cờ là sạch. | **Chưa đo giảm thời gian** |
| **Tầng 4: Gold evaluator** | Hàm so sánh với ground-truth do người dùng cung cấp | Có thể tính metric khi có gold set độc lập và protocol định trước. | **Chưa có pilot gold set** |

---

## 🏗️ SLIDE 6: KIẾN TRÚC HỆ THỐNG & TÍNH KHẢ THI (SYSTEM ARCHITECTURE & FEASIBILITY)

- **Frontend:** UI tĩnh HTML/CSS/JS được phục vụ từ FastAPI ở local và Render; Blazor là giao diện local bổ sung tại `127.0.0.1:5000`.
- **Backend:** FastAPI phục vụ local tại `127.0.0.1:8000` hoặc host/port do deploy platform cấp.
- **Khả năng tương thích đa định dạng:** Parser có adapter cho JSON, XML, TXT, CSV và ZIP theo các schema được hỗ trợ; cần sample phù hợp schema để xác minh từng adapter.
- **Quyền riêng tư theo cấu hình:** Rule và model local có thể chạy tại máy người dùng. Gemini là tùy chọn gửi ảnh ra Google API; CVAT gọi dịch vụ đã cấu hình. Public app chưa có đăng nhập, vì vậy chỉ dùng dữ liệu demo không nhạy cảm.

---

## 🚀 SLIDE 7: CÁC TÍNH NĂNG ĐỘT PHÁ (KEY HIGHLIGHTS)

1. **Auto-Heal cho một số trường hợp xác định:** Có thể kẹp box, xóa box nhỏ/trùng và bổ sung thuộc tính theo cấu hình; reviewer phải kiểm tra file kết quả trước khi dùng.
2. **Guideline citation:** Nội dung phụ thuộc guideline mà đội tải lên; template mặc định không phải tiêu chuẩn chính thức.
3. **Tích hợp CVAT tùy chọn:**
   - Có API lấy dữ liệu/gửi issue khi URL và token được cấu hình. Cần xác minh trên job thử trước khi dùng dữ liệu hoặc job thật.

---

## 📊 SLIDE 8: KẾT QUẢ THỰC NGHIỆM TRÊN BDD100K (EXPERIMENTAL RESULTS)

Lần chạy local ngày 17/10/2026 trên 20 record đầu của file sample BDD100K đóng gói (587 labels):
- **Kết quả của một lần chạy:** 10 frame không có candidate và 10 frame có candidate. Không có candidate không đồng nghĩa sạch; không được bỏ qua review chỉ dựa trên thống kê này.
- **Các loại candidate do rule/model phát ra trong lần chạy này (chưa xác minh là lỗi thật):**
  - 8 trường hợp tỷ lệ khung hình bất thường (`BOX_UNREASONABLE_ASPECT_RATIO`).
  - 5 trường hợp nghi vấn nhầm class (`CLASS_CONFUSION`).
  - 1 candidate vật thể có thể bị bỏ sót (`POTENTIAL_MISSED_OBJECT`).
  - 1 frame gần biên chưa có truncated flag.
- **Giới hạn đo lường:** Chưa có gold set độc lập, nên chưa báo cáo precision, recall, IoU, Recall@20 hay tỷ lệ giảm thời gian.

---

## 💰 SLIDE 9: TÍNH KHẢ THI & GIÁ TRỊ KINH TẾ (FEASIBILITY & BUSINESS VALUE)

1. **Giả thuyết giá trị cần pilot:** So sánh thời gian review toàn bộ tập với thời gian review theo hàng đợi; dùng cùng reviewer, cùng tập frame, cùng tiêu chí lỗi. Chưa có số tiết kiệm hoặc quy đổi chi phí được đo.
2. **Trạng thái triển khai:** Pytest, smoke logic và Blazor build đã được chạy như ghi ở `docs/TESTING.md`. CVAT live handoff và public deployment chưa được xác minh. Chưa gọi sản phẩm production-ready.

---

## 🎬 SLIDE 10: QUY TRÌNH TRẢI NGHIỆM THỰC TẾ (LIVE DEMO WALKTHROUGH)

- **Bước 1: Nạp dữ liệu** $\rightarrow$ Kéo thả file dataset BDD100K JSON, VOC XML hoặc YOLO ZIP vào giao diện.
- **Bước 2: Quét tự động & Xem LQS heuristic** $\rightarrow$ Hệ thống hiển thị finding và thứ tự review; LQS chưa được hiệu chỉnh thành độ chính xác chất lượng.
- **Bước 3: Trực quan hóa & Đối soát** $\rightarrow$ Bấm vào frame có candidate, xem ảnh/nhãn và căn cứ guideline nếu có điều khoản phù hợp; không có guideline thì không gắn trích dẫn.
- **Bước 4: Bấm 1-Click Auto-Heal** $\rightarrow$ Tự động sửa nhanh các lỗi hình học.
- **Bước 5: Xuất CSV hoặc gửi issue CVAT (nếu đã cấu hình)** $\rightarrow$ Kiểm tra trên job thử trước khi dùng tích hợp CVAT.

---

## 🗺️ SLIDE 11: ĐỊNH HƯỚNG PHÁT TRIỂN TƯƠNG LAI (FUTURE ROADMAP)

- **Quý 1/2027:** Mở rộng hỗ trợ dữ liệu 3D LiDAR Point Cloud (3D Bounding Box QC) phục vụ hệ thống tự hành đa cảm biến.
- **Quý 2/2027:** Tích hợp kiểm định chất lượng Video Object Tracking (phát hiện lỗi nhảy ID, mất dấu đối tượng giữa các khung hình liên tiếp).
- **Quý 3/2027:** Mở rộng CI integration và đánh giá các quality-gate workflows; mọi quyết định từ chối job cần policy được phê duyệt và kiểm thử.

---

## 👥 SLIDE 12: ĐỘI NGŨ & LỜI KẾT (TEAM & CONCLUSION)

- **Đội ngũ thực hiện:**
  - **Đoàn Văn Thắng** — Trưởng nhóm, Thiết kế hệ thống & Giao diện tương tác Blazor.
  - **Nguyễn Thanh Đức** — Kỹ sư AI & Core Logic, model candidates và CVAT API tùy chọn.
- **Thông điệp kết luận:**  
  *"QC Studio 2D giúp reviewer rà soát có hệ thống một số bất thường trong nhãn. Độ chính xác, hiệu quả và mức phù hợp phải tiếp tục được đo trên dữ liệu adjudicated và quy trình thực tế."*
- **Kho mã nguồn & Tài liệu:** [https://github.com/vanthang10tin/BP.git](https://github.com/vanthang10tin/BP.git)  
- **Lời cảm ơn:** Chân thành cảm ơn Ban giám khảo, Quý thầy cô và Cộng đồng đã lắng nghe!
