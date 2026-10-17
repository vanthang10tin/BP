# BÁO CÁO TIẾN ĐỘ DỰ ÁN — TUẦN 05

**Dự án:** Hệ thống hỗ trợ kiểm soát và đánh giá chất lượng dữ liệu hình ảnh  
**Thời gian thực hiện:** Tuần 05  
**Nhân sự phụ trách:**
- Đoàn Văn Thắng
- Nguyễn Thanh Đức

---

## 1. TỔNG QUAN MỤC TIÊU TUẦN 05
- Hoàn thiện kế hoạch chi tiết và tài liệu hóa toàn bộ luồng quy trình vận hành của hệ thống.
- Xây dựng hoàn chỉnh khung giao diện người dùng và các phân khu chức năng phục vụ thao tác kiểm tra dữ liệu.
- Thiết lập hệ thống quy tắc nghiệp vụ và cơ chế rà soát tự động ban đầu.
- Tiến hành chạy thử nghiệm liên thông từ đầu vào đến đầu ra trên các tập dữ liệu mẫu để đánh giá tính khả thi và độ ổn định của hệ thống.

---

## 2. PHÂN CÔNG VÀ KẾT QUẢ CÔNG VIỆC CHI TIẾT

### 👤 Đoàn Văn Thắng
* **Xây dựng quy hoạch hệ thống và tài liệu luồng quy trình:**
  - Hoàn thành tài liệu phân tích luồng làm việc tổng thể từ khâu tiếp nhận dữ liệu đầu vào, xử lý tự động, kiểm duyệt thủ công đến khâu kết xuất báo cáo nghiệm thu.
  - Xây dựng kế hoạch triển khai từng giai đoạn và xác định các tiêu chí nghiệm thu cho từng phân hệ.

* **Hoàn thiện thiết kế và xây dựng giao diện người dùng:**
  - Triển khai xong toàn bộ giao diện làm việc trực quan cho người kiểm duyệt, bao gồm khu vực tải lên dữ liệu, màn hình trực quan hóa hình ảnh kèm thông tin đối soát, và bảng điều khiển trung tâm.
  - Thiết kế bảng hàng đợi thông minh giúp phân loại và sắp xếp thứ tự ưu tiên các trường hợp cần rà soát dựa trên mức độ cảnh báo.
  - Xây dựng khu vực xem xét chi tiết cho phép người dùng xác nhận, bác bỏ hoặc để lại ghi chú đối với từng cảnh báo nghi vấn.
  - Tích hợp màn hình quản lý, xem trước và chỉnh sửa bộ tiêu chuẩn/hướng dẫn kiểm tra chất lượng.

* **Phối hợp thử nghiệm và kiểm thử tích hợp:**
  - Thực hiện kiểm tra tính liên kết và độ phản hồi giữa các màn hình chức năng.
  - Thử nghiệm chức năng xuất báo cáo tổng hợp sau kiểm tra dưới nhiều định dạng khác nhau.

---

### 👤 Nguyễn Thanh Đức
* **Thiết lập bộ tiêu chuẩn và quy tắc kiểm tra dữ liệu:**
  - Xây dựng và hoàn thiện bộ tiêu chí đánh giá tự động đối với dữ liệu hình ảnh (kiểm tra tính hợp lệ về hình học, cấu trúc đối tượng, các lỗi trùng lặp và phân loại sai danh mục).
  - Chuẩn hóa các kịch bản kiểm tra đối với các định dạng dữ liệu khác nhau để đảm bảo tính tương thích khi nạp vào hệ thống.

* **Phát triển lõi xử lý và cơ chế đánh giá tự động:**
  - Xây dựng logic phân tích dữ liệu tự động, đối chiếu giữa dữ liệu thực tế và bộ quy tắc đã ban hành để tự động phát hiện các điểm bất thường.
  - Thiết lập cơ chế tích hợp với module hỗ trợ rà soát nâng cao nhằm tạo ra các đề xuất khách quan hỗ trợ người kiểm duyệt.
  - Xây dựng cơ chế trích xuất vị trí nghi ngờ lỗi và tính toán các chỉ số tương đồng giúp khoanh vùng vấn đề nhanh chóng.

* **Thử nghiệm trên dữ liệu mẫu và chuẩn bị môi trường chạy:**
  - Chuẩn bị các gói dữ liệu mẫu chuẩn (bao gồm nhiều tình huống dữ liệu từ đơn giản đến phức tạp) để đưa vào kiểm tra thực tế.
  - Thực hiện các đợt chạy thử nghiệm đo lường hiệu năng và độ ổn định của bộ kiểm tra tự động trên các tập dữ liệu mẫu.

---

## 3. TÓM TẮT NHỮNG GÌ ĐÃ ĐẠT ĐƯỢC (TỔNG KẾT HIỆN TRẠNG)

1. **Về mặt quy trình & Kế hoạch (Planning):**
   - Đã thống nhất và hoàn tất toàn bộ tài liệu đặc tả luồng xử lý của hệ thống.
   - Xác định rõ vai trò của việc kiểm tra tự động là đưa ra cảnh báo và đề xuất hỗ trợ, quyền quyết định nghiệm thu cuối cùng thuộc về người kiểm duyệt.

2. **Về mặt giao diện người dùng (UI/UX):**
   - Đã hoàn thành các phân hệ giao diện:
     - Màn hình tiếp nhận và nạp dữ liệu.
     - Màn hình hiển thị hình ảnh và đối sánh trực quan.
     - Danh sách hàng đợi phân cấp ưu tiên các trường hợp nghi ngờ lỗi.
     - Bảng thao tác phê duyệt / từ chối cảnh báo của người kiểm duyệt.
     - Màn hình quản lý tiêu chuẩn kiểm tra và màn hình xuất báo cáo kết quả.

3. **Về mặt xử lý nghiệp vụ (Core Logic):**
   - Bộ quy tắc kiểm tra cấu trúc dữ liệu đã hoạt động ổn định và lọc được các lỗi sai cơ bản.
   - Module đánh giá tự động bước đầu đã có thể tạo ra các gợi ý và chỉ số so sánh để hỗ trợ việc ra quyết định.
   - Đã có cơ chế liên kết và chuyển giao dữ liệu cảnh báo sang hệ thống quản lý công việc bên ngoài khi cần thiết.

4. **Về mặt thử nghiệm thực tế (Testing & Experimentation):**
   - Đã chạy pipeline QC trên 50 record BDD100K bằng script. Luồng HTTP upload/UI chưa được xác minh end-to-end trong môi trường hiện tại.
   - Các gói dữ liệu mẫu dùng để walkthrough; chưa có tiêu chí gold độc lập để kết luận chất lượng finding.

---

## 4. CÁC CÔNG VIỆC ĐANG TRONG GIAI ĐOẠN THỬ NGHIỆM & TINH CHỈNH

> Cập nhật kiểm chứng ngày 18/10/2026: toàn bộ kiểm thử unit/parser và 5 kiểm thử upload HTTP (kèm kiểm tra giới hạn kích thước payload không gây treo) đều đã pass theo [`docs/TESTING.md`](../docs/TESTING.md); 8/8 smoke logic assertions trong `test_qc_system.py` đã pass. CVAT live trên job thật và public URL trên Render cần được xác nhận khi deploy.

- **Đo lường và tối ưu thời gian xử lý:** Đang thử nghiệm xử lý đồng thời với khối lượng dữ liệu lớn hơn để theo dõi hiệu suất và dung lượng bộ nhớ.
- **Hiệu chỉnh độ nhạy của cảnh báo:** Đang rà soát lại các tiêu chí cảnh báo tự động nhằm hạn chế tình trạng báo nhầm (false positives) đối với các trường hợp dữ liệu hợp lệ.
- **Tối ưu trải nghiệm thao tác:** Thử nghiệm giao diện thực tế để cải thiện tốc độ duyệt của người dùng (bổ sung bộ lọc, phân trang và gom nhóm các lỗi tương đồng).

---

## 5. VƯỚNG MẮC
- Sẽ báo cáo và trao đổi trực tiếp trong buổi mentor duty.
