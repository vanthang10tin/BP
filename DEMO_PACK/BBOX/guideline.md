# QUY CHUẨN GÁN NHÃN 2D BOUNDING BOX (QC STUDIO)

## I. QUY ĐỊNH HÌNH HỌC & ĐỘ KHÍT (BOUNDING BOX)
§1.1. Độ khít Bounding Box: Box phải ôm sát toàn bộ các điểm cực biên ngoài cùng của vật thể (Tightness IoU >= 0.75). Không bao thừa quá 5px hoặc cắt lọt chi tiết bánh xe, gương, đèn, đầu người.
§1.2. Giới hạn viền ảnh: Tọa độ box không được vượt ra ngoài chiều rộng và chiều cao thực tế của ảnh (0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height). Nếu vật thể chạm mép ảnh bắt buộc phải bật thuộc tính truncated=True.
§1.3. Kích thước tối thiểu: Không gán nhãn cho các vật thể quá nhỏ hoặc chấm rác nhiễu (Chiều rộng < 6px hoặc chiều cao < 6px hoặc diện tích < 36px²).
§1.4. Trùng lặp đối tượng: Hai box cùng class chồng lấn nhau với IoU >= 0.70 bị tính là lỗi trùng lặp gán nhãn (Duplicate Bounding Box).
§1.5. Vật thể ảo / Bóng đổ / Phản chiếu: Không gán nhãn lên bóng đổ của xe trên mặt đường hoặc hình ảnh phản chiếu trên gương kính hoặc tranh vẽ quảng cáo trên tường.

## II. DANH MỤC ĐỐI TƯỢNG (ONTOLOGY)
§2.1. Phân loại phương tiện giao thông (Vehicles):
- car: Ô tô con, sedan, hatchback, SUV, taxi, xe bán tải nhỏ gia đình.
- truck: Xe tải chở hàng, xe ben, container, xe bồn lớn.
- bus: Xe khách, xe buýt công cộng nhiều chỗ ngồi và cửa kính đặc trưng.
- motorcycle: Xe máy, mô tô 2 bánh, scooter có gắn động cơ.
- bicycle: Xe đạp thể thao, xe đạp thông thường không động cơ.
- train: Tàu hỏa, toa tàu, tàu điện chạy trên đường ray.
§2.2. Phân loại con người (Humans):
- pedestrian: Người đi bộ ở mọi tư thế (đứng, đi, chạy, ngồi trên vỉa hè).
- rider: Người đang điều khiển phương tiện (lái xe máy, xe đạp). Gán nhãn rider cho người điều khiển và gán nhãn phương tiện riêng cho xe.
§2.3. Biển báo & Đèn giao thông (Signage & Signals):
- traffic light: Cụm đèn tín hiệu giao thông (xanh, vàng, đỏ).
- traffic sign: Biển báo giao thông các loại gắn trên cột hoặc giá long môn.
§2.4. Thuộc tính bắt buộc (Attributes):
- occluded: Đánh dấu True nếu đối tượng bị che khuất một phần bởi vật thể khác (> 20%).
- truncated: Đánh dấu True nếu đối tượng bị cắt một phần bởi viền khung ảnh.

## III. CƠ CHẾ ĐỐI SOÁT HỘI ĐỒNG & PHÂN XỬ
§3.1. Đối soát nội bộ 3 model (YOLO26x, D-FINE-X, Gemini 3.8 Flash) tạo thành tập Box của AI.
§3.2. Khi 2 detector đồng thuận cùng class với IoU >= 0.35: Xác nhận kết quả AI thống nhất.
§3.3. Khi 2 detector bất đồng class hoặc chỉ 1 model phát hiện: Trọng tài Gemini 3.8 Flash phân xử để giữ lại box đúng.
§3.4. Điểm khác biệt giữa Box của Người và Box của AI tạo thành tập Box cần Review.
