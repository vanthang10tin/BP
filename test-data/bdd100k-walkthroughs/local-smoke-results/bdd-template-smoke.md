# TEMPLATE ONLY — BDD100K 2D guideline; not an approved project policy

## 1. QUY CHUẨN HÌNH HỌC & ĐỘ KHÍT (BDD100K BBOX & SEGMENTATION)
§1.1. Bounding Box phải ôm sát các điểm cực biên ngoài cùng của vật thể (Tightness >= 80%). Không vẽ thừa nền đường quá 20% và không cắt cụt các phần thân xe, bánh xe, gương xe.
§1.2. Mọi tọa độ Bbox phải nằm trọn vẹn trong khung ảnh 1280x720 (0 <= x1 < x2 <= 1280, 0 <= y1 < y2 <= 720).
§1.3. Cấm tuyệt đối box có kích thước < 4px hoặc diện tích < 16px^2 (chấm rác).
§1.4. Đa giác Segmentation (Polygon) không được tự cắt chéo cạnh (self-intersect), đỉnh trùng lặp hoặc suy biến.
§1.5. Mask Segmentation và Bounding Box tương ứng phải đồng tâm và có độ chồng lấn IoU >= 0.70.

## 2. QUY ĐỊNH PHÂN LOẠI CATEGORY & THUỘC TÍNH (CLASSES & ATTRIBUTES)
§2.1. Phân biệt rõ giữa 'car' (xe con, sedan, SUV), 'truck' (xe tải có thùng hàng) và 'bus' (xe khách, xe buýt).
§2.2. Phân biệt rõ giữa 'pedestrian' (người đi bộ) và 'rider' (người đang ngồi/điều khiển xe máy, xe đạp).
§2.3. Bắt buộc điền đầy đủ 2 thuộc tính: 'occluded' (bị che khuất) và 'truncated' (bị cắt ở mép ảnh).
§2.4. Không gán nhãn các hình ảnh phản chiếu trên gương/kính hoặc hình vẽ quảng cáo xe trên thân xe buýt/tường nhà.

## 3. Project-specific policy
No project-specific acceptance thresholds are configured in this template. Upload and approve the applicable project guideline before using policy-dependent checks.
