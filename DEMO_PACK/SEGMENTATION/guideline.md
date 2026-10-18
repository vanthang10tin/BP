# QUY CHUẨN GÁN NHÃN SEMANTIC & INSTANCE SEGMENTATION (QC STUDIO)

## I. QUY ĐỊNH HÌNH HỌC & ĐỘ KHÍT MASK
§1.1. Độ khít biên Mask (Boundary Tightness): Mask phân đoạn phải bám sát chính xác từng pixel viền mép của đối tượng. Dung sai lệch viền không vượt quá 3px. Ngưỡng IoU tối thiểu giữa nhãn và đối tượng thực tế phải đạt >= 0.70 (đối với foreground) và >= 0.75 (đối với background). Cấm cắt cụt (under-segmentation) nóc xe, gương xe, chân người hoặc để sót mép vỉa hè.
§1.2. Chống lem / Tràn viền (No Leaking / Over-segmentation): Mask của đối tượng không được lem tràn ra ngoài nền đường hoặc vật thể lân cận quá 3px.
§1.3. Chống trùng lấn giữa các lớp loại trừ lẫn nhau (Zero-Overlap Policy): Các lớp nền tĩnh như road, sidewalk, building, vegetation, sky không được phép chồng lấn lên nhau (Overlap IoU = 0). Nếu phát hiện 2 mask nền đè lên nhau (ví dụ: sidewalk đè lên road), tính là lỗi trùng lấn sai quy định.
§1.4. Chống hở khe tiếp giáp (No Gap Policy): Giữa mép đường (road) và vỉa hè (sidewalk), hoặc giữa vỉa hè và tòa nhà (building) không được để lại các khoảng trống chưa gán nhãn (unlabeled gap > 3px).
§1.5. Tính toàn vẹn Polygon / Mask: Đa giác phân đoạn không được tự cắt chéo cạnh (self-intersecting), không có đỉnh trùng lặp hoặc diện tích suy biến (< 16px²).
§1.6. Quy tắc thứ tự lớp (Painter's Algorithm): Các lớp nền tĩnh (road, sidewalk, building, wall, fence, pole, vegetation, terrain, sky) vẽ trước. Các đối tượng chuyển động / foreground (car, truck, bus, pedestrian, rider, motorcycle, bicycle, traffic sign, traffic light) vẽ đè lên trên lớp nền, không khoét thủng lớp nền một cách tùy tiện.

## II. DANH MỤC 19 LỚP PHÂN ĐOẠN (BDD100K / CITYSCAPES ONTOLOGY)
§2.1. Nhóm Mặt đường & Hạ tầng di chuyển:
- road: Toàn bộ mặt đường xe chạy (nhựa đường, bê tông, vạch kẻ đường, gờ giảm tốc, nắp cống trên đường).
- sidewalk: Vỉa hè lát gạch/bê tông dành cho người đi bộ, bó vỉa (curb), lối đi bộ lát đá.
§2.2. Nhóm Công trình & Vật cản cố định:
- building: Tòa nhà, nhà dân, cao ốc, trạm xăng, công trình kiến trúc xây dựng cố định.
- wall: Bức tường gạch, tường bê tông độc lập không phải tường của tòa nhà.
- fence: Hàng rào kim loại, rào gỗ, lan can ngăn cách trên cầu/đường.
- pole: Cột đèn chiếu sáng, cột điện, cột biển báo giao thông (thanh trụ thẳng đứng).
§2.3. Nhóm Tín hiệu & Biển chỉ dẫn:
- traffic light: Hộp đèn tín hiệu giao thông (gồm cả vỏ hộp và bóng đèn).
- traffic sign: Biển báo giao thông các loại gắn trên cột hoặc giá treo long môn.
§2.4. Nhóm Thiên nhiên & Môi trường:
- vegetation: Cây cối, tán lá, bụi cây, thảm cỏ, cành cây, rặng dừa.
- terrain: Đất trống, đồi cát, sườn núi, mặt đất mấp mô không lát gạch/nhựa.
- sky: Bầu trời thông thoáng, mây, không bao gồm chim/máy bay.
§2.5. Nhóm Con người:
- person / pedestrian: Người đi bộ ở mọi tư thế (đứng, đi, chạy, ngồi, cúi).
- rider: Người đang cưỡi hoặc điều khiển xe đạp, xe máy, ngựa.
§2.6. Nhóm Phương tiện giao thông (Vehicles):
- car: Xe con 4-7 chỗ, sedan, hatchback, SUV, taxi, xe bán tải nhỏ gia đình.
- truck: Xe tải chở hàng, xe ben, container, xe bồn, xe đầu kéo.
- bus: Xe buýt công cộng, xe khách liên tỉnh nhiều chỗ ngồi.
- train: Tàu hỏa, toa tàu, tàu điện chạy trên đường ray.
- motorcycle: Xe máy, xe mô tô 2 bánh, scooter có gắn động cơ.
- bicycle: Xe đạp các loại (xe đạp thể thao, mini, cào cào).

## III. CƠ CHẾ KIỂM TRA & ĐỐI SOÁT HỘI ĐỒNG
§3.1. Đánh giá IoU theo từng class: Class có IoU < 0.65 sẽ bị đánh dấu Candidate cần Review.
§3.2. Phát hiện viền chưa khít hoặc viền lem nhem: So sánh chênh lệch giữa mask người gán và kết quả dự đoán của YOLO26x-Seg / SAM 2.1 Large / SegFormer.
§3.3. Trọng tài Gemini Vision VLM: Khi 1 model đạt mà 1 model báo lỗi viền, Gemini thẩm định vùng crop để xác nhận lỗi do người gán hay do model.
