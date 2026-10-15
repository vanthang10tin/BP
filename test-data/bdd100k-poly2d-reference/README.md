# Bộ walkthrough BDD100K — 4 x 50 frame thật

> **Bộ tham khảo cũ, không dùng cho luồng Segmentation.** Các folder dưới đây minh họa `poly2d` dạng vector trong detection JSON; đây không phải semantic segmentation PNG mask. UI hiện không có Polygon mode. Dùng `test-data/bdd100k-walkthroughs/03-*` hoặc `04-*` để thử Segmentation.

Bốn bộ dữ liệu này được trích từ BDD100K validation có sẵn trong `archive/`. Mỗi bộ có đúng 50 ảnh khớp với 50 frame trong `annotations.json`; các bộ không trùng frame. File nhãn được giữ nguyên từ nguồn, không tự thêm hoặc sửa lỗi.

| Folder | Dùng để xem | Nội dung |
|---|---|---|
| `01-bbox-workflow-50` | Chế độ **BBox** | Frame có nhãn `box2d` |
| `02-polygon-workflow-50` | Chế độ **Polygon** | Frame có polygon `poly2d`, gồm lane/drivable area |
| `03-mixed-geometry-50` | Bật/tắt cả hai chế độ | Frame có cả `box2d` và `poly2d` |
| `04-rule-review-candidates-50` | Hàng đợi review | 50 frame mà rule hiện tại tạo candidate finding |

## Chạy từng luồng trên giao diện

1. Mở QC Studio ở `http://127.0.0.1:5000`.
2. Chọn **BBox** hoặc **Polygon** trên thanh bên phải theo tên bộ.
3. Trong Prepare, tải `annotations.json` của bộ đó lên.
4. Thêm toàn bộ JPG trong folder `images/` để xem ảnh và overlay (multi-select 50 file).
5. Nếu có guideline đã được đội duyệt thì tải lên; guideline mặc định trong app chỉ là template.
6. Chạy Analyze, xem frame queue, ảnh/nhãn và finding ở Review; tải Markdown/CSV trong Export.

Các candidate finding ở bộ 04 được sinh từ rule hình học/cấu trúc hiện tại; đó không phải nhãn lỗi chuẩn đã được con người adjudicate. Bốn bộ dùng để walkthrough và kiểm tra luồng, không dùng để khẳng định model chính xác hoặc tính Recall.

`INDEX.json` ghi tên frame và thứ tự chọn. Mỗi `manifest.json` ghi số lượng nhãn, nguồn, vị trí nguồn và SHA-256 từng ảnh/annotation. Dùng và chia sẻ theo điều khoản dữ liệu BDD100K.

Có thể tạo lại từ archive bằng lệnh sau (chỉ đọc prefix 1.200 frame validation theo thứ tự nguồn):

```powershell
python source-tool/qc-bdd100k/tools/prepare_bdd100k_walkthroughs.py
```
