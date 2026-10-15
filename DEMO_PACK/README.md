# THƯ MỤC CHỨA DỮ LIỆU DEMO QC STUDIO

Thư mục này được tạo riêng để bạn ném dữ liệu kiểm thử (ảnh, nhãn, guideline) vào hoặc dùng ngay bộ dữ liệu mẫu đã được chuẩn bị sẵn.

---

## 📁 1. DEMO BBOX (Phát hiện vật thể 2D)
Nằm tại thư mục: `DEMO_PACK/BBOX/`
- **File nhãn JSON (Input 01):** `DEMO_PACK/BBOX/annotations.json`
- **File quy chuẩn Markdown (Input 02):** `DEMO_PACK/BBOX/guideline.md`
- **Thư mục ảnh JPG:** `DEMO_PACK/BBOX/images/` (đã có sẵn 50 ảnh chuẩn BDD100K)

👉 **Cách dùng trên web:**
1. Chọn chế độ **BBox** (thanh bên phải).
2. Input 01: Chọn `DEMO_PACK/BBOX/annotations.json`.
3. Input 02: Chọn `DEMO_PACK/BBOX/guideline.md`.
4. Image Input: Nhấn Browse và chọn toàn bộ ảnh trong `DEMO_PACK/BBOX/images/`.
5. Bấm **Run local QC**.

---

## 📁 2. DEMO SEGMENTATION (Phân vùng ngữ nghĩa)
Nằm tại thư mục: `DEMO_PACK/SEGMENTATION/`
- **File nhãn JSON (Input 01):** `DEMO_PACK/SEGMENTATION/annotations.json`
- **File quy chuẩn Markdown (Input 02):** `DEMO_PACK/SEGMENTATION/guideline.md`
- **Ảnh JPG:** `DEMO_PACK/SEGMENTATION/images/` (50 ảnh)
- **Masks màu:** `DEMO_PACK/SEGMENTATION/masks_color/` (50 mask hiển thị)
- **Masks index (Train-ID):** `DEMO_PACK/SEGMENTATION/masks_index/` (50 mask đo IoU)

---

## 💡 Ném dữ liệu mới của bạn vào đâu?
- Nếu bạn có dataset BBox riêng: Bạn chỉ cần chép file json và ảnh của bạn vào thư mục `DEMO_PACK/BBOX/` (hoặc tạo thư mục mới tùy thích trong này).
- Sau đó trên giao diện [http://127.0.0.1:5000](http://127.0.0.1:5000), trỏ tới đúng thư mục `DEMO_PACK` này là xong.
