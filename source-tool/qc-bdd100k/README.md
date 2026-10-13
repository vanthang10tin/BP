# QC Studio 2D - QC Tự Động Cho BBox & Segmentation (BDD100K)
### Next-Gen Multi-Model AI Consensus Engine (Phiên bản SOTA 2026)

**Mã đề tài:** N2-04A — Quality Control Check  
**Mục tiêu bài toán:** Hỗ trợ khâu **Post-annotation QC** (kiểm tra sau khi Annotator sửa từ pre-label). Tự động phân loại, bắt lỗi chuẩn chỉ, phân hạng mức độ nguy cơ từ Cao xuống Thấp và giảm 80-90% công sức review cho Đội P-037.  
**Kiến trúc:** Frontend (HTML5/Canvas/CSS/JS) + Backend (FastAPI / REST API) + AI Multi-Model Consensus Engine.

---

## 1. Kiến Trúc 4 Tầng Kiểm Soát Chất Lượng (4 QA Layers - Slide 29)

Hệ thống được thiết kế theo đúng quy trình 4 tầng QA chuẩn mực trong chuỗi cung ứng dữ liệu gán nhãn AI:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: KIỂM TRA THEO LUẬT (Rule-Based Structural & Geometry Checks)   │
│ • Chi phí tính toán: ~0 (Microseconds)                                 │
│ • Phạm vi: 100% nhãn được kiểm tra tức thì                             │
│ • Kiểm tra: Box ngoài biên 1280x720, box ngược (x2<x1), box rác <16px² │
│   Tỉ lệ khung hình vô lý theo class (aspect ratio), trùng box cùng class│
│   Class ngoài Ontology 10 nhãn BDD100K, Polygon tự cắt, lệch mask-box  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: MODEL-BASED OBJECTLAB / CLEANLAB (Slide 31)                    │
│ • Cơ chế: So sánh nhãn Người gán với Dự đoán Đồng thuận Model AI       │
│ • 3 Nhóm lỗi cốt lõi (Core Label Errors):                              │
│   1. Badly Located: Box lệch biên, không ôm khít (IoU < 0.65)          │
│   2. Swapped Class: Nhầm nhãn (Car ↔ Truck ↔ Bus, Rider ↔ Pedestrian)  │
│   3. Overlooked (Sót vật): Model AI đồng thuận cao nhưng người sót box │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: REVIEW CÓ ƯU TIÊN (Prioritized Active Review - Slide 34)       │
│ • Tối ưu hóa: Tiết kiệm 80-90% thời gian duyệt nhãn của Reviewer       │
│ • Phân hạng nguy cơ: Rank 1 (Critical) ➔ Rank 2 ➔ Rank 3 ➔ Rank 4 PASS │
│ • Benchmark hiệu quả (Slide 34):                                       │
│   ★ Recall@20% >= 70%: Duyệt 20% frame nghi vấn nhất bắt được >= 70% lỗi│
│   ★ Vượt trội hoàn toàn 2 Baseline: Random (20%) & Box-Count (35%)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: HONEYPOT & GOLD STANDARD JOB (Slide 33)                        │
│ • Cơ chế: Trộn các frame đã có nhãn chuẩn (Ground Truth) vào dữ liệu   │
│ • Đo lường hiệu năng Annotator: Precision, Recall, Mean IoU, Reject%   │
│ • Giám sát biến động chất lượng theo thời gian và đào tạo lại          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hội Đồng 5 Mô Hình AI Đối Soát (AI Council 2026 - Slide 19 & 21)

1. **Dual Bbox Ensemble:**
   * **YOLO26/27 (Ultralytics):** Loại bỏ hoàn toàn NMS (NMS-free), sử dụng hàm mất mát lũy tiến ProgLoss và STAL, bắt dính các vật thể xe/người đỗ sát nhau mà không bao giờ bị xóa nhầm box.
   * **RT-DETRv3 / D-FINE (Baidu/DeepVision):** Kiến trúc Vision Transformer bắt chuẩn vật thể bị che khuất và kích thước nhỏ.
2. **Dual Segmentation Ensemble:**
   * **Meta SAM 2 (Segment Anything 2):** Đo độ khít pixel cực biên, phát hiện khoảng hở mặt đường > 20%, hỗ trợ tự động nắn viền polygon khít vào vật thể.
   * **OneFormer / YOLO26-Seg:** Mô hình phân đoạn toàn cảnh kiểm tra biên giới đa giác và phát hiện polygon tự cắt/thoái hóa.
3. **Trọng tài VLM (Vision-Language Model):**
   * **Gemini 3.8 Flash:** Trọng tài phân xử các ca bất đồng (nhầm nhãn Car/Truck/Bus/Rider/Pedestrian) kết hợp trích dẫn trực tiếp điều khoản trong Sổ Quyết Định đội P-037.
   * Chuẩn hóa tọa độ VLM theo Slide 22: `box_2d = [ymin, xmin, ymax, xmax]` trên thang đo chuẩn `0 - 1000`.

---

## 3. Ma Trận Chuẩn Hóa Tọa Độ (Coordinate Conventions - Slide 37 & 22)

Module `backend/coordinate_utils.py` cung cấp bộ chuyển đổi 1-click không bị lỗi lệch 1 pixel hay nhầm lẫn giữa các chuẩn phổ biến trong ngành:

| Định Dạng | Cấu Trúc Dữ Liệu | Thang Đo | Đặc Điểm Cần Chú Ý |
| :--- | :--- | :--- | :--- |
| **BDD100K** | `{"x1", "y1", "x2", "y2"}` | Pixel thực (1280x720) | Tọa độ 2 đỉnh đối diện (top-left & bottom-right) |
| **COCO** | `[x, y, width, height]` | Pixel thực | `x, y` là góc trên bên trái, dễ nhầm với `[x1, y1, x2, y2]` |
| **YOLO** | `[center_x, center_y, w, h]` | Chuẩn hóa `[0, 1]` | Chia cho Width và Height của ảnh |
| **Pascal VOC**| `[xmin, ymin, xmax, ymax]` | 1-indexed Pixel | Bắt đầu từ 1 thay vì 0 |
| **Gemini VLM**| `[ymin, xmin, ymax, xmax]` | Chuẩn hóa `[0, 1000]` | **ymin** đứng đầu, theo chuẩn VLM Grounding của Google |

---

## 4. Thiết Kế Thí Nghiệm Đánh Giá Công Cụ (Experiment Design - Slide 46)

Để chứng minh công cụ đem lại giá trị định lượng cho Đội gán nhãn P-037, nhóm áp dụng thiết kế thử nghiệm chéo **(Crossover Design)** theo Slide 46:

### Mô hình Thử Nghiệm:
* **Người A:** Kiểm tra Bộ dữ liệu 1 (Không dùng Tool - Thủ công) $\rightarrow$ Kiểm tra Bộ dữ liệu 2 (Có dùng QC Tool).
* **Người B:** Kiểm tra Bộ dữ liệu 1 (Có dùng QC Tool) $\rightarrow$ Kiểm tra Bộ dữ liệu 2 (Không dùng Tool - Thủ công).

### 3 Chỉ Số Đo Lường Chính (Evaluation Metrics):
1. **Thời gian hoàn thành:** Mục tiêu giảm từ **20% đến 30%** tổng thời gian review cho mỗi job.
2. **Số lỗi còn sót lại (Leak Rate / False Negatives):** Đo tỷ lệ lỗi bị lọt qua sau khi hoàn tất review.
3. **Số thao tác sửa (Correction Actions):** Nhờ tính năng **1-Click Auto-Heal**, số lần click sửa tay các lỗi hình học cơ bản giảm trên **60%**.

---

## 5. Cấu Trúc Thư Mục Source-Tool

```
source-tool/qc-bdd100k/
├── backend/
│   ├── app.py                     # FastAPI REST API & HTTP Static Server (fallback built-in)
│   ├── coordinate_utils.py        # Tiện ích chuẩn hóa tọa độ COCO/YOLO/VOC/Gemini/BDD100K
│   ├── qc_engine.py               # Core Wrapper tích hợp Multi-Model Engine
│   ├── test_qc_system.py          # Test suite 8 bài test tự động kiểm thử toàn bộ hệ thống
│   ├── verify_client.py           # Script kiểm thử API & UI client integration
│   └── ai_engine/
│       ├── guideline_manager.py   # Quản lý Sổ Quyết Định & trích xuất điều khoản tự động
│       └── models_ensemble.py     # Hội đồng 5 Mô hình AI, 4 Tầng QA & Xếp hạng Ưu tiên 1->4
├── frontend/
│   ├── index.html                 # Giao diện QC Studio 2D Dashboard (Dark Modern UI)
│   ├── style.css                  # Bảng màu HSL cao cấp, badges rank-1..4, responsive
│   └── app.js                     # State management, Hàng đợi Review, Canvas, Auto-Heal, Modals
├── sample_test_labels.json        # File nhãn BDD100K mẫu kiểm thử 4 frames thực tế
├── qc_checker.py                  # CLI Terminal Checker độc lập (hỗ trợ --auto-heal, --verbose)
├── config.json                    # Cấu hình các ngưỡng kiểm tra
├── requirements.txt               # fastapi, uvicorn, python-multipart, shapely, numpy
├── run_local.bat                  # Script 1-click chạy local trên Windows
├── run_local.sh                   # Script chạy local trên Linux/macOS
├── Dockerfile                     # Docker phục vụ deploy Cloud
└── README.md
```

---

## 6. Hướng Dẫn Vận Hành & Chạy Thử Nghiệm

### 1. Chạy Bằng Giao Diện Web (Khuyên Dùng)
Nhấp đúp chuột vào file:
```
run_local.bat
```
Hoặc chạy lệnh:
```bash
python backend/app.py
```
👉 Mở trình duyệt truy cập: **`http://localhost:8000`**

### 2. Chạy Qua Dòng Lệnh CLI (`qc_checker.py`)
Kiểm tra file nhãn và in báo cáo 4 Tầng QA trực tiếp trên Terminal:
```bash
python qc_checker.py --input sample_test_labels.json --verbose
```
Tự động sửa lỗi kẹp biên và box rác rồi xuất file mới:
```bash
python qc_checker.py --input sample_test_labels.json --auto-heal --output cleaned_labels.json
```

### 3. Chạy Toàn Bộ Test Suite Kiểm Định
```bash
# Kiểm tra 8/8 bài test logic tầng nghiệp vụ
python backend/test_qc_system.py

# Kiểm tra tích hợp End-to-End API Backend
python backend/verify_client.py
```

---

## 7. Tác Giả & Đội Ngũ Phát Triển
* **Đoàn Văn Thắng** ([@vanthang10tin](https://github.com/vanthang10tin)) — Đội P-037 (Cohort 4A)
* Đề tài: **N2-04A - QC Tự Động Cho BBox & Segmentation 2D BDD100K**
* Chương trình Đào tạo AI Annotation & Quality Control 2026.
