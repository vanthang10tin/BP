# Kiến trúc QC Studio 2D

## Sơ đồ triển khai chính

```mermaid
flowchart LR
  Browser -->|same-origin HTTP| FastAPI
  FastAPI --> StaticUI[frontend/ HTML CSS JS]
  FastAPI --> LegacyQC[POST /api/qc/check · BDDQualityChecker]
  FastAPI --> WorkspaceQC[POST /api/qc/run · BBox / segmentation pipeline]
  FastAPI --> Parser[Flexible dataset parser]
  WorkspaceQC --> Rules[Geometry and mask rules]
  WorkspaceQC -. optional .-> LocalModels[Loopback model provider or local YOLO weights]
  WorkspaceQC -. opt-in, sends selected images .-> Gemini[Gemini API]
  FastAPI --> SQLite[Local .qc-data workspace store]
  FastAPI -. configured local use .-> CVAT[CVAT REST API]
```

## Chạy và deploy

- `python backend/app.py` phục vụ `frontend/` tĩnh và FastAPI cùng một origin trên cổng 8000. Mặc định bind `127.0.0.1`; host/port có thể cấu hình bằng `HOST`/`PORT`.
- `run_qc_studio.ps1` / `.bat` khởi chạy backend trên 8000 và giao diện Blazor riêng trên 5000. Blazor gọi `/api/qc/run`, hỗ trợ workspace, BBox và segmentation. Giao diện tĩnh hiện gọi `/api/qc/check` và là UI mà Render phục vụ.
- `render.yaml` tạo một Python web service; không build Blazor và không có model service riêng. `QC_PUBLIC_DEMO=true` chặn ghi CVAT/Gemini credentials và vô hiệu hóa sử dụng cấu hình CVAT/Gemini trên instance public.
- Dockerfile chạy cùng backend + UI tĩnh như Render. Nó không cài model weights, CVAT/Nuclio containers hay Blazor server.

## Thành phần

- `backend/app.py`: API, phục vụ static UI, route upload/QC/export/workspace và tích hợp ngoài tùy chọn.
- `backend/qc_engine.py` + `backend/ai_engine/models_ensemble.py`: rule checks legacy BDD và hàng đợi heuristic. Các candidate cần reviewer xác nhận.
- `backend/qc_pipeline.py`: chuẩn hóa BBox và tính metrics so sánh mask theo run. Kết quả mask IoU không phải benchmark độc lập.
- `backend/ai_engine/local_model_providers.py`: provider loopback và inference YOLO local nếu đã cài dependency/weights.
- `backend/flexible_dataset_parser.py`: đọc các định dạng được hỗ trợ; giới hạn file/ZIP được đặt trong parser.
- `frontend/`: giao diện tĩnh deploy chính; `frontend/BlazorApp/`: giao diện local riêng.
- `demo_data/`: sample fixtures; không được tự xem là gold set độc lập.
- `.qc-data/`: dữ liệu runtime cục bộ. Render Free không lưu bền vững nội dung tại đây.

## Ý nghĩa số liệu

LQS là điểm heuristic. “Không có cờ” chỉ có nghĩa không rule đang bật nào phát hiện finding. Model disagreement cũng không chứng minh nhãn sai. Chỉ báo cáo precision/recall khi có tập nhãn độc lập được adjudicate và protocol matching được định trước.
