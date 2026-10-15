# BP — QC Studio 2D

Công cụ kiểm soát chất lượng nhãn ảnh cho BDD100K, có hai mode đang hỗ trợ trong giao diện: **BBox detection** và **semantic segmentation**. Repository này chứa Blazor frontend, Python/FastAPI backend, adapter tới local CVAT/Nuclio models, CVAT issue handoff tùy chọn và các pack kiểm tra nhỏ có provenance.

## Bắt đầu trên Windows

1. Cài Python 3.10+ và .NET 10 SDK (hoặc dùng SDK portable `.tools/dotnet` có sẵn trong workspace).
2. Cài backend packages: `python -m pip install -r source-tool/qc-bdd100k/requirements.txt`.
3. Từ PowerShell tại thư mục `BP`, chạy `Set-ExecutionPolicy -Scope Process Bypass` nếu bị chặn, sau đó `./run_qc_studio.ps1`.
4. Mở **http://127.0.0.1:5000/**. API ở `http://127.0.0.1:8000`; dừng ứng dụng bằng `./stop_qc_studio.ps1`.

Hướng dẫn từng bước: [INSTALL_AND_USE.md](INSTALL_AND_USE.md). Lệnh chạy nhanh: [RUN_GUIDE.md](RUN_GUIDE.md).

## Tài liệu

- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) — kiến trúc hiện tại/mục tiêu, quyết định thiết kế, bảo mật, giới hạn và roadmap.
- [PROJECT_FLOW.md](PROJECT_FLOW.md) — luồng nghiệp vụ và trạng thái frontend/backend.
- [FLOW_REVIEW.md](FLOW_REVIEW.md) — audit flow trong slide, rủi ro và acceptance checks.
- [AI_RECOMMENDATIONS.md](AI_RECOMMENDATIONS.md) — model, Gemini/privacy, CVAT handoff và nguồn research.
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) — thay đổi, kiểm chứng và việc còn lại.
- [source-tool/qc-bdd100k/README.md](source-tool/qc-bdd100k/README.md) — chi tiết kỹ thuật ứng dụng.

## Dữ liệu walk-through

`test-data/bdd100k-walkthroughs/` có bốn pack disjoint, mỗi pack 50 frame BDD100K thật, kèm manifest/checksum: hai pack BBox và hai semantic segmentation. Để theo hướng dẫn chọn pack 01 (BBox) hoặc 03 (segmentation). Pack 02/04 bổ sung rule-candidate/class coverage. Chúng kiểm tra parsing, overlay, queue và export; chúng **không phải gold set đã adjudicate** và không chứng minh nhãn sai. Xem [walkthrough README](test-data/bdd100k-walkthroughs/README.md).

`poly2d` detection reference được tách riêng trong `test-data/bdd100k-poly2d-reference`; không dùng nó thay mask segmentation. Không cần đọc toàn bộ archive để chạy các walk-through.

## Trạng thái và giới hạn

- QC API gọi detector/mask providers cục bộ khi endpoint khả dụng. Trạng thái thay đổi theo lúc chạy; kiểm tra `/api/health` và `/api/models`. Lần xác minh gần nhất: API/UI chạy, Docker Engine không kết nối được trong phiên này, model endpoints unavailable và CVAT credentials chưa cấu hình.
- Không có Gemini/hosted VLM call trong build này. Model disagreement chỉ là candidate; chỉ reviewer mới quyết định.
- Workspace/guideline version lưu trong SQLite. Kết quả run và reviewer decisions chưa persist sau refresh.
- Đây là local MVP, chưa có authentication, multi-user isolation, resumable uploads hay hạ tầng production. Không public port 5000/8000 hoặc dùng như dịch vụ LAN/cloud.
- Dockerfile hiện chỉ đóng gói backend API, không phải toàn bộ sản phẩm Blazor + API + CVAT + models.

Kiểm tra Python logic: `python source-tool/qc-bdd100k/backend/test_qc_system.py` và `python source-tool/qc-bdd100k/backend/test_qc_pipeline_normalization.py`. Hướng dẫn build frontend và chạy model packs nằm trong [INSTALL_AND_USE.md](INSTALL_AND_USE.md).
