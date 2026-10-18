# Kiểm thử và kết quả chạy

## Cách chạy trên máy local

Từ thư mục gốc repository:

```powershell
python -m pip install -r requirements-dev.txt
$env:TMP = (Get-Location).Path
$env:TEMP = (Get-Location).Path
python -m pytest backend -q --tb=short
python backend/test_qc_system.py
python backend/test_real_dataset.py --limit 20
dotnet build frontend/BlazorApp/BlazorApp.csproj
```

`TMP`/`TEMP` trỏ vào workspace để các test tạo file tạm trong vùng ghi được trên một số môi trường Windows bị sandbox hóa. Python dependencies cho service nằm trong `requirements.txt`; pytest chỉ nằm trong `requirements-dev.txt`.

## Kiểm tra hai luồng deploy — 2026-10-18

- `python -m pytest backend -q`: **97 passed, 3 skipped**, 5 subtests passed, 1 cảnh báo Starlette/httpx; chạy ngoài sandbox vì TestClient và file `.env` tạm bị giới hạn trong sandbox. Ba test bị skip là tích hợp model local tùy chọn.
- `dotnet run --project tests/BlazorRegression -c Release`: pass (numeric JSON, geometry, object IDs, upload limits, recovery).
- Regression mới trong `backend/test_modal_deployment.py`: input lỗi không gọi GPU; route bbox/segment, mapping và metadata; lỗi model không trả thành công rỗng; status không giả định model đã nạp; đủ 19 lớp semantic kể cả vùng nhỏ; SAM batch 16+1; thiếu detector phụ được ghi nhận; lớp road không bị SAM báo bất đồng giả.
- Các regression endpoint dùng model giả; test semantic/SAM dùng tensor CPU và output giả. Cần cài `modal` để thu thập bộ test deploy, `torch` để chạy các test tensor; không cần tải weights.
- Smoke test endpoint Modal đang cấu hình: status, bbox và segmentation đều HTTP 200, trả list trên ảnh tổng hợp 320×240. Đây là **bản cloud trước các sửa đổi**, không xác nhận image mới build hoặc weights mới chạy đúng.
- Bản sửa chưa được deploy lại. Sau deploy, chạy lại `python test_modal_endpoint.py <url>` và kiểm tra ảnh BDD100K có nhãn; smoke test không chứng minh độ chính xác model.

## Kết quả ghi nhận trước lần sửa deploy

- Pytest: **30 passed, 3 skipped**, 1 cảnh báo deprecation từ `starlette.testclient`, 5 subtests (chạy trong ~0.85s). Bao gồm parser linh hoạt, 5 bài kiểm tra upload HTTP và giới hạn dung lượng tải lên, kiểm tra cấu hình môi trường, đối chiếu honeypot một-một, bảo vệ chế độ public demo và synthetic segmentation mask. 3 bài kiểm tra weights model local tự động skip an toàn khi chưa cấu hình `QC_RUN_MODEL_TESTS=1` hoặc thiếu file weights cục bộ.
- `python backend/test_qc_system.py`: **8/8 logic assertions passed**. Honeypot trong script là fixture tự dựng; kết quả đó không phải phép đo annotator/model hay benchmark dữ liệu thật.
- Blazor build: **pass, 0 warning/error** khi chạy `dotnet build frontend/BlazorApp/BlazorApp.csproj` trên .NET SDK.
- Python compile check: `python -m compileall -q backend` pass.

## Lần chạy dữ liệu demo BDD100K

Lệnh: `python backend/test_real_dataset.py --limit 20`. Chạy trên 20 record đầu theo thứ tự trong file `demo_data/real_val_sample_20.json`, không lấy mẫu ngẫu nhiên. Lần chạy này đọc **587 labels**, 10/20 frame không có candidate và 10/20 frame có candidate. Rule engine phát 9 finding; YOLO26 phát 6 candidate. Các loại finding: aspect ratio 8, class confusion 5, frame gần biên chưa có truncated flag 1, potential missed object 1. Counts theo loại finding không nhất thiết độc lập. CSV snapshot: [`demo_data/real_data_qc_report.csv`](../demo_data/real_data_qc_report.csv); lần chạy mới tạo file trong `.qc-data/reports/`.

Provenance của lần chạy: file nhãn SHA-256 `28e3c05c04519c813c712278ae68ea3f36f6471e78c04a833b79f006fef1c23b`; weights local `models/yolo26n.pt` SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`; Python 3.11.15, PyTorch 2.14.1+cpu, Ultralytics 8.4.177, CUDA unavailable. Output subset/CSV được ghi trong `.qc-data/reports/` (ignored khỏi Git).

Đây là số đếm **đầu ra của một lần chạy**, không phải số lỗi thật trong dữ liệu, precision, recall, accuracy, IoU benchmark hay tỷ lệ tiết kiệm thời gian. Các 20 labels đầu tiên trong validation cũng không phải independent gold cho QC findings. Chỉ sử dụng số liệu để mô tả đúng lần chạy và cấu hình trên, không suy rộng cho toàn bộ BDD100K.

## Còn cần xác minh

- Thử nghiệm public Render chưa chạy vì service/URL chưa được tạo; không có Live URL để báo cáo.
- CVAT live integration chưa được xác minh trên job thử của đội.
- Chưa có independent adjudicated gold set hoặc pilot đo thời gian review.
- Các bài test model inference/weights local (như YOLO26-Seg native) yêu cầu nạp weights và đặt cờ `QC_RUN_MODEL_TESTS=1` khi muốn chạy tích hợp đầy đủ.
