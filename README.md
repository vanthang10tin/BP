# QC Studio 2D

> Công cụ hỗ trợ rà soát annotation BBox và semantic segmentation 2D cho dữ liệu giao thông. Rule findings và model disagreements đều là **candidate để reviewer xác minh**; chúng không tự chứng minh nhãn sai hoặc đạt chuẩn.

## Hồ sơ dự án

- **Mã đề tài:** N2-04A
- **Tên:** QC Studio 2D — QC tự động cho BBox và Segmentation 2D
- **Mô tả:** Nạp nhãn và ảnh, kiểm tra một số điều kiện hình học, so sánh với model khi provider/weights có sẵn, sắp xếp frame cần review và xuất báo cáo.
- **Thumbnail:** [thumbnail.jpg](thumbnail.jpg)
- **Live URL:** https://bp-uaxf.onrender.com

## Chạy nhanh trên máy local

Yêu cầu: Python 3.11 trở lên. Không cần model để dùng parser và rule checks. Model local, CVAT và Gemini là tích hợp tùy chọn.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python backend/app.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000); API docs ở `/docs`. Dùng `run_qc_studio.bat` hoặc `run_qc_studio.ps1` để khởi chạy đầy đủ cả giao diện Blazor (.NET 10) ở cổng 5000 và Backend API ở cổng 8000. Cấu hình Docker/Render đóng gói sẵn giao diện Blazor làm web UI chính.

### Chạy kiểm tra

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest backend -q
python backend/test_qc_system.py
```

Kết quả gần nhất và phạm vi kiểm tra phải được ghi trong [docs/TESTING.md](docs/TESTING.md). Không dùng test fixture làm số đo độ chính xác thực tế.

## Dữ liệu và model

- Bộ nhãn demo BDD100K nằm trong `demo_data/`; ảnh đi kèm nằm trong `demo_data/images/`. Đây là dữ liệu mẫu để chạy thử, không phải tập gold độc lập. [CSV snapshot của lần chạy 20 frame](demo_data/real_data_qc_report.csv).
- Tải dữ liệu BDD100K từ nguồn chính thức và làm theo [hướng dẫn dữ liệu](docs/DATASET_SETUP.md) nếu cần benchmark lớn hơn. Báo cáo phải ghi rõ nguồn, phiên bản, hash, lệnh và giới hạn đánh giá.
- Trọng số `.pt` không được commit vào Git. `requirements.txt` không cài PyTorch/Ultralytics; model inference chỉ hoạt động khi đã cài dependency, có weights phù hợp hoặc cấu hình provider local. Khi thiếu provider, xem số frame bị skip; không diễn giải là model đã đánh giá.
- Gemini gửi ảnh đã chọn và thông tin liên quan lên dịch vụ Google khi được bật. Không gửi dữ liệu nhạy cảm lên bản public.

## Deploy

Cấu hình `Dockerfile` và `render.yaml` đóng gói toàn diện cả Blazor Web App (.NET 10) và Python FastAPI backend trong cùng một container. Bản public bật `QC_PUBLIC_DEMO=true`: các chức năng ghi credential Gemini/CVAT bị khóa; filesystem của dịch vụ không phải nơi lưu dữ liệu bền vững. Xem [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) trước khi tạo dịch vụ.

Tích hợp Modal là tùy chọn. Hãy cấu hình `QC_MODAL_API_URL` bằng endpoint thuộc deployment Modal của bạn; repo không trỏ sẵn tới tài khoản cá nhân. Để trống biến này nếu không dùng Modal.

## Tài liệu

- [Kiến trúc](docs/ARCHITECTURE.md)
- [Đặc tả](docs/SPECIFICATION.md)
- [Kiểm thử và kết quả đo](docs/TESTING.md)
- [Thiết lập dữ liệu](docs/DATASET_SETUP.md)
- [Nhật ký tuần](weekly-logs/README.md)
- [Kịch bản demo](docs/VIDEO_DEMO_SCRIPT.md)

## Giới hạn cần nêu khi trình bày

- “Không có cờ” nghĩa là không điều kiện nào trong cấu hình hiện tại phát hiện vấn đề; không đồng nghĩa dữ liệu sạch.
- LQS là điểm heuristic nội bộ, chưa được hiệu chỉnh trên gold set.
- Chưa báo cáo precision/recall, Recall@20 hoặc phần trăm tiết kiệm thời gian nếu chưa chạy đối chiếu độc lập có adjudication.
- Tích hợp CVAT cần thử trên job test trước khi dùng job thật. Bản public chưa có đăng nhập; chỉ dùng dữ liệu demo không nhạy cảm.
