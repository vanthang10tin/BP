# Cài đặt và sử dụng QC Studio trên máy cá nhân

Tài liệu này chạy phiên bản Blazor + Python API hiện có trong repo `BP`. Ứng dụng bind loopback, phù hợp kiểm thử trên máy cá nhân. Không chạy lệnh trong repo `P307`.

## 1. Yêu cầu

- Windows 10/11 64-bit.
- Python 3.10 trở lên. Repo này đã được kiểm tra với Python 3.11.
- .NET 10 SDK. Script tự dùng SDK portable trong `.tools/dotnet` nếu có; nếu không thì dùng SDK trong `PATH`.
- Kết nối mạng ở lần cài đầu nếu pip/NuGet cần tải dependencies.
- Muốn chạy model: Docker Desktop và CVAT/Nuclio model functions phải đang chạy, truy cập được từ backend trên các cổng đã cấu hình. Giao diện và rule QC vẫn có thể chạy khi model offline, nhưng sẽ báo rõ số frame bị bỏ qua vì inference không sẵn sàng.

> **Dữ liệu riêng tư:** Local QC vẫn là mặc định. Gemini Vision là lựa chọn riêng cho từng lần chạy; khi bật, ảnh trong batch được chọn và guideline sẽ gửi tới Google Gemini API. Không dùng Free Tier cho dữ liệu nhạy cảm hoặc bí mật.

## 2. Cài lần đầu

Mở PowerShell tại thư mục gốc `BP`:

```powershell
python --version
python -m pip install -r source-tool/qc-bdd100k/requirements.txt
```

Kiểm tra .NET:

```powershell
if (Test-Path .tools/dotnet/dotnet.exe) { .\.tools\dotnet\dotnet.exe --version } else { dotnet --version }
```

Nếu máy chặn script trong phiên hiện tại, cho phép script local trong cửa sổ PowerShell này:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Thiết lập này mất hiệu lực khi đóng cửa sổ. Không cần đổi execution policy toàn máy.

## 3. Chạy và dừng

Từ thư mục `BP`:

```powershell
.\run_qc_studio.ps1
```

Ứng dụng tự kiểm tra health, build UI Release khi cần, mở trình duyệt tại **http://127.0.0.1:5000/** và gọi API tại `http://127.0.0.1:8000`. Chỉ đưa các cổng này lên loopback; bản hiện tại chưa có đăng nhập để dùng công khai hoặc trong mạng công ty.

Chạy mà không bật trình duyệt:

```powershell
.\run_qc_studio.ps1 -NoBrowser
```

Dừng đúng hai tiến trình do script khởi chạy:

```powershell
.\stop_qc_studio.ps1
```

CVAT và Docker/model containers không bị dừng. Nhật ký và PID của QC Studio ở `source-tool/qc-bdd100k/.qc-data/runtime/`. Dữ liệu workspace/guideline lưu trong SQLite tại `source-tool/qc-bdd100k/.qc-data/qc_workspaces.sqlite3`; thư mục này bị Git ignore. Kết quả run và quyết định issue hiện nằm trong phiên review của Blazor, refresh trang sẽ mất trạng thái run/quyết định đó. Hãy export báo cáo trước khi đóng phiên.

## 4. Chạy luồng BBox bằng dữ liệu thật (50 ảnh)

Mở [test-data/bdd100k-walkthroughs/README.md](test-data/bdd100k-walkthroughs/README.md). Các pack được lấy từ BDD100K, có manifest/checksum, không phải ảnh demo tự tạo.

1. Ở thanh bên phải chọn **BBox**.
2. Tại **Dataset & labels**, chọn `annotations.json` từ `01-bbox-workflow-50`.
3. Chọn guideline dự án `.md`/`.txt`. Nếu chỉ kiểm tra mechanics, dùng guideline template trong bộ walkthrough; template chưa được phê duyệt và không chứng minh tuân thủ policy của nhóm.
4. Tại **Frame images & masks**, chọn đủ JPG trong thư mục `images/` (50 ảnh). Không bắt buộc profile cho pack BDD mặc định.
5. Nhấn **Run local QC**. Backend chạy kiểm tra cấu trúc nhãn; nếu BBox model local sẵn sàng, so sánh detector proposals với nhãn theo class/IoU.
6. Trong Review, xem frame ưu tiên bên trái; ảnh/annotation/proposal ở giữa; issue, evidence và guideline citation bên phải. Kiểm tra ảnh rồi **Confirm** hoặc **Dismiss**. Mọi khác biệt từ model là **candidate**, chưa phải kết luận nhãn sai.
7. Ở Export tải reviewer Markdown/CSV và guideline draft. Draft chỉ gom nhóm issue đã Confirm; đọc và phê duyệt thủ công trước khi nạp làm guideline active.

## 5. Chạy luồng semantic Segmentation bằng dữ liệu thật (50 ảnh)

1. Chọn **Segmentation** ở thanh bên.
2. Tải `annotations.json` từ `03-segmentation-val-workflow-50`.
3. Tải guideline phù hợp; guideline vẫn cần người phụ trách duyệt.
4. Chọn 50 ảnh trong `images/`, PNG train-ID trong `masks/index/`, và PNG màu trong `masks/color/`.
5. Nhấn **Run local QC**. Backend dùng train-ID mask làm ground truth cho phép so sánh trong run và gọi segmentation provider cục bộ nếu provider khả dụng.
6. Review overlay, class IoU và vùng mismatch. IoU là metric của run này, không chứng minh mask nguồn đúng; người review quyết định issue.
7. Export per-class CSV, reviewer Markdown và guideline draft nếu có quyết định Confirm.

Các pack còn lại: `02-bbox-rule-review-candidates-50` cố ý có nhiều rule candidate để kiểm tra queue; `04-segmentation-class-coverage-50` mở rộng coverage lớp. Candidate chưa adjudicate, không dùng làm bằng chứng dataset sai hoặc benchmark.

## 6. Mô hình và trạng thái dịch vụ

- `GET http://127.0.0.1:8000/api/health` cho biết API/model mode; `GET /api/models` liệt kê provider readiness.
- Mặc định adapter trỏ tới BBox `127.0.0.1:32003`, DINO segmentation `127.0.0.1:32005`, panoptic detector `127.0.0.1:32004`. Có thể cấu hình qua `QC_BBOX_MODEL_URL`, `QC_SEGMENTATION_MODEL_URL`, `QC_DINO_BBOX_MODEL_URL` trong process backend.
- Model endpoint remote bị chặn mặc định. Chỉ đặt `QC_ALLOW_REMOTE_MODEL_ENDPOINTS=true` sau khi duyệt bảo mật và có mạng riêng/TLS/auth tương ứng.
- Gemini tùy chọn: vào [Google AI Studio API keys](https://aistudio.google.com/app/apikey), tạo key rồi dán nguyên key (không cần thêm tên biến) vào `source-tool/qc-bdd100k/backend/GEMINI_API_KEY.txt`. File đã được tạo sẵn và Git ignore. Khởi động lại backend; `/api/models` sẽ báo Gemini đã cấu hình và checkbox **Thêm Gemini vision proposals** sẽ bật. Không dán key vào frontend, git, ảnh chụp màn hình hoặc chat. Model mặc định là `gemini-3.8-flash`.
- Khi key đã cấu hình, checkbox Gemini được chọn sẵn để Gemini tham gia pipeline cùng model local; bỏ tick trước khi Run nếu muốn local-only. Backend gửi guideline và tối đa số frame bạn chọn (mặc định 10, giới hạn 50) lên Gemini. Candidate được gắn nguồn Gemini, không tự sửa nhãn. Khi chưa có key, phần Gemini bị vô hiệu hóa.
- Google AI Studio miễn phí có thể dùng prompts/ảnh/response để cải thiện sản phẩm và cho human reviewers xử lý. Với dữ liệu nhạy cảm, cần xem điều khoản và billing project; Google nêu Paid Services không dùng prompts/ảnh/response để cải thiện sản phẩm, nhưng đây vẫn là dịch vụ cloud và không đồng nghĩa dữ liệu không bao giờ được lưu/giám sát. Hãy xác nhận chính sách tổ chức trước khi gửi ảnh.
- Health endpoint chỉ chứng minh status endpoint trả lời, không thay thế benchmark chất lượng. Cần kiểm overlay, class mapping, latency và gold set.

## 7. Nối CVAT local (tùy chọn)

CVAT của máy hiện có gateway loopback **http://127.0.0.1:8080**. Trong cùng PowerShell dùng để chạy backend, thiết lập URL và token CVAT trước khi chạy:

```powershell
$env:CVAT_URL = 'http://127.0.0.1:8080'
$env:CVAT_ACCESS_TOKEN = 'PASTE_TOKEN_ONLY_IN_THIS_TERMINAL'
.\run_qc_studio.ps1
```

Tạo Personal Access Token có quyền tối thiểu và thời hạn ngắn trong CVAT; không lưu token ở frontend hoặc commit vào Git. Nếu script phát hiện backend đang chạy, nó không khởi chạy backend thứ hai nên đổi env không tự áp dụng. Hãy dừng QC Studio, đặt env rồi chạy lại.

Trong Export, nhập CVAT Job ID và chọn **Load job metadata** để lấy mapping filename→frame index. Ưu tiên mapping chính xác; chỉ dùng offset khi đã xác nhận thứ tự. Push chỉ tạo CVAT issue đã Confirm, không sửa annotation, không đổi trạng thái job, không tự return/reassign job. Dùng disposable CVAT job trước lần ghi đầu tiên.

Thông tin gateway và lệnh dừng riêng gateway: [LOCAL_CVAT_ACCESS.md](LOCAL_CVAT_ACCESS.md).

## 8. Kiểm tra code

Các lệnh dưới đây đọc fixture nhỏ hoặc build code, không chạy model/batch ảnh:

```powershell
python source-tool/qc-bdd100k/backend/test_qc_pipeline_normalization.py
python source-tool/qc-bdd100k/backend/test_qc_system.py
$env:DOTNET_CLI_HOME = (Join-Path $PWD '.tools/dotnet-home')
$env:NUGET_PACKAGES = (Join-Path $PWD '.tools/nuget-packages')
$env:APPDATA = (Join-Path $PWD '.tools/appdata')
if (Test-Path .tools/dotnet/dotnet.exe) {
    .\.tools\dotnet\dotnet.exe restore source-tool/qc-bdd100k/frontend/BlazorApp/BlazorApp.csproj --configfile .tools/NuGet.Config
    if ($LASTEXITCODE -ne 0) { throw 'NuGet restore failed.' }
    .\.tools\dotnet\dotnet.exe build source-tool/qc-bdd100k/frontend/BlazorApp/BlazorApp.csproj --configuration Release --no-restore
} else {
    dotnet restore source-tool/qc-bdd100k/frontend/BlazorApp/BlazorApp.csproj
    if ($LASTEXITCODE -ne 0) { throw 'NuGet restore failed.' }
    dotnet build source-tool/qc-bdd100k/frontend/BlazorApp/BlazorApp.csproj --configuration Release --no-restore
}
```

Inference trên pack 50 ảnh là thao tác riêng, tiêu tốn GPU/thời gian và cần endpoint model hoạt động; không nằm trong unit test.

## 9. Sự cố thường gặp

| Triệu chứng | Kiểm tra |
|---|---|
| Trang không mở | Mở `backend.log`, `backend-error.log`, `frontend.log`, `frontend-server.log`, `frontend-error.log`; thử `http://127.0.0.1:8000/api/health`. |
| API chạy nhưng `models_mode=disabled` | Kiểm tra Docker/Nuclio model containers và cổng local 32003/32005; UI vẫn có thể chạy rule QC nhưng đừng coi model đã inference. |
| `docker ps` báo không kết nối Docker Engine | Mở Docker Desktop và đợi engine báo Running; sau đó kiểm tra lại CVAT/Nuclio containers. Script QC không tự khởi động hay dừng Docker. |
| Thiếu frame image | Chọn ảnh cùng pack; BBox còn có BDD archive fallback nếu đúng filename tồn tại. |
| Segmentation báo thiếu mask/kích thước | Chọn train-ID PNG từ `masks/index/` và color PNG từ `masks/color/` của cùng pack; không thay `poly2d` bằng semantic mask. |
| CVAT báo chưa cấu hình | Kiểm tra env của đúng backend process, token và `/api/cvat/status`; khởi động lại backend sau khi đổi env. |
| Cổng 5000/8000 đang dùng | Dừng QC Studio bằng `stop_qc_studio.ps1`; nếu PID không do script quản lý, xác định chương trình đang dùng cổng trước khi dừng thủ công. |

## Giới hạn hiện tại

Đây là local MVP, chưa có authentication/multi-user, API rate limit, resumable upload hay database cho dataset/run/reviewer decisions. Không public port hoặc dùng chung qua LAN như ứng dụng production. Xem [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) để biết kiến trúc, rủi ro và kế hoạch nâng cấp.
