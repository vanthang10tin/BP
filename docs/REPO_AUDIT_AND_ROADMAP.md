# QC Studio — Nhật ký audit và kế hoạch sửa

Cập nhật: 2026-10-18. Snapshot khảo sát: `ff74937` cùng các thay đổi chưa commit.

**Checkpoint mới nhất:** 31 phát hiện; regression trọng tâm **30 passed (1.19s)**. Có bản sửa local, chưa deploy. A18 đã bỏ ánh xạ citation theo số mục, thêm test mục 1.1/1.2 không liên quan không được dẫn chứng. Tra cứu citation bằng từ khóa còn cần đánh giá độ chính xác ngữ nghĩa. Các dòng MỞ bên dưới là snapshot phát hiện ban đầu; xem phần “Bản sửa và kiểm thử” cùng cập nhật tiếp theo để biết tiến độ sửa. Những việc chưa xong: taxonomy thực tế, decision ledger/UI, đầy đủ luật tùy chỉnh, segmentation, CVAT/export tổng thể, deploy và chatbot/diff.

**Chưa hoàn tất audit toàn repo. Không coi danh sách này là chứng nhận hết lỗi hoặc sẵn sàng production.** Checkout có thêm thay đổi trong lúc khảo sát; phải chạy lại kiểm thử trên phiên bản cuối.

## Hợp đồng hoạt động người dùng yêu cầu

1. Nhập ảnh và nhãn từ máy hoặc CVAT.
2. Có guideline người dùng thì chỉ dùng guideline đó; không có thì dùng guideline mặc định được công bố rõ nguồn.
3. Gemini diễn giải guideline; báo chính xác khả năng và giới hạn thực tế của từng model. Không biến lớp không hỗ trợ thành lỗi nhãn.
4. Hai model đối soát. Đồng thuận mới kết luận; bất đồng chuyển Gemini. Thiếu model hoặc Gemini lỗi phải giữ trạng thái chưa phân xử, không bịa kết quả.
5. Xanh: nhãn người dùng; tím: quyết định hệ thống; đỏ: cần review. Box và issue chọn qua lại đúng frame/object. Issue ngắn, có bằng chứng.
6. Export dữ liệu hoặc gửi issue về đúng job/frame CVAT.
7. Sau khi luồng chính được kiểm chứng: trợ lý guideline nhỏ, trả lời có trích dẫn; đề xuất sửa tạo bản nháp; bước 3 hiển thị diff +/- và chỉ áp dụng sau xác nhận.

## Phát hiện đã đọc trực tiếp từ mã

Trạng thái **MỞ** là chưa sửa/chưa kiểm chứng; không đồng nghĩa mọi tình huống đều tái hiện.

| ID | Mức | Vị trí | Phát hiện / cách xử lý cần thiết | Trạng thái |
|---|---|---|---|---|
| A01 | P0 | backend/gemini_provider.py: arbitrate_dispute | API thất bại trả MODEL_CONFIRMED_ERROR, confidence 0.7 dù không có phán quyết. Phải unresolved, không có confidence giả. | MỞ |
| A02 | P1 | cùng hàm | Không có key trả confidence 0.5 và bảo lưu nhãn như kết luận. Phải phân biệt chưa phân xử với nhãn đúng. | MỞ |
| A03 | P1 | cùng hàm | bool("false") thành True; verdict tùy ý; confidence mặc định 0.9. Cần kiểm tra schema, số hữu hạn, sự nhất quán verdict. | MỞ |
| A04 | P1 | cùng hàm | Cắt guideline còn 1000 ký tự, tự gán tiêu chuẩn BDD100K khi thiếu; có thể bỏ quy tắc người dùng. | MỞ |
| A05 | P1 | cùng hàm | Danh sách model fallback hardcode, thiếu provenance model thực sự trả lời. | MỞ |
| A06 | P0 | backend/qc_pipeline.py: run_bbox_pipeline | Dùng OR cho hai nhãn xung đột rồi gọi là hội đồng đồng thuận; hai class khác nhau vẫn kết luận đổi nhãn. | MỞ |
| A07 | P1 | cùng hàm | Một model đúng, model kia không thấy thì continue, bỏ bước phân xử yêu cầu. | MỞ |
| A08 | P1 | cùng hàm | has_dual = bool(hai danh sách detection): model chạy thành công nhưng không thấy gì bị coi là vắng model. Cần metadata chạy riêng. | MỞ |
| A09 | P1 | cùng hàm | UNRESOLVED_DISPUTE bị loại khỏi findings, mất yêu cầu review. | MỞ |
| A10 | P1 | cùng hàm | match_iou được báo trong output nhưng so khớp dùng hằng 0.30/0.40. | MỞ |
| A11 | P1 | cùng hàm | Ghép vật thể bỏ sót không tiêu thụ detection đối ứng, có thể ghép nhiều-một. | MỞ |
| A12 | P1 | cùng hàm | Cắt single_candidates còn 2/frame và ngưỡng 0.55 cố định, không thông báo phần bỏ qua. | MỞ |
| A13 | P1 | cùng hàm | Nhánh đơn model không loại lớp unsupported như nhánh kép; dùng lời kết luận sai nhãn khi thiếu hội đồng. | MỞ |
| A14 | P1 | cùng hàm | model_predictions là detection thô, chưa phải ledger quyết định cuối; nhãn đúng sau Gemini không được lưu đầy đủ. | MỞ |
| A15 | P1 | deploy_vision.py, guideline_manager.py | RT-DETR được gọi DINOv3; tên kiến trúc/weights không đúng thực tế. Kiểm tra cả tên khi fallback YOLO. | MỞ |
| A16 | P1 | guideline_manager.py DEFAULT_GUIDELINES | Tự gọi guideline nội bộ là tiêu chuẩn quốc tế; cố định viền 1280x720 dù ảnh có thể khác. | MỞ |
| A17 | P1 | guideline_manager.py update_guideline | Thay nội dung nhưng không cập nhật is_default: sai nguồn guideline. | MỞ |
| A18 | P1 | guideline_manager.py find_relevant_clauses | Số mục polygon/mask/attributes không khớp nội dung mặc định; gán citation theo số mục có thể gán sai guideline người dùng. | MỞ |
| A19 | P1 | qc_pipeline.py segmentation | Tự chọn ontology BDD khi thiếu dataset; phải xác nhận hợp đồng mask ID, không suy đoán âm thầm. | MỞ |
| A20 | P0 | local_model_providers.py _modal_base_url | Thiếu cấu hình vẫn gửi ảnh đến URL Modal tài khoản cá nhân. | ĐÃ SỬA: mặc định rỗng, kiểm thử offline đạt; render.yaml vẫn cấu hình endpoint deployment hiện tại rõ ràng |
| A21 | P0 | models_ensemble.py / _base_bbox_frames | Tầng luật tự chạy model thêm lần nữa, phát lỗi ngoài quy trình hội đồng; áp ontology BDD lên class tùy chỉnh. | ĐÃ SỬA ở workspace pipeline: disable model tầng luật, dùng ontology profile; test đạt |
| A22 | P1 | models_ensemble.py | Kích thước 1280x720, luật aspect ratio/attributes/min size/mask mặc định áp lên guideline người dùng. | SỬA MỘT PHẦN: ảnh thật cấp kích thước, vô hiệu policy mặc định không được yêu cầu; còn cần mở rộng compiler hỗ trợ đủ các luật tùy chỉnh |
| A23 | P0 | qc_pipeline.py _base_bbox_frames | Exception tầng luật bị nuốt rồi trả errors=[] như sạch. | ĐÃ SỬA: trả lỗi xử lý, không tạo kết quả QC sạch; test đạt |
| A24 | P1 | app.py push_issues_to_cvat_payload | Lọc non-confirmed trước nhánh resolved khiến không PATCH issue CVAT đã resolve. | ĐÃ SỬA điều kiện lọc; cần thêm test riêng resolved + mapping sai |
| A25 | P1 | gemini_provider.py compile_guideline | Cắt 12000 ký tự, schema bắt buộc min_box_pixels dù văn bản không nêu. | ĐÃ SỬA hai điểm; validation/source verification compiler vẫn cần làm |
| A26 | P1 | gemini_provider.py align_taxonomy_with_gemini | Danh sách khả năng tĩnh, rider/traffic sign được quảng cáo hỗ trợ; heuristic chỉ xét nhóm đường bộ nên bỏ lớp COCO khác; câu “sẵn sàng” không dựa readiness. | MỞ |
| A27 | P1 | app.py QC optional Gemini inspection | Rút guideline xuống key_principles/2000 ký tự; scan thêm độc lập sau consensus có thể gây kết luận mâu thuẫn. | MỞ |
| A28 | P1 | qc_pipeline.py segmentation arbitration | IoU mask được gửi dưới tên confidence; chưa gửi đầy đủ hai mask và mask gốc để Gemini đánh giá biên. | SỬA MỘT PHẦN: đổi tên mask_iou và giữ unresolved; bằng chứng mask còn thiếu |
| A29 | P1 | deploy_vision.py + local_model_providers.py | Bộ lọc hình học ego-hood bỏ detection trong mọi dataset; tên fallback weights không được báo chính xác. | MỞ ở provider; đã bỏ bộ lọc trong qc_pipeline |
| A30 | P1 | guideline_manager.py / app.py | Có API cập nhật guideline global bên cạnh workspace; compile/status global chưa đồng bộ workspace. | MỞ |
| A31 | P1 | local_model_providers.py / deploy_vision.py | Provider metadata chưa mô tả từng lần chạy model: zero detections khác model lỗi; status Modal hardcode ready. | MỞ |

## Bản sửa và kiểm thử trong lượt audit này

- A01–A05: đã sửa arbitration không bịa kết luận/confidence, kiểm tra schema và verdict nhất quán, dùng trọn guideline, chỉ gọi model cấu hình và lưu tên model thực trả lời. Thiếu ảnh/key/guideline trả unresolved.
- A06–A07: chỉ đồng thuận sai khi hai model cùng class thay thế; các trường hợp khác chuyển phân xử. Test car/car, truck/truck, truck/bus, car/truck đạt. Còn A08 (model không có detection) cần metadata contract.
- A09: giữ issue unresolved cho bbox và segmentation; chưa kiểm chứng màu/red trên trình duyệt.
- A10: thay ngưỡng match bbox 0.30/0.40 bằng tham số match_iou; còn cần test biên ngưỡng và truyền từ guideline.
- A11: tiêu thụ detection khi ghép conflict và cặp bỏ sót để tránh tái sử dụng; 4 test consensus không còn issue bỏ sót lặp.
- A12: bỏ cắt 2 ứng viên/frame; ngưỡng 0.55 vẫn cần thay bằng policy rõ ràng. Cần bổ sung ngân sách/rate-limit cho Gemini khi có nhiều bất đồng.
- A13: bỏ lớp unsupported ở vòng nhãn nhánh đơn; cách kết luận đơn model và taxonomy vẫn còn phải sửa.
- A16–A17: bỏ tuyên bố tiêu chuẩn quốc tế, dùng kích thước ảnh thật trong guideline, cập nhật is_default khi người dùng thay guideline. Citation A18 còn mở.
- Bộ test xác nhận gần nhất: `python -m pytest backend/test_arbitration_integrity.py backend/test_guideline_honesty.py backend/test_qc_pipeline_normalization.py backend/test_cvat_push.py -q -p no:cacheprovider` → **27 passed**. Mock API, không chứng minh chất lượng vision hoặc deploy.
- `git diff --check` đạt, có cảnh báo chuyển LF/CRLF. Không sửa/revert các thay đổi có sẵn của người dùng.
- Bộ chạy mở rộng thêm segmentation/upload/flexible parser/honeypot dừng tiến độ sau `27 passed` và 3 dấu skip trên màn hình; đã yêu cầu ngắt, **không ghi nhận là pass toàn bộ**. Cần tách test có network/model khỏi regression offline trước khi chạy lại.
- Scan đường dẫn source chính chưa thấy đường dẫn Windows cá nhân bắt buộc; thấy endpoint tài khoản trong render.yaml (cấu hình triển khai). Các localhost là default có biến môi trường; chưa kiểm chứng toàn bộ script/config archive.

### Thứ tự tiếp tục cụ thể

**Cập nhật tiếp theo trong cùng lượt:**

- Provider bbox local và Modal mới trả `model_runs` với role/status; detection kèm role và tên file weights thật. API `/bbox` hỗ trợ opt-in `include_metadata`, giữ tương thích response list cho client cũ. Cần triển khai Modal mới để live có metadata này; provider cũ vẫn không xác minh được model trả zero detection.
- A08/A31 sửa ở bbox mới; test zero detection một/cả hai model đạt. Segmentation chưa có contract tương đương.
- A13: nhánh thiếu model không còn kết luận sai nhãn; phát yêu cầu review rõ thiếu kết quả hội đồng.
- A12: bỏ cả cap 2/frame và confidence 0.55 riêng; dùng ngưỡng detector đã chọn. Vẫn cần ngân sách API.
- A15/A29: bbox Modal/local trả tên weights thật, bỏ bộ lọc ego-hood ở cả provider và pipeline. Một số nhãn UI/tài liệu/segmentation còn tên cũ cần dọn.
- Status Modal chỉ xác nhận endpoint online, không tự tuyên bố model ready; compute khai báo CPU đúng cấu hình.
- Tập regression trọng tâm cập nhật: **29 passed**; `py_compile` cho deploy_vision.py, qc_pipeline.py và local_model_providers.py đạt. Chưa thực thi weights/cloud để kiểm chứng contract tích hợp.

1. Contract provider trả `model_runs` gồm tên weights thật/status/capabilities kể cả zero detections; pipeline không suy readiness từ số box.
2. Hoàn thiện decision ledger cho mọi object, gồm accepted/disputed/unresolved, bằng chứng hai model + Gemini, revision guideline. Tím chỉ vẽ final decisions, detection thô là lớp riêng.
3. Thay taxonomy suy đoán bằng giao/hiệu của capability thật và ontology guideline; Gemini chỉ diễn giải/mapping có bằng chứng.
4. Sửa citation theo nội dung thực; compile đủ guideline, validate ngưỡng và đánh dấu quy tắc chưa thực thi; không silently ignore.
5. Test CVAT resolved/dedup/frame step, export round-trip, UI hai chiều; rồi build và kiểm tra deployment.
6. Chỉ sau đó triển khai chatbot + diff như thiết kế dưới đây.

## Các mục cần tiếp tục truy vết (chưa kết luận)

- Nguồn guideline theo workspace, compile/cache và thứ tự ưu tiên; parser heuristic có đưa luật không tồn tại vào kết quả không.
- `_base_bbox_frames` và legacy ensemble có phát lỗi model trước hội đồng không.
- Bộ lọc ego-hood, min_pixels, alias class: có áp đặt luật riêng lên dataset bất kỳ không.
- Segmentation: SAM được prompt từ YOLO có thực sự là ý kiến độc lập; IoU không được trình bày thành confidence.
- CVAT: xác thực, frame offset/step/deleted frames, idempotency push, tọa độ và job, retry lỗi từng issue.
- Export: nhãn gốc/nhãn quyết định/review chưa giải quyết, provenance, format round-trip.
- UI: màu, box ↔ issue theo frame/object chính xác, trạng thái pending/error, reconnect và upload nhiều ảnh.
- Deployment: healthcheck, startup supervisor, bộ nhớ, request timeout, concurrent inference, state persistence, CORS/proxy.
- Đường dẫn tuyệt đối Windows/ổ đĩa, dữ liệu demo lẫn dữ liệu thật, secret trong repo, tải weight/phiên bản không pin.
- Frontend JS legacy và Blazor có dùng hai API/state khác nhau không.
- Tất cả README/tuyên bố tốc độ, độ chính xác, tên model: chỉ giữ số có benchmark và nguồn kiểm chứng.

## Kiểm chứng đã làm trước snapshot hiện tại

- Live https://bp-uaxf.onrender.com/: Blazor script và WebSocket kết nối; điều hướng Export và nạp demo hoạt động khi khảo sát. Chưa tái hiện được mọi nút mất tương tác.
- CSS live khác bản local tại thời điểm kiểm tra; chưa đủ bằng chứng để kết luận đây là nguyên nhân duy nhất.
- Bản Docker chẩn đoán trước đó trả HTTP 200 trong giới hạn 512 MB; không phải kiểm chứng inference thật.
- 41 tests Python passed, 3 skipped; build .NET 0 warning/error; Blazor regression và JS syntax pass **trước các thay đổi mới**, cần chạy lại.
- Upload 201 ảnh ở bản cũ gây exception không hiện rõ trên UI; không có bằng chứng rằng exception này làm chết toàn bộ circuit.
- Lần kiểm thử bổ sung yêu cầu escalation bị hệ thống automatic approval review từ chối thực thi do hết hạn mức dịch vụ duyệt. Đây không phải kết luận lệnh không an toàn. Không vượt qua chặn bằng công cụ khác.
- Chưa xác nhận deploy bản sửa; chưa chạy inference cloud trả phí hoặc push issue CVAT thật.

## Checklist hoàn tất (đều còn cần kiểm chứng phiên bản cuối)

- [ ] Audit từng module và ghi bằng chứng; sửa P0 trước P1.
- [ ] Test offline: API lỗi/malformed/missing key không sinh verdict giả.
- [ ] Test bảng chân trị hai model, zero detections, missing provider, unsupported class, một-một matching.
- [ ] Test guideline tùy chỉnh không bị trộn mặc định; dùng đủ nội dung; provenance/citation đúng.
- [ ] Test local upload và CVAT → QC → review → export/push.
- [ ] Kiểm thử UI các nút, màu, selection hai chiều và reconnect.
- [ ] Build/container và live smoke test đúng phiên bản triển khai.
- [ ] Thêm trợ lý guideline + diff có xác nhận sau khi luồng chính ổn.

## Thiết kế trợ lý guideline cần thực hiện sau core

Chat theo workspace và revision guideline. Trả lời dựa trên nội dung đang áp dụng, nêu rõ khi không tìm thấy. Chỉnh sửa chỉ tạo proposed revision cùng diff dòng thêm/xóa, không tự thay luật. Bước 3 cho chấp nhận/từ chối; chấp nhận làm mất hiệu lực compile/cache và QC trước đó, yêu cầu chạy lại hoặc đánh dấu kết quả cũ dùng revision cũ. Lưu lịch sử nguồn, thời điểm, revision và trạng thái xác nhận.
