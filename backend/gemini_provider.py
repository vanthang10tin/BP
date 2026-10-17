"""Opt-in Gemini vision proposals. Credentials stay in the backend environment."""
import base64
import json
import os
import urllib.error
import urllib.request


KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEMINI_API_KEY.txt")


def _api_key() -> str:
    # 1. Environment variable (Render / Cloud / System)
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    # 2. Key files
    candidate_files = [
        KEY_FILE,
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GEMINI_API_KEY.txt")),
        os.path.abspath(os.path.join(os.getcwd(), "GEMINI_API_KEY.txt")),
    ]
    for kf in candidate_files:
        try:
            if os.path.exists(kf):
                with open(kf, "r", encoding="utf-8-sig") as key_file:
                    val = key_file.read().strip().strip('"').strip("'")
                    if val:
                        return val
        except OSError:
            pass
    # 3. Check .env files
    env_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")),
        os.path.abspath(os.path.join(os.getcwd(), ".env")),
    ]
    for ep in env_candidates:
        if os.path.exists(ep):
            try:
                with open(ep, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return ""


def configured() -> bool:
    return bool(_api_key())


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def inspect_frame(
    image_data: bytes,
    guideline: str,
    labels: list[dict],
    classes: list[str],
    model_predictions: list[dict] | None = None
) -> list[dict]:
    key = _api_key()
    if not key:
        raise ValueError("Gemini API key is not configured. Add GEMINI_API_KEY to the backend .env file or environment variables.")
    
    model_preds = model_predictions or []
    
    prompt = (
        "Bạn là trợ lý tạo candidate để reviewer kiểm tra dữ liệu gán nhãn 2D. Bạn không phải ground truth và không được tự quyết định nhãn đúng/sai.\n\n"
        "NHIỆM VỤ CỐT LÕI:\n"
        "Hãy phân tích ảnh camera thực tế và thực hiện ĐỐI SOÁT CHÉO 3 BÊN:\n"
        "1. [NHÃN CỦA ANNOTATOR]: Các bounding box hiện có do Annotator gán.\n"
        "2. [MODEL CANDIDATES]: Các box do provider cục bộ đã cấu hình trả về; danh sách này không chứng minh model đã được huấn luyện riêng cho BDD100K và có thể thiếu/sai.\n"
        "3. [QUY CHUẨN GUIDELINE]: Các quy tắc bắt buộc trong quy chuẩn gán nhãn của dự án.\n\n"
        "HÃY ĐẶC BIỆT CHÚ Ý VÀ THÔNG BÁO RÕ CÁC TRƯỜNG HỢP SAU:\n"
        "- MODEL_DETECTED_BUT_UNLABELED: Model của nhóm phát hiện ra vật thể rõ ràng nhưng Annotator bỏ quên không gán nhãn.\n"
        "- LABEL_CONFLICT_WITH_MODEL: Chỉ nêu bất đồng quan sát được giữa nhãn và model candidate; không kết luận bên nào đúng nếu ảnh/guideline chưa đủ căn cứ.\n"
        "- OUT_OF_MODEL_VOCAB_GUIDELINE_OBJECT: Chỉ dùng khi guideline cung cấp có điều khoản cụ thể và có bằng chứng trực quan; nếu không chắc, bỏ qua candidate.\n"
        "- GUIDELINE_VIOLATION: Chỉ nêu khi có điều khoản trong guideline được cung cấp làm căn cứ; không tự thêm ngưỡng hoặc chính sách.\n\n"
        "DỮ LIỆU ĐỐI SOÁT:\n"
        f"- Danh mục class BDD100K chuẩn: {json.dumps(classes[:50], ensure_ascii=False)}\n"
        f"- Bounding box do Annotator gán ({len(labels)} objects): {json.dumps(labels[:100], ensure_ascii=False)}\n"
        f"- Bounding box do Model cục bộ nhận diện ({len(model_preds)} detections): {json.dumps(model_preds[:100], ensure_ascii=False)}\n"
        f"- Quy chuẩn Guideline của dự án:\n{guideline[:10000]}\n\n"
        "YÊU CẦU ĐẦU RA:\n"
        "Chỉ trả về các candidate thực sự có vi phạm, bất đồng hoặc cần phân xử. Với mỗi candidate, cung cấp:\n"
        "- class_name: tên class chuẩn\n"
        "- box_2d: [ymin, xmin, ymax, xmax] thang chuẩn hóa 0..1000\n"
        "- confidence: độ tin cậy từ 0.0 đến 1.0\n"
        "- discrepancy_type: Một trong ('MODEL_DETECTED_BUT_UNLABELED', 'LABEL_CONFLICT_WITH_MODEL', 'OUT_OF_MODEL_VOCAB_GUIDELINE_OBJECT', 'GUIDELINE_VIOLATION')\n"
        "- model_status: Tóm tắt đúng model candidate được cung cấp, không suy diễn năng lực hoặc dữ liệu huấn luyện của model\n"
        "- label_status: Tóm tắt trạng thái Annotator (Ví dụ: 'Annotator gán: truck' hoặc 'Annotator bỏ sót')\n"
        "- guideline_citation: Điều khoản Guideline liên quan\n"
        "- evidence: Phân tích ngắn gọn bằng tiếng Việt cho Reviewer biết tại sao có bất đồng và cách sửa."
    )
    mime = "image/png" if image_data.startswith(b"\x89PNG") else "image/jpeg"
    body = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {
            "mime_type": mime, "data": base64.b64encode(image_data).decode("ascii")
        }}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT", "properties": {"candidates": {"type": "ARRAY", "items": {
                    "type": "OBJECT", "properties": {
                        "class_name": {"type": "STRING"},
                        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "confidence": {"type": "NUMBER"},
                        "discrepancy_type": {"type": "STRING"},
                        "model_status": {"type": "STRING"},
                        "label_status": {"type": "STRING"},
                        "guideline_citation": {"type": "STRING"},
                        "evidence": {"type": "STRING"}
                    }, "required": ["class_name", "box_2d", "confidence", "discrepancy_type", "model_status", "label_status", "evidence"]
                }}}, "required": ["candidates"]
            }
        }
    }
    primary_model = model_name()
    models_to_try = [primary_model]
    if primary_model != "gemini-2.5-flash":
        models_to_try.append("gemini-2.5-flash")
    if "gemini-1.5-flash" not in models_to_try:
        models_to_try.append("gemini-1.5-flash")

    last_error = None
    output = None
    for current_m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_m}:generateContent"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "x-goog-api-key": key
        }, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = "".join(p.get("text", "") for p in result.get("candidates", [{}])[0].get("content", {}).get("parts", []))
            output = json.loads(text)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = ValueError(f"Gemini API ({current_m}) returned HTTP {exc.code}: {detail}")
            # If 404 model not found, try fallback
            if exc.code == 404:
                continue
            raise last_error from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = ValueError(f"Gemini request failed: {exc}")
            break
        except json.JSONDecodeError as exc:
            last_error = ValueError("Gemini returned invalid structured JSON")
            break

    if output is None:
        raise last_error or ValueError("Gemini request failed without response")

    candidates = []
    for row in output.get("candidates", []):
        box = row.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box]
            confidence = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if ymax <= ymin or xmax <= xmin or confidence < 0.25:
            continue
        candidates.append({
            "category": str(row.get("class_name", "unknown")),
            "box_normalized": [xmin, ymin, xmax, ymax],
            "confidence": max(0.0, min(1.0, confidence)),
            "discrepancy_type": str(row.get("discrepancy_type", "GEMINI_ARBITRATION")),
            "model_status": str(row.get("model_status", "")),
            "label_status": str(row.get("label_status", "")),
            "guideline_citation": str(row.get("guideline_citation", "")),
            "evidence": str(row.get("evidence", ""))[:600]
        })
    return candidates




def _heuristic_compile_guideline(guideline_text: str) -> dict:
    import re
    text = guideline_text or ""

    # 1. Min box / area pixels
    min_box = None
    min_area = None
    m_box = re.search(r"(?:kích thước|size)\s*<\s*(\d+)\s*px", text, re.IGNORECASE)
    if m_box:
        try:
            min_box = int(m_box.group(1))
        except ValueError:
            pass
    m_area = re.search(r"(?:diện tích|area)\s*<\s*(\d+)\s*(?:px\^2|pixel)", text, re.IGNORECASE)
    if m_area:
        try:
            min_area = int(m_area.group(1))
        except ValueError:
            pass

    # 2. Tightness / IoU
    tightness = None
    m_tight = re.search(r"(?:tightness|độ khít)\s*>=\s*(\d+)%", text, re.IGNORECASE)
    if m_tight:
        try:
            tightness = int(m_tight.group(1)) / 100.0
        except ValueError:
            pass

    mask_iou = None
    m_miou = re.search(r"(?:mask|segmentation).*?iou\s*>=\s*0?\.?(\d+)", text, re.IGNORECASE)
    if m_miou:
        val = int(m_miou.group(1))
        mask_iou = val / 100.0 if val > 1 else float(f"0.{val}")

    # 3. Class aliases
    aliases = {
        "person": "pedestrian",
        "human": "pedestrian",
        "cyclist": "rider",
        "bicyclist": "rider",
        "motorcyclist": "rider",
        "van": "truck",
        "suv": "car",
        "pickup": "truck",
        "stop sign": "traffic sign"
    }

    # 4. Extract key principles
    principles = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("§") or re.match(r"^\[?QĐ-\d+\]?", line) or re.match(r"^\d+\.\d+", line):
            clean = re.sub(r"^(\§\d+\.\d+|\[QĐ\-\d+\]|\d+\.\d+)\.?\s*", "", line)
            if len(clean) > 10:
                principles.append(clean[:150])
    return {
        "status": "draft_only" if principles else "no_project_guideline",
        "compiler": "heuristic_engine",
        "target_classes": ["pedestrian", "rider", "car", "truck", "bus", "traffic light", "traffic sign", "motorcycle", "bicycle", "train"],
        "class_aliases": aliases,
        "class_thresholds": {},
        "geometry_constraints": {
            "min_box_pixels": min_box,
            "min_area_pixels": min_area,
            "max_bounds": None,
            "min_tightness_iou": tightness
        },
        "occlusion_policy": {},
        "segmentation_policy": {
            "min_mask_iou": mask_iou,
            "allow_self_intersection": None
        },
        "key_principles": principles[:6]
    }


def compile_guideline(guideline_text: str) -> dict:
    """Pre-compiles natural language guideline into structured, machine-executable inspection parameters."""
    key = _api_key()
    if not key:
        return _heuristic_compile_guideline(guideline_text)

    prompt = (
        "You are an AI Data Quality Orchestrator for Autonomous Driving (BDD100K/COCO datasets).\n"
        "Extract only rules and thresholds stated explicitly in the supplied annotation guideline. Do not invent policies, class lists, thresholds, or defaults. If a value is absent, omit it. This output is a draft for human review, not an approved policy.\n"
        "Extract target classes, class aliases/mapping, detector confidence thresholds, geometry constraints (min box size, min area), "
        "and 3-5 concise key principles.\n\n"
        "Guideline text:\n" + guideline_text[:12000]
    )

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "target_classes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "class_aliases": {
                        "type": "OBJECT",
                        "additionalProperties": {"type": "STRING"}
                    },
                    "class_thresholds": {
                        "type": "OBJECT",
                        "additionalProperties": {"type": "NUMBER"}
                    },
                    "min_box_pixels": {"type": "INTEGER"},
                    "min_area_pixels": {"type": "INTEGER"},
                    "min_tightness_iou": {"type": "NUMBER"},
                    "min_mask_iou": {"type": "NUMBER"},
                    "key_principles": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["target_classes", "class_aliases", "min_box_pixels", "key_principles"]
            }
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name()}:generateContent"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "x-goog-api-key": key
    }, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = "".join(p.get("text", "") for p in result.get("candidates", [{}])[0].get("content", {}).get("parts", []))
        data = json.loads(text)
        return {
            "status": "compiled",
            "compiler": f"gemini ({model_name()})",
            "target_classes": data.get("target_classes", []),
            "class_aliases": data.get("class_aliases", {}),
            "class_thresholds": data.get("class_thresholds", {}),
            "geometry_constraints": {
            "min_box_pixels": int(data["min_box_pixels"]) if data.get("min_box_pixels") is not None else None,
                "min_area_pixels": int(data["min_area_pixels"]) if data.get("min_area_pixels") is not None else None,
                "max_bounds": data.get("max_bounds"),
                "min_tightness_iou": float(data["min_tightness_iou"]) if data.get("min_tightness_iou") is not None else None
            },
            "occlusion_policy": {
                "max_tolerated_occlusion": data.get("max_tolerated_occlusion"),
                "require_occluded_attribute": data.get("require_occluded_attribute")
            },
            "segmentation_policy": {
                "min_mask_iou": float(data["min_mask_iou"]) if data.get("min_mask_iou") is not None else None,
                "allow_self_intersection": data.get("allow_self_intersection")
            },
            "key_principles": data.get("key_principles", [])
        }
    except Exception:
        res = _heuristic_compile_guideline(guideline_text)
        res["compiler"] = "heuristic_fallback (Gemini offline)"
        return res

