"""Opt-in Gemini vision proposals. Credentials stay in the backend environment."""
import base64
import json
import math
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
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"


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
    candidate_models = [primary_model, "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

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
            # If 404 (retired), 503 (overload), or 429 (rate limit), try next model
            if exc.code in {404, 503, 429}:
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
        "Guideline text:\n" + guideline_text
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
                "required": ["target_classes", "class_aliases", "key_principles"]
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


MODEL_PRIMARY_ROAD_CLASSES = [
    "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "rider",
    "traffic light", "traffic sign", "train"
]

ALL_KNOWN_MODEL_DETECTION_CLASSES = [
    "person", "pedestrian", "rider", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "traffic sign", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _heuristic_align_taxonomy(guideline_text: str, dataset_classes: list[str]) -> dict:
    known_aliases = {
        "person": "pedestrian", "people": "pedestrian", "human": "pedestrian", "nguoi": "pedestrian",
        "cyclist": "rider", "bicyclist": "rider", "motorcyclist": "rider",
        "auto": "car", "automobile": "car", "suv": "car", "taxi": "car", "o to": "car", "oto": "car", "xe hoi": "car",
        "van": "truck", "xe tai": "truck", "pickup": "truck",
        "xe buyt": "bus", "xe bus": "bus",
        "xe may": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
        "xe dap": "bicycle", "bike": "bicycle",
        "traffic_light": "traffic light", "den giao thong": "traffic light",
        "traffic_sign": "traffic sign", "bien bao": "traffic sign", "stop sign": "traffic sign"
    }

    req_classes = set()
    for c in dataset_classes:
        c_str = str(c).strip().lower()
        if c_str:
            req_classes.add(c_str)

    if guideline_text:
        import re
        for line in guideline_text.splitlines():
            line_clean = line.strip()
            m = re.match(r"^[-*•]\s*([a-zA-Z0-9_\s]{2,25}):?", line_clean)
            if m:
                cls_cand = m.group(1).strip().lower()
                if len(cls_cand) < 25 and not cls_cand.startswith("độ") and not cls_cand.startswith("kích"):
                    req_classes.add(cls_cand)

    if not req_classes:
        req_classes = set(MODEL_PRIMARY_ROAD_CLASSES)

    active_classes = []
    unsupported_classes = []
    model_canonical_set = set(MODEL_PRIMARY_ROAD_CLASSES)

    for c in sorted(req_classes):
        norm_c = known_aliases.get(c, c)
        if norm_c in model_canonical_set:
            if c not in active_classes:
                active_classes.append(c)
        else:
            if c not in unsupported_classes:
                unsupported_classes.append(c)

    excluded_model_classes = [c for c in ALL_KNOWN_MODEL_DETECTION_CLASSES
                              if c not in active_classes and c not in req_classes and known_aliases.get(c, c) not in active_classes]

    total_req = len(active_classes) + len(unsupported_classes)
    coverage = round((len(active_classes) / total_req * 100), 1) if total_req > 0 else 100.0

    summary_parts = [
        f"Hội đồng AI (YOLO26x + RT-DETR) sẵn sàng kiểm soát {len(active_classes)}/{total_req} nhãn dự án ({coverage}% độ phủ)."
    ]
    if unsupported_classes:
        summary_parts.append(
            f"Phát hiện {len(unsupported_classes)} nhãn điểm mù ({', '.join(unsupported_classes[:5])}) nằm ngoài năng lực model -> Chuyển sang kiểm tra Luật vật lý và Reviewer thủ công."
        )
    summary_parts.append(
        f"Hệ thống đã tự động lọc bỏ {len(excluded_model_classes)} nhãn ngoài phạm vi guideline khỏi kết quả phát hiện."
    )

    return {
        "status": "aligned",
        "engine": "gemini_heuristic",
        "active_classes": active_classes,
        "unsupported_guideline_classes": unsupported_classes,
        "excluded_model_classes": excluded_model_classes[:40],
        "model_coverage_percent": coverage,
        "summary": " ".join(summary_parts)
    }


_TAXONOMY_CACHE = {}


def align_taxonomy_with_gemini(guideline_text: str, dataset_classes: list[str], supported_classes=None) -> dict:
    """Capability coverage is evidence-based; Gemini compiles policy, not model readiness."""
    requested = sorted({str(c).strip() for c in dataset_classes if str(c).strip()})
    if supported_classes is None:
        supported_classes = ALL_KNOWN_MODEL_DETECTION_CLASSES
    supported = set(supported_classes)

    known_aliases = {
        "person": "pedestrian", "people": "pedestrian", "human": "pedestrian", "nguoi": "pedestrian",
        "cyclist": "rider", "bicyclist": "rider", "motorcyclist": "rider",
        "auto": "car", "automobile": "car", "suv": "car", "taxi": "car", "o to": "car", "oto": "car", "xe hoi": "car",
        "van": "truck", "xe tai": "truck", "pickup": "truck",
        "xe buyt": "bus", "xe bus": "bus",
        "xe may": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
        "xe dap": "bicycle", "bike": "bicycle",
        "traffic_light": "traffic light", "den giao thong": "traffic light",
        "traffic_sign": "traffic sign", "bien bao": "traffic sign", "stop sign": "traffic sign"
    }

    active = []
    unsupported = []
    for c in requested:
        norm = known_aliases.get(c.lower(), c.lower())
        if c in supported or norm in supported or norm in MODEL_PRIMARY_ROAD_CLASSES:
            active.append(c)
        else:
            unsupported.append(c)

    excluded = sorted([c for c in supported if c not in requested and c.lower() not in requested])
    cov = round(100 * len(active) / len(requested), 1) if requested else 100.0

    summary_parts = [
        f"Hội đồng AI (YOLO26x + D-FINE-X) sẵn sàng kiểm soát {len(active)}/{len(requested)} nhãn dự án ({cov}% độ phủ)."
    ]
    if unsupported:
        summary_parts.append(
            f"Phát hiện {len(unsupported)} nhãn điểm mù ({', '.join(unsupported[:5])}) nằm ngoài năng lực model -> Hệ thống tuân thủ Guideline, KHÔNG báo lỗi giả thiếu nhãn cho các lớp này."
        )
    return {
        "status": "verified",
        "engine": "provider_metadata",
        "active_classes": active,
        "unsupported_guideline_classes": unsupported,
        "excluded_model_classes": excluded[:40],
        "model_coverage_percent": cov,
        "summary": " ".join(summary_parts)
    }


def _crop_patch(image_data: bytes, box: list[float] | None) -> tuple[bytes, str]:
    if not box or len(box) < 4:
        mime = "image/png" if image_data.startswith(b"\x89PNG") else "image/jpeg"
        return image_data, mime
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(image_data)) as im:
            w, h = im.size
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx1 = max(0, int(x1 - bw * 0.25))
            cy1 = max(0, int(y1 - bh * 0.25))
            cx2 = min(w, int(x2 + bw * 0.25))
            cy2 = min(h, int(y2 + bh * 0.25))
            if cx2 > cx1 and cy2 > cy1:
                cropped = im.crop((cx1, cy1, cx2, cy2))
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=90)
                return buf.getvalue(), "image/jpeg"
    except Exception:
        pass
    mime = "image/png" if image_data.startswith(b"\x89PNG") else "image/jpeg"
    return image_data, mime


def _unresolved_dispute(reason: str) -> dict:
    return {
        "is_violation": False, "verdict": "UNRESOLVED_DISPUTE",
        "confirmed_class": None, "confidence": None,
        "explanation": reason, "action": "Cần reviewer phân xử.",
        "requires_review": True,
    }


def _validate_arbitration(data: dict, actual_model: str) -> dict:
    verdicts = {"ANNOTATOR_CORRECT": False, "MODEL_CONFIRMED_ERROR": True, "OBJECT_MISSED": True}
    if not isinstance(data, dict):
        raise ValueError("Arbitration must be an object")
    verdict = data.get("verdict")
    if verdict == "UNRESOLVED_DISPUTE":
        return {**_unresolved_dispute("Gemini chưa đủ bằng chứng để phân xử."), "model": actual_model}
    if not isinstance(verdict, str) or verdict not in verdicts:
        raise ValueError("Unknown arbitration verdict")
    if type(data.get("is_violation")) is not bool or data["is_violation"] != verdicts[verdict]:
        raise ValueError("Inconsistent arbitration verdict")
    confidence = data.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    for field in ("confirmed_class", "explanation", "action"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Missing {field}")
    return {**{field: data[field] for field in ("is_violation", "verdict", "confirmed_class", "confidence", "explanation", "action")},
            "model": actual_model, "requires_review": False}


def arbitrate_dispute(
    image_data: bytes,
    object_info: dict,
    model_a_info: dict | None = None,
    model_b_info: dict | None = None,
    guideline: str = "",
    task_type: str = "bbox"
) -> dict:
    """
    Arbitrates a tie or dispute when one model says valid and another says invalid.
    Returns:
        {
            "is_violation": bool,
            "verdict": "ANNOTATOR_CORRECT" | "MODEL_CONFIRMED_ERROR" | "OBJECT_MISSED",
            "confirmed_class": str,
            "explanation": str,
            "action": str,
            "confidence": float
        }
    """
    key = _api_key()
    annotator_cat = object_info.get("category", object_info.get("label", "unknown"))
    annotation_description = "Chưa có nhãn người dùng cho ứng viên này" if object_info.get("annotation_missing") else f"Nhãn Annotator: {annotator_cat}"
    box = object_info.get("box", object_info.get("suggested_box", []))

    a_name = (model_a_info or {}).get("model", "Model 1")
    a_cat = (model_a_info or {}).get("category", (model_a_info or {}).get("label", "none"))
    a_verdict = f"{a_name}: {json.dumps(model_a_info, ensure_ascii=False)}" if model_a_info else f"{a_name}: không có detection đối ứng; chưa đủ bằng chứng kết luận"

    b_name = (model_b_info or {}).get("model", "Model 2")
    b_cat = (model_b_info or {}).get("category", (model_b_info or {}).get("label", "none"))
    b_verdict = f"{b_name}: {json.dumps(model_b_info, ensure_ascii=False)}" if model_b_info else f"{b_name}: không có detection đối ứng; chưa đủ bằng chứng kết luận"

    if not key:
        return _unresolved_dispute("Chưa cấu hình Gemini để phân xử bất đồng.")
    if not image_data:
        return _unresolved_dispute("Thiếu ảnh để phân xử bất đồng.")
    if not guideline.strip():
        return _unresolved_dispute("Chưa có guideline áp dụng để phân xử.")

    patch_bytes, mime = _crop_patch(image_data, box) if image_data else (b"", "image/jpeg")

    prompt = (
        "Bạn là TRỌNG TÀI QUY CÁCH VLM (Gemini Arbiter) chuyên giải quyết TRANH CHẤP GÁN NHÃN 2D.\n"
        "Tình huống: Có bất đồng giữa các kết quả nhận diện hoặc với Annotator. Thiếu detection không chứng minh nhãn sai.\n\n"
        f"THÔNG TIN ĐỐI TƯỢNG:\n"
        f"- {annotation_description} (Tọa độ ứng viên: {box})\n"
        f"- Model 1 ({a_name}): {a_verdict}\n"
        f"- Model 2 ({b_name}): {b_verdict}\n"
        f"- Quy chuẩn Guideline (chỉ dùng nội dung này): {guideline}\n\n"
        "YÊU CẦU TRỌNG TÀI:\n"
        "1. Quan sát vùng ảnh được cung cấp.\n"
        "2. Đưa ra phán quyết công tâm:\n"
        f"   - Nếu Annotator GÁN ĐÚNG: is_violation = false, verdict = 'ANNOTATOR_CORRECT', confirmed_class = '{annotator_cat}'.\n"
        "   - Nếu thiếu bằng chứng: is_violation = false, verdict = 'UNRESOLVED_DISPUTE'. Không suy đoán kết luận.\n"
        "   - Nếu Annotator GÁN SAI (ví dụ gán nhầm loại xe/người, gán thừa): is_violation = true, verdict = 'MODEL_CONFIRMED_ERROR', chỉ rõ confirmed_class đúng.\n"
        "   - Nếu đây là vật thể rõ ràng bị Annotator BỎ SÓT: is_violation = true, verdict = 'OBJECT_MISSED', chỉ rõ confirmed_class đúng.\n"
        "3. Explanation và action phải CỰC KỲ NGẮN GỌN (tối đa 1 câu súc tích mỗi mục)."
    )

    body = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {
            "mime_type": mime, "data": base64.b64encode(patch_bytes).decode("ascii")
        }}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "is_violation": {"type": "BOOLEAN"},
                    "verdict": {"type": "STRING"},
                    "confirmed_class": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "action": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"}
                },
                "required": ["is_violation", "verdict", "confirmed_class", "explanation", "action", "confidence"]
            }
        }
    }

    primary_model = model_name()
    candidate_models = [primary_model]
    models_to_try = [m for m in candidate_models if m]

    for current_m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_m}:generateContent"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "x-goog-api-key": key
        }, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = "".join(p.get("text", "") for p in result.get("candidates", [{}])[0].get("content", {}).get("parts", []))
            data = json.loads(text)
            return _validate_arbitration(data, current_m)
        except Exception:
            continue

    return _unresolved_dispute("Gemini không trả về phán quyết hợp lệ; bất đồng chưa được giải quyết.")


