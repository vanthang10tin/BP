"""Dataset normalization and local model-to-annotation comparisons."""
from __future__ import annotations

import base64
import concurrent.futures
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

from ai_engine.local_model_providers import infer_bbox, infer_segmentation, ProviderError
from coordinate_utils import bdd_to_xyxy
from ai_engine.models_ensemble import compute_iou
from gemini_provider import arbitrate_dispute, _unresolved_dispute, configured as gemini_configured


def _normal(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _unresolved_finding(name, object_id, category, box, verdict):
    return {"issue_id": f"{name}:{object_id}:UNRESOLVED_DISPUTE", "object_id": object_id,
            "category": category, "severity": "MEDIUM", "error_type": "UNRESOLVED_DISPUTE",
            "qa_layer": "Cần phân xử", "component": "Trọng tài", "model_name": verdict.get("model", ""),
            "models_flagged": [], "explanation": verdict.get("explanation", "Bất đồng chưa được phân xử."),
            "action": "Cần reviewer kiểm tra.", "suggested_box": box, "review_status": "pending",
            "auto_healable": False, "requires_review": True, "model_confidence": None}


def _ontology(profile: dict[str, Any], fallback: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    classes = profile.get("classes", fallback or [])
    names, alias_map = [], {}
    if isinstance(classes, dict):
        classes = [{"name": k, "aliases": v if isinstance(v, list) else []} for k, v in classes.items()]
    for entry in classes:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        canonical = str(entry["name"]).strip()
        names.append(canonical)
        for alias in [canonical, *entry.get("aliases", []), *entry.get("model_aliases", [])]:
            alias_map[_normal(alias)] = canonical
    norm_names = {_normal(n): n for n in names}
    domain_aliases = {
        "person": "pedestrian", "nguoi": "pedestrian", "people": "pedestrian", "human": "pedestrian",
        "traffic light": "traffic_light", "den giao thong": "traffic_light", "trafficlight": "traffic_light",
        "traffic sign": "traffic_sign", "bien bao": "traffic_sign", "trafficsign": "traffic_sign", "stop sign": "traffic_sign",
        "xanh sm": "car", "xanh_sm": "car", "xanh-sm": "car", "taxi": "car", "cab": "car",
        "automobile": "car", "auto": "car", "o to": "car", "oto": "car", "xe hoi": "car", "vehicle": "car", "sedan": "car", "suv": "car",
        "motorbike": "motorcycle", "xe may": "motorcycle", "scooter": "motorcycle", "moped": "motorcycle",
        "bike": "bicycle", "xe dap": "bicycle", "cyclist": "bicycle",
        "van": "truck", "xe tai": "truck", "xe buyt": "bus", "xe bus": "bus", "pickup": "truck", "pickup truck": "truck"
    }
    for alias, target in domain_aliases.items():
        if _normal(target) in norm_names and _normal(alias) not in alias_map:
            alias_map[_normal(alias)] = norm_names[_normal(target)]
        elif _normal(alias) in norm_names and _normal(target) not in alias_map:
            alias_map[_normal(target)] = norm_names[_normal(alias)]
    return names, alias_map


def compute_intersection_area(b1: tuple[float, float, float, float] | list[float],
                              b2: tuple[float, float, float, float] | list[float]) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_box_area(b: tuple[float, float, float, float] | list[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def is_center_inside(b_inner: tuple[float, float, float, float] | list[float],
                     b_outer: tuple[float, float, float, float] | list[float]) -> bool:
    cx = 0.5 * (float(b_inner[0]) + float(b_inner[2]))
    cy = 0.5 * (float(b_inner[1]) + float(b_inner[3]))
    return float(b_outer[0]) <= cx <= float(b_outer[2]) and float(b_outer[1]) <= cy <= float(b_outer[3])


def is_spatial_match(b1: tuple[float, float, float, float] | list[float],
                     b2: tuple[float, float, float, float] | list[float],
                     match_iou: float = 0.35) -> tuple[bool, float]:
    iou = compute_iou(tuple(b1), tuple(b2))
    # Containment alone is not a match: a road-sized label can enclose
    # several cars while having almost no overlap with any one object.
    return bool(iou >= float(match_iou)), float(iou)



def is_same_category(cat1: Any, cat2: Any, aliases: dict[str, str] | None = None) -> bool:
    c1 = _normal(cat1)
    c2 = _normal(cat2)
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True
    if aliases:
        a1 = _normal(aliases.get(c1, c1))
        a2 = _normal(aliases.get(c2, c2))
        if a1 == a2:
            return True
        c1, c2 = a1, a2

    synonym_groups = [
        {"car", "vehicle", "automobile", "auto", "sedan", "suv", "taxi", "cab", "o to", "oto", "xe hoi", "xanh sm", "xanh_sm", "xanh-sm"},
        {"truck", "van", "pickup", "pickup truck", "xe tai", "lorry"},
        {"bus", "xe buyt", "xe bus", "coach", "minibus"},
        {"motorcycle", "motorbike", "scooter", "moped", "xe may"},
        {"bicycle", "bike", "cyclist", "xe dap"},
        {"pedestrian", "person", "people", "human", "walker", "nguoi"},
        {"traffic light", "traffic_light", "trafficlight", "den giao thong", "signal"},
        {"traffic sign", "traffic_sign", "trafficsign", "stop sign", "bien bao", "sign"},
    ]
    for group in synonym_groups:
        if c1 in group and c2 in group:
            return True

    if len(c1) >= 4 and len(c2) >= 4 and (c1 in c2 or c2 in c1):
        return True

    return False


def normalize_bbox_dataset(data: Any, annotation_format: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    if annotation_format == "coco":
        if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
            raise ValueError("COCO needs images, annotations, and categories arrays")
        categories = {c["id"]: c["name"] for c in data.get("categories", []) if isinstance(c, dict) and "id" in c and "name" in c}
        by_image = {im["id"]: {"name": Path(str(im.get("file_name", im["id"]))).name, "width": im.get("width"), "height": im.get("height"), "labels": []}
                    for im in data["images"] if isinstance(im, dict) and "id" in im}
        for annotation in data["annotations"]:
            if not isinstance(annotation, dict) or annotation.get("image_id") not in by_image:
                continue
            box = annotation.get("bbox")
            category = categories.get(annotation.get("category_id"))
            if not isinstance(box, list) or len(box) < 4 or not category:
                continue
            x, y, w, h = (float(v) for v in box[:4])
            by_image[annotation["image_id"]]["labels"].append({"id": str(annotation.get("id", "")), "category": category, "box2d": {"x1": x, "y1": y, "x2": x + w, "y2": y + h}})
        return list(by_image.values())

    if annotation_format == "yolo_json":
        if not isinstance(data, dict) or not isinstance(data.get("images"), list):
            raise ValueError("YOLO JSON must provide images with normalized cx,cy,w,h annotations")
        cats = {int(c["id"]): c["name"] for c in data.get("categories", []) if isinstance(c, dict) and "id" in c and "name" in c}
        frames = []
        for im in data["images"]:
            width, height = int(im.get("width", 0)), int(im.get("height", 0))
            if width <= 0 or height <= 0:
                raise ValueError("YOLO JSON images must include width and height")
            labels = []
            for obj in im.get("annotations", []):
                cid = int(obj["class_id"]); cx, cy, w, h = map(float, obj["bbox"])
                labels.append({"id": str(obj.get("id", "")), "category": cats.get(cid, str(cid)), "box2d": {
                    "x1": (cx - w / 2) * width, "y1": (cy - h / 2) * height,
                    "x2": (cx + w / 2) * width, "y2": (cy + h / 2) * height}})
            frames.append({"name": Path(str(im["name"])).name, "width": width, "height": height, "labels": labels})
        return frames

    VALID_BBOX_FORMATS = {"bdd100k", "normalized", "voc", "cvat_xml", "yolo_txt", "csv", "zip", "custom", "flexible", "pascal_voc_xml"}
    if annotation_format not in VALID_BBOX_FORMATS:
        annotation_format = "normalized"
    frames = data if isinstance(data, list) else data.get("frames", data.get("data", [data])) if isinstance(data, dict) else None
    if not isinstance(frames, list):
        raise ValueError("Expected a BDD100K/normalized frame array")
    normalized = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {index} is not an object")
        raw_name = frame.get("name", frame.get("frame_name", f"frame_{index:04d}.jpg"))
        if ".." in str(raw_name) or str(raw_name).startswith("/") or str(raw_name).startswith("\\"):
            raise ValueError(f"Unsafe frame name: {raw_name}")
        name = Path(str(raw_name)).name
        labels = []
        for object_index, obj in enumerate(frame.get("labels", [])):
            if not isinstance(obj, dict):
                continue
            category = obj.get("category", obj.get("label", ""))
            box = obj.get("box2d", obj.get("bbox"))
            if isinstance(box, (list, tuple)) and len(box) == 4:
                box = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
            if not isinstance(box, dict) or not category:
                continue
            labels.append({**obj, "id": str(obj.get("id", object_index)), "category": str(category), "box2d": box})
        normalized.append({**frame, "name": name, "labels": labels})
    return normalized


def _image_bytes(name: str, supplied: dict[str, str]) -> bytes | None:
    encoded = supplied.get(name)
    if encoded:
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image for {name}") from exc
    # The uploaded image belongs to this run; a same-named demo image is only a fallback.
    try:
        from app import find_image_file_path
        path = find_image_file_path(name)
        if path and os.path.exists(path):
            return Path(path).read_bytes()
    except Exception:
        pass
    return None


def _rank(frame: dict[str, Any]) -> None:
    errors = frame.get("errors", [])
    counts = {"missing_object": 0, "class_confusion": 0, "boundary_error": 0}

    for err in errors:
        etype = str(err.get("error_type", "")).upper()
        # Direction 1: ObjectLab Triad Classification (NeurIPS 2023)
        if "UNLABELED" in etype or "MISSING" in etype or "NOT_LABELED" in etype:
            olab = "MISSING_OBJECT"
            counts["missing_object"] += 1
        elif "CLASS" in etype or "CATEGORY" in etype or "MISMATCH" in etype or "CONFUSION" in etype:
            olab = "CLASS_CONFUSION"
            counts["class_confusion"] += 1
        else:
            olab = "BOUNDARY_ERROR"
            counts["boundary_error"] += 1
        err["objectlab_category"] = olab

    # Direction 2: Label Quality Score (LQS) [0.0 - 100.0]
    # Penalties: Missing Object = -25.0, Class Confusion = -15.0, Boundary/Geometry = -8.0
    penalty = (counts["missing_object"] * 25.0) + (counts["class_confusion"] * 15.0) + (counts["boundary_error"] * 8.0)
    lqs = max(0.0, round(100.0 - penalty, 1))

    if lqs < 50.0:
        rank, label, status = 1, "CRITICAL REVIEW", "CRITICAL_REVIEW"
    elif lqs < 80.0:
        rank, label, status = 2, "HIGH REVIEW", "REVIEW_RECOMMENDED"
    elif errors:
        rank, label, status = 3, "REVIEW CANDIDATE", "MINOR_ISSUES"
    else:
        rank, label, status = 4, "NO FLAGS FROM ENABLED CHECKS", "CLEAN_CANDIDATE"

    frame.update(
        lqs=lqs,
        lqs_status=status,
        objectlab_counts=counts,
        priority_rank=rank,
        priority_label=label,
        priority_level=label,
        error_count=len(errors),
        recommendation=f"LQS heuristic: {lqs:.1f}/100.0 - {'Ưu tiên review cao' if lqs < 50 else 'Khuyến nghị review' if lqs < 80 or errors else 'Không có cờ từ các kiểm tra đang bật; chưa xác minh nhãn'}"
    )


def _base_bbox_frames(frames: list[dict[str, Any]], profile: dict[str, Any], guideline_manager: Any) -> list[dict[str, Any]]:
    # Layer 1: Luôn chạy bộ kiểm tra luật hình học & vật lý chuẩn quốc tế (BDDQualityChecker)
    from qc_engine import BDDQualityChecker
    try:
        compiled = guideline_manager.get_compiled_rules() if guideline_manager else {}
        geometry = compiled.get("geometry_constraints", {})
        classes, _ = _ontology(profile, sorted({l.get("category", "") for f in frames for l in f.get("labels", [])}))
        config = {
            "enable_model_checks": False,
            "enable_aspect_heuristic": False,
            "allowed_classes": [c.lower() for c in classes],
            "min_box_width": geometry.get("min_box_pixels") or 2.0,
            "min_box_height": geometry.get("min_box_pixels") or 2.0,
            "min_box_area": geometry.get("min_area_pixels") or 4.0,
            "mandatory_attributes": compiled.get("mandatory_attributes", []),
            "require_truncated": compiled.get("require_truncated", False),
            "duplicate_box_iou_threshold": compiled.get("duplicate_box_iou_threshold", 0.95),
            "mask_box_min_iou": 0,
            "max_mask_box_drift_px": float("inf"),
        }
        if guideline_manager and not guideline_manager.is_default:
            config.update(min_box_width=geometry.get("min_box_pixels") or 0,
                          min_box_height=geometry.get("min_box_pixels") or 0,
                          min_box_area=geometry.get("min_area_pixels") or 0,
                          mandatory_attributes=compiled.get("mandatory_attributes", []),
                          require_truncated=compiled.get("require_truncated", False),
                          duplicate_box_iou_threshold=compiled.get("duplicate_box_iou_threshold", float("inf")),
                          mask_box_min_iou=0, max_mask_box_drift_px=float("inf"))
        results = BDDQualityChecker(config, guideline_manager=guideline_manager).check_dataset_advanced(frames)["sorted_frames"]
        for output in results:
            source = frames[output["frame_index"]]
            output["width"] = int(source.get("width", profile.get("image_width", 1280)))
            output["height"] = int(source.get("height", profile.get("image_height", 720)))
        return results
    except Exception as exc:
        raise ValueError("Rule evaluation failed; no verified QC result was produced") from exc



def run_bbox_pipeline(data: Any, images: dict[str, str], profile: dict[str, Any], guideline_manager: Any,
                      threshold: float = 0.35, match_iou: float = 0.5) -> dict[str, Any]:
    def arbitrate(*args, **kwargs):
        if profile.get("use_gemini") is False or not gemini_configured():
            return _unresolved_dispute("Gemini chưa được kích hoạt hoặc chưa cấu hình API key.")
        return arbitrate_dispute(*args, **kwargs)

    profile = {"annotation_format": "bdd100k", "dataset": "bdd100k", **profile}
    frames = normalize_bbox_dataset(data, profile["annotation_format"], profile)
    if not frames:
        raise ValueError("No frames in dataset")
    class_names, aliases = _ontology(profile, sorted({str(l.get("category", "")) for f in frames for l in f.get("labels", []) if l.get("category")}))
    if not class_names:
        raise ValueError("No class ontology found; supply classes in the workspace profile")

    # Direction 3: Inject pre-compiled guideline rules (dynamic thresholds, aliases, constraints)
    compiled = getattr(guideline_manager, "get_compiled_rules", lambda: {})() if guideline_manager else {}
    if compiled and isinstance(compiled, dict):
        for raw_alias, target in compiled.get("class_aliases", {}).items():
            if target in class_names and _normal(raw_alias) not in aliases:
                aliases[_normal(raw_alias)] = target
        class_thresh_map = compiled.get("class_thresholds", {})
        min_pixels = compiled.get("geometry_constraints", {}).get("min_box_pixels")
        min_pixels = 0 if min_pixels is None else min_pixels
    else:
        class_thresh_map = {}
        min_pixels = 0

    guideline_txt = getattr(guideline_manager, "get_guideline_text", lambda: "")() if guideline_manager else ""
    if not guideline_txt or not guideline_txt.strip():
        from ai_engine.guideline_manager import DEFAULT_GUIDELINES
        guideline_txt = DEFAULT_GUIDELINES

    # Đối soát danh mục Taxonomy với Gemini / Heuristic
    from gemini_provider import align_taxonomy_with_gemini
    taxonomy_alignment = align_taxonomy_with_gemini(guideline_txt, class_names)
    active_classes = set(taxonomy_alignment.get("active_classes", class_names))
    unsupported_classes = set(taxonomy_alignment.get("unsupported_guideline_classes", []))
    excluded_model_classes = set(taxonomy_alignment.get("excluded_model_classes", []))

    # Tầng 1: Đánh giá các quy chuẩn hình học, biên, và thuộc tính vật lý
    for frame in frames:
        image_bytes = _image_bytes(frame["name"], images)
        if image_bytes:
            try:
                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    frame["width"], frame["height"] = source_image.size
            except (OSError, ValueError):
                pass
    evaluated = _base_bbox_frames(frames, profile, guideline_manager)
    source_by_name = {f["name"]: f for f in frames}
    provider_meta = None
    model_skips = []
    n_model_issues = 0
    min_conf = float(profile.get("model_threshold", threshold))
    preferred_bbox_model = profile.get("preferred_model") or profile.get("model")

    def _process_bbox_frame(result: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
        name = result["frame_name"]
        frame_unsupported = set(unsupported_classes)
        image_data = _image_bytes(name, images)
        if image_data is None:
            return None, {"frame_name": name, "reason": "image_not_uploaded_or_found_locally"}, 0
        try:
            with Image.open(io.BytesIO(image_data)) as im:
                width, height = im.size
            predictions, p_meta = infer_bbox(image_data, min_conf, class_names, profile.get("model_label_mapping", {}), preferred_model=preferred_bbox_model)
            runs = p_meta.get("model_runs", [])
            vocabularies = [set(aliases.get(_normal(c), c) for c in run["supported_classes"])
                            for run in runs if run.get("status") == "success" and isinstance(run.get("supported_classes"), list)]
            if vocabularies:
                # Use capabilities from every successful provider. A single
                # surviving detector still has a known vocabulary; classes it
                # cannot predict must not be treated as failed annotations.
                supported = sorted(set.intersection(*vocabularies))
                result["taxonomy_alignment"] = align_taxonomy_with_gemini(guideline_txt, class_names, supported)
                frame_unsupported = set(result["taxonomy_alignment"]["unsupported_guideline_classes"])
        except (ProviderError, OSError, ValueError) as exc:
            return None, {"frame_name": name, "reason": str(exc)[:400]}, 0

        labels = result.get("labels", [])
        gt_objects = []
        for label in labels:
            box = label.get("box2d")
            if not isinstance(box, dict):
                continue
            gt_objects.append({"id": str(label.get("id", "")), "category": aliases.get(_normal(label.get("category")), str(label.get("category", ""))),
                               "box": [float(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2")]})
        model_boxes = []
        for prediction in predictions:
            if prediction.get("type") != "rectangle" or len(prediction.get("points", [])) < 4:
                continue
            category_raw = str(prediction.get("label", ""))
            category = aliases.get(_normal(category_raw), category_raw)
            try:
                box = [float(x) for x in prediction["points"][:4]]
                conf = float(prediction.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            bw = abs(box[2] - box[0])
            bh = abs(box[3] - box[1])
            target_thresh = class_thresh_map.get(category, min_conf)
            if conf < target_thresh or bw < min_pixels or bh < min_pixels:
                continue
            # Bỏ qua các class bị loại trừ bởi Taxonomy Alignment (model có nhưng guideline/dự án không cần)
            if category in excluded_model_classes:
                continue
            overlaps_gt = any(compute_iou(tuple(box), tuple(obj["box"])) >= match_iou for obj in gt_objects)
            if category not in class_names and not overlaps_gt:
                continue
            m_source = prediction.get("model") or (p_meta.get("name") if p_meta else "local-detector")
            model_boxes.append({"category": category, "model_label": category_raw, "box": box, "confidence": conf,
                                "model": m_source, "model_role": prediction.get("model_role"), "type": "rectangle"})

        # Separate predictions from YOLO26 and the second detector (D-FINE / DETR).
        def second_detector(m):
            return m.get("model_role") == "detector_b" or any(tag in str(m.get("model", "")).upper() for tag in ("DFINE", "D-FINE", "DINO", "RTDETR", "RT-DETR"))
        yolo_boxes = [m for m in model_boxes if not second_detector(m)]
        dino_boxes = [m for m in model_boxes if second_detector(m)]
        model_runs = (p_meta or {}).get("model_runs", [])
        successful_roles = {r.get("role") for r in model_runs if r.get("status") == "success"}
        has_dual = {"detector_a", "detector_b"}.issubset(successful_roles) if model_runs else bool(yolo_boxes and dino_boxes)
        result["model_runs"] = model_runs
        findings = []
        use_gemini = profile.get("use_gemini") is True and gemini_configured()
        unresolved_disputes = []
        if has_dual:
            # =========================================================================
            # BƯỚC 1: KẾT HỢP 2 MODEL (YOLO26x + D-FINE-X) -> DUY NHẤT 1 TẬP "BOX CỦA AI"
            # Quy tắc người dùng: "những cái nào chung thì lấy, riêng cũng lấy cho nó đủ thừa còn hơn thiếu"
            # =========================================================================
            ai_boxes = []
            unmatched_yolo = list(range(len(yolo_boxes)))
            unmatched_dino = list(range(len(dino_boxes)))

            # 1.1 Tìm các cặp 2 model nhận diện trùng vị trí (IoU >= 0.30 hoặc spatial match)
            # A. Trường hợp 2 model CÙNG CLASS (hoặc cùng nhóm synonym) -> Đồng thuận tuyệt đối
            for yi in list(unmatched_yolo):
                yb = yolo_boxes[yi]
                d_same = []
                for di in unmatched_dino:
                    db = dino_boxes[di]
                    same_cat = is_same_category(yb["category"], db["category"], aliases)
                    m_ok, m_score = is_spatial_match(yb["box"], db["box"], 0.30)
                    if same_cat and m_ok:
                        d_same.append((m_score, di))
                best_iou, best_di = max(d_same, default=(0.0, -1))
                if best_di >= 0:
                    unmatched_yolo.remove(yi)
                    unmatched_dino.remove(best_di)
                    db = dino_boxes[best_di]
                    fused_box = [round(0.5 * (yb["box"][k] + db["box"][k]), 2) for k in range(4)]
                    ai_boxes.append({
                        "id": f"SYS_AI_{len(ai_boxes)}",
                        "category": yb["category"],
                        "model_label": yb.get("model_label", yb["category"]),
                        "box": fused_box,
                        "confidence": round(max(yb.get("confidence", 0), db.get("confidence", 0)), 3),
                        "model": "Hội đồng AI (Đồng thuận YOLO26x + D-FINE-X)",
                        "model_role": "consensus",
                        "type": "rectangle"
                    })

            # B. Trường hợp 2 model KHÁC CLASS tại cùng vị trí
            for yi in list(unmatched_yolo):
                yb = yolo_boxes[yi]
                d_diff = []
                for di in unmatched_dino:
                    db = dino_boxes[di]
                    same_cat = is_same_category(yb["category"], db["category"], aliases)
                    m_ok, m_score = is_spatial_match(yb["box"], db["box"], 0.30)
                    if not same_cat and m_ok:
                        d_diff.append((m_score, di))
                best_iou, best_di = max(d_diff, default=(0.0, -1))
                if best_di >= 0:
                    unmatched_yolo.remove(yi)
                    unmatched_dino.remove(best_di)
                    db = dino_boxes[best_di]
                    fused_box = [round(0.5 * (yb["box"][k] + db["box"][k]), 2) for k in range(4)]
                    if use_gemini:
                        arb = arbitrate(image_data, {"category": yb["category"], "box": fused_box},
                                        model_a_info=yb, model_b_info=db, guideline=guideline_txt)
                        if arb.get("verdict") == "UNRESOLVED_DISPUTE":
                            unresolved_disputes.append({"box": fused_box, "cand": yb, "dispute": arb})
                        else:
                            conf_cls = arb.get("confirmed_class") or (yb["category"] if yb.get("confidence", 0) >= db.get("confidence", 0) else db["category"])
                            ai_boxes.append({
                                "id": f"SYS_AI_{len(ai_boxes)}",
                                "category": conf_cls,
                                "model_label": conf_cls,
                                "box": fused_box,
                                "confidence": float(arb.get("confidence") or max(yb.get("confidence", 0), db.get("confidence", 0))),
                                "model": f"Hội đồng AI (Gemini 3.8 Flash phân xử: {conf_cls})",
                                "model_role": "arbitrated",
                                "type": "rectangle"
                            })
                    else:
                        chosen_cat = yb["category"] if yb.get("confidence", 0) >= db.get("confidence", 0) else db["category"]
                        chosen_conf = round(max(yb.get("confidence", 0), db.get("confidence", 0)), 3)
                        ai_boxes.append({
                            "id": f"SYS_AI_{len(ai_boxes)}",
                            "category": chosen_cat,
                            "model_label": chosen_cat,
                            "box": fused_box,
                            "confidence": chosen_conf,
                            "model": f"Hội đồng AI (YOLO26x + D-FINE-X: {chosen_cat})",
                            "model_role": "consensus_diff_class",
                            "type": "rectangle"
                        })

            # 1.2 Xử lý các detection riêng của từng model
            if use_gemini:
                single_cands = [yolo_boxes[i] for i in unmatched_yolo] + [dino_boxes[i] for i in unmatched_dino]
                single_cands = [c for c in single_cands if c.get("confidence", 0) >= min_conf and c["category"] in class_names and c["category"] not in frame_unsupported]
                for cand in single_cands:
                    cand_gt = [obj for obj in gt_objects if compute_iou(tuple(cand["box"]), tuple(obj["box"])) >= match_iou]
                    has_annotator_box = len(cand_gt) > 0
                    obj_query = {"category": cand_gt[0]["category"], "box": cand["box"]} if has_annotator_box else {"category": cand["category"], "box": cand["box"], "annotation_missing": True}
                    arb = arbitrate(image_data, obj_query,
                                    model_a_info=cand if not second_detector(cand) else None,
                                    model_b_info=cand if second_detector(cand) else None,
                                    guideline=guideline_txt)
                    if arb.get("verdict") == "UNRESOLVED_DISPUTE":
                        unresolved_disputes.append({"box": cand["box"], "cand": cand, "dispute": arb})
                    elif arb.get("verdict") in ("OBJECT_MISSED", "MODEL_CONFIRMED_ERROR") or (has_annotator_box and arb.get("verdict") == "ANNOTATOR_CORRECT"):
                        conf_cls = arb.get("confirmed_class", cand["category"])
                        ai_boxes.append({
                            "id": f"SYS_AI_{len(ai_boxes)}",
                            "category": conf_cls,
                            "model_label": cand.get("model_label", conf_cls),
                            "box": cand["box"],
                            "confidence": float(arb.get("confidence") or cand.get("confidence") or 0.85),
                            "model": "Hội đồng AI (Gemini Flash giữ lại)",
                            "model_role": "arbitrated",
                            "type": "rectangle"
                        })
                for obj in gt_objects:
                    has_det_overlap = any(compute_iou(tuple(obj["box"]), tuple(m["box"])) >= match_iou for m in model_boxes)
                    if not has_det_overlap and obj["category"] not in frame_unsupported:
                        arb = arbitrate(image_data, obj, None, None, guideline=guideline_txt)
                        if arb.get("verdict") == "UNRESOLVED_DISPUTE":
                            unresolved_disputes.append({"box": obj["box"], "cand": obj, "dispute": arb})
                        elif not arb.get("is_violation") and arb.get("verdict") == "ANNOTATOR_CORRECT":
                            ai_boxes.append({
                                "id": f"SYS_AI_{len(ai_boxes)}",
                                "category": obj["category"],
                                "model_label": obj["category"],
                                "box": obj["box"],
                                "confidence": float(arb.get("confidence") or 0.85),
                                "model": "Hội đồng AI (Gemini 3.8 Flash xác nhận)",
                                "model_role": "arbitrated",
                                "type": "rectangle"
                            })
            else:
                # "Riêng cũng lấy cho nó đủ thừa còn hơn thiếu"
                for yi in unmatched_yolo:
                    yb = yolo_boxes[yi]
                    if yb["category"] not in frame_unsupported:
                        ai_boxes.append({
                            "id": f"SYS_AI_{len(ai_boxes)}",
                            "category": yb["category"],
                            "model_label": yb.get("model_label", yb["category"]),
                            "box": yb["box"],
                            "confidence": round(float(yb.get("confidence", 0.8)), 3),
                            "model": yb.get("model", "YOLO26x"),
                            "model_role": "single_yolo",
                            "type": "rectangle"
                        })
                for di in unmatched_dino:
                    db = dino_boxes[di]
                    if db["category"] not in frame_unsupported:
                        ai_boxes.append({
                            "id": f"SYS_AI_{len(ai_boxes)}",
                            "category": db["category"],
                            "model_label": db.get("model_label", db["category"]),
                            "box": db["box"],
                            "confidence": round(float(db.get("confidence", 0.8)), 3),
                            "model": db.get("model", "D-FINE-X"),
                            "model_role": "single_dfine",
                            "type": "rectangle"
                        })

            unified_system_predictions = ai_boxes
        else:
            # Single model fallback: lấy toàn bộ model_boxes vào ai_boxes
            ai_boxes = []
            for m in model_boxes:
                if m["category"] not in frame_unsupported:
                    ai_boxes.append({
                        "id": f"SYS_AI_{len(ai_boxes)}",
                        "category": m["category"],
                        "model_label": m.get("model_label", m["category"]),
                        "box": m["box"],
                        "confidence": round(float(m.get("confidence", 0.8)), 3),
                        "model": m.get("model", "Local-Detector"),
                        "model_role": "single_detector",
                        "type": "rectangle"
                    })
            unified_system_predictions = ai_boxes

        # =========================================================================
        # BƯỚC 2: "BOX CỦA NGƯỜI" (gt_objects) ĐỂ NGUYÊN
        # =========================================================================

        # =========================================================================
        # BƯỚC 3: "BOX CẦN REVIEW" LÀ CÁI KHÁC NHAU GIỮA CỦA NGƯỜI VÀ AI
        # =========================================================================
        matched_ai_indices = set()
        accounted_ai_indices = set()

        for obj in sorted(gt_objects, key=lambda obj: max(
                (compute_iou(tuple(obj["box"]), tuple(ai["box"])) for ai in ai_boxes
                 if is_same_category(obj["category"], ai["category"], aliases)), default=0), reverse=True):
            gt_cat = obj["category"]
            obj_id = str(obj.get("id", ""))

            # Tìm trong Box của AI:
            # A. Trùng vị trí cùng class (hoặc cùng nhóm synonym)
            same_cands = []
            for ai_i, ai_b in enumerate(ai_boxes):
                if ai_i in matched_ai_indices:
                    continue
                same_cat = is_same_category(gt_cat, ai_b["category"], aliases)
                sp_match, sp_score = is_spatial_match(obj["box"], ai_b["box"], min(match_iou, 0.25))
                if same_cat and sp_match:
                    same_cands.append((sp_score, ai_i))

            best_same_score, best_same_idx = max(same_cands, default=(0.0, -1))

            # B. Trùng vị trí khác class (xung đột class thật sự)
            diff_cands = []
            for ai_i, ai_b in enumerate(ai_boxes):
                if ai_i in matched_ai_indices:
                    continue
                same_cat = is_same_category(gt_cat, ai_b["category"], aliases)
                sp_match, sp_score = is_spatial_match(obj["box"], ai_b["box"], match_iou)
                if not same_cat and sp_match:
                    diff_cands.append((sp_score, ai_i))

            best_diff_score, best_diff_idx = max(diff_cands, default=(0.0, -1))

            # Prefer same-class association, then check boundary agreement.
            if best_same_idx >= 0:
                matched_ai_indices.add(best_same_idx)
                accounted_ai_indices.add(best_same_idx)

                if best_same_score < match_iou:
                    ai_cand = ai_boxes[best_same_idx]
                    findings.append({
                        "issue_id": f"{name}:{obj_id}:MODEL_BOUNDARY_DISAGREEMENT",
                        "object_id": obj_id, "category": gt_cat,
                        "severity": "MEDIUM", "error_type": "MODEL_BOUNDARY_DISAGREEMENT",
                        "explanation": f"Box người và AI cùng lớp nhưng khác biên (IoU={best_same_score:.2f}); cần kiểm tra.",
                        "action": "Đối chiếu biên với ảnh gốc.", "suggested_box": obj["box"],
                        "model_box": ai_cand["box"], "iou": best_same_score,
                        "review_status": "pending", "auto_healable": False,
                    })
                continue

            # 3.2 Khác nhau về Class (Người gán một đằng, AI xác định một nẻo) -> BOX CẦN REVIEW!
            elif best_diff_idx >= 0:
                matched_ai_indices.add(best_diff_idx)
                accounted_ai_indices.add(best_diff_idx)
                ai_cand = ai_boxes[best_diff_idx]
                ai_cat = ai_cand["category"]
                findings.append({
                    "issue_id": f"{name}:{obj_id}:MODEL_CLASS_DISAGREEMENT", "object_id": obj_id, "category": gt_cat,
                    "severity": "HIGH", "error_type": "MODEL_CLASS_DISAGREEMENT", "qa_layer": "Hội đồng AI",
                    "component": "Hội đồng AI", "model_name": ai_cand.get("model", "Hội đồng AI"),
                    "models_flagged": sorted({m["model"] for m in model_boxes}) or ["YOLO26x", "D-FINE-X"],
                    "explanation": f"Bất đồng nhãn: Người gán '{gt_cat}', AI xác định '{ai_cat}'.",
                    "action": f"Đổi nhãn thành '{ai_cat}' nếu AI đúng.", "suggested_box": ai_cand["box"],
                    "review_status": "pending", "auto_healable": False,
                    "guideline_rule": getattr(guideline_manager, "format_citation", lambda *a: "")("MODEL_CLASS_DISAGREEMENT", gt_cat) if guideline_manager else ""
                })

            # 3.3 Box người gán nhưng AI hoàn toàn không thấy (Người gán thừa / AI trượt) -> BOX CẦN REVIEW!
            else:
                if gt_cat in frame_unsupported:
                    continue
                findings.append({
                    "issue_id": f"{name}:{obj_id}:ANNOTATION_NOT_CONFIRMED_BY_MODEL", "object_id": obj_id, "category": gt_cat,
                    "severity": "MEDIUM", "error_type": "ANNOTATION_NOT_CONFIRMED_BY_MODEL", "qa_layer": "Hội đồng AI",
                    "component": "Hội đồng AI", "model_name": "",
                    "models_flagged": sorted({m["model"] for m in model_boxes}) or ["YOLO26x", "D-FINE-X"],
                    "explanation": f"AI không tìm thấy đối tượng '{gt_cat}' tại vị trí này (gán thừa hoặc lệch).",
                    "action": "Kiểm tra gán thừa hoặc lệch vị trí.", "suggested_box": obj["box"],
                    "review_status": "pending", "auto_healable": False,
                    "guideline_rule": getattr(guideline_manager, "format_citation", lambda *a: "")("ANNOTATION_NOT_CONFIRMED_BY_MODEL", gt_cat) if guideline_manager else ""
                })

        # 3.4 Box của AI phát hiện nhưng Người không gán (Người bỏ sót đối tượng) -> BOX CẦN REVIEW!
        for ai_i in range(len(ai_boxes)):
            if ai_i in matched_ai_indices or ai_i in accounted_ai_indices:
                continue
            ai_b = ai_boxes[ai_i]
            # Nếu AI box này có trùng lặp với bất kỳ box người nào thì người đã gán rồi, không thể coi là bỏ sót
            has_gt_overlap = any(is_spatial_match(obj["box"], ai_b["box"], 0.20)[0] for obj in gt_objects)
            if has_gt_overlap:
                continue

            findings.append({
                "issue_id": f"{name}:unlabeled:{ai_i}", "object_id": ai_b["id"], "category": ai_b["category"],
                "severity": "HIGH", "error_type": "POSSIBLE_UNLABELED_OBJECT", "qa_layer": "Hội đồng AI",
                "component": "Hội đồng AI", "model_name": ai_b.get("model", "Hội đồng AI"),
                "models_flagged": [ai_b.get("model", "Hội đồng AI")],
                "explanation": f"Có thể thiếu nhãn đối tượng '{ai_b['category']}'.",
                "action": f"Kiểm tra ảnh trước khi bổ sung nhãn '{ai_b['category']}'.", "suggested_box": ai_b["box"],
                "review_status": "pending", "auto_healable": False,
                "guideline_rule": getattr(guideline_manager, "format_citation", lambda *a: "")("class", ai_b["category"]) if guideline_manager else ""
            })

        # Model agreement cannot invalidate independent geometry/attribute rules.
        result["model_predictions"] = unified_system_predictions
        result["raw_model_boxes"] = model_boxes
        result["errors"].extend(findings)
        result["model_coverage"] = {"classes_in_project": class_names, "classes_predicted": sorted({p["category"] for p in unified_system_predictions}),
                                    "annotated_objects": len(gt_objects), "predictions": len(unified_system_predictions), "match_iou": match_iou}
        _rank(result)
        return p_meta, None, len(findings)

    max_workers = min(8, max(1, len(evaluated)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(executor.map(_process_bbox_frame, evaluated))

    for p_meta, skip, f_count in batch_results:
        if p_meta and not provider_meta:
            provider_meta = p_meta
        if skip:
            model_skips.append(skip)
        n_model_issues += f_count

    evaluated.sort(key=lambda f: (f["lqs"], f["priority_rank"], -f["error_count"], f.get("frame_index", 0)))
    return {"status": "success", "task_type": "bbox", "annotation_format": profile["annotation_format"], "guideline_status": profile.get("guideline_status", "not_configured"),
            "models": [provider_meta] if provider_meta else [], "model_issues": n_model_issues, "model_skips": model_skips,
            "taxonomy_alignment": taxonomy_alignment,
            "summary": {"total_frames": len(evaluated), "total_errors": sum(f["error_count"] for f in evaluated),
                        "mean_lqs": round(sum(f.get("lqs", 100.0) for f in evaluated) / max(1, len(evaluated)), 1),
                        "objectlab_breakdown": {
                            "missing_objects": sum(f.get("objectlab_counts", {}).get("missing_object", 0) for f in evaluated),
                            "class_confusions": sum(f.get("objectlab_counts", {}).get("class_confusion", 0) for f in evaluated),
                            "boundary_errors": sum(f.get("objectlab_counts", {}).get("boundary_error", 0) for f in evaluated)
                        },
                        "model_candidate_issues": n_model_issues,
                        "frames_skipped_for_model": len(model_skips), "benchmark": {"available": False, "reason": "Independent adjudicated gold labels are required."}},
            "sorted_frames": evaluated}


def _decode_mask_item(item: dict[str, Any], width: int, height: int):
    import numpy as np
    from PIL import Image, ImageDraw

    # 1. Polygon format (Meta SAM 2.1, YOLO26-Seg)
    points = item.get("points")
    if isinstance(points, list) and len(points) >= 6:
        coords = []
        for i in range(0, len(points) - 1, 2):
            coords.append((float(points[i]), float(points[i + 1])))
        if len(coords) >= 3:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(width - 1, int(max(xs)))
            y2 = min(height - 1, int(max(ys)))
            if x1 <= x2 and y1 <= y2:
                w_box = x2 - x1 + 1
                h_box = y2 - y1 + 1
                poly_local = [(px - x1, py - y1) for px, py in coords]
                img_mask = Image.new("1", (w_box, h_box), 0)
                ImageDraw.Draw(img_mask).polygon(poly_local, outline=1, fill=1)
                roi = np.asarray(img_mask, dtype=bool)
                return x1, y1, x2, y2, roi

    # 2. CVAT ROI dense mask [val0, val1, ..., x1, y1, x2, y2]
    mask = item.get("mask")
    if isinstance(mask, list) and len(mask) >= 5:
        x1, y1, x2, y2 = map(int, mask[-4:])
        if 0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height:
            expected = (x2 - x1 + 1) * (y2 - y1 + 1)
            if len(mask) - 4 == expected:
                roi = np.asarray(mask[:-4], dtype=np.uint8).reshape((y2 - y1 + 1, x2 - x1 + 1)) > 0
                return x1, y1, x2, y2, roi
    return None


def run_segmentation_pipeline(records: list[dict[str, Any]], images: dict[str, str], masks: dict[str, str], profile: dict[str, Any],
                              guideline_manager: Any, threshold: float = 0.35, review_iou_threshold: float = 0.5) -> dict[str, Any]:
    def arbitrate(*args, **kwargs):
        if profile.get("use_gemini") is False or not gemini_configured():
            return _unresolved_dispute("Gemini chưa được kích hoạt hoặc chưa cấu hình API key.")
        return arbitrate_dispute(*args, **kwargs)

    import numpy as np
    if not isinstance(records, list) or not records:
        raise ValueError("Segmentation manifest must be a non-empty array")
    if not profile.get("classes") and not profile.get("class_names") and not profile.get("mask_id_to_class") and profile.get("dataset", "bdd100k").lower() == "bdd100k":
        profile = {**profile, "class_names": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
                   "mask_id_to_class": {str(i): name for i, name in enumerate(["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])}}
    class_names, aliases = _ontology(profile, profile.get("class_names", []))
    if not class_names:
        # Walkthrough manifest rows can carry their class IDs and names.
        found = {}
        for row in records:
            for item in row.get("classes_present", []):
                if isinstance(item, dict) and "id" in item and item.get("name"):
                    found[int(item["id"])] = str(item["name"])
        class_names = [found[i] for i in sorted(found)]
        _, aliases = _ontology({"classes": class_names})
    if not class_names:
        raise ValueError("Semantic segmentation requires class_names or a workspace ontology")
    class_to_id = {name: i for i, name in enumerate(class_names)}
    raw_map = profile.get("mask_id_to_class", {})
    if raw_map:
        if isinstance(raw_map, list):
            id_to_class = {i: str(name) for i, name in enumerate(raw_map)}
        else:
            id_to_class = {int(k): str(v) for k, v in raw_map.items()}
    else:
        id_to_class = {i: name for i, name in enumerate(class_names)}
    id_to_class = {i: aliases.get(_normal(v), v) for i, v in id_to_class.items()}
    model_label_mapping = profile.get("model_label_mapping", {})
    ignore_ids = {int(x) for x in profile.get("ignore_mask_ids", [255])}
    output_frames, providers, model_issues, skips = [], [], 0, []
    min_iou = float(profile.get("review_mask_iou_threshold", review_iou_threshold))

    def _process_seg_frame(arg: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        frame_index, row = arg
        name = Path(str(row.get("name", ""))).name
        if not name:
            raise ValueError(f"Segmentation manifest row {frame_index} requires name")
        image_data = _image_bytes(name, images) or _image_bytes(Path(name).name, images) or _image_bytes(str(row.get("name", "")), images)
        raw_mask_path = str(row.get("mask_index", row.get("mask", "")))
        mask_name = Path(raw_mask_path).name
        mask_encoded = (
            masks.get(mask_name)
            or masks.get(raw_mask_path)
            or masks.get(raw_mask_path.replace("\\", "/"))
            or masks.get(raw_mask_path.replace("/", "\\"))
        )
        if not image_data or not mask_encoded:
            raise ValueError(f"Image or index mask missing for segmentation frame {name}")
        if mask_encoded.startswith("data:"):
            mask_encoded = mask_encoded.split(",", 1)[1]
        try:
            mask_data = base64.b64decode(mask_encoded, validate=True)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            gt = np.asarray(Image.open(io.BytesIO(mask_data)))
        except Exception as exc:
            raise ValueError(f"Could not decode image/index mask for {name}: {exc}") from exc
        if gt.ndim == 3:
            if not profile.get("color_mask_to_id"):
                raise ValueError(f"Mask {mask_name} is color RGB; provide an explicit color_mask_to_id profile or upload train-ID masks")
            rgb_map = {tuple(map(int, k.split(","))): int(v) for k, v in profile["color_mask_to_id"].items()}
            converted = np.full(gt.shape[:2], 255, dtype=np.uint8)
            for rgb, mask_id in rgb_map.items():
                converted[np.all(gt[:, :, :3] == rgb, axis=2)] = mask_id
            gt = converted
        height, width = gt.shape[:2]
        if image.size != (width, height):
            raise ValueError(f"Image/index mask dimensions differ for {name}: {image.size} vs {(width, height)}")
        preferred_seg_model = profile.get("preferred_model") or profile.get("model")
        model_results, info = infer_segmentation(image_data, threshold, class_names, model_label_mapping, preferred_model=preferred_seg_model)
        pred_yolo = np.full((height, width), -1, dtype=np.int16)
        pred_sam = np.full((height, width), -1, dtype=np.int16)
        pred = np.full((height, width), -1, dtype=np.int16)
        mapped_classes_yolo = set()
        mapped_classes_sam = set()
        mapped_classes = set()

        # Thuật toán Painter's Algorithm: Lớp nền (đường, trời, nhà, cây...) vẽ trước,
        # lớp foreground (xe cộ, người, biển báo...) vẽ đè lên trên để không bị che khuất.
        BACKGROUND_STUFF = {"road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky", "snow", "water", "water-other"}
        def _raster_priority(itm: dict[str, Any]) -> int:
            lbl = _normal(itm.get("label", ""))
            return 0 if any(bg in lbl for bg in BACKGROUND_STUFF) else 1

        for item in sorted(model_results, key=_raster_priority):
            raw_label = str(item.get("label", ""))
            canonical = model_label_mapping.get(raw_label, aliases.get(_normal(raw_label), raw_label))
            if canonical not in class_to_id:
                continue
            mask_item = _decode_mask_item(item, width, height)
            if mask_item is None:
                continue
            x1, y1, x2, y2, roi = mask_item
            cid = class_to_id[canonical]
            pred[y1:y2 + 1, x1:x2 + 1][roi] = cid
            mapped_classes.add(canonical)

            m_tag = str(item.get("model", "")).upper()
            if "SAM" in m_tag:
                pred_sam[y1:y2 + 1, x1:x2 + 1][roi] = cid
                mapped_classes_sam.add(canonical)
            elif "YOLO" in m_tag:
                pred_yolo[y1:y2 + 1, x1:x2 + 1][roi] = cid
                mapped_classes_yolo.add(canonical)

        advertised_classes = info.get("supported_classes")
        supported_classes = ({model_label_mapping.get(c, aliases.get(_normal(c), c))
                              for c in advertised_classes} if isinstance(advertised_classes, list)
                             else set(mapped_classes))
        unsupported_classes = set(class_names) - supported_classes
        has_dual_seg = bool(mapped_classes_yolo and mapped_classes_sam)
        guideline_txt = getattr(guideline_manager, "get_guideline_text", lambda: "")() if guideline_manager else ""
        classes_metrics = []
        issues = []
        valid = np.isin(gt, list(id_to_class.keys())) & ~np.isin(gt, list(ignore_ids))

        for mask_id, class_name in sorted(id_to_class.items()):
            if class_name not in class_to_id:
                continue
            gt_pixels = (gt == mask_id) & valid
            pred_pixels = (pred == class_to_id[class_name]) & valid
            union = int(np.logical_or(gt_pixels, pred_pixels).sum())
            intersection = int(np.logical_and(gt_pixels, pred_pixels).sum())
            if union == 0:
                continue
            iou = intersection / union
            classes_metrics.append({"class_id": mask_id, "class_name": class_name, "iou": round(iou, 5),
                                    "intersection_pixels": intersection, "union_pixels": union, "ground_truth_pixels": int(gt_pixels.sum()),
                                    "prediction_pixels": int(pred_pixels.sum()), "prediction_supported": class_name in supported_classes})

            if gt_pixels.any() or pred_pixels.any():
                mismatch_y, mismatch_x = np.where(np.logical_xor(gt_pixels, pred_pixels))
                evidence_box = ([int(mismatch_x.min()), int(mismatch_y.min()), int(mismatch_x.max() + 1), int(mismatch_y.max() + 1)]
                                if mismatch_x.size else None)

                if has_dual_seg and class_name in mapped_classes_yolo and class_name in mapped_classes_sam:
                    # Tính IoU riêng cho YOLO26-Seg và SAM2.1
                    py_pixels = (pred_yolo == class_to_id[class_name]) & valid
                    u_y = int(np.logical_or(gt_pixels, py_pixels).sum())
                    i_y = int(np.logical_and(gt_pixels, py_pixels).sum())
                    iou_y = (i_y / u_y) if u_y > 0 else 0.0

                    ps_pixels = (pred_sam == class_to_id[class_name]) & valid
                    u_s = int(np.logical_or(gt_pixels, ps_pixels).sum())
                    i_s = int(np.logical_and(gt_pixels, ps_pixels).sum())
                    iou_s = (i_s / u_s) if u_s > 0 else 0.0

                    y_fail = (iou_y < min_iou)
                    s_fail = (iou_s < min_iou)
                    # Evidence must include the failing source, even when the
                    # final painted mask matches GT exactly.
                    disagreement = np.logical_xor(gt_pixels, py_pixels) | np.logical_xor(gt_pixels, ps_pixels)
                    dy, dx = np.where(disagreement)
                    evidence_box = [int(dx.min()), int(dy.min()), int(dx.max() + 1), int(dy.max() + 1)] if dx.size else None


                    # Tình huống 1: Cả 2 cùng bắt lỗi sai viền -> HỘI ĐỒNG ĐỒNG THUẬN
                    if y_fail and s_fail:
                        issues.append({
                            "issue_id": f"{name}:mask-class:{mask_id}", "object_id": f"MASK_CLASS_{mask_id}", "category": class_name,
                            "severity": "HIGH", "error_type": "SEGMENTATION_BOUNDARY_SUSPECT",
                            "qa_layer": "⚖️ Hội đồng (YOLO26x-Seg + SAM 2.1)",
                            "component": "YOLO26x-Seg + SAM 2.1", "model_name": "YOLO26x-Seg + SAM 2.1 Large",
                            "models_flagged": ["YOLO26x-Seg", "SAM 2.1"],
                            "model_confidence": None, "iou": round(min(iou_y, iou_s), 5), "error_pixels": union - intersection,
                            "explanation": f"YOLO và SAM dùng prompt từ YOLO cùng khác mask '{class_name}' (IoU {min(iou_y, iou_s):.2f} < {min_iou:.2f}).",
                            "action": f"Kiểm tra biên '{class_name}' trên ảnh gốc; hai kết quả phụ thuộc cùng detector.", "suggested_box": evidence_box,
                            "review_threshold": min_iou,
                            "guideline_rule": guideline_manager.format_citation("mask", class_name) if guideline_manager else "", "review_status": "pending", "auto_healable": False
                        })
                    # Tình huống 2: 1 model đạt, 1 model lỗi -> Thừa còn hơn thiếu: báo candidate để reviewer kiểm tra
                    elif (y_fail and not s_fail) or (s_fail and not y_fail):
                        flagged_m = "YOLO26x-Seg" if y_fail else "SAM 2.1"
                        min_detected_iou = min(iou_y, iou_s)
                        issues.append({
                            "issue_id": f"{name}:mask-class:{mask_id}", "object_id": f"MASK_CLASS_{mask_id}", "category": class_name,
                            "severity": "MEDIUM", "error_type": "SEGMENTATION_BOUNDARY_SUSPECT",
                            "qa_layer": "Hội đồng Segmentation",
                            "component": "YOLO26x-Seg + SAM 2.1", "model_name": flagged_m,
                            "models_flagged": [flagged_m],
                            "model_confidence": None, "iou": round(min_detected_iou, 5), "error_pixels": union - intersection,
                            "explanation": f"Lệch viền mask '{class_name}' phát hiện bởi {flagged_m} (IoU {min_detected_iou:.2f} < {min_iou:.2f}).",
                            "action": f"Kiểm tra lại biên '{class_name}'.",
                            "suggested_box": evidence_box, "review_threshold": min_iou,
                            "guideline_rule": guideline_manager.format_citation("mask", class_name) if guideline_manager else "", "review_status": "pending", "auto_healable": False
                        })

                elif iou < min_iou and (class_name in supported_classes or advertised_classes is None):
                    # An absent/low-overlap prediction is not proof of a bad
                    # mask. Keep it visible for review, including zero-output
                    # model runs, instead of returning a clean frame.
                    issues.append({
                        **_unresolved_finding(
                            name, f"MASK_CLASS_{mask_id}", class_name, evidence_box,
                            {"explanation": f"Kết quả segmentation không đủ xác nhận mask '{class_name}' (IoU={iou:.2f}, ngưỡng review={min_iou:.2f}); cần reviewer kiểm tra."},
                        ),
                        "iou": round(iou, 5), "error_pixels": union - intersection,
                        "review_threshold": min_iou,
                        "guideline_rule": guideline_manager.format_citation("mask", class_name) if guideline_manager else "",
                    })
        macro_all = sum(m["iou"] for m in classes_metrics) / len(classes_metrics) if classes_metrics else None
        supported_metrics = [m for m in classes_metrics if m.get("prediction_supported")]
        macro_supported = sum(m["iou"] for m in supported_metrics) / len(supported_metrics) if supported_metrics else None
        macro = macro_all
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        palette = [(93, 224, 188, 145), (255, 180, 76, 145), (174, 149, 255, 145), (255, 108, 125, 145), (93, 173, 238, 145)]
        for cname, cid in class_to_id.items():
            rgb = palette[(cid * 47 + len(cname)) % len(palette)]
            rgba[pred == class_to_id[cname]] = rgb
        color_img = Image.fromarray(rgba, mode="RGBA")
        png = io.BytesIO(); color_img.save(png, format="PNG")
        out = {"frame_name": name, "frame_index": frame_index, "width": width, "height": height, "priority_rank": 3 if issues else 4,
               "priority_label": "MASK REVIEW CANDIDATE" if issues else "MASKS COMPARED",
               "recommendation": "Review class disagreements" if issues else "No configured mask threshold crossed; not a verified pass",
               "error_count": len(issues), "errors": issues, "issues": issues, "labels": [], "mask_color": Path(str(row.get("mask_color", ""))).name,
               "classes_present": row.get("classes_present", []), "model_segmentation_metrics": {
                   "macro_iou": round(macro, 5) if macro is not None else None,
                   "macro_iou_supported": round(macro_supported, 5) if macro_supported is not None else None,
                   "class_iou": classes_metrics, "classes_model_supported": sorted(supported_classes),
                   "classes_not_evaluated": sorted(unsupported_classes), "mask_iou_threshold": min_iou},
               "model_predictions": model_results if model_results else [{"label": k, "mask_id": v} for v, k in id_to_class.items() if k in mapped_classes],
               "prediction_mask_png_base64": base64.b64encode(png.getvalue()).decode("ascii"),
               "_provider_info": info}
        _rank(out)
        return out

    max_workers = min(6, max(1, len(records)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        output_frames = list(executor.map(_process_seg_frame, enumerate(records)))

    providers = [output_frames[0]["_provider_info"]] if output_frames and "_provider_info" in output_frames[0] else []
    for f in output_frames:
        f.pop("_provider_info", None)
    model_issues = sum(len(f.get("issues", [])) for f in output_frames)
    output_frames.sort(key=lambda f: (f["lqs"], f["priority_rank"], -f["error_count"], f["frame_index"]))
    supported_scores = [f["model_segmentation_metrics"]["macro_iou_supported"] for f in output_frames
                        if f["model_segmentation_metrics"]["macro_iou_supported"] is not None]
    return {"status": "success", "task_type": "semantic_segmentation", "models": providers, "model_issues": model_issues,
            "summary": {"total_frames": len(output_frames), "total_errors": model_issues,
                        "macro_iou_supported": round(sum(supported_scores) / len(supported_scores), 5) if supported_scores else None,
                        "mean_lqs": round(sum(f.get("lqs", 100.0) for f in output_frames) / max(1, len(output_frames)), 1),
                        "objectlab_breakdown": {
                            "missing_objects": sum(f.get("objectlab_counts", {}).get("missing_object", 0) for f in output_frames),
                            "class_confusions": sum(f.get("objectlab_counts", {}).get("class_confusion", 0) for f in output_frames),
                            "boundary_errors": sum(f.get("objectlab_counts", {}).get("boundary_error", 0) for f in output_frames)
                        },
                        "macro_iou": round(sum(f["model_segmentation_metrics"]["macro_iou"] or 0 for f in output_frames) / max(1, sum(f["model_segmentation_metrics"]["macro_iou"] is not None for f in output_frames)), 5),
                        "benchmark": {"available": False, "reason": "Independent adjudicated masks and a fixed class mapping are required for benchmark claims."}},
            "sorted_frames": output_frames}
