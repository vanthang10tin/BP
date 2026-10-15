"""Dataset normalization and local model-to-annotation comparisons."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

from ai_engine.local_model_providers import infer_bbox, infer_segmentation, ProviderError
from coordinate_utils import bdd_to_xyxy
from ai_engine.models_ensemble import compute_iou


def _normal(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


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
    # Widely used CV label aliases, applied only if that canonical label exists.
    for alias, target in {"person": "pedestrian", "traffic light": "traffic_light", "traffic sign": "traffic_sign"}.items():
        if target in names and alias not in alias_map:
            alias_map[alias] = target
    return names, alias_map


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

    if annotation_format not in {"bdd100k", "normalized"}:
        raise ValueError(f"Unsupported bbox annotation format: {annotation_format}")
    frames = data if isinstance(data, list) else data.get("frames", data.get("data", [data])) if isinstance(data, dict) else None
    if not isinstance(frames, list):
        raise ValueError("Expected a BDD100K/normalized frame array")
    normalized = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {index} is not an object")
        name = frame.get("name", frame.get("frame_name"))
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Frame {index} needs a safe image filename in name")
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
    # Optional configured archive lookup remains local and filename-scoped.
    try:
        from app import find_image_file_path
        path = find_image_file_path(name)
        return Path(path).read_bytes() if path else None
    except Exception:
        return None


def _rank(frame: dict[str, Any]) -> None:
    errors = frame.get("errors", [])
    severity = {e.get("severity") for e in errors}
    if "CRITICAL" in severity:
        rank, label = 1, "CRITICAL REVIEW"
    elif "HIGH" in severity:
        rank, label = 2, "HIGH REVIEW"
    elif errors:
        rank, label = 3, "REVIEW CANDIDATE"
    else:
        rank, label = 4, "NO FLAGS FROM ENABLED CHECKS"
    frame.update(priority_rank=rank, priority_label=label, priority_level=label,
                 error_count=len(errors), recommendation="Human review required" if errors else "No findings from enabled checks; not a verified pass")


def _base_bbox_frames(frames: list[dict[str, Any]], profile: dict[str, Any], guideline_manager: Any) -> list[dict[str, Any]]:
    is_bdd = profile.get("annotation_format", "bdd100k") == "bdd100k" and profile.get("dataset", "bdd100k").lower() == "bdd100k"
    if is_bdd:
        from qc_engine import BDDQualityChecker
        results = BDDQualityChecker(guideline_manager=guideline_manager).check_dataset_advanced(frames)["sorted_frames"]
        for output, source in zip(results, frames):
            output["width"] = int(source.get("width", profile.get("image_width", 1280)))
            output["height"] = int(source.get("height", profile.get("image_height", 720)))
        return results
    out = []
    for index, source in enumerate(frames):
        labels = source.get("labels", [])
        out.append({"frame_name": source["name"], "frame_index": index, "priority_rank": 4,
                    "priority_label": "NO FLAGS FROM ENABLED CHECKS", "recommendation": "No findings from enabled checks; not a verified pass",
                    "error_count": 0, "errors": [], "labels": labels,
                    "width": int(source.get("width", profile.get("image_width", 0))),
                    "height": int(source.get("height", profile.get("image_height", 0)))})
    return out


def run_bbox_pipeline(data: Any, images: dict[str, str], profile: dict[str, Any], guideline_manager: Any,
                      threshold: float = 0.35, match_iou: float = 0.5) -> dict[str, Any]:
    profile = {"annotation_format": "bdd100k", "dataset": "bdd100k", **profile}
    frames = normalize_bbox_dataset(data, profile["annotation_format"], profile)
    if not frames:
        raise ValueError("No frames in dataset")
    class_names, aliases = _ontology(profile, sorted({str(l.get("category", "")) for f in frames for l in f.get("labels", []) if l.get("category")}))
    if not class_names:
        raise ValueError("No class ontology found; supply classes in the workspace profile")
    evaluated = _base_bbox_frames(frames, profile, guideline_manager)
    source_by_name = {f["name"]: f for f in frames}
    provider_meta = None
    model_skips = []
    n_model_issues = 0
    min_conf = float(profile.get("model_threshold", threshold))
    for result in evaluated:
        name = result["frame_name"]
        image_data = _image_bytes(name, images)
        if image_data is None:
            model_skips.append({"frame_name": name, "reason": "image_not_uploaded_or_found_locally"})
            continue
        try:
            with Image.open(io.BytesIO(image_data)) as im:
                width, height = im.size
            predictions, provider_meta = infer_bbox(image_data, min_conf, class_names, profile.get("model_label_mapping", {}))
        except (ProviderError, OSError, ValueError) as exc:
            model_skips.append({"frame_name": name, "reason": str(exc)[:400]})
            continue
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
            if category not in class_names or conf < min_conf:
                continue
            model_boxes.append({"category": category, "model_label": category_raw, "box": box, "confidence": conf,
                                "model": provider_meta["name"], "type": "rectangle"})

        unmatched = set(range(len(model_boxes)))
        findings = []
        for obj in gt_objects:
            class_candidates = [(compute_iou(tuple(obj["box"]), tuple(model_boxes[j]["box"])), j) for j in unmatched if model_boxes[j]["category"] == obj["category"]]
            best_iou, best_idx = max(class_candidates, default=(0.0, -1))
            overlap_candidates = [(compute_iou(tuple(obj["box"]), tuple(model_boxes[j]["box"])), j) for j in unmatched]
            overlap_iou, overlap_idx = max(overlap_candidates, default=(0.0, -1))
            if best_iou >= match_iou:
                unmatched.discard(best_idx)
                continue
            issue_type = "MODEL_CLASS_DISAGREEMENT" if overlap_iou >= 0.30 and model_boxes[overlap_idx]["category"] != obj["category"] else "ANNOTATION_NOT_CONFIRMED_BY_MODEL"
            evidence_box = model_boxes[overlap_idx]["box"] if overlap_iou >= 0.30 else obj["box"]
            issue = {"issue_id": f"{name}:{obj['id']}:{issue_type}", "object_id": obj["id"], "category": obj["category"],
                     "severity": "MEDIUM", "error_type": issue_type, "qa_layer": "Local model candidate",
                     "component": provider_meta["name"], "model_name": provider_meta["name"], "model_confidence": model_boxes[overlap_idx]["confidence"] if overlap_iou >= 0.30 else None,
                     "models_flagged": [provider_meta["name"]], "explanation": f"Local detector agreement below IoU {match_iou:.2f} (best same-class IoU={best_iou:.3f}). This is an unverified review candidate, not proof the annotation is wrong.",
                     "action": "Compare the source image and guideline; confirm, correct, or dismiss the candidate.", "suggested_box": evidence_box,
                     "review_status": "pending", "auto_healable": False, "guideline_rule": guideline_manager.format_citation(issue_type, obj["category"])}
            findings.append(issue)
        for j in sorted(unmatched):
            pred = model_boxes[j]
            findings.append({"issue_id": f"{name}:prediction:{j}", "object_id": f"MODEL_{j}", "category": pred["category"],
                "severity": "MEDIUM", "error_type": "POSSIBLE_UNLABELED_OBJECT", "qa_layer": "Local model candidate",
                "component": provider_meta["name"], "model_name": provider_meta["name"], "model_confidence": pred["confidence"],
                "models_flagged": [provider_meta["name"]], "explanation": f"Local detector proposed {pred['category']} (confidence={pred['confidence']:.3f}) without a class/IoU match in the uploaded labels. Confirm against guideline before changing annotations.",
                "action": "Inspect the proposal and confirm or dismiss it.", "suggested_box": pred["box"], "review_status": "pending",
                "auto_healable": False, "guideline_rule": guideline_manager.format_citation("class", pred["category"])})
        result["model_predictions"] = model_boxes
        result["errors"].extend(findings)
        result["model_coverage"] = {"classes_in_project": class_names, "classes_predicted": sorted({p["category"] for p in model_boxes}),
                                    "annotated_objects": len(gt_objects), "predictions": len(model_boxes), "match_iou": match_iou}
        n_model_issues += len(findings)
        _rank(result)
    evaluated.sort(key=lambda f: (f["priority_rank"], -f["error_count"], f.get("frame_index", 0)))
    return {"status": "success", "task_type": "bbox", "annotation_format": profile["annotation_format"], "guideline_status": profile.get("guideline_status", "not_configured"),
            "models": [provider_meta] if provider_meta else [], "model_issues": n_model_issues, "model_skips": model_skips,
            "summary": {"total_frames": len(evaluated), "total_errors": sum(f["error_count"] for f in evaluated), "model_candidate_issues": n_model_issues,
                        "frames_skipped_for_model": len(model_skips), "benchmark": {"available": False, "reason": "Independent adjudicated gold labels are required."}},
            "sorted_frames": evaluated}


def _decode_mask_item(item: dict[str, Any], width: int, height: int):
    import numpy as np
    mask = item.get("mask")
    if not isinstance(mask, list) or len(mask) < 5:
        return None
    x1, y1, x2, y2 = map(int, mask[-4:])
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        return None
    expected = (x2 - x1 + 1) * (y2 - y1 + 1)
    if len(mask) - 4 != expected:
        return None
    roi = np.asarray(mask[:-4], dtype=np.uint8).reshape((y2 - y1 + 1, x2 - x1 + 1)) > 0
    return x1, y1, x2, y2, roi


def run_segmentation_pipeline(records: list[dict[str, Any]], images: dict[str, str], masks: dict[str, str], profile: dict[str, Any],
                              guideline_manager: Any, threshold: float = 0.35, review_iou_threshold: float = 0.5) -> dict[str, Any]:
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
    for frame_index, row in enumerate(records):
        name = Path(str(row.get("name", ""))).name
        if not name:
            raise ValueError(f"Segmentation manifest row {frame_index} requires name")
        image_data = _image_bytes(name, images)
        mask_name = Path(str(row.get("mask_index", row.get("mask", "")))).name
        mask_encoded = masks.get(mask_name)
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
            raise ValueError(f"Image and mask dimensions differ for {name}: {image.size} vs {(width, height)}")
        model_results, info = infer_segmentation(image_data, threshold, class_names, model_label_mapping)
        providers = [info]
        pred = np.full((height, width), -1, dtype=np.int16)
        mapped_classes = set()
        for item in model_results:
            raw_label = str(item.get("label", ""))
            canonical = model_label_mapping.get(raw_label, aliases.get(_normal(raw_label), raw_label))
            if canonical not in class_to_id:
                continue
            mask_item = _decode_mask_item(item, width, height)
            if mask_item is None:
                continue
            x1, y1, x2, y2, roi = mask_item
            view = pred[y1:y2 + 1, x1:x2 + 1]
            view[roi] = class_to_id[canonical]
            mapped_classes.add(canonical)
        classes_metrics = []
        issues = []
        valid = np.isin(gt, list(id_to_class.keys())) & ~np.isin(gt, list(ignore_ids))
        for mask_id, class_name in sorted(id_to_class.items()):
            if class_name not in class_to_id:
                continue
            gt_pixels = (gt == mask_id) & valid
            pred_pixels = pred == class_to_id[class_name]
            union = int(np.logical_or(gt_pixels, pred_pixels & valid).sum())
            intersection = int(np.logical_and(gt_pixels, pred_pixels & valid).sum())
            if union == 0:
                continue
            iou = intersection / union
            classes_metrics.append({"class_id": mask_id, "class_name": class_name, "iou": round(iou, 5),
                                    "intersection_pixels": intersection, "union_pixels": union, "ground_truth_pixels": int(gt_pixels.sum()),
                                    "prediction_pixels": int((pred_pixels & valid).sum()), "prediction_supported": class_name in mapped_classes})
            if iou < min_iou and gt_pixels.any():
                mismatch_y, mismatch_x = np.where(np.logical_xor(gt_pixels, pred_pixels & valid))
                evidence_box = ([int(mismatch_x.min()), int(mismatch_y.min()), int(mismatch_x.max() + 1), int(mismatch_y.max() + 1)]
                                if mismatch_x.size else None)
                issues.append({"issue_id": f"{name}:mask-class:{mask_id}", "object_id": f"MASK_CLASS_{mask_id}", "category": class_name,
                    "severity": "MEDIUM", "error_type": "SEGMENTATION_MODEL_DISAGREEMENT", "qa_layer": "Local model candidate",
                    "component": info["name"], "model_name": info["name"], "models_flagged": [info["name"]],
                    "model_confidence": None, "iou": round(iou, 5), "error_pixels": union - intersection,
                    "explanation": f"Local panoptic model versus uploaded train-ID mask for {class_name}: IoU={iou:.3f}, below review threshold {min_iou:.2f}. This is a comparison candidate, not proof the mask is wrong.",
                    "action": "Inspect this class in the image and mask; confirm or dismiss before correction.", "suggested_box": evidence_box,
                    "review_threshold": min_iou,
                    "guideline_rule": guideline_manager.format_citation("mask", class_name), "review_status": "pending", "auto_healable": False})
        macro = sum(m["iou"] for m in classes_metrics) / len(classes_metrics) if classes_metrics else None
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        palette = [(93, 224, 188, 145), (255, 180, 76, 145), (174, 149, 255, 145), (255, 108, 125, 145), (93, 173, 238, 145)]
        for cname, cid in class_to_id.items():
            rgb = palette[(cid * 47 + len(cname)) % len(palette)]
            rgba[pred == class_to_id[cname]] = rgb
        color_img = Image.fromarray(rgba, mode="RGBA")
        png = io.BytesIO(); color_img.save(png, format="PNG")
        out = {"frame_name": name, "frame_index": frame_index, "priority_rank": 3 if issues else 4,
               "priority_label": "MASK REVIEW CANDIDATE" if issues else "MASKS COMPARED",
               "recommendation": "Review class disagreements" if issues else "No configured mask threshold crossed; not a verified pass",
               "error_count": len(issues), "errors": issues, "labels": [], "mask_color": Path(str(row.get("mask_color", ""))).name,
               "classes_present": row.get("classes_present", []), "model_segmentation_metrics": {"macro_iou": round(macro, 5) if macro is not None else None,
               "class_iou": classes_metrics, "classes_model_supported": sorted(mapped_classes), "mask_iou_threshold": min_iou},
               "model_predictions": [{"label": k, "mask_id": v} for v, k in id_to_class.items() if k in mapped_classes],
               "prediction_mask_png_base64": base64.b64encode(png.getvalue()).decode("ascii")}
        output_frames.append(out); model_issues += len(issues)
    output_frames.sort(key=lambda f: (f["priority_rank"], -f["error_count"], f["frame_index"]))
    return {"status": "success", "task_type": "semantic_segmentation", "models": providers, "model_issues": model_issues,
            "summary": {"total_frames": len(output_frames), "total_errors": model_issues,
                        "macro_iou": round(sum(f["model_segmentation_metrics"]["macro_iou"] or 0 for f in output_frames) / max(1, sum(f["model_segmentation_metrics"]["macro_iou"] is not None for f in output_frames)), 5),
                        "benchmark": {"available": False, "reason": "Independent adjudicated masks and a fixed class mapping are required for benchmark claims."}},
            "sorted_frames": output_frames}
