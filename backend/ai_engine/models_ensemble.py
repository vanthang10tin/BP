#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy deterministic proposal placeholders for the planned multi-stage QC flow.
These classes do not load YOLO, RT-DETR, SAM, or Gemini models. They are
disabled by default and must not be described as model evidence or consensus.
"""

import math
import os
from typing import Dict, List, Any, Tuple, Optional, Set

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from .guideline_manager import global_guideline_manager
from coordinate_utils import to_xyxy, bdd_to_xyxy, xyxy_to_bdd, xyxy_to_gemini

# Danh mục 10 nhãn đối tượng chuẩn của BDD100K (Detection Ontology)
BDD100K_STANDARD_CLASSES = {
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign"
}

# BDD100K detection JSON also stores semantic regions in the same labels array.
# These are legitimate polygons, but are not object-detection classes.
BDD100K_POLYGON_CLASSES = {"lane", "drivable area"}

# Ánh xạ alias phổ biến về class chuẩn
BDD100K_CLASS_ALIASES = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
    "auto": "car",
    "automobile": "car",
    "van": "truck",
    "xanh_sm": "car",
    "xanh sm": "car",
    "xanh-sm": "car",
    "taxi": "car",
    "cab": "car",
    "o to": "car",
    "oto": "car",
    "xe hoi": "car",
    "xe may": "motorcycle",
    "xe dap": "bicycle",
    "xe tai": "truck",
    "xe buyt": "bus",
    "xe bus": "bus",
}

# Heuristic application ranges only; these are not BDD100K or project standards.
CLASS_ASPECT_RATIO_RANGES = {
    "car": (0.35, 4.2),
    "truck": (0.35, 4.8),
    "bus": (0.40, 5.0),
    "pedestrian": (0.12, 1.40),
    "rider": (0.15, 1.60),
    "bicycle": (0.20, 3.0),
    "motorcycle": (0.20, 3.0),
    "traffic light": (0.15, 3.8),
    "traffic sign": (0.20, 3.5),
    "train": (0.30, 8.0),
}


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / union


def check_polygon_validity(vertices: List[List[float]]) -> Tuple[bool, str]:
    """Kiểm tra đa giác có tự cắt, suy biến hoặc có đỉnh trùng lặp không (Slide 30)."""
    if len(vertices) < 3:
        return True, "Polygon có ít hơn 3 đỉnh"

    # Kiểm tra trùng đỉnh liên tiếp
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        if abs(p1[0] - p2[0]) < 1e-4 and abs(p1[1] - p2[1]) < 1e-4:
            return True, "Có đỉnh liên tiếp trùng nhau"

    if SHAPELY_AVAILABLE:
        try:
            poly = ShapelyPolygon(vertices)
            if not poly.is_valid:
                return True, "Polygon tự cắt cạnh (Self-intersecting) hoặc viền phức"
            if poly.area <= 1.0:
                return True, "Diện tích polygon quá nhỏ hoặc suy biến (Area <= 1px²)"
            return False, ""
        except Exception as e:
            return True, f"Lỗi cấu trúc đa giác: {str(e)}"
    return False, ""


def get_polygon_bbox(vertices: List[List[float]]) -> Tuple[float, float, float, float]:
    """Lấy bounding box nhỏ nhất bao quanh polygon."""
    xs = [pt[0] for pt in vertices]
    ys = [pt[1] for pt in vertices]
    return min(xs), min(ys), max(xs), max(ys)


# =====================================================================
# LOCAL YOLO26 DETECTOR (GENUINE INFERENCE - NO SIMULATION)
# =====================================================================

class LocalYOLO26Detector:
    """
    Detector BBox cục bộ sử dụng Ultralytics YOLO26 Native (yolo26n.pt / yolo26x.pt).
    Chỉ chạy khi có ảnh thực tế (file trên đĩa hoặc bytes upload).
    Tuyệt đối không giả lập tọa độ hay fake kết quả inference.
    """
    def __init__(self):
        self.model_name = "Ultralytics YOLO26"

    def detect_frame(self, frame_name: str, image_bytes: Optional[bytes] = None, conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Chạy inference YOLO26 thật trên ảnh nếu có ảnh cục bộ hoặc image_bytes."""
        if not image_bytes:
            from coordinate_utils import find_image_path
            path = find_image_path(frame_name)
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        image_bytes = f.read()
                except Exception:
                    image_bytes = None

        if not image_bytes:
            return []

        try:
            from .local_model_providers import _infer_yolo26_native
            preds, info = _infer_yolo26_native(image_bytes, conf_threshold, list(BDD100K_STANDARD_CLASSES))
            return preds
        except Exception:
            return []


# =====================================================================
# TỔNG CHỈ HUY KIỂM SOÁT 4 TẦNG (MULTI-MODEL QC ENGINE)
# =====================================================================

class MultiModelQCEngine:
    """
    Core Engine tích hợp toàn diện 4 Tầng QA:
    Tầng 1: Rule Engine (Cấu trúc, hình học, quy chuẩn BDD100K)
    Tầng 2: Model-Based Error Detection (ObjectLab / Cleanlab)
    Tầng 3: Prioritized Review Queue (Đo lường Recall@20% >= 70%)
    Tầng 4: Honeypot & Gold Standard Quality Tracker
    """
    def __init__(self, guideline_manager=None):
        self.detector = LocalYOLO26Detector()
        self.guideline_manager = guideline_manager or global_guideline_manager

    def evaluate_dataset_with_consensus(self, dataset: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Đánh giá toàn diện tập dữ liệu qua 4 Tầng QA."""
        cfg = {
            "image_width": 1280,
            "image_height": 720,
            "min_box_width": 4.0,
            "min_box_height": 4.0,
            "min_box_area": 16.0,
            "duplicate_box_iou_threshold": 0.85,
            "mandatory_attributes": ["occluded", "truncated"],
            "mask_box_min_iou": 0.70,
            "max_mask_box_drift_px": 15.0,
        }
        if config:
            cfg.update(config)

        evaluated_frames = []
        total_objects = 0
        total_errors = 0
        auto_pass_count = 0
        need_review_count = 0
        critical_count = 0

        # Thống kê hiệu suất từng tầng QA (Slide 29 & Slide 30)
        layer_stats = {
            "layer_1_rules": 0,
            "layer_2_models": 0,
            "layer_3_prioritized": 0,
            "layer_4_honeypot": 0,
        }
        rule_telemetry = {}

        for frame_idx, frame in enumerate(dataset):
            frame_name = frame.get("name", f"frame_{frame_idx:04d}.jpg")
            labels = frame.get("labels", [])
            total_objects += len(labels)
            frame_errors = []

            # -----------------------------------------------------------------
            # TẦNG 1: QC THEO LUẬT (CẤU TRÚC, HÌNH HỌC, QUY CHUẨN BDD100K - Slide 30)
            # -----------------------------------------------------------------
            valid_boxes_for_dup = []

            for idx, label in enumerate(labels):
                lbl_id = str(label.get("id", f"obj_{idx}"))
                cat = str(label.get("category", "unspecified")).strip().lower()
                attrs = label.get("attributes", {})
                b = label.get("box2d")

                # 1.1 Kiểm tra BDD100K Ontology Class chuẩn (Slide 30)
                canonical_cat = BDD100K_CLASS_ALIASES.get(cat, cat)
                is_semantic_region = cat in BDD100K_POLYGON_CLASSES and not b
                if canonical_cat not in BDD100K_STANDARD_CLASSES and not is_semantic_region:
                    cit = self.guideline_manager.format_citation("class", cat)
                    err = {
                        "object_id": lbl_id,
                        "category": cat,
                        "severity": "CRITICAL",
                        "error_type": "BOX_INVALID_CATEGORY",
                        "qa_layer": "Tầng 1 (Luật)",
                        "component": "Ontology Class",
                        "models_flagged": ["Rule-Engine", "BDD-Ontology"],
                        "explanation": f"Class '{cat}' nằm ngoài danh mục 10 lớp chuẩn BDD100K. {cit}",
                        "action": "Đổi sang class hợp lệ trong ontology",
                        "auto_healable": False
                    }
                    frame_errors.append(err)
                    layer_stats["layer_1_rules"] += 1
                    rule_telemetry["BOX_INVALID_CATEGORY"] = rule_telemetry.get("BOX_INVALID_CATEGORY", 0) + 1

                if not b:
                    continue

                try:
                    x1, y1 = float(b.get("x1", 0)), float(b.get("y1", 0))
                    x2, y2 = float(b.get("x2", 0)), float(b.get("y2", 0))
                except (ValueError, TypeError):
                    continue

                box_coords = (x1, y1, x2, y2)
                w = x2 - x1
                h = y2 - y1
                area = w * h

                # 1.2 Inverted Box (Tọa độ ngược)
                if x2 <= x1 or y2 <= y1:
                    err = {
                        "object_id": lbl_id,
                        "category": cat,
                        "severity": "CRITICAL",
                        "error_type": "BOX_INVERTED",
                        "qa_layer": "Tầng 1 (Luật)",
                        "component": "Bbox Geometry",
                        "models_flagged": ["Rule-Engine"],
                        "explanation": f"Box bị ngược tọa độ: x1={x1:.1f} >= x2={x2:.1f} hoặc y1={y1:.1f} >= y2={y2:.1f}.",
                        "action": "Tự đảo lại tọa độ",
                        "auto_healable": True
                    }
                    frame_errors.append(err)
                    layer_stats["layer_1_rules"] += 1
                    rule_telemetry["BOX_INVERTED"] = rule_telemetry.get("BOX_INVERTED", 0) + 1

                # 1.3 Box tràn ngoài ảnh 1280x720
                elif x1 < 0 or y1 < 0 or x2 > cfg["image_width"] or y2 > cfg["image_height"]:
                    cit = self.guideline_manager.format_citation("out_of_image")
                    err = {
                        "object_id": lbl_id,
                        "category": cat,
                        "severity": "CRITICAL",
                        "error_type": "BOX_OUT_OF_IMAGE",
                        "qa_layer": "Tầng 1 (Luật)",
                        "component": "Bbox Geometry",
                        "models_flagged": ["Rule-Engine"],
                        "explanation": f"Box tràn ngoài biên ảnh [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}].",
                        "guideline_rule": cit,
                        "action": "Tự động kẹp biên (Auto-Clamp)",
                        "auto_healable": True
                    }
                    frame_errors.append(err)
                    layer_stats["layer_1_rules"] += 1
                    rule_telemetry["BOX_OUT_OF_IMAGE"] = rule_telemetry.get("BOX_OUT_OF_IMAGE", 0) + 1

                # 1.4 Box quá nhỏ / chấm rác diện tích < 16px²
                elif w < cfg["min_box_width"] or h < cfg["min_box_height"] or area < cfg["min_box_area"]:
                    cit = self.guideline_manager.format_citation("too_small")
                    err = {
                        "object_id": lbl_id,
                        "category": cat,
                        "severity": "CRITICAL",
                        "error_type": "BOX_TOO_SMALL",
                        "qa_layer": "Tầng 1 (Luật)",
                        "component": "Bbox Geometry",
                        "models_flagged": ["Rule-Engine"],
                        "explanation": f"Box quá nhỏ / chấm rác: {w:.0f}x{h:.0f}px (diện tích={area:.0f}px²).",
                        "guideline_rule": cit,
                        "action": "Xóa chấm rác (Auto-Delete)",
                        "auto_healable": True
                    }
                    frame_errors.append(err)
                    layer_stats["layer_1_rules"] += 1
                    rule_telemetry["BOX_TOO_SMALL"] = rule_telemetry.get("BOX_TOO_SMALL", 0) + 1

                else:
                    valid_boxes_for_dup.append((lbl_id, canonical_cat, box_coords))

                    # 1.5 Kiểm tra tỉ lệ khung vô lý theo class (Slide 30: Aspect ratio check)
                    aspect_ratio = w / max(0.1, h)
                    ar_range = CLASS_ASPECT_RATIO_RANGES.get(canonical_cat)
                    if ar_range and (aspect_ratio < ar_range[0] or aspect_ratio > ar_range[1]):
                        err = {
                            "object_id": lbl_id,
                            "category": cat,
                            "severity": "HIGH",
                            "error_type": "BOX_UNREASONABLE_ASPECT_RATIO",
                            "qa_layer": "Tầng 1 (Luật)",
                            "component": "Shape Geometry",
                            "models_flagged": ["Rule-Engine", "Aspect-Ratio-Guard"],
                            "explanation": f"Tỉ lệ khung hình (w/h = {aspect_ratio:.2f}) nằm ngoài phạm vi heuristic ứng dụng cho class '{canonical_cat}' ({ar_range[0]}-{ar_range[1]}); cần reviewer đối chiếu guideline dự án.",
                            "action": "Kiểm tra lại hình dáng và loại xe",
                            "auto_healable": False
                        }
                        frame_errors.append(err)
                        layer_stats["layer_1_rules"] += 1
                        rule_telemetry["BOX_UNREASONABLE_ASPECT_RATIO"] = rule_telemetry.get("BOX_UNREASONABLE_ASPECT_RATIO", 0) + 1

                    # 1.6 Kiểm tra sát mép ảnh mà quên đánh dấu truncated (Slide 30)
                    near_border = (x1 <= 2.0 or y1 <= 2.0 or x2 >= (cfg["image_width"] - 2.0) or y2 >= (cfg["image_height"] - 2.0))
                    if near_border and not attrs.get("truncated", False):
                        err = {
                            "object_id": lbl_id,
                            "category": cat,
                            "severity": "MEDIUM",
                            "error_type": "BOX_NEAR_BOUNDARY_UNMARKED",
                            "qa_layer": "Tầng 1 (Luật)",
                            "component": "Attributes",
                            "models_flagged": ["Rule-Engine", "Boundary-Guard"],
                            "explanation": f"Box nằm sát mép ảnh (cách mép <= 2px) nhưng thuộc tính 'truncated' chưa được bật.",
                            "action": "Bật thuộc tính truncated=True",
                            "auto_healable": True
                        }
                        frame_errors.append(err)
                        layer_stats["layer_1_rules"] += 1
                        rule_telemetry["BOX_NEAR_BOUNDARY_UNMARKED"] = rule_telemetry.get("BOX_NEAR_BOUNDARY_UNMARKED", 0) + 1

                # 1.7 Thiếu thuộc tính bắt buộc (occluded, truncated)
                for req_attr in cfg.get("mandatory_attributes", []):
                    if req_attr not in attrs or attrs[req_attr] is None or attrs[req_attr] == "":
                        cit = self.guideline_manager.format_citation("attribute")
                        err = {
                            "object_id": lbl_id,
                            "category": cat,
                            "severity": "HIGH",
                            "error_type": "MISSING_MANDATORY_ATTRIBUTE",
                            "qa_layer": "Tầng 1 (Luật)",
                            "component": "Metadata",
                            "models_flagged": ["Rule-Engine"],
                            "explanation": f"Thiếu trường thuộc tính bắt buộc '{req_attr}'.",
                            "guideline_rule": cit,
                            "action": f"Bổ sung thuộc tính {req_attr}",
                            "auto_healable": True
                        }
                        frame_errors.append(err)
                        layer_stats["layer_1_rules"] += 1
                        rule_telemetry["MISSING_MANDATORY_ATTRIBUTE"] = rule_telemetry.get("MISSING_MANDATORY_ATTRIBUTE", 0) + 1

                # 1.8 Kiểm tra đa giác Segmentation (Polygon self-intersection & degenerate - Slide 30)
                poly_boxes = []
                if "poly2d" in label and isinstance(label["poly2d"], list):
                    for p_idx, poly_entry in enumerate(label["poly2d"]):
                        raw_verts = poly_entry.get("vertices", [])
                        vertices = [[float(pt[0]), float(pt[1])] for pt in raw_verts if len(pt) >= 2]
                        if not vertices:
                            continue

                        is_bad_poly, reason = check_polygon_validity(vertices)
                        if is_bad_poly:
                            cit = self.guideline_manager.format_citation("polygon")
                            err = {
                                "object_id": lbl_id,
                                "category": cat,
                                "severity": "CRITICAL",
                                "error_type": "POLYGON_SELF_INTERSECT",
                                "qa_layer": "Tầng 1 (Luật)",
                                "component": "Segmentation Polygon",
                                "models_flagged": ["Rule-Engine", "Shapely-Geometry"],
                                "explanation": f"Đa giác #{p_idx} lỗi hình học: {reason}.",
                                "guideline_rule": cit,
                                "action": "Vẽ lại đa giác polygon không để tự cắt",
                                "auto_healable": False
                            }
                            frame_errors.append(err)
                            layer_stats["layer_1_rules"] += 1
                            rule_telemetry["POLYGON_SELF_INTERSECT"] = rule_telemetry.get("POLYGON_SELF_INTERSECT", 0) + 1
                        else:
                            poly_boxes.append(get_polygon_bbox(vertices))

                # 1.9 Mask / Polygon lệch Bbox (Slide 30: Mask drift & IoU)
                if box_coords and poly_boxes:
                    p_min_x = min(pb[0] for pb in poly_boxes)
                    p_min_y = min(pb[1] for pb in poly_boxes)
                    p_max_x = max(pb[2] for pb in poly_boxes)
                    p_max_y = max(pb[3] for pb in poly_boxes)
                    poly_union_box = (p_min_x, p_min_y, p_max_x, p_max_y)

                    mask_iou = compute_iou(box_coords, poly_union_box)
                    drift_dist = max(
                        abs(box_coords[0] - p_min_x),
                        abs(box_coords[1] - p_min_y),
                        abs(box_coords[2] - p_max_x),
                        abs(box_coords[3] - p_max_y)
                    )

                    if mask_iou < cfg["mask_box_min_iou"] or drift_dist > cfg["max_mask_box_drift_px"]:
                        cit = self.guideline_manager.format_citation("mask")
                        err = {
                            "object_id": lbl_id,
                            "category": cat,
                            "severity": "CRITICAL",
                            "error_type": "MASK_BOX_MISMATCH",
                            "qa_layer": "Tầng 1 (Luật)",
                            "component": "Mask & Bbox Sync",
                            "models_flagged": ["Rule-Engine"],
                            "explanation": f"Mask lệch Bbox: IoU={mask_iou:.2f}, lệch={drift_dist:.0f}px.",
                            "guideline_rule": cit,
                            "action": "Đồng bộ lại Bbox và Mask",
                            "auto_healable": False
                        }
                        frame_errors.append(err)
                        layer_stats["layer_1_rules"] += 1
                        rule_telemetry["MASK_BOX_MISMATCH"] = rule_telemetry.get("MASK_BOX_MISMATCH", 0) + 1

            # 1.10 Box trùng cùng Class (Slide 30: IoU >= duplicate_threshold)
            num_vboxes = len(valid_boxes_for_dup)
            for i in range(num_vboxes):
                ida, cata, boxa = valid_boxes_for_dup[i]
                for j in range(i + 1, num_vboxes):
                    idb, catb, boxb = valid_boxes_for_dup[j]
                    if cata == catb:
                        dup_iou = compute_iou(boxa, boxb)
                        if dup_iou >= cfg["duplicate_box_iou_threshold"]:
                            err = {
                                "object_id": f"{ida} & {idb}",
                                "category": cata,
                                "severity": "CRITICAL",
                                "error_type": "BOX_DUPLICATE_SAME_CLASS",
                                "qa_layer": "Tầng 1 (Luật)",
                                "component": "Bbox Duplicate",
                                "models_flagged": ["Rule-Engine"],
                                "explanation": f"Hai box cùng class '{cata}' bị vẽ đè trùng lặp (IoU={dup_iou:.2f} >= ngưỡng {cfg['duplicate_box_iou_threshold']:.2f}).",
                                "action": "Xóa bớt 1 box trùng",
                                "auto_healable": True
                            }
                            frame_errors.append(err)
                            layer_stats["layer_1_rules"] += 1
                            rule_telemetry["BOX_DUPLICATE_SAME_CLASS"] = rule_telemetry.get("BOX_DUPLICATE_SAME_CLASS", 0) + 1

            # -----------------------------------------------------------------
            # TẦNG 2: PHÁT HIỆN LỖI BẰNG MODEL (OBJECTLAB / CLEANLAB)
            # -----------------------------------------------------------------
            # Chạy model inference thật khi có ảnh cục bộ hoặc ảnh upload
            model_preds = self.detector.detect_frame(
                frame_name,
                frame.get("image_bytes"),
                conf_threshold=float(cfg.get("model_threshold", 0.35))
            )

            if model_preds:
                gt_labeled = []
                for lbl_idx, lbl in enumerate(labels):
                    b = lbl.get("box2d")
                    if b:
                        try:
                            x1, y1 = float(b.get("x1", 0)), float(b.get("y1", 0))
                            x2, y2 = float(b.get("x2", 0)), float(b.get("y2", 0))
                            gt_cat = str(lbl.get("category", "")).strip().lower()
                            canonical_gt = BDD100K_CLASS_ALIASES.get(gt_cat, gt_cat)
                            gt_labeled.append({
                                "id": str(lbl.get("id", f"obj_{lbl_idx}")),
                                "category": canonical_gt,
                                "raw_category": gt_cat,
                                "box": (x1, y1, x2, y2)
                            })
                        except (ValueError, TypeError):
                            pass

                matched_model_indices = set()

                # 2.1 Kiểm tra Swapped Class (Nghi nhầm class)
                for gt_obj in gt_labeled:
                    best_iou = 0.0
                    best_pred = None
                    best_pred_idx = -1
                    for p_idx, pred in enumerate(model_preds):
                        p_box = tuple(pred.get("points", [0, 0, 0, 0]))
                        iou = compute_iou(gt_obj["box"], p_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = pred
                            best_pred_idx = p_idx

                    if best_pred and best_iou >= 0.40 and best_pred.get("confidence", 0) >= 0.50:
                        matched_model_indices.add(best_pred_idx)
                        pred_cat = str(best_pred.get("label", "")).lower()
                        canonical_pred = BDD100K_CLASS_ALIASES.get(pred_cat, pred_cat)
                        if canonical_pred != gt_obj["category"] and canonical_pred in BDD100K_STANDARD_CLASSES:
                            cit = self.guideline_manager.format_citation("class", gt_obj["category"])
                            conf_pct = int(best_pred.get("confidence", 0) * 100)
                            err = {
                                "object_id": gt_obj["id"],
                                "category": gt_obj["category"],
                                "severity": "HIGH",
                                "error_type": "CLASS_CONFUSION",
                                "qa_layer": "Tầng 2 (Model YOLO26)",
                                "component": "Bbox Detector (YOLO26)",
                                "models_flagged": ["Ultralytics YOLO26"],
                                "explanation": f"Nghi vấn nhầm class: Annotator gán '{gt_obj['category']}', nhưng YOLO26 nhận diện '{canonical_pred}' ({conf_pct}%, IoU={best_iou:.2f}). {cit}",
                                "action": f"Kiểm tra lại xem là '{gt_obj['category']}' hay '{canonical_pred}'",
                                "auto_healable": False,
                                "suggested_box": list(best_pred.get("points", gt_obj["box"])),
                                "guideline_rule": cit
                            }
                            frame_errors.append(err)
                            layer_stats["layer_2_models"] += 1

                # 2.2 Bỏ sót vật thể (Overlooked Object)
                for p_idx, pred in enumerate(model_preds):
                    if p_idx in matched_model_indices:
                        continue
                    pred_conf = pred.get("confidence", 0)
                    if pred_conf < 0.45:
                        continue
                    p_box = tuple(pred.get("points", [0, 0, 0, 0]))
                    max_iou = max([compute_iou(p_box, gt["box"]) for gt in gt_labeled], default=0.0)
                    if max_iou < 0.20:
                        pred_cat = str(pred.get("label", "car")).lower()
                        canonical_pred = BDD100K_CLASS_ALIASES.get(pred_cat, pred_cat)
                        if canonical_pred in BDD100K_STANDARD_CLASSES:
                            cit = self.guideline_manager.format_citation("class", canonical_pred)
                            conf_pct = int(pred_conf * 100)
                            err = {
                                "object_id": f"MISSING_{p_idx+1}",
                                "category": canonical_pred,
                                "severity": "CRITICAL" if pred_conf >= 0.60 else "HIGH",
                                "error_type": "POTENTIAL_MISSED_OBJECT",
                                "qa_layer": "Tầng 2 (Model YOLO26)",
                                "component": "Bbox Detector (YOLO26)",
                                "models_flagged": ["Ultralytics YOLO26"],
                                "explanation": f"Vật thể bị bỏ sót: YOLO26 phát hiện '{canonical_pred}' ({conf_pct}%) tại [{p_box[0]:.0f}, {p_box[1]:.0f}, {p_box[2]:.0f}, {p_box[3]:.0f}] nhưng chưa được gán nhãn. {cit}",
                                "action": f"Kiểm tra và bổ sung nhãn cho '{canonical_pred}'",
                                "auto_healable": False,
                                "suggested_box": list(p_box),
                                "guideline_rule": cit
                            }
                            frame_errors.append(err)
                            layer_stats["layer_2_models"] += 1

            # -----------------------------------------------------------------
            # TẦNG 3: REVIEW CÓ ƯU TIÊN (PRIORITIZED SEVERITY RANKING - Slide 34)
            # -----------------------------------------------------------------
            has_critical = any(e["severity"] == "CRITICAL" for e in frame_errors)
            has_high = any(e["severity"] == "HIGH" for e in frame_errors)
            has_medium = any(e["severity"] == "MEDIUM" for e in frame_errors)

            if has_critical:
                priority_rank = 1
                priority_label = "CỰC CAO"
                recommendation = "REVIEW REQUIRED — CRITICAL RULE FLAG"
                badge_class = "badge-danger"
                critical_count += 1
                need_review_count += 1
            elif has_high:
                priority_rank = 2
                priority_label = "CAO"
                recommendation = "REVIEW REQUIRED — HIGH RULE FLAG"
                badge_class = "badge-warning"
                need_review_count += 1
            elif has_medium:
                priority_rank = 3
                priority_label = "VỪA"
                recommendation = "REVIEW REQUIRED — MEDIUM RULE FLAG"
                badge_class = "badge-info"
                need_review_count += 1
            else:
                priority_rank = 4
                priority_label = "KHÔNG CÓ CỜ RULE"
                recommendation = "NO CANDIDATE FINDINGS FROM ENABLED CHECKS — NOT A VERIFIED PASS"
                badge_class = "badge-success"
                auto_pass_count += 1

            total_errors += len(frame_errors)
            layer_stats["layer_3_prioritized"] += 1

            all_models = list({m for e in frame_errors for m in e.get("models_flagged", [])})
            for e in frame_errors:
                if "guideline_rule" not in e:
                    e["guideline_rule"] = e.get("explanation", "")

            # Direction 1: ObjectLab Triad Classification & Direction 2: Label Quality Score (LQS)
            counts = {"missing_object": 0, "class_confusion": 0, "boundary_error": 0}
            for e in frame_errors:
                etype = str(e.get("error_type", "")).upper()
                if "UNLABELED" in etype or "MISSING" in etype or "NOT_LABELED" in etype:
                    olab = "MISSING_OBJECT"
                    counts["missing_object"] += 1
                elif "CLASS" in etype or "CATEGORY" in etype or "MISMATCH" in etype or "CONFUSION" in etype:
                    olab = "CLASS_CONFUSION"
                    counts["class_confusion"] += 1
                else:
                    olab = "BOUNDARY_ERROR"
                    counts["boundary_error"] += 1
                e["objectlab_category"] = olab

            penalty = (counts["missing_object"] * 25.0) + (counts["class_confusion"] * 15.0) + (counts["boundary_error"] * 8.0)
            lqs = max(0.0, round(100.0 - penalty, 1))

            evaluated_frames.append({
                "frame_name": frame_name,
                "frame_index": frame_idx,
                "lqs": lqs,
                "lqs_status": "CRITICAL_REVIEW" if lqs < 50 else "REVIEW_RECOMMENDED" if lqs < 80 else "CLEAN_CANDIDATE",
                "objectlab_counts": counts,
                "priority_rank": priority_rank,
                "priority_label": priority_label,
                "priority_level": priority_label,
                "recommendation": f"LQS: {lqs:.1f}/100.0 - {recommendation}",
                "badge_class": badge_class,
                "error_count": len(frame_errors),
                "errors": frame_errors,
                "flagged_models": all_models,
                "labels": labels,
                "consensus_score": None
            })

        # SẮP XẾP HÀNG ĐỢI REVIEW: Ưu tiên LQS thấp nhất (lỗi nặng nhất) lên đầu
        evaluated_frames.sort(key=lambda x: (x["lqs"], x["priority_rank"], -x["error_count"]))

        total_frames = len(dataset)
        no_rule_findings_pct = round((auto_pass_count / total_frames) * 100, 1) if total_frames > 0 else 0.0
        defect_rate_pct = round((need_review_count / total_frames) * 100, 1) if total_frames > 0 else 0.0

        # No independent adjudicated gold set is provided by this analysis API.
        benchmark_metrics = self.calculate_review_prioritization_curve(evaluated_frames)

        summary = {
            "total_frames": total_frames,
            "total_objects": total_objects,
            "total_errors": total_errors,
            "mean_lqs": round(sum(f.get("lqs", 100.0) for f in evaluated_frames) / max(1, total_frames), 1),
            "objectlab_breakdown": {
                "missing_objects": sum(f.get("objectlab_counts", {}).get("missing_object", 0) for f in evaluated_frames),
                "class_confusions": sum(f.get("objectlab_counts", {}).get("class_confusion", 0) for f in evaluated_frames),
                "boundary_errors": sum(f.get("objectlab_counts", {}).get("boundary_error", 0) for f in evaluated_frames)
            },
            "auto_pass_count": auto_pass_count,
            "need_review_count": need_review_count,
            "critical_count": critical_count,
            "workload_reduction_pct": None,
            "defect_rate_pct": None,
            "no_rule_findings_pct": no_rule_findings_pct,
            "rule_flagged_frame_pct": defect_rate_pct,
            "job_verdict": "REVIEW_REQUIRED" if need_review_count else "NO_RULE_FINDINGS",
            "layer_stats": {
                **layer_stats,
                "layer_1_rules_count": layer_stats["layer_1_rules"],
                "layer_2_models_count": layer_stats["layer_2_models"],
                "layer_3_prioritized_frames": layer_stats["layer_3_prioritized"],
                "layer_4_honeypot_status": "Requires an independent adjudicated reference set"
            },
            "rule_telemetry": rule_telemetry,
            "benchmark": benchmark_metrics
        }

        return {
            "summary": summary,
            "sorted_frames": evaluated_frames,
            "guideline_citation_sample": self.guideline_manager.format_citation("general")
        }

    def calculate_review_prioritization_curve(self, evaluated_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Return unavailable until the run is evaluated against independent,
        adjudicated frame-level ground truth. Counting this engine's own flags
        as defects makes Recall@20 tautological and invalid.
        """
        total_frames = len(evaluated_frames)
        return {
            "status": "unavailable_without_independent_gold",
            "reason": "Recall@20 requires adjudicated defect labels independent of this QC run.",
            "total_frames": total_frames,
            "review_top_20_frames": math.ceil(total_frames * 0.20) if total_frames else 0,
            "total_defective_frames": None,
            "recall_at_20_pct": None,
            "random_recall_at_20_pct": None,
            "box_count_recall_at_20_pct": None,
            "prioritized_recall_at_20": None,
            "random_baseline_recall_at_20": None,
            "box_count_baseline_recall_at_20": None,
            "meets_slide34_target": None
        }

    def auto_heal_dataset(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        1-Click Auto-Heal (Slide 29, 30):
        Tự động sửa lỗi hình học, kẹp biên, xóa rác, xóa trùng và điền thuộc tính.
        """
        healed_dataset = []
        stats = {
            "boxes_clamped": 0,
            "trash_boxes_removed": 0,
            "inverted_boxes_fixed": 0,
            "duplicate_boxes_removed": 0,
            "attributes_healed": 0
        }

        for frame in dataset:
            frame_copy = dict(frame)
            labels = frame.get("labels", [])
            clean_labels = []
            seen_boxes_by_class: Dict[str, List[Tuple[float, float, float, float]]] = {}

            for lbl in labels:
                b = lbl.get("box2d")
                if not b:
                    clean_labels.append(lbl)
                    continue

                try:
                    x1 = float(b.get("x1", 0))
                    y1 = float(b.get("y1", 0))
                    x2 = float(b.get("x2", 0))
                    y2 = float(b.get("y2", 0))
                except (ValueError, TypeError):
                    continue

                # 1. Bỏ qua box chấm rác
                if (x2 - x1) < 4.0 or (y2 - y1) < 4.0 or (x2 - x1) * (y2 - y1) < 16.0:
                    stats["trash_boxes_removed"] += 1
                    continue

                # 2. Đảo lại box ngược tọa độ
                if x2 < x1:
                    x1, x2 = x2, x1
                    stats["inverted_boxes_fixed"] += 1
                if y2 < y1:
                    y1, y2 = y2, y1
                    stats["inverted_boxes_fixed"] += 1

                # 3. Kẹp biên theo kích thước ảnh thực tế của frame
                frame_w = float(frame.get("width") or 1280.0)
                frame_h = float(frame.get("height") or 720.0)
                clamped = False
                if x1 < 0:
                    x1 = 0.0
                    clamped = True
                if y1 < 0:
                    y1 = 0.0
                    clamped = True
                if x2 > frame_w:
                    x2 = frame_w
                    clamped = True
                if y2 > frame_h:
                    y2 = frame_h
                    clamped = True

                if clamped:
                    stats["boxes_clamped"] += 1

                # 4. Loại bỏ box trùng cùng class
                cat = str(lbl.get("category", "car")).lower()
                curr_box = (x1, y1, x2, y2)
                is_dup = False
                if cat in seen_boxes_by_class:
                    for prev_box in seen_boxes_by_class[cat]:
                        if compute_iou(curr_box, prev_box) >= 0.85:
                            is_dup = True
                            break

                if is_dup:
                    stats["duplicate_boxes_removed"] += 1
                    continue

                if cat not in seen_boxes_by_class:
                    seen_boxes_by_class[cat] = []
                seen_boxes_by_class[cat].append(curr_box)

                # 5. Điền thuộc tính bắt buộc nếu thiếu
                new_lbl = dict(lbl)
                attrs = dict(new_lbl.get("attributes", {}))
                healed_attr = False
                if "occluded" not in attrs:
                    attrs["occluded"] = False
                    healed_attr = True
                if "truncated" not in attrs:
                    attrs["truncated"] = (x1 <= 2.0 or y1 <= 2.0 or x2 >= 1278.0 or y2 >= 718.0)
                    healed_attr = True
                if healed_attr:
                    stats["attributes_healed"] += 1
                new_lbl["attributes"] = attrs

                new_lbl["box2d"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                clean_labels.append(new_lbl)

            frame_copy["labels"] = clean_labels
            healed_dataset.append(frame_copy)

        return healed_dataset, stats

    def generate_reviewer_feedback(self, evaluated_frames: List[Dict[str, Any]], annotator_name: str = "Annotator") -> str:
        """Export rule/model candidate findings for human review; never decide job acceptance."""
        candidates = [f for f in evaluated_frames if f.get("errors")]
        total = len(evaluated_frames)

        md = [
            "# Annotation QC candidate findings",
            f"- **Annotator:** {annotator_name}",
            "- **Decision status:** REVIEW REQUIRED — this report is not an acceptance or rejection decision.",
            f"- **Frames with candidate findings:** {len(candidates)}/{total}",
            "- Confirm, correct, or dismiss each finding against the approved project guideline before requesting annotation changes.",
            "",
            "### Candidate findings (prioritized)",
        ]

        for idx, f in enumerate(candidates[:15], 1):
            fname = f.get("frame_name", f.get("name", f"frame_{idx}"))
            p_label = f.get("priority_label", "MEDIUM")
            rec = f.get("recommendation", "Cần kiểm tra lại theo guideline")
            md.append(f"\n**{idx}. Frame: `{fname}`** [Mức độ: {p_label} - Khuyến nghị: {rec}]")
            for err in f.get("errors", []):
                models_str = ", ".join(err.get("models_flagged", [])) or "Rule Engine"
                err_type = err.get("error_type", "UNKNOWN_ERROR")
                expl = err.get("explanation", "")
                md.append(f"   • [{err_type}] {expl} (Phát hiện: {models_str})")

        if len(candidates) > 15:
            md.append(f"\n*(...and {len(candidates) - 15} more frames in the full findings export.)*")

        md.append("\nOnly findings confirmed by a reviewer should be sent to CVAT as annotation feedback.")
        return "\n".join(md)


# Global Engine Instance
global_multi_model_engine = MultiModelQCEngine()
