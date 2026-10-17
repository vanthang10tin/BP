#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDD100K 2D quality-control engine.
Deterministic rules are active; legacy simulated model proposals are disabled by
default. Human review and independent gold labels are required before reporting
model-error or annotator-quality metrics.
"""

import math
from typing import List, Dict, Any, Tuple, Optional

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from ai_engine.guideline_manager import global_guideline_manager
from ai_engine.models_ensemble import global_multi_model_engine, MultiModelQCEngine, compute_iou
from coordinate_utils import to_xyxy, coco_to_xyxy, yolo_to_xyxy, voc_to_xyxy, bdd_to_xyxy, xyxy_to_bdd

DEFAULT_CONFIG = {
    "image_width": 1280,
    "image_height": 720,
    "min_box_width": 4.0,
    "min_box_height": 4.0,
    "min_box_area": 16.0,
    "duplicate_box_iou_threshold": 0.85,
    "mandatory_attributes": [
        "occluded",
        "truncated"
    ],
    "mask_box_min_iou": 0.70,
    "max_mask_box_drift_px": 15.0
}


def _honeypot_box(label: Dict[str, Any]) -> Tuple[float, float, float, float]:
    box = label["box2d"]
    coords = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    if not all(math.isfinite(value) for value in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("Invalid honeypot box")
    return coords


def _valid_honeypot_label(label: Any) -> bool:
    if not isinstance(label, dict) or not isinstance(label.get("box2d"), dict) or not str(label.get("category", "")).strip():
        return False
    try:
        _honeypot_box(label)
        return True
    except (KeyError, TypeError, ValueError):
        return False


class BDDQualityChecker:
    """Bộ kiểm tra chất lượng BDD100K tích hợp Hội đồng Đa Mô hình & Phân luồng Review 4 Tầng."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, guideline_manager=None):
        self.cfg = DEFAULT_CONFIG.copy()
        if config:
            self.cfg.update(config)
        self.guideline_manager = guideline_manager or global_guideline_manager
        self.ai_engine = global_multi_model_engine if guideline_manager is None else MultiModelQCEngine(self.guideline_manager)

    def check_dataset_advanced(self, data: Any) -> Dict[str, Any]:
        """
        Phương thức kiểm tra toàn diện tích hợp Hội Đồng 5 Mô Hình & 4 Tầng QA.
        Trả về:
        - Số frame bị gắn cờ bởi luật hiện tại và breakdown theo rule
        - Danh sách Frame đã được SẮP XẾP TỪ CAO XUỐNG THẤP
        - Cột giải thích chi tiết lỗi, tên model và trích dẫn điều khoản guideline.
        - Recall@20 chỉ có thể đo sau khi có frame-level gold độc lập
        """
        if isinstance(data, list):
            frames = data
        elif isinstance(data, dict):
            frames = data.get("frames", [data])
        else:
            raise ValueError("Dữ liệu JSON không đúng cấu trúc danh sách BDD100K.")

        # Chạy qua Multi-Model Engine với config
        result = self.ai_engine.evaluate_dataset_with_consensus(frames, config=self.cfg)
        return result

    def check_dataset(self, data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Hàm tương thích ngược với API hiện tại, bổ sung các trường nâng cao và sắp xếp từ Cao xuống Thấp.
        """
        adv_result = self.check_dataset_advanced(data)
        summary = adv_result["summary"]
        sorted_frames = adv_result["sorted_frames"]

        all_errors = []
        enriched_frames = []

        for f in sorted_frames:
            f_copy = dict(f)
            f_copy["name"] = f["frame_name"]
            f_copy["has_error"] = f["error_count"] > 0
            enriched_frames.append(f_copy)
            all_errors.extend(f.get("errors", []))

        # Bổ sung các chỉ số thống kê
        summary["error_counts_by_type"] = {}
        for err in all_errors:
            t = err.get("error_type", "GENERAL")
            summary["error_counts_by_type"][t] = summary["error_counts_by_type"].get(t, 0) + 1

        summary["estimated_review_rounds_saved"] = None

        return all_errors, summary, enriched_frames

    def auto_heal(self, data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """1-Click Auto-Heal: Tự sửa lỗi hình học, kẹp biên, xóa trùng và xuất JSON sạch."""
        if isinstance(data, list):
            frames = data
        elif isinstance(data, dict):
            frames = data.get("frames", [data])
        else:
            raise ValueError("Dữ liệu JSON không hợp lệ.")

        return self.ai_engine.auto_heal_dataset(frames)

    def generate_feedback_report(self, sorted_frames: List[Dict[str, Any]], annotator: str = "Annotator") -> str:
        """Tạo đoạn text báo cáo trả job 1-click cho Reviewer."""
        return self.ai_engine.generate_reviewer_feedback(sorted_frames, annotator)

    def evaluate_honeypot(self, annotator_frames: List[Dict[str, Any]], ground_truth_frames: List[Dict[str, Any]],
                          annotator_name: str = "Annotator") -> Dict[str, Any]:
        """
        Tầng 4: Đánh giá chất lượng Annotator qua Ground Truth Job / Honeypot Frames (Slide 33).
        Tính toán:
        - Độ chính xác trên frame honeypot (Precision, Recall, F1)
        - Tỉ lệ bị reviewer trả lại
        - Điểm IoU trung bình
        - Đề xuất đào tạo lại và hiệu chỉnh
        """
        gt_map = {f.get("name", ""): f.get("labels", []) for f in ground_truth_frames if f.get("name")}
        matched_frames = 0
        total_gt_boxes = 0
        total_pred_boxes = 0
        true_positives = 0
        iou_sum = 0.0

        for frame in annotator_frames:
            fname = frame.get("name", "")
            if fname not in gt_map:
                continue
            matched_frames += 1
            raw_gt = gt_map[fname] if isinstance(gt_map[fname], list) else []
            raw_predictions = frame.get("labels", [])
            raw_predictions = raw_predictions if isinstance(raw_predictions, list) else []
            gt_labels = [label for label in raw_gt if _valid_honeypot_label(label)]
            pred_labels = [label for label in raw_predictions if _valid_honeypot_label(label)]

            total_gt_boxes += len(gt_labels)
            total_pred_boxes += len(pred_labels)

            # Global greedy one-to-one match: class must match and IoU >= 0.50.
            pairs = []
            for p_idx, prediction in enumerate(pred_labels):
                p_box = _honeypot_box(prediction)
                p_cat = str(prediction.get("category", "")).strip().casefold()
                for g_idx, truth in enumerate(gt_labels):
                    if p_cat != str(truth.get("category", "")).strip().casefold():
                        continue
                    iou = compute_iou(p_box, _honeypot_box(truth))
                    if iou >= 0.50:
                        pairs.append((iou, p_idx, g_idx))
            used_predictions, used_truth = set(), set()
            for iou, p_idx, g_idx in sorted(pairs, reverse=True):
                if p_idx in used_predictions or g_idx in used_truth:
                    continue
                used_predictions.add(p_idx)
                used_truth.add(g_idx)
                true_positives += 1
                iou_sum += iou

        precision = round((true_positives / max(1, total_pred_boxes)) * 100, 1)
        recall = round((true_positives / max(1, total_gt_boxes)) * 100, 1)
        f1 = round((2 * precision * recall / max(0.1, precision + recall)), 1)
        mean_iou = round((iou_sum / max(1, true_positives)) * 100, 1) if true_positives > 0 else 0.0

        passed = matched_frames > 0 and precision >= 90.0 and recall >= 85.0

        return {
            "annotator_name": annotator_name,
            "honeypot_frames_evaluated": matched_frames,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_iou": mean_iou,
            "gold_standard_status": "MEETS_CONFIGURED_THRESHOLDS" if passed else "BELOW_CONFIGURED_THRESHOLDS",
            "recommendation": "Metric trên tập đối chiếu được cung cấp; chỉ có ý nghĩa nếu tập này độc lập và đã được adjudicate. Ngưỡng hiện tại là 90% precision và 85% recall."
        }

