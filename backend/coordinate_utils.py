#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate Conventions & Transformation Utilities
Tuân thủ chuẩn Slide 37 & Slide 22 (VinUniversity Annotation & QC Curriculum):
- COCO Box: [x, y, w, h] (top-left, width, height)
- YOLO Box: [cx, cy, w, h] normalized in [0, 1]
- Pascal VOC: [xmin, ymin, xmax, ymax] 1-indexed pixel
- Gemini VLM: box_2d = [ymin, xmin, ymax, xmax] thang 0-1000
- BDD100K: {"x1": float, "y1": float, "x2": float, "y2": float} 0-indexed pixel
"""

from typing import Tuple, List, Dict, Any, Union, Optional


def to_xyxy(box_2d: List[Union[int, float]], W: float = 1280.0, H: float = 720.0) -> Tuple[float, float, float, float]:
    """
    Quy đổi tọa độ VLM Gemini (box_2d = [ymin, xmin, ymax, xmax], thang 0-1000)
    sang pixel chuẩn [xmin, ymin, xmax, ymax] theo công thức Slide 22:
        ymin, xmin, ymax, xmax = box_2d
        return [xmin*W/1000, ymin*H/1000, xmax*W/1000, ymax*H/1000]
    """
    if len(box_2d) != 4:
        raise ValueError(f"Gemini box_2d phải có 4 phần tử [ymin, xmin, ymax, xmax], nhận được: {box_2d}")
    ymin, xmin, ymax, xmax = box_2d
    return (
        float(xmin) * float(W) / 1000.0,
        float(ymin) * float(H) / 1000.0,
        float(xmax) * float(W) / 1000.0,
        float(ymax) * float(H) / 1000.0,
    )


def xyxy_to_gemini(box: Tuple[float, float, float, float], W: float = 1280.0, H: float = 720.0) -> List[int]:
    """Chuyển đổi [x1, y1, x2, y2] sang Gemini box_2d [ymin, xmin, ymax, xmax] thang 0-1000."""
    x1, y1, x2, y2 = box
    xmin = int(round(max(0.0, min(1.0, x1 / W)) * 1000.0))
    ymin = int(round(max(0.0, min(1.0, y1 / H)) * 1000.0))
    xmax = int(round(max(0.0, min(1.0, x2 / W)) * 1000.0))
    ymax = int(round(max(0.0, min(1.0, y2 / H)) * 1000.0))
    return [ymin, xmin, ymax, xmax]


def coco_to_xyxy(coco_box: List[float]) -> Tuple[float, float, float, float]:
    """
    COCO: [x, y, w, h] (gốc trái trên, bề rộng, chiều cao)
    -> [x1, y1, x2, y2]
    Tránh lỗi phổ biến nhầm COCO box với [x1, y1, x2, y2] (Slide 37).
    """
    x, y, w, h = coco_box
    return (float(x), float(y), float(x + w), float(y + h))


def xyxy_to_coco(box: Tuple[float, float, float, float]) -> List[float]:
    """[x1, y1, x2, y2] -> COCO [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def yolo_to_xyxy(yolo_box: List[float], W: float = 1280.0, H: float = 720.0) -> Tuple[float, float, float, float]:
    """
    YOLO: [cx, cy, w, h] chuẩn hóa 0-1
    -> [x1, y1, x2, y2] pixel.
    Tránh lỗi dùng sai kích thước ảnh gốc / resize (Slide 37).
    """
    cx, cy, w, h = yolo_box
    pixel_cx = cx * W
    pixel_cy = cy * H
    pixel_w = w * W
    pixel_h = h * H
    x1 = pixel_cx - pixel_w / 2.0
    y1 = pixel_cy - pixel_h / 2.0
    x2 = pixel_cx + pixel_w / 2.0
    y2 = pixel_cy + pixel_h / 2.0
    return (float(x1), float(y1), float(x2), float(y2))


def xyxy_to_yolo(box: Tuple[float, float, float, float], W: float = 1280.0, H: float = 720.0) -> List[float]:
    """[x1, y1, x2, y2] pixel -> YOLO [cx, cy, w, h] chuẩn hóa 0-1."""
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return [round(cx / W, 6), round(cy / H, 6), round(w / W, 6), round(h / H, 6)]


def voc_to_xyxy(voc_box: List[float]) -> Tuple[float, float, float, float]:
    """
    Pascal VOC: [xmin, ymin, xmax, ymax] 1-indexed
    -> 0-indexed [x1, y1, x2, y2] (Tránh lệch 1 pixel theo Slide 37).
    """
    xmin, ymin, xmax, ymax = voc_box
    return (float(xmin - 1.0), float(ymin - 1.0), float(xmax), float(ymax))


def xyxy_to_voc(box: Tuple[float, float, float, float]) -> List[float]:
    """0-indexed [x1, y1, x2, y2] -> Pascal VOC 1-indexed [xmin, ymin, xmax, ymax]."""
    x1, y1, x2, y2 = box
    return [float(x1 + 1.0), float(y1 + 1.0), float(x2), float(y2)]


def bdd_to_xyxy(box2d: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """BDD100K box2d dict {'x1', 'y1', 'x2', 'y2'} -> tuple (x1, y1, x2, y2)."""
    return (
        float(box2d.get("x1", 0.0)),
        float(box2d.get("y1", 0.0)),
        float(box2d.get("x2", 0.0)),
        float(box2d.get("y2", 0.0)),
    )


def xyxy_to_bdd(box: Tuple[float, float, float, float]) -> Dict[str, float]:
    """Tuple (x1, y1, x2, y2) -> BDD100K box2d dict {'x1', 'y1', 'x2', 'y2'}."""
    return {
        "x1": round(float(box[0]), 2),
        "y1": round(float(box[1]), 2),
        "x2": round(float(box[2]), 2),
        "y2": round(float(box[3]), 2),
    }


def find_image_path(filename: str) -> Optional[str]:
    """Tìm đường dẫn ảnh thực tế trên ổ đĩa từ tên file (hỗ trợ SSD cache & BDD100K archive)."""
    import os
    if not filename:
        return None
    clean_name = os.path.basename(filename)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
    candidates_dirs = [
        os.path.abspath(os.path.join(repo_dir, "demo_data", "images")),
        os.path.abspath(os.path.join(os.getcwd(), "demo_data", "images")),
        os.path.abspath(os.path.join(repo_dir, "demo_images")),
        os.path.abspath(os.path.join(repo_dir, ".qc-data", "images")),
        os.path.abspath(os.path.join(repo_dir, "DEMO_PACK", "BBOX", "images")),
        os.path.abspath(os.path.join(repo_dir, "DEMO_PACK", "SEGMENTATION", "images")),
        os.path.abspath(os.path.join(os.getcwd(), "DEMO_PACK", "BBOX", "images")),
        os.path.abspath(os.path.join(os.getcwd(), "DEMO_PACK", "SEGMENTATION", "images")),
        os.path.abspath(os.path.join(repo_dir, "..", "archive", "bdd100k", "bdd100k", "images", "100k", "val")),
        os.path.abspath(os.path.join(repo_dir, "..", "archive", "bdd100k", "bdd100k", "images", "10k", "val")),
    ]
    for sdir in candidates_dirs:
        if os.path.exists(sdir):
            target = os.path.join(sdir, clean_name)
            if os.path.exists(target) and os.path.isfile(target):
                return target
    return None
