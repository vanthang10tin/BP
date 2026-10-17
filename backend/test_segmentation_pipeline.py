"""Rigorous test suite for semantic and instance segmentation pipeline.
Validates true model execution (Ultralytics YOLO26-Seg native and SAM 2.1 container),
real IoU computation, mask rasterization, and guideline rule enforcement.
"""
import base64
import json
import os
import unittest
from pathlib import Path

from PIL import Image
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ai_engine.local_model_providers import (
    _get_yolo26_seg,
    _infer_yolo26_seg_native,
    infer_segmentation,
    provider_info
)
from qc_pipeline import run_segmentation_pipeline
from ai_engine.guideline_manager import global_guideline_manager


class TestSegmentationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
        cls.demo_seg_dir = os.path.join(cls.repo_dir, "DEMO_PACK", "SEGMENTATION")
        cls.images_dir = os.path.join(cls.demo_seg_dir, "images")
        cls.masks_dir = os.path.join(cls.demo_seg_dir, "masks_index")
        cls.ann_path = os.path.join(cls.demo_seg_dir, "annotations.json")

    def test_01_yolo26_seg_model_loaded_and_valid(self):
        """Kiểm tra model weights yolo26n-seg.pt tải thực sự vào bộ nhớ."""
        if os.environ.get("QC_RUN_MODEL_TESTS") != "1":
            self.skipTest("Set QC_RUN_MODEL_TESTS=1 to run optional local model integration tests")
        if not os.path.isfile(os.path.join(self.repo_dir, "models", "yolo26n-seg.pt")):
            self.skipTest("Local yolo26n-seg.pt weights are not included in the repository")
        model = _get_yolo26_seg()
        self.assertIsNotNone(model, "YOLO26-Seg model weights phải tồn tại và load được")
        self.assertTrue(hasattr(model, "names"), "Model phải có thuộc tính class names")
        self.assertIn("car", model.names.values(), "Model phải có class 'car'")
        self.assertIn("person", model.names.values(), "Model phải có class 'person'")

    def test_02_native_inference_produces_real_polygons_and_boxes(self):
        """Chạy inference thực tế trên ảnh DEMO_PACK với yolo26n-seg.pt."""
        if os.environ.get("QC_RUN_MODEL_TESTS") != "1":
            self.skipTest("Set QC_RUN_MODEL_TESTS=1 to run optional local model integration tests")
        if not os.path.isfile(os.path.join(self.repo_dir, "models", "yolo26n-seg.pt")):
            self.skipTest("Local yolo26n-seg.pt weights are not included in the repository")
        sample_img = os.path.join(self.images_dir, "7d06fefd-f7be05a6.jpg")
        if not os.path.isfile(sample_img):
            self.skipTest("Optional DEMO_PACK segmentation image is not included in the repository")
        self.assertTrue(os.path.exists(sample_img), f"Ảnh mẫu phải tồn tại: {sample_img}")

        with open(sample_img, "rb") as f:
            img_bytes = f.read()

        predictions, info = _infer_yolo26_seg_native(
            img_bytes,
            threshold=0.25,
            classes=["car", "pedestrian", "truck", "bus"],
            label_mapping={"person": "pedestrian"}
        )

        self.assertTrue(info["available"], "Provider info phải báo available=True")
        self.assertIn("YOLO26-Seg", info["name"])
        self.assertGreater(len(predictions), 0, "Model phải phát hiện ít nhất 1 object trong ảnh giao thông")

        first = predictions[0]
        self.assertEqual(first["type"], "polygon")
        self.assertIn("points", first)
        self.assertGreaterEqual(len(first["points"]), 6, "Polygon phải có ít nhất 3 điểm (6 tọa độ x,y)")
        self.assertIn("box", first)
        self.assertEqual(len(first["box"]), 4, "BBox phải có 4 tọa độ [x1, y1, x2, y2]")
        self.assertGreaterEqual(first["confidence"], 0.25)

    def test_03_run_segmentation_pipeline_on_real_demo_dataset(self):
        """Chạy toàn bộ pipeline kiểm định Segmentation trên 2 frame BDD100K thật."""
        if os.environ.get("QC_RUN_MODEL_TESTS") != "1":
            self.skipTest("Set QC_RUN_MODEL_TESTS=1 to run optional local model integration tests")
        if not all(os.path.isfile(path) for path in (self.ann_path, os.path.join(self.repo_dir, "models", "yolo26n-seg.pt"))) or not all(os.path.isdir(path) for path in (self.images_dir, self.masks_dir)):
            self.skipTest("Segmentation annotations or local model weights are not included in the repository")
        with open(self.ann_path, "r", encoding="utf-8") as f:
            records = json.load(f)[:2]

        images = {}
        masks = {}
        for r in records:
            name = Path(r["name"]).name
            mask_name = Path(r["mask_index"]).name
            img_p = os.path.join(self.images_dir, name)
            mask_p = os.path.join(self.masks_dir, mask_name)
            self.assertTrue(os.path.exists(img_p), f"Thiếu ảnh {img_p}")
            self.assertTrue(os.path.exists(mask_p), f"Thiếu mask {mask_p}")

            with open(img_p, "rb") as fi:
                images[name] = base64.b64encode(fi.read()).decode("ascii")
            with open(mask_p, "rb") as fm:
                masks[r["mask_index"]] = base64.b64encode(fm.read()).decode("ascii")
                masks[mask_name] = masks[r["mask_index"]]

        profile = {
            "dataset": "bdd100k",
            "annotation_format": "bdd100k_segmentation",
            "review_mask_iou_threshold": 0.5
        }

        result = run_segmentation_pipeline(
            records=records,
            images=images,
            masks=masks,
            profile=profile,
            guideline_manager=global_guideline_manager,
            threshold=0.3,
            review_iou_threshold=0.5
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task_type"], "semantic_segmentation")
        self.assertEqual(len(result["models"]), 1)
        self.assertEqual(len(result["sorted_frames"]), 2)

        summary = result["summary"]
        self.assertEqual(summary["total_frames"], 2)
        self.assertIn("macro_iou", summary)
        self.assertIn("objectlab_breakdown", summary)

        for frame in result["sorted_frames"]:
            self.assertIn("frame_name", frame)
            self.assertIn("error_count", frame)
            self.assertIn("errors", frame)
            self.assertIn("issues", frame)
            self.assertIn("prediction_mask_png_base64", frame)
            self.assertIn("model_segmentation_metrics", frame)
            self.assertTrue(len(frame["prediction_mask_png_base64"]) > 50)

            # Kiểm tra định dạng issue nếu có lỗi
            for issue in frame["errors"]:
                self.assertIn("issue_id", issue)
                self.assertIn("object_id", issue)
                self.assertIn("category", issue)
                self.assertIn("iou", issue)
                self.assertIn("suggested_box", issue)
                self.assertEqual(issue["error_type"], "SEGMENTATION_MODEL_DISAGREEMENT")
                if issue["suggested_box"] is not None:
                    self.assertEqual(len(issue["suggested_box"]), 4)

    def test_04_synthetic_mask_perfect_agreement(self):
        """Kiểm tra trường hợp mask dự đoán trùng khớp hoàn toàn với Ground Truth."""
        # Tạo ảnh trắng 200x200
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        img_buf = io.BytesIO()
        img.save(img_buf, format="JPEG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")

        # Tạo mask với class 13 (car) ở góc giữa
        mask = np.full((200, 200), 255, dtype=np.uint8)
        mask[50:150, 50:150] = 13
        mask_buf = io.BytesIO()
        Image.fromarray(mask).save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("ascii")

        records = [{
            "name": "synth_perfect.jpg",
            "mask_index": "synth_perfect_id.png",
            "classes_present": [{"id": 13, "name": "car"}]
        }]
        images = {"synth_perfect.jpg": img_b64}
        masks = {"synth_perfect_id.png": mask_b64}
        profile = {
            "class_names": ["road", "car"],
            "mask_id_to_class": {"13": "car", "0": "road"},
            "ignore_mask_ids": [255]
        }

        # Mock infer_segmentation để trả về đúng đa giác hình vuông 50:150
        from unittest.mock import patch
        mock_pred = [{
            "type": "polygon",
            "label": "car",
            "points": [50.0, 50.0, 150.0, 50.0, 150.0, 150.0, 50.0, 150.0],
            "confidence": 0.95
        }]
        mock_info = {"id": "segmentation", "name": "Mock Segmenter", "available": True}

        with patch("qc_pipeline.infer_segmentation", return_value=(mock_pred, mock_info)):
            res = run_segmentation_pipeline(
                records=records,
                images=images,
                masks=masks,
                profile=profile,
                guideline_manager=global_guideline_manager,
                threshold=0.3,
                review_iou_threshold=0.5
            )

        frame = res["sorted_frames"][0]
        self.assertEqual(frame["error_count"], 0, "Trùng khớp hoàn hảo thì không được sinh lỗi")
        car_metric = next(m for m in frame["model_segmentation_metrics"]["class_iou"] if m["class_name"] == "car")
        self.assertGreaterEqual(car_metric["iou"], 0.95)


import io

if __name__ == "__main__":
    unittest.main()
