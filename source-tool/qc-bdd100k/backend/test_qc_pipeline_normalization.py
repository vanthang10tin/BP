"""Unit checks for the supported annotation adapters; no dataset or model is loaded."""
import unittest

from qc_pipeline import normalize_bbox_dataset


class AnnotationNormalizationTests(unittest.TestCase):
    def test_coco_xywh_becomes_xyxy_and_keeps_image_dimensions(self):
        result = normalize_bbox_dataset(
            {"images": [{"id": 7, "file_name": "scene/a.jpg", "width": 640, "height": 480}],
             "categories": [{"id": 3, "name": "vehicle"}],
             "annotations": [{"id": 9, "image_id": 7, "category_id": 3, "bbox": [10, 20, 30, 40]}]},
            "coco", {})
        self.assertEqual(result[0]["name"], "a.jpg")
        self.assertEqual((result[0]["width"], result[0]["height"]), (640, 480))
        self.assertEqual(result[0]["labels"][0]["box2d"], {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0})

    def test_structured_yolo_json_uses_normalized_center_width_height(self):
        result = normalize_bbox_dataset(
            {"categories": [{"id": 0, "name": "car"}],
             "images": [{"name": "frame.jpg", "width": 1000, "height": 500,
                         "annotations": [{"class_id": 0, "bbox": [0.5, 0.5, 0.2, 0.4]}]}]},
            "yolo_json", {})
        self.assertEqual(result[0]["labels"][0]["box2d"], {"x1": 400.0, "y1": 150.0, "x2": 600.0, "y2": 350.0})

    def test_bdd_labels_preserve_pixels_and_safe_image_name(self):
        result = normalize_bbox_dataset(
            [{"name": "frame.jpg", "labels": [{"id": "x", "category": "car",
                                                  "box2d": {"x1": 2, "y1": 4, "x2": 12, "y2": 20}}]}],
            "bdd100k", {})
        self.assertEqual(result[0]["labels"][0]["box2d"]["x2"], 12)
        with self.assertRaises(ValueError):
            normalize_bbox_dataset([{"name": "../outside.jpg", "labels": []}], "bdd100k", {})


if __name__ == "__main__":
    unittest.main()
