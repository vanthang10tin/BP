import io
import base64
import json
import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

import gemini_provider
from fastapi.testclient import TestClient
import app


class GeminiArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.fastapi_app)

    def test_api_key_discovery_from_environ(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTestKeyFromEnv"}):
            self.assertEqual(gemini_provider._api_key(), "AIzaSyTestKeyFromEnv")
            self.assertTrue(gemini_provider.configured())

    def test_endpoint_returns_400_when_gemini_not_configured(self):
        with patch.object(gemini_provider, "configured", return_value=False), \
             patch.object(app, "gemini_configured", return_value=False):
            res = self.client.post("/api/qc/gemini-inspect-frame", json={"frame_name": "test.jpg"})
            self.assertEqual(res.status_code, 400)
            self.assertIn("chưa được cấu hình", res.json()["detail"])

    def test_endpoint_returns_404_when_image_missing(self):
        with patch.object(gemini_provider, "configured", return_value=True), \
             patch.object(app, "gemini_configured", return_value=True), \
             patch.object(app, "find_image_file_path", return_value=None):
            res = self.client.post("/api/qc/gemini-inspect-frame", json={"frame_name": "non_existent_image.jpg"})
            self.assertEqual(res.status_code, 404)
            self.assertIn("Không tìm thấy dữ liệu ảnh", res.json()["detail"])

    def test_arbitration_prompt_includes_model_predictions_and_guideline(self):
        fake_response_data = {
            "candidates": [
                {
                    "class_name": "bus",
                    "box_2d": [100, 200, 500, 600],
                    "confidence": 0.92,
                    "discrepancy_type": "LABEL_CONFLICT_WITH_MODEL",
                    "model_status": "Model detector nhận diện là bus với độ tin cậy 0.88",
                    "label_status": "Annotator gán nhầm là truck",
                    "guideline_citation": "§2.1 / QĐ-018: Phân biệt bus chở khách có cửa kính với truck thùng kín",
                    "evidence": "Đối tượng là xe khách 29 chỗ có cửa kính bên, Annotator gán nhầm truck."
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{"text": json.dumps(fake_response_data)}]
                }
            }]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.object(gemini_provider, "_api_key", return_value="fake_key"), \
             patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:

            dummy_img = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # JPEG header
            labels = [{"class": "truck", "box": [200, 100, 600, 500]}]
            model_preds = [{"class": "bus", "box": [195, 102, 605, 498], "confidence": 0.88}]

            proposals = gemini_provider.inspect_frame(
                image_data=dummy_img,
                guideline="QĐ-018: Bus vs Truck",
                labels=labels,
                classes=["car", "truck", "bus"],
                model_predictions=model_preds
            )

            self.assertEqual(len(proposals), 1)
            p = proposals[0]
            self.assertEqual(p["category"], "bus")
            self.assertEqual(p["discrepancy_type"], "LABEL_CONFLICT_WITH_MODEL")
            self.assertIn("bus", p["model_status"])
            self.assertIn("truck", p["label_status"])

            # Verify that urllib was called with the model predictions in the payload prompt
            call_args = mock_urlopen.call_args[0][0]
            req_body = json.loads(call_args.data.decode("utf-8"))
            prompt_text = req_body["contents"][0]["parts"][0]["text"]
            self.assertIn("MODEL_DETECTED_BUT_UNLABELED", prompt_text)
            self.assertIn("LABEL_CONFLICT_WITH_MODEL", prompt_text)
            self.assertIn("OUT_OF_MODEL_VOCAB_GUIDELINE_OBJECT", prompt_text)
            self.assertIn("1 detections", prompt_text)

    def test_arbitrate_dispute_annotator_correct(self):
        fake_response = {
            "is_violation": False,
            "verdict": "ANNOTATOR_CORRECT",
            "confirmed_class": "car",
            "explanation": "Xe con 4 chỗ gán car là hoàn toàn chính xác.",
            "action": "Giữ nguyên nhãn car.",
            "confidence": 0.95
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [{"text": json.dumps(fake_response)}]}}]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.object(gemini_provider, "_api_key", return_value="fake_key"), \
             patch("urllib.request.urlopen", return_value=mock_response):
            res = gemini_provider.arbitrate_dispute(
                image_data=b"fake_bytes",
                object_info={"category": "car", "box": [10, 10, 100, 100]},
                model_a_info={"model": "YOLO26", "category": "car", "confidence": 0.9},
                model_b_info={"model": "DINOv3", "category": "truck", "confidence": 0.8}
            )
            self.assertFalse(res["is_violation"])
            self.assertEqual(res["verdict"], "ANNOTATOR_CORRECT")
            self.assertEqual(res["confirmed_class"], "car")

    def test_dual_model_consensus_creates_council_issue(self):
        from qc_pipeline import run_bbox_pipeline
        dummy_data = [{
            "name": "f1.jpg",
            "labels": [{"id": "1", "category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 200, "y2": 200}}]
        }]
        # Both YOLO26 and DINOv3 say it's truck
        mock_predictions = [
            {"type": "rectangle", "label": "truck", "points": [50, 50, 200, 200], "confidence": 0.92, "model": "YOLO26"},
            {"type": "rectangle", "label": "truck", "points": [50, 50, 200, 200], "confidence": 0.88, "model": "DINOv3"}
        ]
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(buf, format="JPEG")
        valid_jpeg_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        with patch("qc_pipeline.infer_bbox", return_value=(mock_predictions, {"name": "Ensemble", "id": "bbox"})):
            res = run_bbox_pipeline(dummy_data, {"f1.jpg": valid_jpeg_b64}, {"classes": ["car", "truck"]}, None)
            frame = res["sorted_frames"][0]
            council_issues = [e for e in frame["errors"] if "Hội đồng" in e.get("qa_layer", "")]
            self.assertTrue(len(council_issues) > 0)
            self.assertIn("YOLO26", council_issues[0]["models_flagged"])
            self.assertIn("DINOv3", council_issues[0]["models_flagged"])


if __name__ == "__main__":
    unittest.main()
