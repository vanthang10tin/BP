"""Offline regressions for native model isolation and segmentation display."""
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image
from ai_engine import local_model_providers as providers
import qc_pipeline as qp


def encoded_image(size, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, 0).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_task_ontology_preserves_person_for_segmentation():
    assert providers._task_label_mapping(["person"])["person"] == "person"
    assert providers._task_label_mapping(["pedestrian"])["person"] == "pedestrian"
    assert providers._task_label_mapping(["person"], {"person": "human"})["person"] == "human"


def test_shared_native_predictor_is_serialized():
    active = 0
    peak = 0
    @providers._serialized_native
    def predict():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(.01)
        active -= 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: predict(), range(8)))
    assert peak == 1


def test_segmentation_returns_actual_canvas_dimensions(monkeypatch):
    monkeypatch.setattr(qp, "infer_segmentation", lambda *a, **kw: ([{
        "label": "person", "type": "polygon", "points": [0, 0, 31, 0, 31, 15, 0, 15],
        "model": "YOLO26-Seg", "confidence": .8}], {"name": "fixture"}))
    result = qp.run_segmentation_pipeline(
        [{"name": "frame.png", "mask_index": "mask.png"}],
        {"frame.png": encoded_image((32, 16))}, {"mask.png": encoded_image((32, 16), "L")},
        {"class_names": ["person"], "mask_id_to_class": {"0": "person"}}, None)
    frame = result["sorted_frames"][0]
    assert (frame["width"], frame["height"]) == (32, 16)
    overlay = Image.open(io.BytesIO(base64.b64decode(frame["prediction_mask_png_base64"])))
    assert overlay.size == (32, 16)
    assert overlay.getchannel("A").getextrema()[0] > 0


def test_segmentation_rejects_misaligned_image_and_mask(monkeypatch):
    def unexpected(*a, **kw):
        pytest.fail("Do not run inference against a misaligned mask")
    monkeypatch.setattr(qp, "infer_segmentation", unexpected)
    with pytest.raises(ValueError, match="dimensions differ"):
        qp.run_segmentation_pipeline(
            [{"name": "frame.png", "mask_index": "mask.png"}],
            {"frame.png": encoded_image((64, 32))}, {"mask.png": encoded_image((32, 16), "L")},
            {"class_names": ["person"]}, None)


@pytest.mark.parametrize("supported,expected_iou,expected_issues", [(["person"], 0.0, 1), (["car"], None, 0)])
def test_zero_predictions_distinguish_missed_and_unsupported_classes(monkeypatch, supported, expected_iou, expected_issues):
    monkeypatch.setattr(qp, "infer_segmentation", lambda *a, **kw: ([], {"supported_classes": supported}))
    result = qp.run_segmentation_pipeline(
        [{"name": "frame.png", "mask_index": "mask.png"}],
        {"frame.png": encoded_image((32, 16))}, {"mask.png": encoded_image((32, 16), "L")},
        {"class_names": ["person"], "mask_id_to_class": {"0": "person"}}, None)
    frame = result["sorted_frames"][0]
    assert frame["model_segmentation_metrics"]["macro_iou_supported"] == expected_iou
    assert result["summary"]["macro_iou_supported"] == expected_iou
    assert len(frame["issues"]) == expected_issues
    assert ("person" in frame["model_segmentation_metrics"]["classes_not_evaluated"]) == (not expected_issues)
