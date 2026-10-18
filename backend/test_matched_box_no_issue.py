"""
Test suite verifying that when both Human and AI detect an object,
it is NEVER flagged as an issue or review candidate (no red box).
"""
import pytest
from qc_pipeline import run_bbox_pipeline, is_same_category, is_spatial_match
from ai_engine.guideline_manager import GuidelineManager

def test_is_same_category():
    # Direct equality & case insensitive
    assert is_same_category("car", "car")
    assert is_same_category("Car", "car")
    assert is_same_category("CAR", "car")

    # Synonyms in vehicle group
    assert is_same_category("car", "vehicle")
    assert is_same_category("vehicle", "car")
    assert is_same_category("automobile", "car")
    assert is_same_category("auto", "car")
    assert is_same_category("sedan", "car")
    assert is_same_category("suv", "car")

    # Person / Pedestrian
    assert is_same_category("pedestrian", "person")
    assert is_same_category("Person", "pedestrian")
    assert is_same_category("human", "pedestrian")

    # Traffic light / sign with underscore or spaces
    assert is_same_category("traffic_light", "traffic light")
    assert is_same_category("traffic light", "traffic_light")
    assert is_same_category("traffic_sign", "traffic sign")

    # Incompatible classes should return False
    assert not is_same_category("car", "pedestrian")
    assert not is_same_category("car", "traffic light")
    assert not is_same_category("bicycle", "bus")


def test_is_spatial_match():
    # High overlap
    b1 = [100.0, 100.0, 300.0, 300.0]
    b2 = [105.0, 95.0, 305.0, 295.0]
    matched, score = is_spatial_match(b1, b2, match_iou=0.35)
    assert matched is True
    assert score > 0.8

    # Containment / Partial overlap (e.g. IoU around 0.40 in rain/occlusion)
    b_human = [200.0, 200.0, 400.0, 400.0]
    b_ai = [220.0, 210.0, 390.0, 390.0]
    matched, score = is_spatial_match(b_human, b_ai, match_iou=0.5)
    assert matched is True

    # Completely disjoint
    b3 = [600.0, 600.0, 700.0, 700.0]
    matched, score = is_spatial_match(b1, b3, match_iou=0.35)
    assert matched is False


def test_matched_box_produces_zero_issues(monkeypatch):
    """
    Core invariant: If both Human and AI detect an object at the same location,
    that object is valid/confirmed and must NEVER be flagged as an error!
    """
    human_frames = [{
        "name": "test_car_frame.jpg",
        "width": 1280,
        "height": 720,
        "labels": [
            {
                "id": "human_car_01",
                "category": "car",
                "box2d": {"x1": 590.0, "y1": 490.0, "x2": 780.0, "y2": 615.0}
            }
        ]
    }]

    # Mock AI predictions returning the matching car
    ai_predictions = [
        {
            "type": "rectangle",
            "label": "car",
            "points": [592.0, 488.0, 785.0, 618.0],
            "confidence": 0.92,
            "model": "YOLO26x",
            "model_role": "detector_a"
        }
    ]

    p_meta = {
        "id": "bbox",
        "name": "YOLO26x",
        "available": True,
        "model_runs": [{"role": "detector_a", "name": "YOLO26x", "status": "success", "supported_classes": ["car", "truck"]}]
    }

    import qc_pipeline
    monkeypatch.setattr(qc_pipeline, "infer_bbox", lambda *a, **k: (ai_predictions, p_meta))
    monkeypatch.setattr(qc_pipeline, "_image_bytes", lambda *a, **k: b"fake_bytes")
    
    # Mock PIL Image.open
    class FakeImage:
        size = (1280, 720)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr("PIL.Image.open", lambda *a, **k: FakeImage())

    profile = {"classes": ["car", "truck", "bus"], "use_gemini": False}
    result = run_bbox_pipeline(human_frames, {"test_car_frame.jpg": "dummy"}, profile, None)

    f0 = result["sorted_frames"][0]
    assert len(f0["errors"]) == 0, f"Expected 0 errors on matched car, got: {f0['errors']}"
    assert len(f0["model_predictions"]) == 1
    assert f0["model_predictions"][0]["category"] == "car"


def test_matched_box_synonym_casing_produces_zero_issues(monkeypatch):
    """
    Case where human labeled 'Car' or 'vehicle' and AI detected 'car'.
    Must match cleanly without class disagreement error.
    """
    human_frames = [{
        "name": "test_synonym_frame.jpg",
        "width": 1280,
        "height": 720,
        "labels": [
            {
                "id": "human_veh_01",
                "category": "Vehicle",  # Capitalized synonym
                "box2d": {"x1": 500.0, "y1": 300.0, "x2": 700.0, "y2": 500.0}
            }
        ]
    }]

    ai_predictions = [
        {
            "type": "rectangle",
            "label": "car",
            "points": [505.0, 298.0, 702.0, 503.0],
            "confidence": 0.88,
            "model": "YOLO26x",
            "model_role": "detector_a"
        }
    ]

    p_meta = {
        "id": "bbox",
        "name": "YOLO26x",
        "available": True,
        "model_runs": [{"role": "detector_a", "name": "YOLO26x", "status": "success", "supported_classes": ["car", "vehicle"]}]
    }

    import qc_pipeline
    monkeypatch.setattr(qc_pipeline, "infer_bbox", lambda *a, **k: (ai_predictions, p_meta))
    monkeypatch.setattr(qc_pipeline, "_image_bytes", lambda *a, **k: b"fake_bytes")
    class FakeImage:
        size = (1280, 720)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr("PIL.Image.open", lambda *a, **k: FakeImage())

    profile = {"classes": ["car", "vehicle"], "use_gemini": False}
    result = run_bbox_pipeline(human_frames, {"test_synonym_frame.jpg": "dummy"}, profile, None)

    f0 = result["sorted_frames"][0]
    assert len(f0["errors"]) == 0, f"Expected 0 errors on synonym matched car, got: {f0['errors']}"


def test_extra_overlapping_ai_box_not_marked_as_unlabeled(monkeypatch):
    """
    When detector produces 2 boxes on the same car (e.g. YOLO and D-FINE or multiple proposals),
    the second box must NOT be flagged as 'POSSIBLE_UNLABELED_OBJECT' on the car!
    """
    human_frames = [{
        "name": "test_double_ai.jpg",
        "width": 1280,
        "height": 720,
        "labels": [
            {
                "id": "human_car_01",
                "category": "car",
                "box2d": {"x1": 500.0, "y1": 300.0, "x2": 700.0, "y2": 500.0}
            }
        ]
    }]

    ai_predictions = [
        # AI prediction 1 (YOLO)
        {
            "type": "rectangle",
            "label": "car",
            "points": [505.0, 298.0, 702.0, 503.0],
            "confidence": 0.90,
            "model": "YOLO26x",
            "model_role": "detector_a"
        },
        # AI prediction 2 (D-FINE or duplicate) on the SAME car
        {
            "type": "rectangle",
            "label": "car",
            "points": [510.0, 305.0, 695.0, 498.0],
            "confidence": 0.85,
            "model": "D-FINE-X",
            "model_role": "detector_b"
        }
    ]

    p_meta = {
        "id": "bbox",
        "name": "YOLO26x + D-FINE-X",
        "available": True,
        "model_runs": [
            {"role": "detector_a", "name": "YOLO26x", "status": "success", "supported_classes": ["car"]},
            {"role": "detector_b", "name": "D-FINE-X", "status": "success", "supported_classes": ["car"]}
        ]
    }

    import qc_pipeline
    monkeypatch.setattr(qc_pipeline, "infer_bbox", lambda *a, **k: (ai_predictions, p_meta))
    monkeypatch.setattr(qc_pipeline, "_image_bytes", lambda *a, **k: b"fake_bytes")
    class FakeImage:
        size = (1280, 720)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr("PIL.Image.open", lambda *a, **k: FakeImage())

    profile = {"classes": ["car"], "use_gemini": False}
    result = run_bbox_pipeline(human_frames, {"test_double_ai.jpg": "dummy"}, profile, None)

    f0 = result["sorted_frames"][0]
    unlabeled_errors = [e for e in f0["errors"] if e["error_type"] == "POSSIBLE_UNLABELED_OBJECT"]
    assert len(unlabeled_errors) == 0, f"Expected 0 unlabeled errors on already-annotated car, got: {unlabeled_errors}"
