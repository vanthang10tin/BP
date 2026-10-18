"""Offline regressions: no cloud calls, weights or credentials required."""
import json
from unittest.mock import patch

import pytest
import gemini_provider as gp
from ai_engine.guideline_manager import GuidelineManager
from qc_pipeline import _unresolved_finding


def valid_result():
    return dict(is_violation=False, verdict="ANNOTATOR_CORRECT", confirmed_class="car",
                explanation="Nhãn phù hợp.", action="Giữ nhãn.", confidence=0.8)


@pytest.mark.parametrize("change", [
    {"is_violation": "false"}, {"confidence": None}, {"confidence": float("nan")},
    {"confidence": 1.1}, {"confidence": True}, {"verdict": "invented"},
    {"verdict": "MODEL_CONFIRMED_ERROR"}, {"explanation": ""},
])
def test_malformed_verdict_is_rejected(change):
    with pytest.raises(ValueError):
        gp._validate_arbitration({**valid_result(), **change}, "configured-model")


def test_valid_verdict_records_actual_model():
    assert gp._validate_arbitration(valid_result(), "configured-model")["model"] == "configured-model"


@pytest.mark.parametrize("key", ["", "test-key"])
def test_missing_key_or_api_failure_never_invents_error(key):
    with patch.object(gp, "_api_key", return_value=key), patch.object(gp.urllib.request, "urlopen", side_effect=OSError("offline")):
        result = gp.arbitrate_dispute(b"image", {"category": "car"}, guideline="User rule")
    assert result["verdict"] == "UNRESOLVED_DISPUTE"
    assert result["confidence"] is None
    assert result["confirmed_class"] is None
    assert result["is_violation"] is False
    issue = _unresolved_finding("frame", "obj", "car", [0, 0, 10, 10], result)
    assert issue["requires_review"] and issue["review_status"] == "pending"


def test_entire_user_guideline_reaches_arbiter():
    guideline = "User policy " * 200 + "END_RULE_MUST_SURVIVE"
    seen = []
    def respond(request, **kwargs):
        seen.append(json.loads(request.data))
        raise OSError("offline")
    with patch.object(gp, "_api_key", return_value="test"), patch.object(gp.urllib.request, "urlopen", side_effect=respond):
        gp.arbitrate_dispute(b"image", {"category": "car"}, guideline=guideline)
    assert len(seen) == 1
    assert guideline in seen[0]["contents"][0]["parts"][0]["text"]


def test_custom_guideline_replaces_default_source():
    manager = GuidelineManager()
    manager.update_guideline("Only the user's policy.")
    assert not manager.is_default
    assert manager.raw_text == "Only the user's policy."


def test_modal_cloud_requires_explicit_configuration(monkeypatch):
    from ai_engine.local_model_providers import _modal_base_url
    monkeypatch.delenv("QC_MODAL_API_URL", raising=False)
    assert _modal_base_url() == ""


def test_custom_rules_do_not_inherit_bdd_policy_or_run_extra_model():
    from qc_pipeline import _base_bbox_frames
    from ai_engine.models_ensemble import LocalYOLO26Detector
    manager = GuidelineManager("Inspect widgets; no minimum size or attribute requirement.")
    manager.compiled_rules = {"geometry_constraints": {}}
    frame = {"name": "a.png", "width": 2000, "height": 1000, "labels": [
        {"id": "1", "category": "widget", "box2d": dict(x1=1500, y1=800, x2=1502, y2=802)}]}
    with patch.object(LocalYOLO26Detector, "detect_frame", side_effect=AssertionError("Unexpected model inference")):
        result = _base_bbox_frames([frame], {"classes": ["widget"]}, manager)
    assert result[0]["errors"] == []


def test_rule_failure_cannot_be_reported_as_clean():
    from qc_pipeline import _base_bbox_frames
    from qc_engine import BDDQualityChecker
    with patch.object(BDDQualityChecker, "check_dataset_advanced", side_effect=RuntimeError("failure")):
        with pytest.raises(ValueError, match="no verified QC result"):
            _base_bbox_frames([], {}, None)


def test_clause_number_does_not_invent_a_citation():
    manager = GuidelineManager("§1.1 Tên file dùng chữ thường.\n§1.2 Lưu tác giả khi export.")
    assert manager.find_relevant_clauses("tightness") == []
    assert manager.find_relevant_clauses("out_of_image") == []


def test_two_model_fusion_without_gemini():
    """Kiểm tra nguyên tắc: 'những cái nào chung thì lấy, riêng cũng lấy cho nó đủ thừa còn hơn thiếu' khi tắt Gemini."""
    import io
    from PIL import Image
    import qc_pipeline as qp
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, format="PNG")
    # Annotator gán 1 car
    labels = [{"id": "1", "category": "car", "box2d": dict(x1=10, y1=10, x2=50, y2=50)}]
    frames = [{"name": "a.png", "labels": labels}]
    base = [{"frame_name": "a.png", "frame_index": 0, "labels": labels, "errors": []}]
    # Model 1 (YOLO) thấy car tại [10, 10, 50, 50] conf 0.90 và truck tại [100, 100, 200, 200] conf 0.85
    # Model 2 (D-FINE) thấy car tại [11, 10, 51, 50] conf 0.95 và bus tại [300, 300, 400, 400] conf 0.80
    predictions = [
        dict(type="rectangle", label="car", points=[10, 10, 50, 50], confidence=.90, model="YOLO26x", model_role="detector_a"),
        dict(type="rectangle", label="truck", points=[100, 100, 200, 200], confidence=.85, model="YOLO26x", model_role="detector_a"),
        dict(type="rectangle", label="car", points=[11, 10, 51, 50], confidence=.95, model="D-FINE-X", model_role="detector_b"),
        dict(type="rectangle", label="bus", points=[300, 300, 400, 400], confidence=.80, model="D-FINE-X", model_role="detector_b"),
    ]
    with patch.object(qp, "_base_bbox_frames", return_value=base), \
         patch.object(qp, "_image_bytes", return_value=image.getvalue()), \
         patch.object(qp, "infer_bbox", return_value=(predictions, {"name": "fixture", "model_runs": [
             {"role": "detector_a", "status": "success"}, {"role": "detector_b", "status": "success"}]})), \
         patch.object(gp, "align_taxonomy_with_gemini", return_value={}), \
         patch.object(qp, "arbitrate_dispute") as arb:
        result = qp.run_bbox_pipeline(frames, {}, {"classes": ["car", "truck", "bus"], "use_gemini": False}, None)

    # KHÔNG được gọi Gemini khi tắt Gemini!
    assert arb.call_count == 0

    frame_res = result["sorted_frames"][0]
    # AI boxes phải gồm: 1 box chung (fused car) + 1 box riêng YOLO (truck) + 1 box riêng D-FINE (bus) = 3 box!
    assert len(frame_res["model_predictions"]) == 3
    cats = [b["category"] for b in frame_res["model_predictions"]]
    assert "car" in cats
    assert "truck" in cats
    assert "bus" in cats

    # Raw model boxes giữ đủ 4 detections
    assert len(frame_res["raw_model_boxes"]) == 4

    # Human box car [10, 10, 50, 50] khớp với AI fused car -> không bị báo lỗi sai nhãn
    # Nhưng 2 box riêng của AI (truck và bus) không có nhãn người gán -> báo POSSIBLE_UNLABELED_OBJECT
    errors = frame_res["errors"]
    unlabeled = [e for e in errors if e["error_type"] == "POSSIBLE_UNLABELED_OBJECT"]
    assert len(unlabeled) == 2


def test_gemini_arbitration_when_explicitly_requested():
    """Khi người dùng cố ý bật use_gemini=True, Gemini được gọi để phân xử."""
    import io
    from PIL import Image
    import qc_pipeline as qp
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, format="PNG")
    labels = [{"id": "1", "category": "car", "box2d": dict(x1=10, y1=10, x2=50, y2=50)}]
    frames = [{"name": "a.png", "labels": labels}]
    base = [{"frame_name": "a.png", "frame_index": 0, "labels": labels, "errors": []}]
    # YOLO says car, D-FINE says truck -> conflict!
    predictions = [
        dict(type="rectangle", label="car", points=[10, 10, 50, 50], confidence=.90, model="YOLO26x", model_role="detector_a"),
        dict(type="rectangle", label="truck", points=[10, 10, 50, 50], confidence=.88, model="D-FINE-X", model_role="detector_b"),
    ]
    with patch.object(qp, "_base_bbox_frames", return_value=base), \
         patch.object(qp, "_image_bytes", return_value=image.getvalue()), \
         patch.object(qp, "infer_bbox", return_value=(predictions, {"name": "fixture", "model_runs": [
             {"role": "detector_a", "status": "success"}, {"role": "detector_b", "status": "success"}]})), \
         patch.object(gp, "align_taxonomy_with_gemini", return_value={}), \
         patch.object(qp, "gemini_configured", return_value=True), \
         patch.object(qp, "arbitrate_dispute", return_value=gp._unresolved_dispute("arbitrated")) as arb:
        result = qp.run_bbox_pipeline(frames, {}, {"classes": ["car", "truck"], "use_gemini": True}, None)
    assert arb.call_count == 1


