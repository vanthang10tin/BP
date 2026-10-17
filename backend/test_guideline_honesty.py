from ai_engine.guideline_manager import GuidelineManager
from gemini_provider import _heuristic_compile_guideline


def test_empty_template_does_not_claim_policy_or_unrelated_citations():
    manager = GuidelineManager()

    assert manager.rules == []
    assert manager.format_citation("out_of_image") == "Chưa có điều khoản khớp trong guideline đã nạp"


def test_heuristic_compiler_does_not_invent_missing_thresholds_or_principles():
    compiled = _heuristic_compile_guideline("")

    assert compiled["status"] == "no_project_guideline"
    assert compiled["class_thresholds"] == {}
    assert compiled["geometry_constraints"]["min_box_pixels"] is None
    assert compiled["geometry_constraints"]["min_area_pixels"] is None
    assert compiled["geometry_constraints"]["min_tightness_iou"] is None
    assert compiled["segmentation_policy"]["min_mask_iou"] is None
    assert compiled["key_principles"] == []
