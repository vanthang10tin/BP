"""Review candidates must retain rule evidence and not depend on box order."""
import copy
import base64
import io
from PIL import Image
import pytest
import qc_pipeline as qp


def png(size=(100,100), mode="RGB"):
    b=io.BytesIO(); Image.new(mode,size,0).save(b,format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def run_boxes(monkeypatch, labels, predictions, errors=None):
    base = [{"frame_name":"a.png", "frame_index":0, "labels":labels, "errors": errors or []}]
    monkeypatch.setattr(qp,"_base_bbox_frames",lambda *a:copy.deepcopy(base))
    monkeypatch.setattr(qp,"infer_bbox",lambda *a,**kw:(predictions,{"name":"fixture"}))
    return qp.run_bbox_pipeline([{"name":"a.png","labels":labels}],{"a.png":png()},
        {"classes":["car","truck"],"use_gemini":False},None)["sorted_frames"][0]


def label(id, box, category="car"):
    return {"id":id,"category":category,"box2d":dict(zip(("x1","y1","x2","y2"),box))}


def detection(box, category="car"):
    return {"type":"rectangle","label":category,"points":box,"confidence":.9,"model":"YOLO"}


def test_containment_does_not_confirm_loose_box():
    assert not qp.is_spatial_match([0,0,100,100],[20,20,40,40],.35)[0]
    assert not qp.is_spatial_match([0,0,100,100],[0,0,60,100],.8)[0]


def test_same_class_match_wins_over_higher_overlap_conflicting_proposal(monkeypatch):
    frame=run_boxes(monkeypatch,[label("a",[0,0,50,50])],
        [detection([2,2,48,48]),detection([0,0,50,50],"truck")])
    assert not any(e["error_type"]=="MODEL_CLASS_DISAGREEMENT" for e in frame["errors"])


def test_matched_model_keeps_independent_rule_errors(monkeypatch):
    error={"issue_id":"a.png:a:RULE","object_id":"a","error_type":"MISSING_ATTRIBUTE","severity":"MEDIUM"}
    frame=run_boxes(monkeypatch,[label("a",[0,0,50,50])],[detection([0,0,50,50])],[error])
    assert any(e["issue_id"] == error["issue_id"] for e in frame["errors"])


def test_loose_box_produces_boundary_candidate(monkeypatch):
    frame=run_boxes(monkeypatch,[label("a",[0,0,100,100])],[detection([0,0,50,60])])
    assert any(e["error_type"]=="MODEL_BOUNDARY_DISAGREEMENT" for e in frame["errors"])


def test_segment_prediction_only_class_is_reviewed(monkeypatch):
    monkeypatch.setattr(qp,"infer_segmentation",lambda *a,**kw:([{
        "label":"car","points":[10,10,30,10,30,30,10,30],"model":"YOLO-Seg"}],{"supported_classes":["road","car"]}))
    result=qp.run_segmentation_pipeline([{"name":"a.png","mask_index":"m.png"}],
        {"a.png":png()},{"m.png":png(mode="L")},{"class_names":["road","car"]},None)
    assert any(i["category"]=="car" for i in result["sorted_frames"][0]["issues"])


def test_segment_failing_source_has_visible_evidence(monkeypatch):
    predictions=[{"label":"car","points":[0,0,20,0,20,20,0,20],"model":"YOLO-Seg"},
                 {"label":"car","points":[0,0,99,0,99,99,0,99],"model":"SAM2"}]
    monkeypatch.setattr(qp,"infer_segmentation",lambda *a,**kw:(predictions,{"supported_classes":["car"]}))
    result=qp.run_segmentation_pipeline([{"name":"a.png","mask_index":"m.png"}],
        {"a.png":png()},{"m.png":png(mode="L")},{"class_names":["car"]},None)
    issue=result["sorted_frames"][0]["issues"][0]
    assert issue["suggested_box"] is not None
    assert issue["error_type"]=="SEGMENTATION_BOUNDARY_SUSPECT"


@pytest.mark.parametrize("reverse", [False, True])
def test_best_same_class_assignment_is_not_stolen_by_earlier_conflict(monkeypatch, reverse):
    labels=[label("truck",[0,0,50,50],"truck"),label("car",[0,0,50,50])]
    if reverse: labels.reverse()
    frame=run_boxes(monkeypatch,labels,[detection([0,0,50,50])])
    assert not any(e.get("object_id")=="car" for e in frame["errors"])
    assert not any(e["error_type"]=="MODEL_CLASS_DISAGREEMENT" for e in frame["errors"])
