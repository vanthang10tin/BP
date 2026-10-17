from qc_engine import BDDQualityChecker


def frame(name, boxes):
    return {"name": name, "labels": [
        {"category": category, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
        for category, x1, y1, x2, y2 in boxes
    ]}


def test_honeypot_uses_one_to_one_matching_and_valid_box_denominators():
    checker = BDDQualityChecker()
    truth = [frame("a.jpg", [("car", 0, 0, 10, 10)])]
    predictions = [frame("a.jpg", [("car", 0, 0, 10, 10), ("car", 0, 0, 10, 10)])]
    predictions[0]["labels"].append({"category": "car"})  # invalid box must not inflate denominator

    result = checker.evaluate_honeypot(predictions, truth)

    assert result["honeypot_frames_evaluated"] == 1
    assert result["precision"] == 50.0
    assert result["recall"] == 100.0
    assert result["mean_iou"] == 100.0


def test_honeypot_without_matching_frames_does_not_pass_thresholds():
    result = BDDQualityChecker().evaluate_honeypot(
        [frame("submission.jpg", [("car", 0, 0, 10, 10)])],
        [frame("reference.jpg", [("car", 0, 0, 10, 10)])],
    )

    assert result["honeypot_frames_evaluated"] == 0
    assert result["gold_standard_status"] == "BELOW_CONFIGURED_THRESHOLDS"
