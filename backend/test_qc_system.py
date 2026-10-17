#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification & Test Suite for Next-Gen QC System (VinUniversity 4 QA Layers)
Kiểm thử toàn diện:
1. Guideline Manager & Parsing điều khoản
2. Multi-Model Consensus Engine (4 Tầng QA)
3. Cột chỉ rõ lỗi gì, lý do vi phạm, và trích dẫn điều khoản
4. 1-Click Auto-Heal (kẹp biên, xóa rác, gỡ trùng, điền thuộc tính)
5. Xuất danh sách finding cần reviewer xác nhận (không tự quyết định trả job)
6. Coordinate Conventions & Transformations (Slide 37, Slide 22)
7. Honeypot & Gold Standard Quality Tracker (Slide 33)
8. Prioritized Review Benchmark vs Baselines (Slide 34)
"""

import sys
import os
import json
import math

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from qc_engine import BDDQualityChecker
from ai_engine.guideline_manager import global_guideline_manager
from ai_engine.models_ensemble import global_multi_model_engine
from coordinate_utils import to_xyxy, coco_to_xyxy, xyxy_to_coco, yolo_to_xyxy, xyxy_to_yolo, voc_to_xyxy, bdd_to_xyxy, xyxy_to_bdd


def run_tests():
    print("=" * 65)
    print("🧪 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN HỆ THỐNG AUTO-QC 2D BDD100K")
    print("=" * 65)

    # 1. Test Guideline Manager
    print("\n[TEST 1] Kiểm tra Guideline Manager...")
    assert len(global_guideline_manager.rules) == 0, "Default template không được tự tạo điều khoản"
    from ai_engine.guideline_manager import GuidelineManager
    sample_rules = """# QUY CHUẨN GÁN NHÃN MẪU
§1.1 Bounding box phải bao sát đối tượng.
§1.2 Tọa độ không được vượt quá khung hình 1280x720.
§1.3 Không gán nhãn các đối tượng nhỏ hơn 16x16 pixel (rác).
§1.4 Đa giác không được tự cắt nhau.
"""
    test_manager = GuidelineManager(sample_rules)
    assert len(test_manager.rules) >= 4, "Chưa parse được các điều khoản khi nạp guideline"
    citation = test_manager.format_citation("out_of_image")
    print(f"  ✓ Guideline parsed {len(test_manager.rules)} điều khoản")
    print(f"  ✓ Trích dẫn mẫu: {citation}")

    # 2. Test Multi-Model Consensus Engine với Sample Data
    print("\n[TEST 2] Kiểm tra Multi-Model Consensus Engine & 4 Tầng QA...")
    sample_path = os.path.join(CURRENT_DIR, "..", "demo_data", "sample_test_labels.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    checker = BDDQualityChecker()
    res = checker.check_dataset_advanced(sample_data)

    summary = res["summary"]
    sorted_frames = res["sorted_frames"]

    print(f"  ✓ Tổng số frames: {summary['total_frames']}")
    print(f"  ✓ Frames without active rule flags: {summary['auto_pass_count']} ({summary['no_rule_findings_pct']}%)")
    print(f"  ✓ Số frames cần Review: {summary['need_review_count']}")
    print(f"  ✓ Trạng thái rule review: {summary['job_verdict']}")

    # Kiểm tra tính sắp xếp từ Cao xuống Thấp
    ranks = [f["priority_rank"] for f in sorted_frames]
    print(f"  ✓ Thứ tự Priority Ranks: {ranks}")
    assert ranks == sorted(ranks), "Danh sách chưa được sắp xếp từ Cao xuống Thấp!"
    print("  ✓ Priority ranks are sorted from highest to lowest review priority.")

    # 3. Test Cột Chi Tiết Lỗi
    print("\n[TEST 3] Kiểm tra nội dung Cột Chỉ Rõ Lỗi Gì & Lý Do Vi Phạm...")
    for idx, f in enumerate(sorted_frames[:3], 1):
        print(f"  Frame #{idx}: {f['frame_name']} -> {f['priority_label']} ({f['recommendation']})")
        for err in f["errors"]:
            print(f"    • [{err['error_type']}] (QA Layer: {err.get('qa_layer', 'N/A')})")
            print(f"      -> {err['explanation']}")
            print(f"      -> Đề xuất: {err['action']}")

    # 4. Test 1-Click Auto-Heal
    print("\n[TEST 4] Kiểm tra tính năng 1-Click Auto-Heal...")
    healed_data, heal_stats = checker.auto_heal(sample_data)
    print(f"  ✓ Kết quả Auto-Heal: {heal_stats}")
    assert heal_stats["boxes_clamped"] > 0 or heal_stats["trash_boxes_removed"] > 0, "Chưa heal được box nào"
    assert heal_stats.get("duplicate_boxes_removed", 0) > 0, "Chưa xóa được box trùng"
    assert heal_stats.get("attributes_healed", 0) > 0, "Chưa điền được thuộc tính thiếu"
    print("  ✓ ĐÃ XÁC NHẬN: Auto-Heal tự động kẹp biên, xóa box rác, gỡ trùng và điền thuộc tính thành công!")

    # 5. Test candidate feedback export
    print("\n[TEST 5] Kiểm tra xuất danh sách finding cần reviewer xác nhận...")
    feedback_md = checker.generate_feedback_report(sorted_frames, "Đoàn Văn Thắng")
    assert "not an acceptance or rejection decision" in feedback_md, "Report must not make a job acceptance decision"
    assert "TRẢ NGUYÊN JOB" not in feedback_md and "P-037" not in feedback_md, "Report contains unapproved policy"
    print("  ✓ ĐÃ XÁC NHẬN: Export labels findings as reviewer candidates (10 dòng đầu):")
    for line in feedback_md.splitlines()[:10]:
        print(f"    {line}")

    # 6. Test Coordinate Conversions (Slide 37 & Slide 22)
    print("\n[TEST 6] Kiểm tra Chuẩn Hóa Tọa Độ & Chuyển Đổi Không Lỗi (Slide 37, 22)...")
    # Gemini VLM thang 0-1000
    gemini_box = [200, 100, 500, 300]  # ymin, xmin, ymax, xmax
    xyxy = to_xyxy(gemini_box, W=1280, H=720)
    assert abs(xyxy[0] - 128.0) < 1e-3 and abs(xyxy[1] - 144.0) < 1e-3, f"Sai to_xyxy: {xyxy}"
    print(f"  ✓ Gemini VLM to_xyxy(box_2d=[200, 100, 500, 300]): {xyxy}")

    # COCO conversion
    coco_b = [100.0, 150.0, 50.0, 80.0]  # x, y, w, h
    xyxy_coco = coco_to_xyxy(coco_b)
    assert xyxy_coco == (100.0, 150.0, 150.0, 230.0), f"Sai COCO: {xyxy_coco}"
    assert xyxy_to_coco(xyxy_coco) == coco_b, "Sai đảo COCO"
    print(f"  ✓ COCO [x, y, w, h] <-> [x1, y1, x2, y2]: {coco_b} <-> {xyxy_coco}")

    # YOLO conversion
    yolo_b = [0.5, 0.5, 0.2, 0.2]  # cx, cy, w, h
    xyxy_yolo = yolo_to_xyxy(yolo_b, 1000, 1000)
    assert xyxy_yolo == (400.0, 400.0, 600.0, 600.0), f"Sai YOLO: {xyxy_yolo}"
    assert xyxy_to_yolo(xyxy_yolo, 1000, 1000) == yolo_b, "Sai đảo YOLO"
    print(f"  ✓ YOLO [cx, cy, w, h] <-> [x1, y1, x2, y2]: {yolo_b} <-> {xyxy_yolo}")

    # Pascal VOC
    voc_b = [101.0, 201.0, 300.0, 400.0]
    xyxy_voc = voc_to_xyxy(voc_b)
    assert xyxy_voc == (100.0, 200.0, 300.0, 400.0), f"Sai VOC: {xyxy_voc}"
    print(f"  ✓ Pascal VOC 1-based to 0-based: {voc_b} -> {xyxy_voc}")

    # 7. Synthetic logic fixture only; these values are not an annotator/model benchmark.
    print("\n[TEST 7] Kiểm tra logic so khớp honeypot trên fixture tự dựng (không phải benchmark)...")
    gt_frames = [
        {
            "name": "0003d45e-1122aa33.jpg",
            "labels": [
                {"box2d": {"x1": 320.0, "y1": 280.0, "x2": 550.0, "y2": 420.0}, "category": "car"},
                {"box2d": {"x1": 700.0, "y1": 150.0, "x2": 740.0, "y2": 190.0}, "category": "traffic sign"}
            ]
        }
    ]
    honeypot_res = checker.evaluate_honeypot(sample_data, gt_frames, "Đoàn Văn Thắng")
    print(f"  ✓ Honeypot Evaluated: {honeypot_res['honeypot_frames_evaluated']} frames")
    print("  ✓ Metric function returned finite values for a toy fixture; do not interpret as quality results.")
    assert honeypot_res['precision'] >= 90.0, "Độ chính xác Honeypot không đạt chuẩn"

    # 8. Benchmark must remain unavailable without independent adjudicated gold.
    print("\n[TEST 8] Xác nhận không tự chấm Recall@20 bằng rule flag của chính hệ thống...")
    bench = summary.get("benchmark", {})
    assert bench.get("status") == "unavailable_without_independent_gold"
    assert bench.get("prioritized_recall_at_20") is None
    assert bench.get("random_baseline_recall_at_20") is None
    assert bench.get("meets_slide34_target") is None
    print("  ✓ Recall, baselines và target chưa được báo cáo nếu thiếu gold độc lập.")

    print("\n" + "=" * 65)
    print("🎉 8/8 logic assertions passed; this is a smoke test, not a quality benchmark.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
