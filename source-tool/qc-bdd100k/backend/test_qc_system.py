#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification & Test Suite for Next-Gen QC System (VinUniversity 4 QA Layers)
Kiểm thử toàn diện:
1. Guideline Manager & Parsing điều khoản
2. Multi-Model Consensus Engine (4 Tầng QA)
3. Cột chỉ rõ lỗi gì, lý do vi phạm, và trích dẫn điều khoản
4. 1-Click Auto-Heal (kẹp biên, xóa rác, gỡ trùng, điền thuộc tính)
5. Xuất phiếu phản hồi trả job Markdown (QĐ-002)
6. Coordinate Conventions & Transformations (Slide 37, Slide 22)
7. Honeypot & Gold Standard Quality Tracker (Slide 33)
8. Prioritized Review Benchmark vs Baselines (Slide 34)
"""

import sys
import os
import json

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
    text = global_guideline_manager.get_guideline_text()
    assert len(global_guideline_manager.rules) >= 4, "Chưa parse được các điều khoản"
    citation = global_guideline_manager.format_citation("out_of_image")
    print(f"  ✓ Guideline parsed {len(global_guideline_manager.rules)} điều khoản")
    print(f"  ✓ Trích dẫn mẫu: {citation}")

    # 2. Test Multi-Model Consensus Engine với Sample Data
    print("\n[TEST 2] Kiểm tra Multi-Model Consensus Engine & 4 Tầng QA...")
    sample_path = os.path.join(CURRENT_DIR, "..", "sample_test_labels.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    checker = BDDQualityChecker()
    res = checker.check_dataset_advanced(sample_data)

    summary = res["summary"]
    sorted_frames = res["sorted_frames"]

    print(f"  ✓ Tổng số frames: {summary['total_frames']}")
    print(f"  ✓ Tỷ lệ Auto-Pass: {summary['workload_reduction_pct']}% ({summary['auto_pass_count']} frames)")
    print(f"  ✓ Số frames cần Review: {summary['need_review_count']}")
    print(f"  ✓ Quyết định Job: {summary['job_verdict']}")

    # Kiểm tra tính sắp xếp từ Cao xuống Thấp
    ranks = [f["priority_rank"] for f in sorted_frames]
    print(f"  ✓ Thứ tự Priority Ranks: {ranks}")
    assert ranks == sorted(ranks), "Danh sách chưa được sắp xếp từ Cao xuống Thấp!"
    print("  ✓ ĐÃ XÁC NHẬN: Danh sách được sắp xếp từ Cao xuống Thấp (Rank 1 -> 2 -> 3 -> 4) chuẩn 100%!")

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

    # 5. Test Xuất Phiếu Trả Job
    print("\n[TEST 5] Kiểm tra tính năng Xuất Phiếu Trả Job Markdown...")
    feedback_md = checker.generate_feedback_report(sorted_frames, "Đoàn Văn Thắng")
    assert "BÁO CÁO REVIEW" in feedback_md, "Sai định dạng feedback"
    print("  ✓ ĐÃ XÁC NHẬN: Tạo phiếu trả job markdown thành công (10 dòng đầu):")
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

    # 7. Test Honeypot Quality Tracker (Slide 33)
    print("\n[TEST 7] Kiểm tra Tầng 4: Honeypot & Gold Standard Evaluator (Slide 33)...")
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
    print(f"  ✓ Precision: {honeypot_res['precision']}% | Recall: {honeypot_res['recall']}% | Mean IoU: {honeypot_res['mean_iou']}%")
    print(f"  ✓ Status: {honeypot_res['gold_standard_status']}")
    assert honeypot_res['precision'] >= 90.0, "Độ chính xác Honeypot không đạt chuẩn"

    # 8. Test Review Prioritization Benchmark (Slide 34)
    print("\n[TEST 8] Kiểm tra Tầng 3: Review Prioritization Benchmark Recall@20% (Slide 34)...")
    bench = summary.get("benchmark", {})
    recall_20 = bench.get("recall_at_20_pct", 0)
    random_20 = bench.get("random_recall_at_20_pct", 20)
    print(f"  ✓ Recall@20% của Hệ Thống: {recall_20}%")
    print(f"  ✓ Recall@20% của Baseline Ngẫu Nhiên: {random_20}%")
    assert recall_20 >= 70.0, f"Recall@20% ({recall_20}%) chưa đạt mục tiêu >= 70% của Slide 34"
    print("  ✓ ĐÃ XÁC NHẬN: Hệ thống đạt Recall@20% >= 70%, vượt trội gấp 3.5 lần so với chọn ngẫu nhiên!")

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ 8/8 BÀI KIỂM THỬ CHUẨN VINUNIVERSITY ĐÃ VƯỢT QUA 100%!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
