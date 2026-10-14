#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Tự Động Cho Bounding Box & Segmentation 2D (BDD100K)
Đề tài: N2-04A - Quality Control Check
Tác giả: Nhóm P-037
Tuân thủ toàn diện chương trình VinUniversity (Slides 19, 21, 22, 29, 30, 31, 33, 34, 37, 46)

4 TẦNG KIỂM SOÁT CHẤT LƯỢNG (QA LAYERS - Slide 29):
1. Tầng 1: QC theo luật (cấu trúc, hình học, ontology 10 classes BDD100k, aspect ratio, polygon self-intersection)
2. Tầng 2: Lỗi nghi ngờ từ Model (ObjectLab/Cleanlab: Badly located, Swapped class, Overlooked)
3. Tầng 3: Review có ưu tiên (Hàng đợi xếp từ Cao xuống Thấp, Recall@20% >= 70%)
4. Tầng 4: Honeypot & Gold Standard Benchmark
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi charmap
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(CURRENT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from qc_engine import BDDQualityChecker, DEFAULT_CONFIG


def print_summary(summary: Dict[str, Any], sorted_frames: List[Dict[str, Any]], verbose: bool = False):
    print("\n" + "=" * 75)
    print(" 🛡️ BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG GÁN NHÃN 2D (4 TẦNG QA) - BDD100K")
    print("=" * 75)
    print(f" Tổng số frames kiểm tra : {summary['total_frames']}")
    print(f" Tổng số đối tượng (Objs): {summary['total_objects']}")
    print(f" Số frames đạt chuẩn 100%: {summary['auto_pass_count']} ({summary['workload_reduction_pct']}%) -> Reviewer bỏ qua!")
    print(f" Số frames cần review lại: {summary['need_review_count']} ({summary['defect_rate_pct']}%)")
    print(f" Số frames CỰC CAO (Nguy cơ): {summary['critical_count']}")
    print(f" Quyết định nghiệm thu   : {summary['job_verdict']} (Theo QĐ-002: Ngưỡng 10%)")
    print("-" * 75)

    l_stats = summary.get("layer_stats", {})
    print(" 📊 Hiệu suất theo 4 Tầng QA (Slide 29 VinUni):")
    print(f"  • Tầng 1 (QC Luật - Cấu trúc & Hình học): {l_stats.get('layer_1_rules', 0)} lỗi bắt được")
    print(f"  • Tầng 2 (Model ObjectLab / Cleanlab)    : {l_stats.get('layer_2_models', 0)} lỗi phát hiện")
    print(f"  • Tầng 3 (Hàng đợi Review có ưu tiên)  : Đã xếp hạng 1 -> 4 theo mức độ nghiêm trọng")
    bench = summary.get("benchmark", {})
    print(f"    -> Hiệu quả Review top 20% (Recall@20%): {bench.get('recall_at_20_pct', 0)}% (So với Random: {bench.get('random_recall_at_20_pct', 20)}%)")
    print("-" * 75)

    r_tele = summary.get("rule_telemetry", {})
    if r_tele:
        print(" 🔍 Thống kê chi tiết lỗi theo luật:")
        for r_type, count in r_tele.items():
            print(f"  • {r_type:<32} : {count:>3} lỗi")
        print("-" * 75)

    if sorted_frames and verbose:
        print(" 📋 Danh sách Hàng Đợi Review (Đã sắp từ Cao xuống Thấp):")
        for idx, f in enumerate(sorted_frames, 1):
            status = f"{f['priority_label']} ({f['recommendation']})"
            print(f"\n  [{idx:02d}] Frame: {f['frame_name']} | Mức độ: {status} | Số lỗi: {f['error_count']}")
            for err in f.get("errors", []):
                models_str = ", ".join(err.get("models_flagged", []))
                print(f"       • [{err['error_type']}] ({err['qa_layer']}) - Models: {models_str}")
                print(f"         Lý do: {err['explanation']}")
                print(f"         Đề xuất: {err['action']}")
    elif sorted_frames and not verbose:
        print(" (Gợi ý: Dùng cờ --verbose để in chi tiết từng lỗi của từng frame trên console)")
    print("=" * 75 + "\n")


def export_csv(sorted_frames: List[Dict[str, Any]], output_path: str):
    fieldnames = ["priority_rank", "priority_label", "frame_name", "recommendation", "error_type", "qa_layer", "severity", "models_flagged", "explanation", "action"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for f in sorted_frames:
            errors = f.get("errors", [])
            if not errors:
                writer.writerow({
                    "priority_rank": f.get("priority_rank", 4),
                    "priority_label": f.get("priority_label", "PASS"),
                    "frame_name": f.get("frame_name", ""),
                    "recommendation": f.get("recommendation", "ĐẠT"),
                    "error_type": "NONE",
                    "qa_layer": "Pass",
                    "severity": "NONE",
                    "models_flagged": "ALL_AGREED",
                    "explanation": "Đồng thuận cao 99%, không có lỗi.",
                    "action": "Pass"
                })
            else:
                for err in errors:
                    writer.writerow({
                        "priority_rank": f.get("priority_rank", 1),
                        "priority_label": f.get("priority_label", "CỰC CAO"),
                        "frame_name": f.get("frame_name", ""),
                        "recommendation": f.get("recommendation", "TRẢ VỀ"),
                        "error_type": err.get("error_type", ""),
                        "qa_layer": err.get("qa_layer", ""),
                        "severity": err.get("severity", ""),
                        "models_flagged": ", ".join(err.get("models_flagged", [])),
                        "explanation": err.get("explanation", ""),
                        "action": err.get("action", "")
                    })
    print(f"[OK] Đã xuất báo cáo CSV xếp hạng ưu tiên: {output_path}")


def export_json(adv_res: Dict[str, Any], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adv_res, f, ensure_ascii=False, indent=2)
    print(f"[OK] Đã xuất báo cáo JSON chi tiết 4 Tầng QA: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tool kiểm tra chất lượng nhãn BDD100K 2D (N2-04A - 4 QA Layers)")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn file JSON nhãn BDD100K")
    parser.add_argument("--config", "-c", default=None, help="Đường dẫn file config JSON tùy chỉnh")
    parser.add_argument("--output", "-o", default=None, help="Đường dẫn file xuất ra (kết quả auto-heal hoặc báo cáo)")
    parser.add_argument("--output-json", "-oj", default=None, help="Đường dẫn xuất file JSON báo cáo lỗi")
    parser.add_argument("--output-csv", "-oc", default=None, help="Đường dẫn xuất file CSV báo cáo lỗi")
    parser.add_argument("--auto-heal", "-ah", nargs="?", const="__FLAG__", default=None, help="Tự động sửa lỗi kẹp biên & box rác. Có thể kèm tên file xuất hoặc dùng chung với --output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Hiển thị chi tiết từng lỗi trên console")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[LỖI] File nhãn không tồn tại: {args.input}")
        sys.exit(1)

    # Đọc cấu hình
    cfg = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    # Đọc dữ liệu nhãn
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    checker = BDDQualityChecker(cfg)
    res = checker.check_dataset_advanced(data)

    print_summary(res["summary"], res["sorted_frames"], verbose=args.verbose)

    # Xuất JSON báo cáo
    target_json = args.output_json
    if not target_json and args.output and not args.auto_heal:
        if args.output.lower().endswith(".json"):
            target_json = args.output
        elif args.output.lower().endswith(".csv") and not args.output_csv:
            args.output_csv = args.output

    if target_json:
        export_json(res, target_json)
    if args.output_csv:
        export_csv(res["sorted_frames"], args.output_csv)

    # Xử lý Auto-Heal
    if args.auto_heal is not None:
        healed_data, stats = checker.auto_heal(data)
        
        # Xác định file xuất
        if args.auto_heal != "__FLAG__":
            heal_out = args.auto_heal
        elif args.output:
            heal_out = args.output
        else:
            base_name = os.path.basename(args.input)
            heal_out = os.path.join(os.path.dirname(args.input), f"cleaned_{base_name}")

        with open(heal_out, "w", encoding="utf-8") as f:
            json.dump(healed_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] ⚡ 1-Click Auto-Heal hoàn tất! Đã lưu file: {heal_out}")
        print(f"     Thống kê đã sửa: {stats}")

    # Exit code: 0 nếu không có lỗi, 1 nếu có lỗi cần review
    sys.exit(0 if res["summary"]["need_review_count"] == 0 else 1)


if __name__ == "__main__":
    main()
