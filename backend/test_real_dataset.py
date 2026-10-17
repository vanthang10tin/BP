#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra có giới hạn trên một số frame đầu của BDD100K validation.
Chỉ giải mã số frame được yêu cầu, chạy các luật đang bật và xuất CSV UTF-8 BOM.
"""

import os
import sys
import json
import csv
import argparse
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..", "tools")))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from qc_engine import BDDQualityChecker, DEFAULT_CONFIG
from prepare_bdd100k_val_pack import first_json_array_items

DEFAULT_INPUT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_data", "real_val_sample_20.json"))
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".qc-data", "reports"))

def run_real_dataset_test(num_frames: int = 50, input_path: str = DEFAULT_INPUT_PATH):
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY QC TRÊN BDD100K LABEL SAMPLE")
    print("=" * 70)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file nhãn đầu vào: {input_path}")

    print(f"[*] Đang nạp tối đa {num_frames} frames từ: {input_path}...")
    subset = first_json_array_items(input_path, num_frames)
    print(f"  ✓ Đã nạp thành công {len(subset)} frames.")

    # Giữ report tái tạo trong thư mục local, không ghi đè demo fixture được commit.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_subset_path = os.path.join(OUTPUT_DIR, "real_val_subset.json")
    output_csv_path = os.path.join(OUTPUT_DIR, "real_data_qc_report.csv")
    with open(output_subset_path, "w", encoding="utf-8") as f:
        json.dump(subset[:20], f, indent=2, ensure_ascii=False)
    print(f"  ✓ Đã lưu tối đa 20 frame vào: {output_subset_path}.")

    # Chạy QC Engine
    checker = BDDQualityChecker(DEFAULT_CONFIG)
    adv_res = checker.check_dataset_advanced(subset)

    summary = adv_res["summary"]
    sorted_frames = adv_res["sorted_frames"]

    print("\n" + "-" * 70)
    print("📊 KẾT QUẢ PHÂN TÍCH CHẤT LƯỢNG (4 TẦNG QA) TRÊN DỮ LIỆU THẬT:")
    print("-" * 70)
    print(f"  • Tổng số frame kiểm tra : {summary['total_frames']}")
    print(f"  • Tổng số vật thể (Labels): {summary['total_objects']}")
    print(f"  • Frame không có candidate: {summary['auto_pass_count']} ({summary['no_rule_findings_pct']}%) — chưa được xác nhận là sạch.")
    print(f"  • Frame có candidate      : {summary['need_review_count']} ({summary['rule_flagged_frame_pct']}%)")
    print(f"  • Trạng thái review       : {summary['job_verdict']}")
    print("  • Recall@20%: chưa đo (cần gold set độc lập được adjudicate).")

    print("\n🛡️ PHÂN BỐ LỖI THEO 4 TẦNG QA:")
    print(f"  • Tầng 1 (QC Luật - Hình học): {summary['layer_stats']['layer_1_rules_count']} lỗi")
    print(f"  • Tầng 2 (Model ObjectLab)   : {summary['layer_stats']['layer_2_models_count']} lỗi")
    print(f"  • Tầng 3 (Hàng đợi Ưu tiên)  : {len(sorted_frames)} frames xếp hạng")

    # Đếm các loại lỗi thực tế
    err_types = Counter()
    for f in sorted_frames:
        for err in f.get("errors", []):
            err_types[err["error_type"]] += 1

    print("\n🔍 TOP CÁC LOẠI LỖI THỰC TẾ TRONG TẬP BDD100K:")
    for err_type, count in err_types.most_common(10):
        print(f"  - [{err_type:28s}]: {count:4d} trường hợp")

    # Xuất file CSV UTF-8 với BOM chuẩn Microsoft Excel
    export_excel_compatible_csv(sorted_frames, output_csv_path)
    print(f"\n  ✓ Đã xuất CSV UTF-8 BOM: {output_csv_path}")
    print("=" * 70)

def export_excel_compatible_csv(sorted_frames, file_path):
    """
    Xuất file CSV chuẩn hóa với UTF-8 BOM (\ufeff).
    Dùng UTF-8 BOM để tăng khả năng Excel nhận diện encoding tiếng Việt.
    """
    fieldnames = [
        "Mức Độ Ưu Tiên",
        "Tên Frame Ảnh",
        "Khuyến Nghị Xử Lý",
        "Tầng QA Phát Hiện",
        "Mã Loại Lỗi",
        "Mức Độ Nghiêm Trọng",
        "Hội Đồng Model Phát Hiện",
        "Chi Tiết Vi Phạm & Điều Khoản Guideline",
        "Đề Xuất Sửa Đổi Cho Annotator"
    ]

    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for f_data in sorted_frames:
            errors = f_data.get("errors", [])
            p_label = f"#{f_data.get('priority_rank', 4)} {f_data.get('priority_label', 'PASS')}"
            f_name = f_data.get("frame_name", "")
            rec = f_data.get("recommendation", "ĐẠT (PASS)")

            if not errors:
                writer.writerow({
                    "Mức Độ Ưu Tiên": p_label,
                    "Tên Frame Ảnh": f_name,
                    "Khuyến Nghị Xử Lý": rec,
                    "Tầng QA Phát Hiện": "Tầng 3 (Pass)",
                    "Mã Loại Lỗi": "NONE",
                    "Mức Độ Nghiêm Trọng": "NONE",
                    "Hội Đồng Model Phát Hiện": "Không áp dụng (chưa nối model thật)",
                    "Chi Tiết Vi Phạm & Điều Khoản Guideline": "Không có lỗi theo các luật đang bật; chưa được xác nhận bằng model hoặc gold set độc lập.",
                    "Đề Xuất Sửa Đổi Cho Annotator": "Không cần sửa (Pass)"
                })
            else:
                for err in errors:
                    writer.writerow({
                        "Mức Độ Ưu Tiên": p_label,
                        "Tên Frame Ảnh": f_name,
                        "Khuyến Nghị Xử Lý": rec,
                        "Tầng QA Phát Hiện": err.get("qa_layer", "Tầng 1 (Luật)"),
                        "Mã Loại Lỗi": err.get("error_type", ""),
                        "Mức Độ Nghiêm Trọng": err.get("severity", ""),
                        "Hội Đồng Model Phát Hiện": ", ".join(err.get("models_flagged", [])),
                        "Chi Tiết Vi Phạm & Điều Khoản Guideline": err.get("explanation", ""),
                        "Đề Xuất Sửa Đổi Cho Annotator": err.get("action", "")
                    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QC on a real, explicitly selected BDD100K labels JSON file.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSON path (default: bundled 20-frame demo labels)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of leading records to process")
    args = parser.parse_args()
    run_real_dataset_test(num_frames=max(1, args.limit), input_path=os.path.abspath(args.input))
