#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo 2 bộ test chuyên biệt với các lỗi người gán nhãn cố tình:
1. Bộ test BBox (test_bbox_suite_flawed.json):
   - Thiếu box (Unlabeled / Missing object)
   - Lệch box / Viền lỏng (Loose box / Tightness poor)
   - Thừa box / Box ma trên khoảng trống (Ghost box / False annotation)
   - Sai class (Wrong class: xe tải gắn nhãn là car)
   - Box quá bé / Nhiễu (Tiny box < 6px)
   - Box trùng lặp (Duplicate box IoU > 0.85)

2. Bộ test Segmentation (test_seg_suite_flawed.json + masks_flawed/):
   - Viền chưa khít (Under-segmentation: cắt cụt xe)
   - Tràn viền / Lem nhem (Over-segmentation / Leaking mask)
   - Trùng lấn lớp cấm (Mutual exclusion overlap: sidewalk lấn vào road)
   - Thiếu mask (Missing mask: xe có hình nhưng mask bị xóa)
   - Sai class mask (Wrong class: vỉa hè gán thành building)
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def generate_flawed_bbox_suite():
    print("=== Generating Flawed BBox Test Suite ===")
    src_json = Path("DEMO_PACK/BBOX/annotations.json")
    if not src_json.exists():
        src_json = Path("demo_data/real_val_sample_20.json")

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chọn frame tiêu biểu: b1c66a42-6f7d68ca.jpg (có nhiều xe và đường phố phong phú)
    target_frame_name = "b1c66a42-6f7d68ca.jpg"
    base_frame = None
    for f in data:
        if f.get("name") == target_frame_name:
            base_frame = json.loads(json.dumps(f))
            break
    if not base_frame:
        base_frame = json.loads(json.dumps(data[0]))
        target_frame_name = base_frame["name"]

    print(f"Using base frame: {target_frame_name}")

    # Xây dựng danh sách nhãn có lỗi cố ý
    # Giữ lại một số box chuẩn làm ground truth đối chiếu
    flawed_labels = []

    # 1. Box chuẩn (Correct baseline)
    flawed_labels.append({
        "id": "CORRECT_CAR_01",
        "category": "car",
        "box2d": {"x1": 1205.5, "y1": 412.1, "x2": 1279.7, "y2": 524.3},
        "attributes": {"occluded": False, "truncated": True},
        "_test_case": "CHUẨN: Cả người và AI đều thống nhất (Pass)"
    })

    # 2. Lỗi 1: LỆCH BOX / VIỀN LỎNG (Loose Box - Tightness IoU < 0.60)
    # Xe thực tế tại [253.59, 346.44, 347.13, 395.05] (kích thước ~ 94x49px)
    # Người gán vẽ thừa hẳn ra ngoài nền đường: [200.0, 310.0, 400.0, 440.0]
    flawed_labels.append({
        "id": "ERROR_LOOSE_BOX_02",
        "category": "car",
        "box2d": {"x1": 200.0, "y1": 310.0, "x2": 400.0, "y2": 440.0},
        "attributes": {"occluded": False, "truncated": False},
        "_test_case": "LỖI 1: Lệch box / Viền lỏng (Tightness kém, bao thừa nền đường > 20px)"
    })

    # 3. Lỗi 2: THỪA BOX / BOX MA (Ghost Box trên khoảng trống)
    # Người gán vẽ 1 box xe ở ngay giữa mặt đường trống ở làn trước camera
    flawed_labels.append({
        "id": "ERROR_GHOST_BOX_03",
        "category": "car",
        "box2d": {"x1": 480.0, "y1": 520.0, "x2": 660.0, "y2": 650.0},
        "attributes": {"occluded": False, "truncated": False},
        "_test_case": "LỖI 2: Thừa box / Box ma (Vẽ box xe trên mặt đường trống không có vật thể)"
    })

    # 4. Lỗi 3: SAI CLASS (Wrong Class: Xe tải gắn nhãn thành car)
    # Xe tải lớn ở vị trí [939.18, 336.56, 1205.70, 484.04] (AI xác định 73% truck)
    # Người gán lại gán nhầm là 'car'
    flawed_labels.append({
        "id": "ERROR_WRONG_CLASS_04",
        "category": "car",  # Sai class, đúng phải là truck!
        "box2d": {"x1": 939.0, "y1": 336.0, "x2": 1205.0, "y2": 484.0},
        "attributes": {"occluded": True, "truncated": False},
        "_test_case": "LỖI 3: Sai nhãn class (Xe tải lớn có thùng hàng nhưng gán là 'car')"
    })

    # 5. Lỗi 4: BOX QUÁ BÉ / NHIỄU (Tiny Box < 6px)
    # Box kích thước 4x4 px
    flawed_labels.append({
        "id": "ERROR_TINY_BOX_05",
        "category": "car",
        "box2d": {"x1": 350.0, "y1": 250.0, "x2": 354.0, "y2": 254.0},
        "attributes": {"occluded": False, "truncated": False},
        "_test_case": "LỖI 4: Box quá bé / Nhiễu vi phạm quy chuẩn (Kích thước 4x4px < 6px)"
    })

    # 6. Lỗi 5: BOX TRÙNG LẶP (Duplicate Box)
    # 2 box cùng gán cho chiếc xe tại [748.5, 360.2, 812.2, 410.6] với IoU > 0.90
    flawed_labels.append({
        "id": "ERROR_DUP_BOX_06A",
        "category": "car",
        "box2d": {"x1": 748.0, "y1": 360.0, "x2": 812.0, "y2": 410.0},
        "attributes": {"occluded": True, "truncated": False},
        "_test_case": "LỖI 5A: Box gốc của xe 06"
    })
    flawed_labels.append({
        "id": "ERROR_DUP_BOX_06B",
        "category": "car",
        "box2d": {"x1": 750.0, "y1": 361.0, "x2": 813.0, "y2": 411.0},
        "attributes": {"occluded": True, "truncated": False},
        "_test_case": "LỖI 5B: Box trùng lặp đè lên xe 06 (Duplicate IoU > 0.90)"
    })

    # Lưu ý về LỖI 6: THIẾU BOX (Unlabeled object)
    # Chiếc xe rất rõ ràng tại [791.79, 360.54, 899.48, 426.72] (conf 0.90)
    # CỐ TÌNH KHÔNG ĐƯỢC THÊM VÀO flawed_labels để hệ thống AI phát hiện và cắm cờ POSSIBLE_UNLABELED_OBJECT!

    test_suite = [
        {
            "name": target_frame_name,
            "width": 1280,
            "height": 720,
            "labels": flawed_labels,
            "_test_suite_metadata": {
                "description": "Bộ Test Bounding Box chuyên dụng chứa đầy đủ các lỗi người gán nhãn điển hình",
                "flaws_included": [
                    {"code": "MISSING_BOX", "desc": "Thiếu box: Xe tại [792, 361, 899, 427] không được gán nhãn"},
                    {"code": "LOOSE_BOX", "desc": "Lệch box: Xe ID ERROR_LOOSE_BOX_02 vẽ bao thừa hơn 20px ra ngoài"},
                    {"code": "GHOST_BOX", "desc": "Thừa box: Box ID ERROR_GHOST_BOX_03 vẽ trên mặt đường trống"},
                    {"code": "WRONG_CLASS", "desc": "Sai class: Xe tải lớn ID ERROR_WRONG_CLASS_04 bị gán nhãn 'car'"},
                    {"code": "TINY_BOX", "desc": "Box quá bé: Box ID ERROR_TINY_BOX_05 kích thước 4x4px vi phạm §1.3"},
                    {"code": "DUPLICATE_BOX", "desc": "Box trùng lặp: Hai box 06A và 06B đè lên nhau vi phạm §1.4"}
                ]
            }
        }
    ]

    out_paths = [
        Path("DEMO_PACK/BBOX/test_bbox_suite_flawed.json"),
        Path("demo_data/test_bbox_suite_flawed.json")
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(test_suite, f, indent=2, ensure_ascii=False)
        print(f"Saved: {p}")


def generate_flawed_seg_suite():
    print("\n=== Generating Flawed Segmentation Test Suite ===")
    src_json = Path("DEMO_PACK/SEGMENTATION/annotations.json")
    if not src_json.exists():
        print("DEMO_PACK/SEGMENTATION/annotations.json not found!")
        return

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame = data[0]
    frame_name = frame["name"]
    orig_mask_path = Path("DEMO_PACK/SEGMENTATION/masks_index") / frame_name.replace(".jpg", "_train_id.png")

    if not orig_mask_path.exists():
        print(f"Mask file not found: {orig_mask_path}")
        return

    orig_mask_img = Image.open(orig_mask_path)
    mask_arr = np.array(orig_mask_img, copy=True)
    height, width = mask_arr.shape

    # BDD100K Train IDs:
    # 0: road, 1: sidewalk, 2: building, 4: fence, 5: pole, 7: traffic sign, 8: vegetation, 10: sky, 13: car

    # Tìm các vùng của car (id 13) và road (id 0) và sidewalk (id 1)
    car_mask = (mask_arr == 13)
    road_mask = (mask_arr == 0)
    sidewalk_mask = (mask_arr == 1)

    flawed_mask_arr = mask_arr.copy()

    # 1. Lỗi 1: VIỀN CHƯA KHÍT / CẮT CỤT MASK (Under-segmentation)
    # Lấy 1 góc của car và cắt cụt 1 nửa chuyển thành road (id 0)
    car_ys, car_xs = np.where(car_mask)
    if len(car_ys) > 0:
        # Cắt góc trên của xe
        y_mid = int(np.median(car_ys))
        top_half = car_mask & (mask_arr == 13) & (np.arange(height)[:, None] < y_mid)
        flawed_mask_arr[top_half] = 0  # Cắt cụt nóc xe chuyển thành đường/nền

    # 2. Lỗi 2: TRÀN VIỀN / LEM NHEM (Over-segmentation / Leaking mask)
    # Lấy một vùng đường (id 0) gần vỉa hè hoặc xe và tô tràn màu xe (id 13) rộng 25px
    if len(car_ys) > 0:
        x_max = int(np.max(car_xs))
        y_center = int(np.median(car_ys))
        # Tạo vết lem hình chữ nhật 30x25 px tràn ra mặt đường
        flawed_mask_arr[max(0, y_center-15):min(height, y_center+15), x_max:min(width, x_max+30)] = 13

    # 3. Lỗi 3: TRÙNG LẤN LỚP CẤM (Mutual Exclusion Overlap: Vỉa hè đè lên giữa lòng đường)
    # Vẽ 1 mảng vỉa hè (id 1) lấn sâu vào giữa lòng đường xe chạy (road id 0)
    if road_mask.any():
        road_ys, road_xs = np.where(road_mask)
        ry_mid = int(np.median(road_ys))
        rx_mid = int(np.median(road_xs))
        flawed_mask_arr[ry_mid:ry_mid+40, rx_mid:rx_mid+60] = 1  # Đè sidewalk lên road!

    # 4. Lỗi 4: SAI CLASS MASK (Wrong Class: Vỉa hè gán thành building)
    if sidewalk_mask.any():
        sw_ys, sw_xs = np.where(sidewalk_mask)
        sw_y = int(np.min(sw_ys))
        sw_x = int(np.min(sw_xs))
        flawed_mask_arr[sw_y:sw_y+30, sw_x:sw_x+50] = 2  # Gán nhầm thành building!

    # Lưu mask index bị lỗi
    out_dir_masks = Path("DEMO_PACK/SEGMENTATION/masks_flawed")
    out_dir_masks.mkdir(parents=True, exist_ok=True)
    flawed_mask_file = out_dir_masks / f"flawed_{frame_name.replace('.jpg', '_train_id.png')}"
    Image.fromarray(flawed_mask_arr).save(flawed_mask_file)
    print(f"Saved flawed mask index: {flawed_mask_file}")

    # Tạo cả mask màu để xem trực quan
    palette_colors = {
        0: (128, 64, 128),   # road
        1: (244, 35, 232),   # sidewalk
        2: (70, 70, 70),     # building
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        7: (220, 220, 0),    # traffic sign
        8: (107, 142, 35),   # vegetation
        10: (70, 130, 180),  # sky
        13: (0, 0, 142),     # car
        255: (0, 0, 0)       # ignore
    }
    color_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for tid, col in palette_colors.items():
        color_arr[flawed_mask_arr == tid] = col
    color_mask_file = out_dir_masks / f"flawed_{frame_name.replace('.jpg', '_train_color.png')}"
    Image.fromarray(color_arr).save(color_mask_file)
    print(f"Saved flawed color mask: {color_mask_file}")

    # Tạo file manifest test segmentation
    test_seg_suite = [
        {
            "name": frame_name,
            "width": width,
            "height": height,
            "mask_index": str(flawed_mask_file).replace("\\", "/"),
            "mask_color": str(color_mask_file).replace("\\", "/"),
            "classes_present": frame.get("classes_present", []),
            "_test_suite_metadata": {
                "description": "Bộ Test Segmentation chuyên dụng chứa các lỗi phân đoạn cố tình",
                "flaws_included": [
                    {"code": "UNDER_SEGMENTATION", "desc": "Viền chưa khít / Cắt cụt nóc xe ô tô"},
                    {"code": "OVER_SEGMENTATION", "desc": "Tràn viền: Mask xe bị lem tràn ra mặt đường 30px"},
                    {"code": "MUTUAL_EXCLUSION_OVERLAP", "desc": "Trùng lấn lớp cấm: Mask sidewalk đè lên giữa lòng đường road"},
                    {"code": "WRONG_CLASS_MASK", "desc": "Sai class mask: Vùng vỉa hè bị gán nhãn thành building"}
                ]
            }
        }
    ]

    out_seg_paths = [
        Path("DEMO_PACK/SEGMENTATION/test_seg_suite_flawed.json"),
        Path("demo_data/test_seg_suite_flawed.json")
    ]
    for p in out_seg_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(test_seg_suite, f, indent=2, ensure_ascii=False)
        print(f"Saved: {p}")

if __name__ == "__main__":
    generate_flawed_bbox_suite()
    generate_flawed_seg_suite()
    print("\nSUCCESS: Created both flawed test suites with complete error taxonomies!")
