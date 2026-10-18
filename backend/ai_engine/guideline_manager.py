#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guideline Manager - Quản lý Quy chuẩn Gán nhãn & Sổ Quyết Định
Hỗ trợ nạp động file Markdown, text, PDF text để VLM và Rule Engine đối soát trực tiếp.
"""

import os
import re
from typing import Dict, List, Any, Optional

DEFAULT_BBOX_GUIDELINES = """# QUY CHUẨN GÁN NHÃN 2D BOUNDING BOX (QC STUDIO)

## I. QUY ĐỊNH HÌNH HỌC & ĐỘ KHÍT (BOUNDING BOX)
§1.1 Độ khít Bounding Box: Box phải ôm sát toàn bộ các điểm cực biên ngoài cùng của vật thể (Tightness IoU >= 0.75). Không bao thừa quá 5px hoặc cắt lọt chi tiết bánh xe, gương, đèn, đầu người.
§1.2 Giới hạn viền ảnh: Tọa độ box không được vượt ra ngoài chiều rộng và chiều cao thực tế của ảnh (0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height). Nếu vật thể chạm mép ảnh bắt buộc phải bật thuộc tính truncated=True.
§1.3 Kích thước tối thiểu: Không gán nhãn cho các vật thể quá nhỏ hoặc chấm rác nhiễu (Chiều rộng < 6px hoặc chiều cao < 6px hoặc diện tích < 36px²).
§1.4 Trùng lặp đối tượng: Hai box cùng class chồng lấn nhau với IoU >= 0.70 bị tính là lỗi trùng lặp gán nhãn (Duplicate Bounding Box).
§1.5 Vật thể ảo / Bóng đổ: Không gán nhãn lên bóng đổ của xe trên mặt đường hoặc hình ảnh phản chiếu trên gương kính.

## II. DANH MỤC ĐỐI TƯỢNG (ONTOLOGY)
§2.1 Phân loại phương tiện giao thông (Vehicles):
- car: Ô tô con, sedan, hatchback, SUV, taxi, xe bán tải nhỏ gia đình.
- truck: Xe tải chở hàng, xe ben, container, xe bồn lớn.
- bus: Xe khách, xe buýt công cộng nhiều chỗ ngồi và cửa kính đặc trưng.
- motorcycle: Xe máy, mô tô 2 bánh, scooter có gắn động cơ.
- bicycle: Xe đạp thể thao, xe đạp thông thường không động cơ.
- train: Tàu hỏa, toa tàu, tàu điện chạy trên đường ray.
§2.2 Phân loại con người (Humans):
- pedestrian: Người đi bộ ở mọi tư thế (đứng, đi, chạy, ngồi trên vỉa hè).
- rider: Người đang điều khiển phương tiện (lái xe máy, xe đạp). Gán nhãn rider cho người điều khiển và gán nhãn phương tiện riêng cho xe.
§2.3 Biển báo & Đèn giao thông (Signage & Signals):
- traffic light: Cụm đèn tín hiệu giao thông (xanh, vàng, đỏ).
- traffic sign: Biển báo giao thông các loại gắn trên cột hoặc giá long môn.
§2.4 Thuộc tính bắt buộc (Attributes):
- occluded: Đánh dấu True nếu đối tượng bị che khuất một phần bởi vật thể khác (> 20%).
- truncated: Đánh dấu True nếu đối tượng bị cắt một phần bởi viền khung ảnh.

## III. CƠ CHẾ ĐỐI SOÁT HỘI ĐỒNG & PHÂN XỬ
§3.1 Đối soát nội bộ 3 model (YOLO26x, D-FINE-X, Gemini 3.8 Flash) tạo thành tập Box của AI.
§3.2 Khi 2 detector đồng thuận cùng class với IoU >= 0.35: Xác nhận kết quả AI thống nhất.
§3.3 Khi 2 detector bất đồng class hoặc chỉ 1 model phát hiện: Trọng tài Gemini 3.8 Flash phân xử để giữ lại box đúng.
§3.4 Điểm khác biệt giữa Box của Người và Box của AI tạo thành tập Box cần Review.
"""

DEFAULT_SEGMENTATION_GUIDELINES = """# QUY CHUẨN GÁN NHÃN SEMANTIC & INSTANCE SEGMENTATION (QC STUDIO)

## I. QUY ĐỊNH HÌNH HỌC & ĐỘ KHÍT MASK
§1.1 Độ khít biên Mask (Boundary Tightness): Mask phân đoạn phải bám sát chính xác từng pixel viền mép của đối tượng. Dung sai lệch viền không vượt quá 3px. Ngưỡng IoU tối thiểu giữa nhãn và đối tượng thực tế phải đạt >= 0.70 (đối với foreground) và >= 0.75 (đối với background).
§1.2 Chống trùng lấn giữa các lớp mutually exclusive (Zero-Overlap Policy): Các lớp nền như road, sidewalk, building, vegetation, sky không được phép chồng lấn lên nhau (Overlap IoU = 0). Nếu phát hiện 2 mask nền đè lên nhau, tính là lỗi trùng lấn sai quy định.
§1.3 Chống hở viền nền (No Gap Policy): Giữa mép đường (road) và vỉa hè (sidewalk), hoặc giữa vỉa hè và tòa nhà (building) không được để lại các khoảng trống chưa gán nhãn (unlabeled gap > 3px).
§1.4 Tính toàn vẹn Polygon / Mask: Đa giác phân đoạn không được tự cắt chéo cạnh (self-intersecting), không có đỉnh trùng lặp hoặc diện tích suy biến (< 16px²).
§1.5 Quy tắc thứ tự lớp (Painter's Algorithm): Các lớp nền tĩnh (road, sidewalk, building, wall, fence, pole, vegetation, terrain, sky) vẽ trước. Các đối tượng chuyển động / foreground (car, truck, bus, pedestrian, rider, motorcycle, bicycle, traffic sign, traffic light) vẽ đè lên trên lớp nền.

## II. DANH MỤC 19 LỚP PHÂN ĐOẠN (BDD100K / CITYSCAPES ONTOLOGY)
§2.1 Nhóm Mặt đường & Hạ tầng di chuyển:
- road: Toàn bộ mặt đường xe chạy (nhựa đường, bê tông, vạch kẻ đường, nắp cống trên đường).
- sidewalk: Vỉa hè lát gạch/bê tông dành cho người đi bộ, bó vỉa (curb).
§2.2 Nhóm Công trình & Vật cản cố định:
- building: Tòa nhà, nhà ở, trạm xăng, công trình kiến trúc xây dựng.
- wall: Bức tường gạch, tường bê tông độc lập không phải tường của tòa nhà.
- fence: Hàng rào kim loại, rào gỗ, lan can ngăn cách.
- pole: Cột đèn chiếu sáng, cột điện, cột biển báo giao thông (thanh trụ thẳng đứng).
§2.3 Nhóm Tín hiệu & Biển chỉ dẫn:
- traffic light: Hộp đèn tín hiệu giao thông (gồm cả vỏ hộp và bóng đèn).
- traffic sign: Biển báo giao thông các loại gắn trên cột hoặc giá treo.
§2.4 Nhóm Thiên nhiên & Môi trường:
- vegetation: Cây cối, tán lá, bụi cây, thảm cỏ, cành cây.
- terrain: Đất trống, đồi cát, sườn núi, mặt đất không lát gạch/nhựa.
- sky: Bầu trời thông thoáng, mây, không bao gồm chim/máy bay.
§2.5 Nhóm Con người:
- person / pedestrian: Người đi bộ, người đứng, trẻ em, người đi dạo.
- rider: Người đang cưỡi hoặc điều khiển xe đạp, xe máy, ngựa.
§2.6 Nhóm Phương tiện giao thông (Vehicles):
- car: Xe con 4-7 chỗ, sedan, hatchback, SUV, taxi, bán tải nhỏ.
- truck: Xe tải chở hàng, xe bồn, container, xe đầu kéo.
- bus: Xe buýt công cộng, xe khách liên tỉnh.
- train: Tàu hỏa, toa tàu đường sắt.
- motorcycle: Xe máy, xe mô tô 2 bánh, scooter.
- bicycle: Xe đạp các loại.

## III. CƠ CHẾ KIỂM TRA & ĐỐI SOÁT
§3.1 Đánh giá IoU theo từng class: Class có IoU < 0.65 sẽ bị đánh dấu Candidate cần Review.
§3.2 Phát hiện viền chưa khít hoặc viền lem nhem: So sánh chênh lệch giữa mask người gán và kết quả dự đoán của YOLO26x-Seg / SAM 2.1 Large.
§3.3 Trọng tài Gemini Vision VLM: Khi 1 model đạt mà 1 model báo lỗi viền, Gemini thẩm định vùng crop để xác nhận lỗi do người gán hay do model.
"""

DEFAULT_GUIDELINES = DEFAULT_BBOX_GUIDELINES

class GuidelineManager:
    """Quản lý các quy chuẩn gán nhãn và trích xuất điều khoản phục vụ VLM / Rule audit."""

    def __init__(self, custom_text: Optional[str] = None, allow_empty: bool = False, task_type: str = "bbox"):
        self.task_type = task_type
        default_for_task = DEFAULT_SEGMENTATION_GUIDELINES if task_type in ("segmentation", "semantic_segmentation") else DEFAULT_BBOX_GUIDELINES
        if allow_empty and custom_text is not None:
            self.raw_text = custom_text
            self.is_default = False
        elif custom_text is None or not str(custom_text).strip():
            self.raw_text = default_for_task
            self.is_default = True
        else:
            self.raw_text = str(custom_text)
            self.is_default = False
        self.rules: List[Dict[str, str]] = []
        self._parse_rules()

    def _parse_rules(self):
        """Phân tích các điều khoản từ văn bản quy chuẩn."""
        self.rules = []
        lines = self.raw_text.splitlines()
        current_section = "Quy định chung"

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_section = line.lstrip("#").strip()
                continue

            # Bắt các điều khoản dạng §1.1, [QĐ-001]...
            match_clause = re.match(r"^(\§\d+\.\d+|\[QĐ\-\d+\]|\d+\.\d+)\.?\s*(.+)$", line)
            if match_clause:
                clause_id = match_clause.group(1)
                content = match_clause.group(2)
                self.rules.append({
                    "id": clause_id,
                    "section": current_section,
                    "content": content,
                    "full_text": line
                })
        self.compiled_rules = None

    def update_guideline(self, new_text: str):
        """Cập nhật văn bản quy chuẩn mới từ người dùng."""
        self.raw_text = new_text
        self.is_default = False
        self._parse_rules()

    def get_guideline_text(self) -> str:
        return self.raw_text

    def get_compiled_rules(self, force_recompile: bool = False) -> Dict[str, Any]:
        """Lấy các quy tắc đã được biên dịch sang định dạng máy có thể thực thi."""
        if self.compiled_rules is None or force_recompile:
            try:
                from gemini_provider import compile_guideline
                self.compiled_rules = compile_guideline(self.raw_text)
            except Exception:
                from gemini_provider import _heuristic_compile_guideline
                self.compiled_rules = _heuristic_compile_guideline(self.raw_text)
        return self.compiled_rules

    def find_relevant_clauses(self, error_type: str, category: str = "") -> List[Dict[str, str]]:
        """Tìm các điều khoản liên quan nhất tới loại lỗi."""
        error = error_type.lower()
        groups = [
            (("tight", "khít", "loose"), ("tightness", "độ khít", "ôm sát")),
            (("out_of_image", "tràn", "viền"), ("viền ảnh", "kích thước khung ảnh", "image bounds")),
            (("too_small", "rác"), ("kích thước tối thiểu", "minimum size", "min_box", "diện tích tối thiểu")),
            (("polygon", "intersect", "self"), ("đa giác", "polygon", "self-intersect")),
            (("mask", "drift"), ("mask", "phân đoạn")),
            (("occluded", "truncated", "attribute"), ("thuộc tính", "attribute", "occluded", "truncated")),
            (("duplicate",), ("trùng lặp", "duplicate")),
            (("class", "miss", "nhầm"), ("phân loại", "ontology", "danh mục", "class", "bỏ sót")),
        ]
        keywords = next((words for errors, words in groups if any(term in error for term in errors)), ())
        # Clause numbers have no shared meaning across user policies.
        return [rule for rule in self.rules
                if any(word in rule["content"].lower() for word in keywords)]

    def format_citation(self, error_type: str, category: str = "") -> str:
        """Tạo đoạn trích dẫn điều khoản ngắn gọn cho cột lỗi."""
        clauses = self.find_relevant_clauses(error_type, category)
        if clauses:
            top_clause = clauses[0]
            txt = top_clause['content'].strip()
            if len(txt) > 50:
                txt = txt[:47] + "..."
            return f"{top_clause['id']}: {txt}"
        return "Chưa có điều khoản khớp trong guideline đã nạp"

# Instance mặc định dùng chung
global_guideline_manager = GuidelineManager()
