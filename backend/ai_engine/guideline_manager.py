#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guideline Manager - Quản lý Quy chuẩn Gán nhãn & Sổ Quyết Định
Hỗ trợ nạp động file Markdown, text, PDF text để VLM và Rule Engine đối soát trực tiếp.
"""

import os
import re
from typing import Dict, List, Any, Optional

DEFAULT_GUIDELINES = """# QUY CHUẨN GÁN NHÃN 2D MẶC ĐỊNH CỦA QC STUDIO

## I. QUY ĐỊNH HÌNH HỌC & VẬT LÝ
§1.1 Độ khít Bounding Box: Box phải ôm sát toàn bộ đường biên vật thể (Tightness IoU >= 0.50). Không bao thừa quá 5px hoặc cắt lọt chi tiết.
§1.2 Giới hạn viền ảnh: Tọa độ box không được vượt ra ngoài chiều rộng và chiều cao thực tế của ảnh. Nếu vật thể chạm mép ảnh bắt buộc phải bật thuộc tính truncated=True.
§1.3 Kích thước tối thiểu: Không gán nhãn cho các vật thể quá nhỏ hoặc chấm rác nhiễu (Chiều rộng < 4px hoặc chiều cao < 4px hoặc diện tích < 16px²).
§1.4 Trùng lặp đối tượng: Hai box cùng class chồng lấn nhau với IoU >= 0.85 bị tính là lỗi trùng lặp gán nhãn (Duplicate Bounding Box).
§1.5 Tính toàn vẹn Polygon / Mask: Đa giác phân đoạn không được tự cắt cạnh (Self-intersecting), diện tích polygon phải > 1px².
§1.6 Đồng bộ Mask và Bounding Box: Độ lệch giữa Bounding Box và Mask không vượt quá 15px, IoU giữa Mask và Box phải đạt tối thiểu 0.70.

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
- rider: Người đang điều khiển phương tiện (lái xe máy, xe đạp, cưỡi ngựa). Lưu ý: Gán nhãn rider cho người điều khiển và gán nhãn phương tiện riêng cho xe.
§2.3 Biển báo & Đèn giao thông (Signage & Signals):
- traffic light: Cụm đèn tín hiệu giao thông (xanh, vàng, đỏ).
- traffic sign: Biển báo giao thông các loại gắn trên cột hoặc giá long môn.
§2.4 Thuộc tính bắt buộc (Attributes):
- occluded: Đánh dấu True nếu đối tượng bị che khuất một phần bởi vật thể khác (> 20%).
- truncated: Đánh dấu True nếu đối tượng bị cắt một phần bởi viền khung ảnh.

## III. CƠ CHẾ ĐỐI SOÁT HỘI ĐỒNG & PHÂN XỬ
§3.1 Đối soát bởi hai model; tên và trạng thái model phải lấy từ provider thực tế.
§3.2 Khi phát hiện có sự trùng khớp cùng class với IoU >= 0.30: Xác nhận gán đúng (PASS).
§3.3 Khi có bất đồng class hoặc nghi ngờ bỏ sót/gán thừa: Trọng tài Gemini Vision VLM phân xử và đưa ra phán quyết cuối cùng.
"""

class GuidelineManager:
    """Quản lý các quy chuẩn gán nhãn và trích xuất điều khoản phục vụ VLM / Rule audit."""

    def __init__(self, custom_text: Optional[str] = None, allow_empty: bool = False):
        if allow_empty and custom_text is not None:
            self.raw_text = custom_text
            self.is_default = False
        elif custom_text is None or not str(custom_text).strip():
            self.raw_text = DEFAULT_GUIDELINES
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
