#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guideline Manager - Quản lý Quy chuẩn Gán nhãn & Sổ Quyết Định
Hỗ trợ nạp động file Markdown, text, PDF text để VLM và Rule Engine đối soát trực tiếp.
"""

import os
import re
from typing import Dict, List, Any, Optional

DEFAULT_GUIDELINES = """# QUY CHUẨN GÁN NHÃN BDD100K 2D & SỔ QUYẾT ĐỊNH ĐỘI P-037

## 1. QUY CHUẨN HÌNH HỌC & ĐỘ KHÍT (BDD100K BBOX & SEGMENTATION)
§1.1. Bounding Box phải ôm sát các điểm cực biên ngoài cùng của vật thể (Tightness >= 80%). Không vẽ thừa nền đường quá 20% và không cắt cụt các phần thân xe, bánh xe, gương xe.
§1.2. Mọi tọa độ Bbox phải nằm trọn vẹn trong khung ảnh 1280x720 (0 <= x1 < x2 <= 1280, 0 <= y1 < y2 <= 720).
§1.3. Cấm tuyệt đối box có kích thước < 4px hoặc diện tích < 16px^2 (chấm rác).
§1.4. Đa giác Segmentation (Polygon) không được tự cắt chéo cạnh (self-intersect), đỉnh trùng lặp hoặc suy biến.
§1.5. Mask Segmentation và Bounding Box tương ứng phải đồng tâm và có độ chồng lấn IoU >= 0.70.

## 2. QUY ĐỊNH PHÂN LOẠI CATEGORY & THUỘC TÍNH (CLASSES & ATTRIBUTES)
§2.1. Phân biệt rõ giữa 'car' (xe con, sedan, SUV), 'truck' (xe tải có thùng hàng) và 'bus' (xe khách, xe buýt).
§2.2. Phân biệt rõ giữa 'pedestrian' (người đi bộ) và 'rider' (người đang ngồi/điều khiển xe máy, xe đạp).
§2.3. Bắt buộc điền đầy đủ 2 thuộc tính: 'occluded' (bị che khuất) và 'truncated' (bị cắt ở mép ảnh).
§2.4. Không gán nhãn các hình ảnh phản chiếu trên gương/kính hoặc hình vẽ quảng cáo xe trên thân xe buýt/tường nhà.

## 3. SỔ QUYẾT ĐỊNH ĐỘI P-037 (CÁC QUYẾT ĐỊNH ĐÃ BAN HÀNH)
[QĐ-001] Người ngồi sau xe máy có box 'pedestrian/rider' riêng: Mỗi người trên xe máy, kể cả người ngồi sau chỉ lộ đầu, bắt buộc phải có một box riêng biệt, không được gộp chung một box với xe.
[QĐ-002] Quy tắc nghiệm thu & trả job: Reviewer kiểm tra toàn bộ hoặc lấy mẫu. Nếu tỷ lệ ảnh sai > 10% thì TRẢ NGUYÊN JOB về cho Annotator sửa lại.
[QĐ-003] Quy tắc vật thể che khuất: Vật thể bị che khuất hơn 50%, box chỉ ôm phần nhìn thấy thực tế, không vẽ ước lượng phồng ra ngoài khoảng không.
[QĐ-004] Tính nhất quán: Số lượng box và vị trí các vật thể tĩnh phải đồng nhất qua các frame liên tiếp, không nhảy giật cục.
"""

class GuidelineManager:
    """Quản lý các quy chuẩn gán nhãn và trích xuất điều khoản phục vụ VLM / Rule audit."""

    def __init__(self, custom_text: Optional[str] = None):
        self.raw_text = custom_text if custom_text else DEFAULT_GUIDELINES
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

    def update_guideline(self, new_text: str):
        """Cập nhật văn bản quy chuẩn mới từ người dùng."""
        self.raw_text = new_text
        self._parse_rules()

    def get_guideline_text(self) -> str:
        return self.raw_text

    def find_relevant_clauses(self, error_type: str, category: str = "") -> List[Dict[str, str]]:
        """Tìm các điều khoản liên quan nhất tới loại lỗi."""
        matched = []
        err_lower = error_type.lower()
        cat_lower = category.lower()

        for r in self.rules:
            text_lower = (r["id"] + " " + r["content"]).lower()
            if "tight" in err_lower or "khít" in err_lower or "loose" in err_lower:
                if "1.1" in r["id"] or "khít" in text_lower:
                    matched.append(r)
            elif "out_of_image" in err_lower or "tràn" in err_lower or "viền" in err_lower:
                if "1.2" in r["id"] or "1280" in text_lower:
                    matched.append(r)
            elif "too_small" in err_lower or "rác" in err_lower:
                if "1.3" in r["id"] or "rác" in text_lower or "16" in text_lower:
                    matched.append(r)
            elif "polygon" in err_lower or "intersect" in err_lower or "self" in err_lower:
                if "1.4" in r["id"] or "cắt" in text_lower:
                    matched.append(r)
            elif "mask" in err_lower or "drift" in err_lower:
                if "1.5" in r["id"] or "mask" in text_lower:
                    matched.append(r)
            elif "rider" in err_lower or "pedestrian" in err_lower or "qđ-001" in err_lower:
                if "qđ-001" in r["id"].lower() or "2.2" in r["id"]:
                    matched.append(r)
            elif "class" in err_lower or "miss" in err_lower or "nhầm" in err_lower:
                if "2.1" in r["id"] or "2.2" in r["id"] or "qđ-001" in r["id"].lower():
                    matched.append(r)
            elif "occluded" in err_lower or "truncated" in err_lower or "attribute" in err_lower:
                if "2.3" in r["id"] or "qđ-003" in r["id"].lower():
                    matched.append(r)

        return matched if matched else self.rules[:3]

    def format_citation(self, error_type: str, category: str = "") -> str:
        """Tạo đoạn trích dẫn điều khoản ngắn gọn cho cột lỗi."""
        clauses = self.find_relevant_clauses(error_type, category)
        if clauses:
            top_clause = clauses[0]
            return f"{top_clause['id']}: {top_clause['content'][:85]}..."
        return "Quy chuẩn chất lượng BDD100K & Đội P-037"

# Instance mặc định dùng chung
global_guideline_manager = GuidelineManager()
