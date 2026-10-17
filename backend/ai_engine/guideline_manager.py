#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guideline Manager - Quản lý Quy chuẩn Gán nhãn & Sổ Quyết Định
Hỗ trợ nạp động file Markdown, text, PDF text để VLM và Rule Engine đối soát trực tiếp.
"""

import os
import re
from typing import Dict, List, Any, Optional

DEFAULT_GUIDELINES = """# TEMPLATE ONLY — no approved project guideline is configured

No acceptance policy or numeric thresholds are supplied by this template.
Upload the applicable project guideline and have its owner approve it before
using policy-dependent checks. Application defaults are software heuristics,
not official BDD100K rules or project requirements.
"""

class GuidelineManager:
    """Quản lý các quy chuẩn gán nhãn và trích xuất điều khoản phục vụ VLM / Rule audit."""

    def __init__(self, custom_text: Optional[str] = None):
        self.raw_text = custom_text if custom_text is not None else DEFAULT_GUIDELINES
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

        # Never attach an unrelated clause just to make the UI look cited.
        return matched

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
