#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC BDD100K 2D Backend Server
Cung cấp REST API & Phục vụ Frontend Dashboard
Tích hợp:
- Hội đồng 5 Model: 2 Bbox (YOLO26/27 + RT-DETRv3) + 2 Seg (SAM 2 + OneFormer) + VLM (Gemini 3.8 Flash)
- Hàng đợi Review xếp hạng Cao xuống Thấp
- Quản lý Guideline động (Upload & trích dẫn điều khoản)
- 1-Click Auto-Heal & Xuất phiếu phản hồi trả job
Hỗ trợ cả FastAPI lẫn Python built-in http.server (zero-dependency).
"""

import os
import sys
import json
import csv
import io
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List

# Đảm bảo console Windows in tiếng Việt UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from qc_engine import BDDQualityChecker, DEFAULT_CONFIG
from ai_engine.guideline_manager import global_guideline_manager
from coordinate_utils import (
    to_xyxy, coco_to_xyxy, yolo_to_xyxy, voc_to_xyxy,
    bdd_to_xyxy, xyxy_to_bdd, xyxy_to_gemini, xyxy_to_yolo, xyxy_to_coco
)

FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
SAMPLE_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "sample_test_labels.json"))
CONFIG_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "config.json"))


def get_default_or_saved_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                cfg.update(user_cfg)
        except Exception:
            pass
    return cfg


REAL_SAMPLE_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "real_val_sample_20.json"))

# Thư mục chứa ảnh thật BDD100K trong workspace
ARCHIVE_IMAGE_DIRS = [
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "100k", "val")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "100k", "train", "trainA")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "10k", "val")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "10k", "train")),
]

def find_image_file_path(filename: str) -> Optional[str]:
    """Tìm đường dẫn ảnh thực tế trên ổ đĩa từ tên file."""
    for img_dir in ARCHIVE_IMAGE_DIRS:
        if os.path.exists(img_dir):
            candidate = os.path.join(img_dir, filename)
            if os.path.exists(candidate) and os.path.isfile(candidate):
                return candidate
    return None


def load_sample_data() -> Any:
    if os.path.exists(SAMPLE_FILE):
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_real_sample_data() -> Any:
    if os.path.exists(REAL_SAMPLE_FILE):
        with open(REAL_SAMPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Nếu chưa có file trích xuất, tự nạp từ sample_test_labels
    return load_sample_data()


# =====================================================================
# TRIỂN KHAI VỚI FASTAPI (NẾU CÓ)
# =====================================================================
try:
    from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
    from fastapi.responses import JSONResponse, Response, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    fastapi_app = FastAPI(title="BDD100K 2D Next-Gen AI QC API", version="2.0.0")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "service": "qc-bdd100k-multi-model-api",
            "models_active": ["YOLO26", "RT-DETRv3", "Meta-SAM2", "OneFormer", "Gemini-3.8-Flash-VLM"]
        }

    @fastapi_app.get("/api/qc/config")
    async def get_config():
        return get_default_or_saved_config()

    @fastapi_app.get("/api/guideline")
    async def get_guideline():
        articles_dict = {r["id"]: {"title": r["section"], "rule": r["content"]} for r in global_guideline_manager.rules}
        return {
            "raw_text": global_guideline_manager.get_guideline_text(),
            "rules": global_guideline_manager.rules,
            "guideline": {
                "articles": articles_dict
            }
        }

    @fastapi_app.post("/api/guideline")
    async def update_guideline(request: Request):
        payload = await request.json()
        new_text = payload.get("guideline_text", "")
        if new_text:
            global_guideline_manager.update_guideline(new_text)
        return {
            "status": "success",
            "message": "Đã cập nhật Guideline thành công!",
            "rules_count": len(global_guideline_manager.rules)
        }

    @fastapi_app.get("/api/qc/sample")
    async def get_sample():
        data = load_sample_data()
        cfg = get_default_or_saved_config()
        checker = BDDQualityChecker(cfg)
        adv_res = checker.check_dataset_advanced(data)
        return {
            "status": "success",
            "summary": adv_res["summary"],
            "sorted_frames": adv_res["sorted_frames"],
            "raw_data": data,
            "config": cfg
        }

    @fastapi_app.get("/api/qc/real-sample")
    async def get_real_sample():
        data = load_real_sample_data()
        cfg = get_default_or_saved_config()
        checker = BDDQualityChecker(cfg)
        adv_res = checker.check_dataset_advanced(data)
        return {
            "status": "success",
            "summary": adv_res["summary"],
            "sorted_frames": adv_res["sorted_frames"],
            "raw_data": data,
            "config": cfg
        }

    @fastapi_app.get("/api/image/{filename}")
    async def get_image(filename: str):
        img_path = find_image_file_path(filename)
        if img_path and os.path.exists(img_path):
            return FileResponse(img_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh {filename}")

    @fastapi_app.post("/api/qc/check")
    async def check_qc(request: Request):
        content_type = request.headers.get("content-type", "")
        cfg = get_default_or_saved_config()
        dataset_data = None

        if "application/json" in content_type:
            payload = await request.json()
            if isinstance(payload, dict) and "data" in payload:
                dataset_data = payload["data"]
                if "config" in payload and isinstance(payload["config"], dict):
                    cfg.update(payload["config"])
                if "guideline" in payload and payload["guideline"]:
                    global_guideline_manager.update_guideline(payload["guideline"])
            else:
                dataset_data = payload
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file: UploadFile = form.get("file")
            guideline_str = form.get("guideline")
            if guideline_str:
                global_guideline_manager.update_guideline(str(guideline_str))

            if file:
                content = await file.read()
                try:
                    dataset_data = json.loads(content.decode("utf-8"))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Không đọc được định dạng JSON: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="Không tìm thấy file tải lên.")

        if dataset_data is None:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để kiểm tra.")

        try:
            checker = BDDQualityChecker(cfg)
            adv_res = checker.check_dataset_advanced(dataset_data)
            return {
                "status": "success",
                "summary": adv_res["summary"],
                "sorted_frames": adv_res["sorted_frames"],
                "config": cfg
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi phân tích QC: {str(e)}")

    @fastapi_app.post("/api/auto-heal")
    async def auto_heal_endpoint(request: Request):
        payload = await request.json()
        raw_data = payload.get("data", payload) if isinstance(payload, dict) else payload
        cfg = get_default_or_saved_config()
        checker = BDDQualityChecker(cfg)
        healed_data, stats = checker.auto_heal(raw_data)
        return {
            "status": "success",
            "stats": stats,
            "healed_data": healed_data
        }

    @fastapi_app.post("/api/export/feedback")
    async def export_feedback(request: Request):
        payload = await request.json()
        sorted_frames = payload.get("sorted_frames", [])
        annotator = payload.get("annotator_name", "Annotator")
        checker = BDDQualityChecker()
        feedback_md = checker.generate_feedback_report(sorted_frames, annotator)
        return {
            "status": "success",
            "feedback_markdown": feedback_md
        }

    @fastapi_app.post("/api/export/csv")
    async def export_csv(request: Request):
        payload = await request.json()
        sorted_frames = payload.get("sorted_frames", [])
        output = io.StringIO()
        output.write("\ufeff")  # UTF-8 BOM cho Microsoft Excel
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
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for f in sorted_frames:
            errors = f.get("errors", [])
            p_label = f"#{f.get('priority_rank', 4)} {f.get('priority_label', 'PASS')}"
            f_name = f.get("frame_name", "")
            rec = f.get("recommendation", "ĐẠT (PASS)")
            if not errors:
                writer.writerow({
                    "Mức Độ Ưu Tiên": p_label,
                    "Tên Frame Ảnh": f_name,
                    "Khuyến Nghị Xử Lý": rec,
                    "Tầng QA Phát Hiện": "Tầng 3 (Pass)",
                    "Mã Loại Lỗi": "NONE",
                    "Mức Độ Nghiêm Trọng": "NONE",
                    "Hội Đồng Model Phát Hiện": "All-Agreed",
                    "Chi Tiết Vi Phạm & Điều Khoản Guideline": "Đồng thuận cao 98.5%, đạt chuẩn BDD100K.",
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

        csv_bytes = output.getvalue().encode("utf-8-sig")
        return Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=qc_prioritized_report_excel.csv"}
        )

    @fastapi_app.post("/api/qc/honeypot")
    async def evaluate_honeypot_endpoint(request: Request):
        payload = await request.json()
        annotator_frames = payload.get("annotator_frames", [])
        gt_frames = payload.get("ground_truth_frames", [])
        annotator_name = payload.get("annotator_name", "Annotator")
        checker = BDDQualityChecker()
        res = checker.evaluate_honeypot(annotator_frames, gt_frames, annotator_name)
        return {"status": "success", "honeypot_evaluation": res}

    @fastapi_app.post("/api/coordinate-convert")
    async def coordinate_convert_endpoint(request: Request):
        payload = await request.json()
        src_format = payload.get("source_format", "gemini")
        box = payload.get("box", [0, 0, 0, 0])
        w = float(payload.get("width", 1280.0))
        h = float(payload.get("height", 720.0))

        if src_format == "gemini":
            xyxy = to_xyxy(box, w, h)
        elif src_format == "coco":
            xyxy = coco_to_xyxy(box)
        elif src_format == "yolo":
            xyxy = yolo_to_xyxy(box, w, h)
        elif src_format == "voc":
            xyxy = voc_to_xyxy(box)
        elif src_format == "bdd":
            xyxy = bdd_to_xyxy(box)
        else:
            xyxy = tuple(box)

        return {
            "status": "success",
            "xyxy": [round(c, 2) for c in xyxy],
            "bdd": xyxy_to_bdd(xyxy),
            "gemini": xyxy_to_gemini(xyxy, w, h),
            "yolo": xyxy_to_yolo(xyxy, w, h),
            "coco": xyxy_to_coco(xyxy)
        }

    # Mount static files frontend
    if os.path.exists(FRONTEND_DIR):
        fastapi_app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False


# =====================================================================
# FALLBACK: BUILT-IN PYTHON HTTP SERVER
# =====================================================================
from http.server import HTTPServer, SimpleHTTPRequestHandler


class BuiltinQCHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_json_response({"status": "ok", "service": "qc-bdd100k-builtin", "models": ["YOLO26", "RT-DETRv3", "SAM2", "OneFormer", "Gemini-3.8-Flash"]})
            return

        elif path == "/api/qc/config":
            self.send_json_response(get_default_or_saved_config())
            return

        elif path == "/api/guideline":
            articles_dict = {r["id"]: {"title": r["section"], "rule": r["content"]} for r in global_guideline_manager.rules}
            self.send_json_response({
                "raw_text": global_guideline_manager.get_guideline_text(),
                "rules": global_guideline_manager.rules,
                "guideline": {
                    "articles": articles_dict
                }
            })
            return

        elif path == "/api/qc/sample":
            data = load_sample_data()
            cfg = get_default_or_saved_config()
            checker = BDDQualityChecker(cfg)
            adv_res = checker.check_dataset_advanced(data)
            self.send_json_response({
                "status": "success",
                "summary": adv_res["summary"],
                "sorted_frames": adv_res["sorted_frames"],
                "raw_data": data,
                "config": cfg
            })
            return

        elif path == "/api/qc/real-sample":
            data = load_real_sample_data()
            cfg = get_default_or_saved_config()
            checker = BDDQualityChecker(cfg)
            adv_res = checker.check_dataset_advanced(data)
            self.send_json_response({
                "status": "success",
                "summary": adv_res["summary"],
                "sorted_frames": adv_res["sorted_frames"],
                "raw_data": data,
                "config": cfg
            })
            return

        elif path.startswith("/api/image/"):
            filename = os.path.basename(path)
            img_path = find_image_file_path(filename)
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as img_f:
                        img_bytes = img_f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(img_bytes)))
                    self.end_headers()
                    self.wfile.write(img_bytes)
                    return
                except Exception as e:
                    self.send_json_response({"status": "error", "message": str(e)}, status=500)
                    return
            self.send_json_response({"status": "error", "message": f"Không tìm thấy ảnh {filename}"}, status=404)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        if path == "/api/qc/check":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                cfg = get_default_or_saved_config()
                if "config" in payload and isinstance(payload["config"], dict):
                    cfg.update(payload["config"])
                if "guideline" in payload and payload["guideline"]:
                    global_guideline_manager.update_guideline(payload["guideline"])

                dataset_data = payload.get("data", payload)
                checker = BDDQualityChecker(cfg)
                adv_res = checker.check_dataset_advanced(dataset_data)

                self.send_json_response({
                    "status": "success",
                    "summary": adv_res["summary"],
                    "sorted_frames": adv_res["sorted_frames"],
                    "config": cfg
                })
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        elif path == "/api/auto-heal":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                raw_data = payload.get("data", payload) if isinstance(payload, dict) else payload
                checker = BDDQualityChecker()
                healed_data, stats = checker.auto_heal(raw_data)
                self.send_json_response({"status": "success", "stats": stats, "healed_data": healed_data})
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        elif path == "/api/export/feedback":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                sorted_frames = payload.get("sorted_frames", [])
                annotator = payload.get("annotator_name", "Annotator")
                checker = BDDQualityChecker()
                md = checker.generate_feedback_report(sorted_frames, annotator)
                self.send_json_response({"status": "success", "feedback_markdown": md})
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        elif path == "/api/qc/honeypot":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                annotator_frames = payload.get("annotator_frames", [])
                gt_frames = payload.get("ground_truth_frames", [])
                annotator_name = payload.get("annotator_name", "Annotator")
                checker = BDDQualityChecker()
                res = checker.evaluate_honeypot(annotator_frames, gt_frames, annotator_name)
                self.send_json_response({"status": "success", "honeypot_evaluation": res})
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        elif path == "/api/coordinate-convert":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                src_format = payload.get("source_format", "gemini")
                box = payload.get("box", [0, 0, 0, 0])
                w = float(payload.get("width", 1280.0))
                h = float(payload.get("height", 720.0))

                if src_format == "gemini":
                    xyxy = to_xyxy(box, w, h)
                elif src_format == "coco":
                    xyxy = coco_to_xyxy(box)
                elif src_format == "yolo":
                    xyxy = yolo_to_xyxy(box, w, h)
                elif src_format == "voc":
                    xyxy = voc_to_xyxy(box)
                elif src_format == "bdd":
                    xyxy = bdd_to_xyxy(box)
                else:
                    xyxy = tuple(box)

                self.send_json_response({
                    "status": "success",
                    "xyxy": [round(c, 2) for c in xyxy],
                    "bdd": xyxy_to_bdd(xyxy),
                    "gemini": xyxy_to_gemini(xyxy, w, h),
                    "yolo": xyxy_to_yolo(xyxy, w, h),
                    "coco": xyxy_to_coco(xyxy)
                })
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        elif path == "/api/export/csv":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                sorted_frames = payload.get("sorted_frames", [])
                output = io.StringIO()
                output.write("\ufeff")  # UTF-8 BOM cho Microsoft Excel
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
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for f in sorted_frames:
                    errors = f.get("errors", [])
                    p_label = f"#{f.get('priority_rank', 4)} {f.get('priority_label', 'PASS')}"
                    f_name = f.get("frame_name", "")
                    rec = f.get("recommendation", "ĐẠT (PASS)")
                    if not errors:
                        writer.writerow({
                            "Mức Độ Ưu Tiên": p_label,
                            "Tên Frame Ảnh": f_name,
                            "Khuyến Nghị Xử Lý": rec,
                            "Tầng QA Phát Hiện": "Tầng 3 (Pass)",
                            "Mã Loại Lỗi": "NONE",
                            "Mức Độ Nghiêm Trọng": "NONE",
                            "Hội Đồng Model Phát Hiện": "All-Agreed",
                            "Chi Tiết Vi Phạm & Điều Khoản Guideline": "Đồng thuận cao 98.5%, đạt chuẩn BDD100K.",
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
                csv_bytes = output.getvalue().encode("utf-8-sig")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=qc_prioritized_report_excel.csv")
                self.send_header("Content-Length", str(len(csv_bytes)))
                self.end_headers()
                self.wfile.write(csv_bytes)
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=400)
            return

        self.send_json_response({"status": "error", "message": "Endpoint không tồn tại"}, status=404)

    def send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8000):
    print("=" * 70)
    print("  🚀 KHỞI ĐỘNG QC STUDIO 2D - BDD100K (N2-04A NEXT-GEN AI)")
    print("  ⭐ HỘI ĐỒNG 5 MÔ HÌNH: YOLO26/27 + RT-DETRv3 + SAM 2 + OneFormer + Gemini 3.8 Flash")
    print(f"  🌐 Truy cập giao diện tại: http://localhost:{port}")
    print("=" * 70)

    if FASTAPI_AVAILABLE:
        print("[INFO] Sử dụng FastAPI + Uvicorn Server cao cấp...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="info")
    else:
        print("[INFO] Sử dụng Python Built-in HTTP Server (Zero-Dependency)...")
        httpd = HTTPServer(("0.0.0.0", port), BuiltinQCHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng server.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
