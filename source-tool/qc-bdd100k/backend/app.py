#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC BDD100K 2D Backend Server
Cung cấp REST API & Phục vụ Frontend Dashboard
Backend cung cấp rule QC, chuẩn hóa nhãn, so sánh với local detector/segmentation providers,
workspace/guideline, report và CVAT issue API. Inference chỉ chạy khi provider cục bộ sẵn sàng;
model findings là candidate để reviewer kiểm tra. Luồng mô phỏng cũ tắt mặc định.
FastAPI là server được hỗ trợ; Python built-in HTTP fallback chỉ dành cho compatibility.
"""

import os
import sys
import json
import csv
import io
import urllib.request
import urllib.error
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
from ai_engine.guideline_manager import GuidelineManager
from coordinate_utils import (
    to_xyxy, coco_to_xyxy, yolo_to_xyxy, voc_to_xyxy,
    bdd_to_xyxy, xyxy_to_bdd, xyxy_to_gemini, xyxy_to_yolo, xyxy_to_coco
)
from ai_engine.local_model_providers import available_providers, ProviderError
from qc_pipeline import run_bbox_pipeline, run_segmentation_pipeline
from gemini_provider import configured as gemini_configured, inspect_frame as gemini_inspect_frame, model_name as gemini_model_name
from workspace_store import (create_workspace, get_workspace, list_workspaces, update_profile,
                             save_guideline, activate_guideline, list_guidelines)

FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
SAMPLE_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "sample_test_labels.json"))
CONFIG_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "config.json"))


def load_env_file():
    candidates = [
        os.path.join(CURRENT_DIR, ".env"),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", ".env")),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".env")),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k not in os.environ or not os.environ[k]:
                                os.environ[k] = v
            except Exception:
                pass

load_env_file()

def get_cvat_config() -> tuple[str, str]:
    base_url = os.environ.get("CVAT_URL", "http://127.0.0.1:8080").rstrip("/")
    # Never ship a CVAT credential in source. Configure it in the backend environment.
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    return base_url, token

def get_cvat_headers(token: str) -> dict:
    clean_token = token.strip()
    if clean_token.startswith("Token "):
        clean_token = clean_token[6:].strip()
    elif clean_token.startswith("Bearer "):
        clean_token = clean_token[7:].strip()
    return {
        "Authorization": f"Token {clean_token}",
        "Accept": "application/vnd.cvat+json"
    }


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
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "100k", "val")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "100k", "train", "trainA")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "10k", "val")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "..", "archive", "bdd100k", "bdd100k", "images", "10k", "train")),
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
        allow_origins=["http://localhost:5000", "https://localhost:5001"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/api/health")
    async def health():
        providers = available_providers()
        ready = [p for p in providers if p["available"]]
        return {
            "status": "ok",
            "service": "local-dataset-qc-api",
            "models_mode": "local" if ready else "disabled",
            "models_configured": [p["id"] for p in ready],
            "capabilities": ["rule-based-geometry", "local-bbox-inference", "local-semantic-mask-comparison", "workspace-guideline-versioning", "cvat-issues"]
        }

    @fastapi_app.get("/api/models")
    async def get_models():
        providers = available_providers()
        return {"privacy": "Local inference is the default. Gemini is opt-in per run and sends selected images plus guideline to Google.", "providers": providers,
                "gemini": {"configured": gemini_configured(), "model": gemini_model_name(), "enabled": gemini_configured()}}

    @fastapi_app.get("/api/workspaces")
    async def workspaces_list():
        return {"workspaces": list_workspaces()}

    @fastapi_app.post("/api/workspaces")
    async def workspaces_create(request: Request):
        payload = await request.json()
        try:
            workspace = create_workspace(payload.get("name", ""), payload.get("profile", {}))
            return {"status": "success", "workspace": workspace}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @fastapi_app.get("/api/workspaces/{workspace_id}")
    async def workspaces_get(workspace_id: str):
        workspace = get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"workspace": workspace}

    @fastapi_app.put("/api/workspaces/{workspace_id}")
    async def workspaces_update(workspace_id: str, request: Request):
        try:
            workspace = update_profile(workspace_id, (await request.json()).get("profile", {}))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"status": "success", "workspace": workspace}

    @fastapi_app.post("/api/qc/run")
    async def run_workspace_qc(request: Request):
        payload = await request.json()
        workspace_id = payload.get("workspace_id")
        workspace = get_workspace(workspace_id) if workspace_id else None
        if workspace_id and workspace is None:
            raise HTTPException(status_code=404, detail="Workspace not found")
        profile = dict(workspace["profile"] if workspace else payload.get("profile", {}))
        task = payload.get("task_type", profile.get("task_type"))
        if task not in {"bbox", "semantic_segmentation"}:
            raise HTTPException(status_code=400, detail="task_type must be bbox or semantic_segmentation")
        profile.setdefault("task_type", task)
        if profile["task_type"] != task:
            raise HTTPException(status_code=409, detail="Workspace task_type does not match the current run")
        guideline_text, guideline_version = "", None
        if workspace:
            active = workspace.get("active_guideline")
            if active:
                guideline_text, guideline_version = active["text"], active["version"]
            profile["guideline_status"] = active["status"] if active else "not_configured"
        elif payload.get("guideline"):
            guideline_text = str(payload["guideline"])
            profile["guideline_status"] = "request_draft"
        else:
            profile["guideline_status"] = "not_configured"
        guideline_manager = GuidelineManager(guideline_text)
        images, masks = payload.get("images", {}), payload.get("masks", {})
        try:
            if task == "bbox":
                result = run_bbox_pipeline(payload.get("data"), images, profile, guideline_manager,
                                           threshold=float(payload.get("threshold", profile.get("model_threshold", 0.35))),
                                           match_iou=float(payload.get("match_iou", profile.get("match_iou_threshold", 0.5))))
            else:
                records = payload.get("data")
                if isinstance(records, dict):
                    records = records.get("frames", records.get("data"))
                result = run_segmentation_pipeline(records, images, masks, profile, guideline_manager,
                    threshold=float(payload.get("threshold", profile.get("model_threshold", 0.35))),
                    review_iou_threshold=float(payload.get("review_iou_threshold", profile.get("review_mask_iou_threshold", 0.5))))
        except (ValueError, ProviderError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        result["workspace_id"] = workspace_id
        result["guideline_version"] = guideline_version
        result["guideline_sha256"] = workspace["active_guideline"]["sha256"] if workspace and workspace.get("active_guideline") else None
        result["guideline_status"] = profile.get("guideline_status", "not_configured")
        # Cloud vision is strictly opt-in; the local pipeline runs first and remains the source of truth.
        if payload.get("use_gemini") is True:
            if not gemini_configured():
                raise HTTPException(status_code=400, detail="Gemini chưa có key. Dán key vào backend/GEMINI_API_KEY.txt rồi khởi động lại backend.")
            limit = max(1, min(50, int(payload.get("gemini_frame_limit", 10))))
            image_payloads = images if isinstance(images, dict) else {}
            source_records = payload.get("data")
            if isinstance(source_records, dict):
                source_records = source_records.get("frames", source_records.get("data", []))
            source_by_name = {str(row.get("name", "")): row for row in (source_records or []) if isinstance(row, dict)}
            gemini_skips, added = [], 0
            for frame in result.get("sorted_frames", [])[:limit]:
                name = frame.get("frame_name", "")
                encoded = image_payloads.get(name)
                if encoded:
                    import base64
                    encoded = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
                    try:
                        image_bytes = base64.b64decode(encoded, validate=True)
                    except Exception:
                        image_bytes = None
                else:
                    image_bytes = None
                    path = find_image_file_path(name)
                    if path:
                        with open(path, "rb") as image_file:
                            image_bytes = image_file.read()
                if not image_bytes:
                    gemini_skips.append({"frame_name": name, "reason": "image_not_available"})
                    continue
                source = source_by_name.get(name, {})
                raw_labels = source.get("labels", [])
                prompt_labels = []
                for label in raw_labels[:300] if isinstance(raw_labels, list) else []:
                    box = label.get("box2d") or label.get("bbox")
                    prompt_labels.append({"class": label.get("category", label.get("label", "")), "box": box})
                classes = profile.get("classes", profile.get("class_names", []))
                if not classes:
                    classes = sorted({str(label.get("category", label.get("label", ""))) for label in raw_labels if isinstance(label, dict) and label.get("category", label.get("label"))})
                try:
                    proposals = gemini_inspect_frame(image_bytes, guideline_text, prompt_labels, classes)
                except ValueError as exc:
                    # Preserve partial local results and report the cloud failure explicitly.
                    gemini_skips.append({"frame_name": name, "reason": str(exc)[:500]})
                    continue
                width, height = int(frame.get("width", 0)), int(frame.get("height", 0))
                if width <= 0 or height <= 0:
                    try:
                        from PIL import Image
                        with Image.open(io.BytesIO(image_bytes)) as source_image:
                            width, height = source_image.size
                    except Exception:
                        gemini_skips.append({"frame_name": name, "reason": "could_not_read_image_dimensions"})
                        continue
                for index, proposal in enumerate(proposals):
                    x1, y1, x2, y2 = proposal["box_normalized"]
                    box = [round(x1 * width / 1000), round(y1 * height / 1000), round(x2 * width / 1000), round(y2 * height / 1000)]
                    category = proposal["category"]
                    frame.setdefault("model_predictions", []).append({"category": category, "label": category, "type": "rectangle", "box": box,
                        "confidence": proposal["confidence"], "model": gemini_model_name(), "provider": "gemini"})
                    frame.setdefault("errors", []).append({"issue_id": f"{name}:gemini:{index}", "object_id": f"GEMINI_{index}", "category": category,
                        "severity": "MEDIUM", "error_type": "GEMINI_POSSIBLE_UNLABELED_OBJECT", "qa_layer": "Gemini cloud candidate",
                        "component": gemini_model_name(), "model_name": f"Gemini · {gemini_model_name()}", "model_confidence": proposal["confidence"],
                        "models_flagged": [gemini_model_name()], "explanation": f"Gemini đề xuất {category}: {proposal['evidence']} Đây chỉ là gợi ý chưa xác minh, không phải nhãn đúng.",
                        "action": "Đối chiếu ảnh và guideline; xác nhận hoặc loại bỏ đề xuất.", "suggested_box": box, "review_status": "pending",
                        "auto_healable": False, "guideline_rule": guideline_manager.format_citation("class", category)})
                    added += 1
                frame["gemini_candidate_count"] = len(proposals)
                frame["error_count"] = len(frame.get("errors", []))
                if proposals:
                    frame["priority_rank"] = min(frame.get("priority_rank", 4), 3)
                    if frame["priority_rank"] == 3:
                        frame["priority_label"] = "MODEL REVIEW CANDIDATE"
                        frame["priority_level"] = "MODEL REVIEW CANDIDATE"
                    frame["recommendation"] = "Review local/Gemini candidates against the source image and guideline"
            result["models"] = result.get("models", []) + [{"id": "gemini", "name": f"Gemini · {gemini_model_name()}", "provider": "Google Gemini API", "mode": "cloud_opt_in"}]
            result["gemini"] = {"used": True, "model": gemini_model_name(), "frames_requested": min(limit, len(result.get("sorted_frames", []))),
                                "frames_checked": min(limit, len(result.get("sorted_frames", []))) - len(gemini_skips),
                                "candidate_issues": added, "skips": gemini_skips,
                                "privacy_notice": "Selected image bytes and guideline text were sent to Google Gemini for this run."}
            result["model_issues"] = int(result.get("model_issues", 0)) + added
            result["summary"]["total_errors"] = sum(len(f.get("errors", [])) for f in result.get("sorted_frames", []))
            result["summary"]["model_candidate_issues"] = result["model_issues"]
            result["sorted_frames"].sort(key=lambda f: (f.get("priority_rank", 4), -len(f.get("errors", [])), f.get("frame_index", 0)))
        return result

    @fastapi_app.get("/api/qc/config")
    async def get_config():
        return get_default_or_saved_config()

    @fastapi_app.get("/api/guideline")
    async def get_guideline(request: Request):
        workspace_id = request.query_params.get("workspace_id")
        if workspace_id:
            workspace = get_workspace(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            versions = workspace.get("guidelines", [])
            active = workspace.get("active_guideline")
            manager = GuidelineManager(active["text"] if active else "")
            return {"workspace_id": workspace_id, "raw_text": manager.get_guideline_text(), "rules": manager.rules,
                    "active_version": active["version"] if active else None, "active_status": active["status"] if active else "not_configured",
                    "versions": [{k: v for k, v in row.items() if k != "text"} for row in versions]}
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
        workspace_id = payload.get("workspace_id")
        if workspace_id:
            try:
                status = "active" if payload.get("activate") is True else "draft"
                item = save_guideline(workspace_id, str(new_text), status)
                return {"status": "success", "workspace_id": workspace_id, "version": item["version"],
                        "sha256": item["sha256"], "guideline_status": item["status"]}
            except KeyError:
                raise HTTPException(status_code=404, detail="Workspace not found")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        if new_text:
            global_guideline_manager.update_guideline(new_text)
        return {
            "status": "success",
            "message": "Đã cập nhật Guideline thành công!",
            "rules_count": len(global_guideline_manager.rules)
        }

    @fastapi_app.post("/api/workspaces/{workspace_id}/guidelines/{version}/activate")
    async def activate_workspace_guideline(workspace_id: str, version: int):
        item = activate_guideline(workspace_id, version)
        if not item:
            raise HTTPException(status_code=404, detail="Workspace or guideline version not found")
        return {"status": "success", "guideline": {k: v for k, v in item.items() if k != "text"}}

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
        if os.path.basename(filename) != filename or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Tên file ảnh không hợp lệ.")
        img_path = find_image_file_path(filename)
        if img_path and os.path.exists(img_path):
            return FileResponse(img_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh {filename}")

    @fastapi_app.post("/api/qc/check")
    async def check_qc(request: Request):
        content_type = request.headers.get("content-type", "")
        cfg = get_default_or_saved_config()
        dataset_data = None
        request_guideline = ""

        if "application/json" in content_type:
            payload = await request.json()
            if isinstance(payload, dict) and "data" in payload:
                dataset_data = payload["data"]
                if "config" in payload and isinstance(payload["config"], dict):
                    cfg.update(payload["config"])
                request_guideline = str(payload.get("guideline", ""))
            else:
                dataset_data = payload
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file: UploadFile = form.get("file")
            guideline_str = form.get("guideline")
            request_guideline = str(guideline_str or "")

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
            checker = BDDQualityChecker(cfg, guideline_manager=GuidelineManager(request_guideline) if request_guideline else None)
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
        feedback_md += "\n\n## Run provenance\n\n"
        feedback_md += f"- Guideline version: {payload.get('guideline_version') or 'none'}\n- Guideline SHA-256: {payload.get('guideline_sha256') or 'none'}\n"
        feedback_md += "\n## Reviewer decisions\n\n| Frame | Issue | Decision |\n|---|---|---|\n"
        for frame in sorted_frames:
            for issue in frame.get("errors", []):
                feedback_md += f"| {frame.get('frame_name', '')} | {issue.get('issue_id', issue.get('error_type', 'issue'))} | {issue.get('review_status', 'pending')} |\n"
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
            "Đề Xuất Sửa Đổi Cho Annotator",
            "Reviewer Decision",
            "Guideline Version",
            "Guideline SHA-256"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for f in sorted_frames:
            errors = f.get("errors", [])
            p_label = f"#{f.get('priority_rank', 4)} {f.get('priority_label', 'NO RULE FLAGS')}"
            f_name = f.get("frame_name", "")
            rec = f.get("recommendation", "NO RULE FLAGS — NOT A VERIFIED PASS")
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
                    "Đề Xuất Sửa Đổi Cho Annotator": "Không cần sửa (Pass)",
                    "Reviewer Decision": "no candidate",
                    "Guideline Version": payload.get("guideline_version", ""),
                    "Guideline SHA-256": payload.get("guideline_sha256", "")
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
                        "Đề Xuất Sửa Đổi Cho Annotator": err.get("action", ""),
                        "Reviewer Decision": err.get("review_status", "pending"),
                        "Guideline Version": payload.get("guideline_version", ""),
                        "Guideline SHA-256": payload.get("guideline_sha256", "")
                    })

        csv_bytes = output.getvalue().encode("utf-8-sig")
        return Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=qc_prioritized_report_excel.csv"}
        )

    @fastapi_app.post("/api/export/segmentation-csv")
    async def export_segmentation_csv(request: Request):
        payload = await request.json()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["frame", "class_id", "class", "IoU", "intersection_pixels", "union_pixels", "prediction_supported", "review_threshold", "review_candidate_count", "review_decisions", "guideline_version", "guideline_sha256"])
        for frame in payload.get("sorted_frames", []):
            metrics = frame.get("model_segmentation_metrics", {})
            issues_by_class = {str(i.get("category", "")): i for i in frame.get("errors", [])}
            classes = metrics.get("class_iou", [])
            if not classes:
                writer.writerow([frame.get("frame_name", ""), "", "", "", "", "", "", metrics.get("mask_iou_threshold", ""), 0, "", payload.get("guideline_version", ""), payload.get("guideline_sha256", "")])
            for item in classes:
                candidate = issues_by_class.get(str(item.get("class_name", "")), {})
                writer.writerow([frame.get("frame_name", ""), item.get("class_id", ""), item.get("class_name", ""), item.get("iou", ""), item.get("intersection_pixels", ""), item.get("union_pixels", ""), item.get("prediction_supported", ""), metrics.get("mask_iou_threshold", ""), int(bool(candidate)), candidate.get("review_status", ""), payload.get("guideline_version", ""), payload.get("guideline_sha256", "")])
        return Response(content=output.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=qc-segmentation-class-iou.csv"})

    @fastapi_app.post("/api/export/guideline-draft")
    async def export_guideline_draft(request: Request):
        payload = await request.json()
        base_text = str(payload.get("guideline_text", "")).strip()
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for frame in payload.get("sorted_frames", []):
            for issue in frame.get("errors", []):
                if issue.get("review_status") != "confirmed":
                    continue
                key = (str(issue.get("error_type", "reviewed_issue")), str(issue.get("category", "general")))
                groups.setdefault(key, []).append(issue)
        lines = ["# Guideline update draft", "", "> Draft only. This file is not activated. Review every proposed clause before uploading and activating it.", "",
                 f"- Based on guideline version: {payload.get('guideline_version') or 'none'}",
                 f"- Base guideline SHA-256: {payload.get('guideline_sha256') or 'none'}",
                 f"- Human-confirmed issue groups: {len(groups)}", "", "## Current guideline", "", base_text or "(No base guideline text supplied.)", "", "## Proposed additions from confirmed review findings", ""]
        if not groups:
            lines.append("No additions generated: confirm relevant review issues first. No model-only or pending finding was used.")
        else:
            for (error_type, category), issues in sorted(groups.items()):
                actions = sorted({" ".join(str(i.get("action", "Inspect against the source image and approved ontology.")).split()) for i in issues})
                citations = sorted({" ".join(str(i.get("guideline_rule", "No current clause citation.")).split()) for i in issues})
                lines.extend([f"### {category} — {error_type}", "", f"- Confirmed occurrences in this run: {len(issues)}",
                              f"- Existing clause references: {'; '.join(citations)}",
                              f"- Reviewer action recorded: {'; '.join(actions)}", "",
                              "**Draft clause for policy-owner review:** When this condition is observed, inspect the source image and the applicable project ontology/guideline, then correct the annotation only after human verification.", ""])
        body = "\n".join(lines) + "\n"
        return {"status": "success", "draft_markdown": body, "confirmed_issue_groups": len(groups), "activated": False}

    @fastapi_app.post("/api/cvat/push-issues")
    async def push_issues_to_cvat(request: Request):
        """Create frame-mapped CVAT issues and optionally return the job (update stage/state)."""
        base_url, token = get_cvat_config()
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình. Thiết lập CVAT_URL và CVAT_ACCESS_TOKEN ở backend.")
        payload = await request.json()
        try:
            job_id = int(payload.get("job_id"))
            frame_offset = int(payload.get("frame_offset", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="job_id và frame_offset phải là số nguyên.")
        headers = get_cvat_headers(token)
        try:
            list_req = urllib.request.Request(f"{base_url}/api/issues?job_id={job_id}&page_size=1000", headers=headers)
            with urllib.request.urlopen(list_req, timeout=30) as response:
                page = json.loads(response.read().decode("utf-8"))
            existing = {(int(i.get("frame", -1)), str(i.get("message", ""))) for i in page.get("results", [])}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise HTTPException(status_code=502, detail=f"Không đọc được issue hiện có của CVAT ({exc.code}): {detail}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Không kết nối được CVAT: {exc}")

        frame_map = payload.get("frame_map", {})
        if frame_map and not isinstance(frame_map, dict):
            raise HTTPException(status_code=400, detail="frame_map must map uploaded image names to zero-based CVAT frame indices")
        include_all = payload.get("include_all_unreviewed", False)
        update_job_state = payload.get("update_job_state", True)
        target_stage = payload.get("target_stage", "annotation")
        target_state = payload.get("target_state", "rejected")

        created, skipped, skipped_unreviewed, failures = 0, 0, 0, []
        for frame in payload.get("sorted_frames", []):
            frame_name = str(frame.get("frame_name", "frame"))
            if frame_map:
                if frame_name not in frame_map:
                    if any(issue.get("review_status") == "confirmed" or include_all for issue in frame.get("errors", [])):
                        failures.append({"frame": frame_name, "error": "frame name is absent from the supplied CVAT mapping"})
                    continue
                try:
                    frame_id = int(frame_map[frame_name])
                except (TypeError, ValueError):
                    failures.append({"frame": frame_name, "error": "mapped CVAT frame index must be an integer"})
                    continue
            else:
                frame_id = frame_offset + int(frame.get("frame_index", 0))
            labels_by_id = {str(label.get("id", "")): label for label in frame.get("labels", [])}
            for issue in frame.get("errors", []):
                rev_status = issue.get("review_status", "pending")
                if rev_status == "dismissed":
                    continue
                if not include_all and rev_status != "confirmed":
                    skipped_unreviewed += 1
                    continue
                error_type = str(issue.get("error_type", "QC issue"))
                model_name = str(issue.get("model_name", ""))
                message = f"[QC Studio][{issue.get('severity', 'REVIEW')}][{issue.get('issue_id', error_type)}] {frame_name}: {error_type}. {issue.get('explanation', '')} Model: {model_name or 'rule'}; guideline: {issue.get('guideline_rule', '')}"
                if (frame_id, message) in existing:
                    skipped += 1
                    continue
                box = issue.get("suggested_box")
                if not box and str(issue.get("object_id", "")) in labels_by_id:
                    box = labels_by_id[str(issue.get("object_id"))].get("box2d")
                    if isinstance(box, dict):
                        box = [box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)]
                position = [0.0, 0.0]
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    position = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                issue_body_dict = {"frame": frame_id, "position": position, "job": job_id, "message": message}
                if payload.get("assignee_id") not in (None, ""):
                    try:
                        issue_body_dict["assignee"] = int(payload["assignee_id"])
                    except (TypeError, ValueError):
                        raise HTTPException(status_code=400, detail="assignee_id must be an integer CVAT user id")
                issue_body = json.dumps(issue_body_dict).encode("utf-8")
                issue_req = urllib.request.Request(f"{base_url}/api/issues", data=issue_body, headers={**headers, "Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(issue_req, timeout=30) as response:
                        if response.status not in (200, 201):
                            raise RuntimeError(f"HTTP {response.status}")
                    created += 1
                    existing.add((frame_id, message))
                except Exception as exc:
                    failures.append({"frame": frame_name, "error_type": error_type, "message": str(exc)[:300]})

        job_state_changed = False
        task_id = None
        actual_stage = None
        actual_state = None
        try:
            if update_job_state:
                patch_body = json.dumps({"stage": target_stage, "state": target_state}).encode("utf-8")
                patch_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", data=patch_body, headers={**headers, "Content-Type": "application/json"}, method="PATCH")
                with urllib.request.urlopen(patch_req, timeout=15) as patch_res:
                    if patch_res.status in (200, 204):
                        job_state_changed = True
                        res_data = json.loads(patch_res.read().decode("utf-8"))
                        task_id = res_data.get("task_id")
                        actual_stage = res_data.get("stage")
                        actual_state = res_data.get("state")
            if not task_id:
                job_info_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", headers=headers)
                with urllib.request.urlopen(job_info_req, timeout=10) as j_res:
                    j_info = json.loads(j_res.read().decode("utf-8"))
                    task_id = j_info.get("task_id")
                    actual_stage = actual_stage or j_info.get("stage")
                    actual_state = actual_state or j_info.get("state")
        except Exception as exc:
            failures.append({"error": f"Không cập nhật được trạng thái job trên CVAT: {exc}"})

        cvat_job_url = f"{base_url}/tasks/{task_id}/jobs/{job_id}" if task_id else f"{base_url}/jobs/{job_id}"

        return {
            "status": "partial" if failures else "success",
            "created": created,
            "skipped_existing": skipped,
            "skipped_unreviewed": skipped_unreviewed,
            "frame_mapping": "explicit" if frame_map else "confirmed_offset",
            "job_state_changed": job_state_changed,
            "job_id": job_id,
            "task_id": task_id,
            "job_stage": actual_stage,
            "job_state": actual_state,
            "cvat_job_url": cvat_job_url,
            "annotations_modified": False,
            "failed": failures
        }

    @fastapi_app.get("/api/cvat/jobs")
    async def cvat_list_jobs():
        """Retrieve recent active jobs from CVAT for 1-click selection."""
        base_url, token = get_cvat_config()
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình. Thiết lập CVAT_URL và CVAT_ACCESS_TOKEN.")
        headers = get_cvat_headers(token)
        try:
            req = urllib.request.Request(f"{base_url}/api/jobs?page_size=50", headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
            results = []
            for j in data.get("results", []):
                t_id = j.get("task_id")
                j_id = j.get("id")
                results.append({
                    "id": j_id,
                    "task_id": t_id,
                    "task_name": j.get("task_name") or f"Task #{t_id}",
                    "stage": j.get("stage"),
                    "state": j.get("state"),
                    "frame_count": j.get("frame_count"),
                    "mode": j.get("mode"),
                    "url": f"{base_url}/tasks/{t_id}/jobs/{j_id}" if t_id else f"{base_url}/jobs/{j_id}"
                })
            return {"status": "success", "jobs": results, "count": len(results)}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Không lấy được danh sách job từ CVAT: {exc}")

    @fastapi_app.get("/api/cvat/status")
    async def cvat_status():
        base_url, token = get_cvat_config()
        if not base_url or not token:
            return {"configured": False, "reachable": False, "detail": "Set CVAT_URL and CVAT_ACCESS_TOKEN in the backend process environment."}
        headers = get_cvat_headers(token)
        req = urllib.request.Request(f"{base_url}/api/server/about", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                info = json.loads(response.read().decode("utf-8"))
            return {"configured": True, "reachable": True, "version": info.get("version"), "name": info.get("name"), "url": base_url}
        except Exception as exc:
            return {"configured": True, "reachable": False, "detail": str(exc)[:400]}

    @fastapi_app.get("/api/cvat/jobs/{job_id}/frame-map")
    async def cvat_job_frame_map(job_id: int):
        base_url, token = get_cvat_config()
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT backend credentials are not configured")
        headers = get_cvat_headers(token)
        try:
            meta_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}/data/meta", headers=headers)
            with urllib.request.urlopen(meta_req, timeout=15) as response:
                metadata = json.loads(response.read().decode("utf-8"))
            frames_meta = metadata.get("frames", [])
            frame_map = {}
            for index, item in enumerate(frames_meta):
                if not isinstance(item, dict):
                    continue
                name = item.get("name", item.get("filename"))
                if name:
                    frame_map[os.path.basename(str(name))] = int(item.get("frame", index))
            if not frame_map:
                raise HTTPException(status_code=422, detail="CVAT metadata returned no filenames. Supply a mapping exported from the exact task/job.")
            return {"job_id": job_id, "frame_map": frame_map, "frame_count": len(frame_map), "mapping_source": "CVAT job data metadata"}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise HTTPException(status_code=502, detail=f"Could not read CVAT job metadata ({exc.code}): {detail}")

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
            simulated = os.environ.get("QC_ENABLE_SIMULATED_MODELS", "false").strip().lower() in {"1", "true", "yes"}
            self.send_json_response({"status": "ok", "service": "qc-bdd100k-builtin", "models_mode": "simulated" if simulated else "disabled", "models_configured": []})
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
                    p_label = f"#{f.get('priority_rank', 4)} {f.get('priority_label', 'NO RULE FLAGS')}"
                    f_name = f.get("frame_name", "")
                    rec = f.get("recommendation", "NO RULE FLAGS — NOT A VERIFIED PASS")
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
    print("  ℹ️ QC API local: BBox/semantic segmentation use loopback model providers; candidates need human review.")
    print(f"  🌐 API loopback: http://127.0.0.1:{port}")
    print("=" * 70)

    if FASTAPI_AVAILABLE:
        print("[INFO] Sử dụng FastAPI + Uvicorn Server cao cấp...")
        uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="info")
    else:
        print("[INFO] Sử dụng Python Built-in HTTP Server (Zero-Dependency)...")
        httpd = HTTPServer(("127.0.0.1", port), BuiltinQCHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng server.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
