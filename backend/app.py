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
import math
import base64
import tempfile
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
SAMPLE_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_data", "sample_test_labels.json"))
REAL_SAMPLE_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_data", "real_val_sample_20.json"))
CONFIG_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "config.json"))
if not os.path.exists(SAMPLE_FILE):
    # fallback candidates
    for cand in [os.path.join(CURRENT_DIR, "sample_test_labels.json"), os.path.join(CURRENT_DIR, "..", "sample_test_labels.json")]:
        if os.path.exists(cand):
            SAMPLE_FILE = cand
            break
if not os.path.exists(CONFIG_FILE):
    for cand in [os.path.join(CURRENT_DIR, "..", "config.json"), os.path.join(CURRENT_DIR, "config", "bdd100k_segmentation_profile.json")]:
        if os.path.exists(cand):
            CONFIG_FILE = cand
            break


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

def public_demo_enabled() -> bool:
    return os.environ.get("QC_PUBLIC_DEMO", "false").strip().lower() in {"1", "true", "yes"}

def get_cvat_config(request: Any = None) -> tuple[str, str]:
    if request is not None and hasattr(request, "headers"):
        header_url = request.headers.get("x-cvat-url", "").strip()
        header_token = request.headers.get("x-cvat-token", "").strip()
        if header_url and header_token:
            normalized_url = header_url.rstrip("/")
            if normalized_url.endswith("/api"):
                normalized_url = normalized_url[:-4].rstrip("/")
            return normalized_url, header_token

    # Public demo instances must never use shared CVAT credentials, even if
    # they were accidentally added to the hosting environment.
    if public_demo_enabled():
        return "", ""
    base_url = os.environ.get("CVAT_URL", "http://127.0.0.1:8080").rstrip("/")
    if base_url.endswith("/api"):
        base_url = base_url[:-4].rstrip("/")
    parsed_url = urlparse(base_url)
    if parsed_url.hostname in {"127.0.0.1", "::1"}:
        port = f":{parsed_url.port}" if parsed_url.port else ""
        base_url = parsed_url._replace(netloc=f"localhost{port}").geturl().rstrip("/")
    # Never ship a CVAT credential in source. Configure it in the backend environment.
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    return base_url, token

def save_backend_env_values(values: dict[str, str]) -> None:
    env_path = os.path.join(CURRENT_DIR, ".env")
    try:
        with open(env_path, "r", encoding="utf-8", newline="") as env_file:
            existing = env_file.read()
    except FileNotFoundError:
        existing = ""

    output = []
    found = set()
    for line in existing.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        key, separator, _ = content.partition("=")
        config_key = key.strip()
        if separator and config_key in values:
            if config_key not in found:
                ending = line[len(content):]
                output.append(f"{config_key}={values[config_key]}{ending}")
                found.add(config_key)
        else:
            output.append(line)

    for config_key, value in values.items():
        if config_key not in found:
            if output and not output[-1].endswith(("\n", "\r")):
                output[-1] += os.linesep
            output.append(f"{config_key}={value}{os.linesep}")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=CURRENT_DIR, delete=False) as env_file:
            temporary_path = env_file.name
            env_file.write("".join(output))
        os.replace(temporary_path, env_path)
    except OSError:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

def save_cvat_config(base_url: str, token: str) -> None:
    normalized_url = base_url.strip().rstrip("/")
    if normalized_url.endswith("/api"):
        normalized_url = normalized_url[:-4].rstrip("/")
    clean_token = token.strip()
    parsed_url = urlparse(normalized_url)
    try:
        hostname = parsed_url.hostname
        parsed_url.port
    except ValueError:
        raise ValueError("CVAT_URL must be a valid HTTP(S) URL.")
    if (
        parsed_url.scheme not in {"http", "https"}
        or not hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
        or any(character.isspace() for character in normalized_url)
    ):
        raise ValueError("CVAT_URL must be an HTTP(S) URL without credentials, query, or fragment.")
    if not clean_token or any(character in clean_token for character in "\r\n\x00\u0085\u2028\u2029"):
        raise ValueError("CVAT_ACCESS_TOKEN is required and must be a single line.")

    save_backend_env_values({"CVAT_URL": normalized_url, "CVAT_ACCESS_TOKEN": clean_token})

    os.environ["CVAT_URL"] = normalized_url
    os.environ["CVAT_ACCESS_TOKEN"] = clean_token

def save_gemini_api_key(api_key: str) -> None:
    clean_key = api_key.strip()
    if not clean_key or any(character in clean_key for character in "\r\n\x00\u0085\u2028\u2029"):
        raise ValueError("A non-empty, single-line Gemini API key is required.")

    save_backend_env_values({"GEMINI_API_KEY": clean_key})
    os.environ["GEMINI_API_KEY"] = clean_key

def get_cvat_headers(token: str) -> dict:
    clean_token = token.strip()
    if clean_token.startswith("Token "):
        clean_token = clean_token[6:].strip()
    elif clean_token.startswith("Bearer "):
        clean_token = clean_token[7:].strip()
    return {
        "Authorization": f"Bearer {clean_token}",
        "Accept": "application/json, application/vnd.cvat+json, */*",
        "User-Agent": "QC-Studio-2D/2.0 (CVAT-Client)"
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


# Thư mục lưu cache ảnh cục bộ để load nhanh
CACHE_IMAGE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".qc-data", "images"))
os.makedirs(CACHE_IMAGE_DIR, exist_ok=True)

# Thư mục tìm kiếm ảnh trong workspace (ưu tiên demo_images và cache trước để load siêu tốc)
ARCHIVE_IMAGE_DIRS = [
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_data", "images")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "demo_images")),
    CACHE_IMAGE_DIR,
    os.path.abspath(os.path.join(os.getcwd(), "demo_data", "images")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "DEMO_PACK", "BBOX", "images")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "DEMO_PACK", "SEGMENTATION", "images")),
    os.path.abspath(os.path.join(os.getcwd(), "DEMO_PACK", "BBOX", "images")),
    os.path.abspath(os.path.join(os.getcwd(), "DEMO_PACK", "SEGMENTATION", "images")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "archive", "bdd100k", "bdd100k", "images", "100k", "val")),
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "archive", "bdd100k", "bdd100k", "images", "10k", "val")),
]

def find_image_file_path(filename: str) -> Optional[str]:
    """Tìm đường dẫn ảnh thực tế trên ổ đĩa từ tên file (ưu tiên SSD cache)."""
    from coordinate_utils import find_image_path
    return find_image_path(filename)


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
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "service": "local-dataset-qc-api",
            # A liveness check must not import/load large model checkpoints.
            # Provider availability is reported by /api/models on demand.
            "models_mode": "lazy local providers; see /api/models",
            "capabilities": ["rule-based-geometry", "local-bbox-inference", "local-semantic-mask-comparison", "workspace-guideline-versioning", "cvat-issues"]
        }

    @fastapi_app.get("/api/models")
    async def get_models():
        providers = available_providers()
        return {"privacy": "Local inference is the default. Gemini is opt-in per run and sends selected images plus guideline to Google.", "providers": providers,
                "gemini": {"configured": gemini_configured(), "model": gemini_model_name(), "enabled": gemini_configured()}}

    @fastapi_app.post("/api/gemini/config")
    async def save_gemini_config(request: Request):
        if public_demo_enabled():
            raise HTTPException(status_code=403, detail="Gemini configuration is disabled on the public demo.")
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
        if not isinstance(payload, dict) or not isinstance(payload.get("api_key"), str):
            raise HTTPException(status_code=400, detail="api_key must be provided as a string.")
        try:
            save_gemini_api_key(payload["api_key"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "success", "configured": True}

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
        result["compiled_rules"] = guideline_manager.get_compiled_rules()
        # Cloud vision is strictly opt-in; the local pipeline runs first and remains the source of truth.
        if payload.get("use_gemini") is True:
            if public_demo_enabled():
                raise HTTPException(status_code=403, detail="Gemini is disabled on the public demo.")
            if not gemini_configured():
                raise HTTPException(status_code=400, detail="Gemini API key is not configured. Add GEMINI_API_KEY to the backend .env file.")
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
                
                # Trích xuất phát hiện từ model cục bộ (YOLO / DINO) để Gemini đối soát
                model_preds = frame.get("model_predictions", [])
                prompt_model_preds = []
                for mp in model_preds[:50]:
                    prompt_model_preds.append({
                        "class": mp.get("category", mp.get("label", "")),
                        "box": mp.get("box"),
                        "confidence": mp.get("confidence", 0.0),
                        "model": mp.get("model", "local-detector")
                    })

                classes = profile.get("classes", profile.get("class_names", []))
                if not classes:
                    classes = sorted({str(label.get("category", label.get("label", ""))) for label in raw_labels if isinstance(label, dict) and label.get("category", label.get("label"))})
                try:
                    principles = guideline_manager.get_compiled_rules().get("key_principles", [])
                    compact_guideline = "\n- ".join(principles) if principles else guideline_text[:2000]
                    proposals = gemini_inspect_frame(
                        image_data=image_bytes,
                        guideline=compact_guideline,
                        labels=prompt_labels,
                        classes=classes,
                        model_predictions=prompt_model_preds
                    )
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
                    discrepancy = proposal.get("discrepancy_type", "GEMINI_POSSIBLE_UNLABELED_OBJECT")
                    m_status = proposal.get("model_status", "")
                    l_status = proposal.get("label_status", "")
                    guide_cit = proposal.get("guideline_citation") or guideline_manager.format_citation("class", category)
                    expl = f"[{discrepancy}] {proposal['evidence']}"
                    if m_status or l_status:
                        expl += f" (🔍 {m_status} | 🏷️ {l_status})"

                    severity = "HIGH" if ("UNLABELED" in discrepancy or "CONFLICT" in discrepancy) else "MEDIUM"
                    frame.setdefault("model_predictions", []).append({
                        "category": category, "label": category, "type": "rectangle", "box": box,
                        "confidence": proposal["confidence"], "model": gemini_model_name(), "provider": "gemini"
                    })
                    frame.setdefault("errors", []).append({
                        "issue_id": f"{name}:gemini:{index}", "object_id": f"GEMINI_{index}", "category": category,
                        "severity": severity, "error_type": discrepancy, "qa_layer": "Tầng 2 (Gemini VLM Arbitrator)",
                        "component": gemini_model_name(), "model_name": f"Gemini VLM · {gemini_model_name()}",
                        "model_confidence": proposal["confidence"], "models_flagged": [f"Gemini VLM ({gemini_model_name()})"],
                        "explanation": expl, "action": f"Đối chiếu Guideline ({guide_cit}): Phân xử xác nhận hoặc sửa lại nhãn.",
                        "suggested_box": box, "review_status": "pending", "auto_healable": False,
                        "guideline_rule": guide_cit, "model_status": m_status, "label_status": l_status
                    })
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
                    "compiled_rules": manager.get_compiled_rules(),
                    "active_version": active["version"] if active else None, "active_status": active["status"] if active else "not_configured",
                    "versions": [{k: v for k, v in row.items() if k != "text"} for row in versions]}
        articles_dict = {r["id"]: {"title": r["section"], "rule": r["content"]} for r in global_guideline_manager.rules}
        return {
            "raw_text": global_guideline_manager.get_guideline_text(),
            "rules": global_guideline_manager.rules,
            "compiled_rules": global_guideline_manager.get_compiled_rules(),
            "guideline": {
                "articles": articles_dict
            }
        }

    @fastapi_app.get("/api/guideline/compiled")
    async def get_compiled_guideline():
        return {
            "status": "success",
            "compiled_rules": global_guideline_manager.get_compiled_rules()
        }

    @fastapi_app.post("/api/guideline/compile")
    async def compile_guideline_route(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        new_text = payload.get("guideline_text", "")
        if new_text:
            global_guideline_manager.update_guideline(new_text)
        compiled = global_guideline_manager.get_compiled_rules(force_recompile=True)
        return {
            "status": "success",
            "message": "Đã biên dịch Guideline thành bộ quy tắc máy hiểu (Machine-executable Rules)!",
            "compiled_rules": compiled
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
                mgr = GuidelineManager(str(new_text))
                return {"status": "success", "workspace_id": workspace_id, "version": item["version"],
                        "sha256": item["sha256"], "guideline_status": item["status"],
                        "compiled_rules": mgr.get_compiled_rules(force_recompile=True)}
            except KeyError:
                raise HTTPException(status_code=404, detail="Workspace not found")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        if new_text:
            global_guideline_manager.update_guideline(new_text)
        compiled = global_guideline_manager.get_compiled_rules(force_recompile=True)
        return {
            "status": "success",
            "message": "Đã cập nhật và biên dịch Guideline thành công!",
            "rules_count": len(global_guideline_manager.rules),
            "compiled_rules": compiled
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
            return FileResponse(img_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
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
                from flexible_dataset_parser import MAX_UPLOAD_BYTES
                content = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Tệp upload vượt giới hạn 64 MiB.")
                try:
                    dataset_data = json.loads(content.decode("utf-8"))
                except Exception:
                    from flexible_dataset_parser import parse_flexible_dataset
                    try:
                        parsed = parse_flexible_dataset(content, file.filename or "uploaded_dataset")
                        dataset_data = parsed.get("frames", [])
                    except Exception as parse_err:
                        raise HTTPException(status_code=400, detail=f"Không đọc được định dạng file '{file.filename}': {parse_err}")
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

    @fastapi_app.post("/api/qc/gemini-inspect-frame")
    async def gemini_inspect_frame_endpoint(request: Request):
        if public_demo_enabled():
            raise HTTPException(status_code=403, detail="Gemini is disabled on the public demo.")
        """Đối soát chéo 3 bên (Model vs Nhãn Annotator vs Guideline) cho 1 frame bằng Gemini VLM."""
        if not gemini_configured():
            raise HTTPException(
                status_code=400,
                detail="Gemini API key chưa được cấu hình trên máy chủ. Bạn có thể thêm GEMINI_API_KEY vào biến môi trường hoặc file .env."
            )
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

        frame_name = payload.get("frame_name", "")
        encoded_image = payload.get("image_base64")
        image_bytes = None

        if encoded_image:
            import base64
            encoded_clean = encoded_image.split(",", 1)[-1] if encoded_image.startswith("data:") else encoded_image
            try:
                image_bytes = base64.b64decode(encoded_clean, validate=True)
            except Exception:
                image_bytes = None

        if not image_bytes and frame_name:
            path = find_image_file_path(frame_name)
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        image_bytes = f.read()
                except Exception:
                    image_bytes = None

        if not image_bytes:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu ảnh cho frame '{frame_name}'")

        guideline_text = payload.get("guideline") or global_guideline_manager.get_guideline_text()
        raw_labels = payload.get("labels", [])
        prompt_labels = []
        for l in (raw_labels if isinstance(raw_labels, list) else []):
            b = l.get("box2d") or l.get("bbox")
            prompt_labels.append({"class": l.get("category", l.get("label", "")), "box": b})

        raw_models = payload.get("model_predictions", [])
        prompt_model_preds = []
        for mp in (raw_models if isinstance(raw_models, list) else []):
            prompt_model_preds.append({
                "class": mp.get("category", mp.get("label", "")),
                "box": mp.get("box"),
                "confidence": mp.get("confidence", 0.0),
                "model": mp.get("model", "local-detector")
            })

        classes = payload.get("classes", ["car", "truck", "bus", "pedestrian", "rider", "bicycle", "motorcycle", "traffic light", "traffic sign", "train"])

        try:
            proposals = gemini_inspect_frame(
                image_data=image_bytes,
                guideline=guideline_text,
                labels=prompt_labels,
                classes=classes,
                model_predictions=prompt_model_preds
            )
            return {
                "status": "success",
                "frame_name": frame_name,
                "model": gemini_model_name(),
                "proposals": proposals,
                "total_proposals": len(proposals),
                "message": f"Gemini VLM đã đối soát 3 bên và phát hiện {len(proposals)} điểm bất thường/bất đồng."
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Lỗi khi gọi Gemini VLM: {exc}")


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
        payload = await request.json()
        custom_url = payload.get("cvat_url") if isinstance(payload, dict) else None
        custom_token = payload.get("cvat_access_token") if isinstance(payload, dict) else None
        if custom_url and custom_token:
            base_url = str(custom_url).rstrip("/")
            if base_url.endswith("/api"):
                base_url = base_url[:-4].rstrip("/")
            token = str(custom_token).strip()
        else:
            base_url, token = get_cvat_config(request)
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình. Thiết lập CVAT_URL và CVAT_ACCESS_TOKEN ở backend hoặc client.")
        try:
            job_id = int(payload.get("job_id"))
            frame_offset = int(payload.get("frame_offset", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="job_id và frame_offset phải là số nguyên.")
        if job_id <= 0 or frame_offset < 0:
            raise HTTPException(status_code=400, detail="job_id phải lớn hơn 0 và frame_offset không được âm.")
        frames = payload.get("sorted_frames")
        if not isinstance(frames, list) or not frames:
            raise HTTPException(status_code=400, detail="Không có kết quả QC frames để gửi lên CVAT.")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("errors", []), list):
                raise HTTPException(status_code=400, detail=f"QC frame {index} must be an object with an errors array.")
            try:
                int(frame.get("frame_index", index))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"QC frame {index} has an invalid frame_index.")
            if any(not isinstance(issue, dict) for issue in frame.get("errors", [])):
                raise HTTPException(status_code=400, detail=f"QC frame {index} contains a malformed issue.")
        frame_map = payload.get("frame_map", {})
        if not isinstance(frame_map, dict):
            raise HTTPException(status_code=400, detail="frame_map must map uploaded image names to zero-based CVAT frame indices")
        if not frame_map and payload.get("confirm_offset_mapping") is not True:
            raise HTTPException(status_code=400, detail="Xác nhận frame offset hoặc nạp frame map chính xác trước khi gửi.")
        include_all = payload.get("include_all_unreviewed", False)
        update_job_state = payload.get("update_job_state", True)
        if not isinstance(include_all, bool) or not isinstance(update_job_state, bool):
            raise HTTPException(status_code=400, detail="include_all_unreviewed và update_job_state phải là boolean.")
        target_stage = payload.get("target_stage", "annotation")
        target_state = payload.get("target_state", "rejected")
        if target_stage not in {"annotation", "validation", "acceptance"}:
            raise HTTPException(status_code=400, detail="target_stage không hợp lệ.")
        if target_state not in {"rejected", "in progress", "completed"}:
            raise HTTPException(status_code=400, detail="target_state không hợp lệ.")
        assignee_id = payload.get("assignee_id")
        if assignee_id not in (None, ""):
            try:
                assignee_id = int(assignee_id)
                if assignee_id <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="assignee_id must be a positive CVAT user id")

        headers = get_cvat_headers(token)
        try:
            job_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", headers=headers)
            with urllib.request.urlopen(job_req, timeout=15) as response:
                job_info = json.loads(response.read().decode("utf-8"))
            if int(job_info.get("id", -1)) != job_id:
                raise HTTPException(status_code=502, detail="CVAT returned a different job than requested.")
            task_id = job_info.get("task_id")

            metadata_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}/data/meta", headers=headers)
            with urllib.request.urlopen(metadata_req, timeout=15) as response:
                metadata = json.loads(response.read().decode("utf-8"))
            metadata_frames = metadata.get("frames")
            if not isinstance(metadata_frames, list) or not metadata_frames:
                raise HTTPException(status_code=502, detail="CVAT returned no frame metadata for this job.")
            valid_frame_ids = {
                int(item.get("frame", index))
                for index, item in enumerate(metadata_frames)
                if isinstance(item, dict)
            }
            if frame_map:
                for name, value in frame_map.items():
                    try:
                        frame_id = int(value)
                    except (TypeError, ValueError):
                        raise HTTPException(status_code=400, detail=f"CVAT frame mapping for {name} must be an integer.")
                    if frame_id not in valid_frame_ids:
                        raise HTTPException(status_code=400, detail=f"CVAT frame mapping for {name} points outside Job #{job_id}.")

            list_req = urllib.request.Request(f"{base_url}/api/issues?job_id={job_id}&page_size=1000", headers=headers)
            with urllib.request.urlopen(list_req, timeout=30) as response:
                page = json.loads(response.read().decode("utf-8"))
            cvat_raw_issues = page.get("results", [])
            existing = {(int(i.get("frame", -1)), str(i.get("message", ""))) for i in cvat_raw_issues}
            existing_by_frame = {}
            for item in cvat_raw_issues:
                existing_by_frame.setdefault(int(item.get("frame", -1)), []).append(item)
        except HTTPException:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise HTTPException(status_code=502, detail=f"Không xác minh được Job/frame/issues của CVAT ({exc.code}): {detail}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Không xác minh được Job/frame/issues của CVAT: {exc}")

        created, skipped, skipped_unreviewed, cvat_resolved, failures = 0, 0, 0, 0, []
        eligible_count = 0
        for frame in frames:
            frame_name = str(frame.get("frame_name", "frame"))
            needs_push = any(
                issue.get("review_status") not in {"rejected", "dismissed"}
                and (include_all or issue.get("review_status", "pending") == "confirmed")
                for issue in frame.get("errors", [])
            )
            if frame_map:
                if frame_name not in frame_map:
                    if needs_push:
                        failures.append({"frame": frame_name, "error": "frame name is absent from the supplied CVAT mapping"})
                    continue
                try:
                    frame_id = int(frame_map[frame_name])
                except (TypeError, ValueError):
                    failures.append({"frame": frame_name, "error": "mapped CVAT frame index must be an integer"})
                    continue
            else:
                frame_id = frame_offset + int(frame.get("frame_index", 0))
            if frame_id not in valid_frame_ids:
                if needs_push:
                    failures.append({"frame": frame_name, "error": f"frame index {frame_id} is outside CVAT Job #{job_id}"})
                continue
            labels_by_id = {str(label.get("id", "")): label for label in frame.get("labels", [])}
            for issue in frame.get("errors", []):
                if not isinstance(issue, dict):
                    failures.append({"frame": frame_name, "error": "QC issue must be an object"})
                    continue
                rev_status = issue.get("review_status", "pending")
                if rev_status in {"rejected", "dismissed"}:
                    continue
                if not include_all and rev_status != "confirmed":
                    skipped_unreviewed += 1
                    continue
                eligible_count += 1
                # Nếu issue này vốn từ CVAT kéo về: không tạo mới, mà chỉ cập nhật trạng thái nếu đã resolve
                cvat_orig_id = issue.get("cvat_issue_id")
                if cvat_orig_id:
                    if rev_status == "resolved":
                        try:
                            patch_data = json.dumps({"resolved": True}).encode("utf-8")
                            res_req = urllib.request.Request(f"{base_url}/api/issues/{cvat_orig_id}", data=patch_data, headers={**headers, "Content-Type": "application/json"}, method="PATCH")
                            with urllib.request.urlopen(res_req, timeout=15) as res_res:
                                if res_res.status in (200, 204):
                                    cvat_resolved += 1
                        except Exception as exc:
                            failures.append({"frame": frame_name, "error_type": error_type, "message": f"Could not sync resolved to CVAT issue #{cvat_orig_id}: {exc}"})
                    continue

                if (frame_id, message) in existing:
                    skipped += 1
                    continue

                # Deduplication chống đánh chồng: kiểm tra xem có issue cũ nào trên cùng frame đã có box tương tự (IoU >= 0.70)
                box = issue.get("suggested_box")
                if not box and str(issue.get("object_id", "")) in labels_by_id:
                    box = labels_by_id[str(issue.get("object_id"))].get("box2d")
                    if isinstance(box, dict):
                        box = [box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)]
                if isinstance(box, dict):
                    box = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
                try:
                    position = [float(value) for value in box]
                except (TypeError, ValueError):
                    failures.append({"frame": frame_name, "error_type": error_type, "error": "Issue has no valid evidence box; refusing to invent a CVAT position"})
                    continue
                if len(position) != 4 or not all(math.isfinite(value) for value in position) or position[2] <= position[0] or position[3] <= position[1]:
                    failures.append({"frame": frame_name, "error_type": error_type, "error": "Issue evidence box must contain finite x1,y1,x2,y2 coordinates with positive area"})
                    continue

                # Kiểm tra spatial overlap với các issue đã có trên CVAT
                is_overlap_dup = False
                for ex_iss in existing_by_frame.get(frame_id, []):
                    ex_pos = ex_iss.get("position", [])
                    if isinstance(ex_pos, list) and len(ex_pos) == 4:
                        # Simple IoU check
                        ix1 = max(position[0], ex_pos[0])
                        iy1 = max(position[1], ex_pos[1])
                        ix2 = min(position[2], ex_pos[2])
                        iy2 = min(position[3], ex_pos[3])
                        iw = max(0.0, ix2 - ix1)
                        ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_a = (position[2] - position[0]) * (position[3] - position[1])
                        area_b = (ex_pos[2] - ex_pos[0]) * (ex_pos[3] - ex_pos[1])
                        union = area_a + area_b - inter
                        if union > 0 and (inter / union) >= 0.70:
                            is_overlap_dup = True
                            break
                    for comm in ex_iss.get("comments", []):
                        c_text = comm.get("message", "")
                        if str(issue.get("issue_id", "")) in c_text or (issue.get("object_id") and f":{issue.get('object_id')}:" in c_text):
                            is_overlap_dup = True
                            break
                    if is_overlap_dup:
                        break

                if is_overlap_dup:
                    skipped += 1
                    continue

                issue_body_dict = {"frame": frame_id, "position": position, "job": job_id, "message": message}
                if assignee_id is not None:
                    issue_body_dict["assignee"] = assignee_id
                issue_body = json.dumps(issue_body_dict).encode("utf-8")
                issue_req = urllib.request.Request(f"{base_url}/api/issues", data=issue_body, headers={**headers, "Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(issue_req, timeout=30) as response:
                        if response.status not in (200, 201):
                            raise RuntimeError(f"HTTP {response.status}")
                    created += 1
                    existing.add((frame_id, message))
                    existing_by_frame.setdefault(frame_id, []).append({"position": position, "comments": [{"message": message}]})
                except Exception as exc:
                    failures.append({"frame": frame_name, "error_type": error_type, "message": str(exc)[:300]})

        if eligible_count == 0 and not failures:
            raise HTTPException(status_code=422, detail="Không có issue nào đủ điều kiện gửi lên CVAT. Confirm issue hoặc bật tùy chọn gửi candidate chưa duyệt.")

        job_state_changed = False
        actual_stage = job_info.get("stage")
        actual_state = job_info.get("state")
        if update_job_state and not failures:
            try:
                patch_body = json.dumps({"stage": target_stage, "state": target_state}).encode("utf-8")
                patch_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", data=patch_body, headers={**headers, "Content-Type": "application/json"}, method="PATCH")
                with urllib.request.urlopen(patch_req, timeout=15) as patch_res:
                    if patch_res.status not in (200, 204):
                        raise RuntimeError(f"CVAT returned HTTP {patch_res.status}")
                verify_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", headers=headers)
                with urllib.request.urlopen(verify_req, timeout=15) as verify_res:
                    verified_job = json.loads(verify_res.read().decode("utf-8"))
                actual_stage = verified_job.get("stage")
                actual_state = verified_job.get("state")
                job_state_changed = actual_stage == target_stage and actual_state == target_state
                if not job_state_changed:
                    failures.append({"error": f"CVAT job state did not match requested stage/state (actual: {actual_stage}/{actual_state})"})
            except Exception as exc:
                failures.append({"error": f"Could not confirm CVAT job state update: {exc}"})

        cvat_job_url = f"{base_url}/tasks/{task_id}/jobs/{job_id}" if task_id else None

        return {
            "status": "partial" if failures else "success",
            "created": created,
            "skipped_existing": skipped,
            "skipped_unreviewed": skipped_unreviewed,
            "cvat_resolved": cvat_resolved,
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
    async def cvat_list_jobs(request: Request):
        """Retrieve recent active jobs from CVAT for 1-click selection."""
        base_url, token = get_cvat_config(request)
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình. Thiết lập CVAT_URL và CVAT_ACCESS_TOKEN ở backend hoặc client.")
        headers = get_cvat_headers(token)
        try:
            req = urllib.request.Request(f"{base_url}/api/jobs?page_size=50", headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
            results = []
            for j in data.get("results", []):
                t_id = j.get("task_id")
                j_id = j.get("id")
                frame_cnt = j.get("frame_count")
                if frame_cnt is None and j.get("stop_frame") is not None and j.get("start_frame") is not None:
                    frame_cnt = j.get("stop_frame", 0) - j.get("start_frame", 0) + 1
                results.append({
                    "id": j_id,
                    "task_id": t_id,
                    "task_name": j.get("task_name") or f"Task #{t_id}",
                    "stage": j.get("stage"),
                    "state": j.get("state"),
                    "frame_count": frame_cnt or 0,
                    "mode": j.get("mode"),
                    "url": f"{base_url}/tasks/{t_id}/jobs/{j_id}" if t_id else f"{base_url}/jobs/{j_id}"
                })
            return {"status": "success", "jobs": results, "count": len(results)}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise HTTPException(status_code=502, detail=f"Lỗi từ CVAT ({exc.code}): {detail}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Không lấy được danh sách job từ CVAT: {exc}")

    @fastapi_app.get("/api/cvat/status")
    async def cvat_status(request: Request):
        base_url, token = get_cvat_config(request)
        if not base_url or not token:
            return {"configured": False, "reachable": False, "detail": "Set CVAT_URL and CVAT_ACCESS_TOKEN in the backend process environment or pass X-CVAT-Url / X-CVAT-Token."}
        headers = get_cvat_headers(token)
        req = urllib.request.Request(f"{base_url}/api/server/about", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                info = json.loads(response.read().decode("utf-8"))

            username = None
            try:
                user_req = urllib.request.Request(f"{base_url}/api/users/self", headers=headers)
                with urllib.request.urlopen(user_req, timeout=10) as u_res:
                    u_info = json.loads(u_res.read().decode("utf-8"))
                    username = u_info.get("username")
            except Exception:
                pass

            return {"configured": True, "reachable": True, "version": info.get("version"), "name": info.get("name"), "url": base_url, "username": username}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            return {"configured": True, "reachable": False, "detail": f"CVAT HTTP Error {exc.code}: {detail}"}
        except Exception as exc:
            return {"configured": True, "reachable": False, "detail": str(exc)[:400]}

    @fastapi_app.post("/api/cvat/config")
    async def cvat_save_config(request: Request):
        if public_demo_enabled():
            raise HTTPException(status_code=403, detail="CVAT configuration is disabled on the public demo.")
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        base_url = payload.get("cvat_url")
        token = payload.get("cvat_access_token")
        if not isinstance(base_url, str) or not isinstance(token, str):
            raise HTTPException(status_code=400, detail="CVAT_URL and CVAT_ACCESS_TOKEN must be strings.")
        try:
            save_cvat_config(base_url, token)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "success", "configured": True}

    @fastapi_app.get("/api/cvat/jobs/{job_id}/frame-map")
    async def cvat_job_frame_map(job_id: int, request: Request):
        base_url, token = get_cvat_config(request)
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

    @fastapi_app.post("/api/cvat/issues/{issue_id}/resolve")
    async def cvat_resolve_issue(issue_id: int, request: Request):
        """Allow reviewer to resolve or reopen an annotator issue directly on CVAT."""
        base_url, token = get_cvat_config(request)
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình.")
        headers = get_cvat_headers(token)
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            pass
        resolved = bool(payload.get("resolved", True))
        patch_data = json.dumps({"resolved": resolved}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/issues/{issue_id}", data=patch_data, headers={**headers, "Content-Type": "application/json"}, method="PATCH")
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                if res.status not in (200, 204):
                    raise RuntimeError(f"HTTP {res.status}")
            return {"status": "success", "issue_id": issue_id, "resolved": resolved}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Không thể cập nhật trạng thái issue #{issue_id} trên CVAT: {exc}")

    @fastapi_app.post("/api/cvat/jobs/{job_id}/pull")
    async def cvat_pull_job(job_id: int, request: Request = None):
        """Pull annotations, frame metadata, and frame images directly from a CVAT job for 1-click review."""
        payload = {}
        if request:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
        if payload.get("cvat_url") and payload.get("cvat_access_token"):
            base_url = str(payload["cvat_url"]).rstrip("/")
            if base_url.endswith("/api"):
                base_url = base_url[:-4].rstrip("/")
            token = str(payload["cvat_access_token"]).strip()
        else:
            base_url, token = get_cvat_config(request)
        if not base_url or not token:
            raise HTTPException(status_code=503, detail="CVAT chưa được cấu hình. Thiết lập CVAT_URL và CVAT_ACCESS_TOKEN ở backend hoặc client.")
        headers = get_cvat_headers(token)
        download_images = payload.get("download_images", True)
        max_frames = int(payload.get("max_frames", 100))

        try:
            # 1. Job details
            job_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", headers=headers)
            with urllib.request.urlopen(job_req, timeout=15) as res:
                job_info = json.loads(res.read().decode("utf-8"))
            task_id = job_info.get("task_id")
            task_name = job_info.get("task_name") or f"Task #{task_id}"

            # 2. Labels for this task
            label_map = {}
            try:
                labels_req = urllib.request.Request(f"{base_url}/api/labels?task_id={task_id}&page_size=500", headers=headers)
                with urllib.request.urlopen(labels_req, timeout=15) as res:
                    labels_data = json.loads(res.read().decode("utf-8"))
                label_map = {l["id"]: l["name"] for l in labels_data.get("results", []) if isinstance(l, dict) and "id" in l}
            except Exception:
                try:
                    task_req = urllib.request.Request(f"{base_url}/api/tasks/{task_id}", headers=headers)
                    with urllib.request.urlopen(task_req, timeout=15) as res:
                        task_data = json.loads(res.read().decode("utf-8"))
                    label_map = {l["id"]: l["name"] for l in task_data.get("labels", []) if isinstance(l, dict) and "id" in l}
                except Exception:
                    label_map = {}

            # 3. Frame metadata
            meta_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}/data/meta", headers=headers)
            with urllib.request.urlopen(meta_req, timeout=15) as res:
                meta_data = json.loads(res.read().decode("utf-8"))
            frames_meta = meta_data.get("frames", [])

            # 4. Annotations
            anno_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}/annotations", headers=headers)
            with urllib.request.urlopen(anno_req, timeout=30) as res:
                annotations = json.loads(res.read().decode("utf-8"))

            shapes_by_frame: dict[int, list[dict[str, Any]]] = {}

            # Parse standalone shapes
            for s in annotations.get("shapes", []):
                frame_idx = int(s.get("frame", 0))
                lbl_name = label_map.get(s.get("label_id"), str(s.get("label_id", "object")))
                pts = s.get("points")
                if isinstance(pts, str):
                    pts = [float(p) for p in pts.replace(",", " ").split() if p.strip()]
                if isinstance(pts, list) and len(pts) >= 4:
                    x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                    shapes_by_frame.setdefault(frame_idx, []).append({
                        "id": str(s.get("id", len(shapes_by_frame.get(frame_idx, [])))),
                        "category": lbl_name,
                        "box2d": {"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)},
                        "attributes": {"occluded": bool(s.get("occluded", False)), "truncated": False}
                    })

            # Parse tracks
            for t in annotations.get("tracks", []):
                lbl_name = label_map.get(t.get("label_id"), str(t.get("label_id", "object")))
                for s in t.get("shapes", []):
                    frame_idx = int(s.get("frame", 0))
                    if s.get("outside") is True:
                        continue
                    pts = s.get("points")
                    if isinstance(pts, str):
                        pts = [float(p) for p in pts.replace(",", " ").split() if p.strip()]
                    if isinstance(pts, list) and len(pts) >= 4:
                        x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                        shapes_by_frame.setdefault(frame_idx, []).append({
                            "id": str(s.get("id", f"{t.get('id')}_{frame_idx}")),
                            "category": lbl_name,
                            "box2d": {"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)},
                            "attributes": {"occluded": bool(s.get("occluded", False)), "truncated": False}
                        })

            # 4.5 Existing CVAT issues
            cvat_issues_by_frame: dict[int, list[dict[str, Any]]] = {}
            cvat_total_issues = 0
            try:
                iss_req = urllib.request.Request(f"{base_url}/api/issues?job_id={job_id}&page_size=1000", headers=headers)
                with urllib.request.urlopen(iss_req, timeout=15) as iss_res:
                    cvat_iss_data = json.loads(iss_res.read().decode("utf-8"))
                for item in cvat_iss_data.get("results", []):
                    f_idx = int(item.get("frame", 0))
                    cvat_id = int(item.get("id", 0))
                    comments = item.get("comments", [])
                    comment_text = comments[0].get("message", "Ghi chú trên CVAT") if comments else "Annotator issue trên CVAT"
                    author = comments[0].get("author", {}).get("username", "") if comments else ""
                    position = item.get("position", [])
                    is_res = bool(item.get("resolved", False))
                    cvat_issues_by_frame.setdefault(f_idx, []).append({
                        "issue_id": f"cvat:{cvat_id}",
                        "cvat_issue_id": cvat_id,
                        "object_id": f"CVAT_{cvat_id}",
                        "category": "CVAT Issue",
                        "severity": "LOW" if is_res else "HIGH",
                        "error_type": "CVAT_ANNOTATOR_ISSUE",
                        "qa_layer": "CVAT Issue",
                        "component": f"CVAT User ({author})" if author else "CVAT Platform",
                        "model_name": author or "CVAT",
                        "models_flagged": [author or "CVAT"],
                        "explanation": f"[{'ĐÃ RESOLVE' if is_res else 'CHƯA RESOLVE'}] {comment_text}",
                        "action": "Reviewer kiểm tra và Resolve hoặc Reopen.",
                        "suggested_box": position if len(position) >= 4 else None,
                        "review_status": "resolved" if is_res else "pending",
                        "cvat_resolved": is_res,
                        "auto_healable": False,
                        "guideline_rule": "CVAT Task Issue"
                    })
                    cvat_total_issues += 1
            except Exception as err:
                print(f"Warning: could not fetch existing issues from CVAT: {err}")

            # 5. Build frames list and frame map
            frames = []
            frame_map = {}
            for index, item in enumerate(frames_meta[:max_frames]):
                frame_idx = int(item.get("frame", index))
                name = os.path.basename(str(item.get("name", f"frame_{frame_idx:04d}.jpg")))
                w = int(item.get("width", 1280))
                h = int(item.get("height", 720))
                frame_map[name] = frame_idx
                labels = shapes_by_frame.get(frame_idx, [])
                initial_errors = cvat_issues_by_frame.get(frame_idx, [])
                frames.append({
                    "name": name,
                    "frame_index": frame_idx,
                    "width": w,
                    "height": h,
                    "labels": labels,
                    "errors": initial_errors
                })

            # 6. Download images if requested (Parallel fetch + Disk Cache)
            images = {}
            if download_images:
                from concurrent.futures import ThreadPoolExecutor

                def _fetch_frame(f):
                    idx = f["frame_index"]
                    fname = f["name"]
                    local_cache_path = os.path.join(CACHE_IMAGE_DIR, fname)
                    if os.path.exists(local_cache_path) and os.path.getsize(local_cache_path) > 0:
                        try:
                            with open(local_cache_path, "rb") as cf:
                                data = cf.read()
                            b64 = base64.b64encode(data).decode("ascii")
                            return fname, f"data:image/jpeg;base64,{b64}"
                        except Exception:
                            pass
                    img_url = f"{base_url}/api/jobs/{job_id}/data?type=frame&number={idx}&quality=compressed"
                    try:
                        img_req = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(img_req, timeout=15) as img_res:
                            data = img_res.read()
                            try:
                                with open(local_cache_path, "wb") as cf:
                                    cf.write(data)
                            except Exception:
                                pass
                            b64 = base64.b64encode(data).decode("ascii")
                            return fname, f"data:image/jpeg;base64,{b64}"
                    except Exception as err:
                        print(f"Warning: could not fetch frame {idx} from CVAT: {err}")
                        return fname, None

                with ThreadPoolExecutor(max_workers=8) as executor:
                    for fname, img_val in executor.map(_fetch_frame, frames):
                        if img_val:
                            images[fname] = img_val

            return {
                "status": "success",
                "job_id": job_id,
                "task_id": task_id,
                "task_name": task_name,
                "frame_count": len(frames),
                "frames": frames,
                "images": images,
                "frame_map": frame_map,
                "total_annotations": sum(len(f["labels"]) for f in frames)
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise HTTPException(status_code=502, detail=f"Không thể lấy dữ liệu từ CVAT Job #{job_id} ({exc.code}): {detail}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Lỗi khi nạp dữ liệu từ CVAT Job #{job_id}: {exc}")

    @fastapi_app.post("/api/dataset/parse-flexible")
    async def dataset_parse_flexible(file: UploadFile = File(...)):
        """Parse any uploaded dataset file: .json, .xml (VOC/CVAT), .txt (YOLO), .csv, or .zip archive."""
        from flexible_dataset_parser import MAX_UPLOAD_BYTES, parse_flexible_dataset
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Tệp upload vượt giới hạn 64 MiB.")
        try:
            result = parse_flexible_dataset(content, file.filename)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file '{file.filename}': {exc}")

    @fastapi_app.post("/api/qc/honeypot")
    async def evaluate_honeypot_endpoint(request: Request):
        payload = await request.json()
        annotator_frames = payload.get("annotator_frames", [])
        gt_frames = payload.get("ground_truth_frames", payload.get("honeypot_frames", []))
        annotator_name = payload.get("annotator_name", "Annotator")
        checker = BDDQualityChecker()
        res = checker.evaluate_honeypot(annotator_frames, gt_frames, annotator_name)
        return {"status": "success", "honeypot_evaluation": res, "evaluation": res}

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
    app = fastapi_app

except ImportError:
    FASTAPI_AVAILABLE = False
    app = None


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
                    self.send_header("Cache-Control", "public, max-age=86400")
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

        elif path == "/api/qc/gemini-inspect-frame":
            try:
                if not gemini_configured():
                    self.send_json_response({"status": "error", "message": "Gemini API key chưa được cấu hình"}, status=400)
                    return
                payload = json.loads(post_data.decode("utf-8"))
                frame_name = payload.get("frame_name", "")
                image_bytes = None
                encoded_image = payload.get("image_base64")
                if encoded_image:
                    clean = encoded_image.split(",", 1)[-1] if encoded_image.startswith("data:") else encoded_image
                    try:
                        image_bytes = base64.b64decode(clean, validate=True)
                    except Exception:
                        pass
                if not image_bytes and frame_name:
                    p = find_image_file_path(frame_name)
                    if p and os.path.exists(p):
                        with open(p, "rb") as f:
                            image_bytes = f.read()
                if not image_bytes:
                    self.send_json_response({"status": "error", "message": "Không tìm thấy ảnh"}, status=404)
                    return
                g_text = payload.get("guideline") or global_guideline_manager.get_guideline_text()
                raw_labels = payload.get("labels", [])
                p_labels = [{"class": l.get("category", l.get("label", "")), "box": l.get("box2d") or l.get("bbox")} for l in raw_labels]
                raw_models = payload.get("model_predictions", [])
                p_models = [{"class": mp.get("category", mp.get("label", "")), "box": mp.get("box"), "confidence": mp.get("confidence", 0.0), "model": mp.get("model", "local")} for mp in raw_models]
                classes = payload.get("classes", ["car", "truck", "bus", "pedestrian", "rider", "bicycle", "motorcycle", "traffic light", "traffic sign", "train"])
                proposals = gemini_inspect_frame(image_bytes, g_text, p_labels, classes, model_predictions=p_models)
                self.send_json_response({"status": "success", "frame_name": frame_name, "model": gemini_model_name(), "proposals": proposals, "total_proposals": len(proposals)})
            except Exception as e:
                self.send_json_response({"status": "error", "message": str(e)}, status=500)
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


def run_server(port: int = 8000, host: str = None):
    if host is None:
        host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 70)
    print("  🚀 KHỞI ĐỘNG QC STUDIO 2D - BDD100K (N2-04A NEXT-GEN AI)")
    print("  ℹ️ QC API: BBox/semantic segmentation use model providers; candidates need human review.")
    print(f"  🌐 API Server: http://{host}:{port}")
    print("=" * 70)

    if FASTAPI_AVAILABLE:
        print("[INFO] Sử dụng FastAPI + Uvicorn Server cao cấp...")
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    else:
        print("[INFO] Sử dụng Python Built-in HTTP Server (Zero-Dependency)...")
        httpd = HTTPServer((host, port), BuiltinQCHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐã dừng server.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    run_server(port=port, host=host)
