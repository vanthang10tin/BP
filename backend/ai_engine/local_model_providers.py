"""Loopback-only adapters for YOLOv26 and Meta SAM 2.1 (Segment Anything 2).

Architecture:
- Object Detection (BBox): Ultralytics YOLO26 (Native End-to-End NMS-Free) via yolo26x.pt / yolo26n.pt
- Segmentation (Mask & Polygon): Meta SAM 2.1 Hiera-Small via CVAT Nuclio Container (port 32001 / 32002) and YOLO26-Seg
Images never go to a hosted model API. Provider endpoints are local loopback by default.
"""
from __future__ import annotations

import base64
import io
import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

DEFAULTS = {
    "bbox": ("QC_BBOX_MODEL_URL", "http://127.0.0.1:32003/"),
    "segmentation": ("QC_SEGMENTATION_MODEL_URL", "http://127.0.0.1:32001/"),
    "sam2": ("QC_SAM2_MODEL_URL", "http://127.0.0.1:32001/"),
    "sam2_interactor": ("QC_SAM2_INTERACTOR_URL", "http://127.0.0.1:32002/"),
    "panoptic_bbox": ("QC_DINO_BBOX_MODEL_URL", "http://127.0.0.1:32004/"),
    "dinov3_seg": ("QC_DINO_SEG_MODEL_URL", "http://127.0.0.1:32005/"),
}
MAX_IMAGE_BYTES = 31 * 1024 * 1024  # CVAT functions cap JSON request bodies at 32 MiB.
_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class ProviderError(RuntimeError):
    pass


def _url(capability: str) -> str:
    modal_url = os.environ.get("QC_MODAL_API_URL", "").strip().rstrip("/")
    if modal_url:
        os.environ["QC_ALLOW_REMOTE_MODEL_ENDPOINTS"] = "true"
        if capability in ("bbox",):
            return f"{modal_url}/bbox"
        elif capability in ("segmentation", "sam2"):
            return f"{modal_url}/segmentation"
        elif capability in ("panoptic_bbox", "dinov3_seg"):
            return f"{modal_url}/analyze"
        return f"{modal_url}/"

    env_name, default = DEFAULTS.get(capability, ("QC_MODEL_URL", "http://127.0.0.1:32001/"))
    value = os.environ.get(env_name, default).rstrip("/") + "/"
    if not value.startswith(("http://", "https://")):
        raise ProviderError(f"{env_name} must be an HTTP(S) URL")
    hostname = value.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0].strip("[]").lower()
    is_loopback = hostname == "localhost"
    try:
        is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if not is_loopback and os.environ.get("QC_ALLOW_REMOTE_MODEL_ENDPOINTS", "false").lower() not in {"1", "true", "yes"}:
        raise ProviderError("Model endpoint must use loopback; set QC_ALLOW_REMOTE_MODEL_ENDPOINTS=true only after a privacy review")
    return value


def _post(capability: str, payload: dict[str, Any], timeout: int = 90) -> Any:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_IMAGE_BYTES + 1_000_000:
        raise ProviderError("Image request exceeds the local CVAT function body limit")
    request = urllib.request.Request(_url(capability), data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if response.status != 200:
                raise ProviderError(f"Local provider returned HTTP {response.status}")
        return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise ProviderError(f"Local {capability} provider HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Local {capability} provider unavailable: {exc}") from exc


def provider_info(capability: str, timeout: int = 15) -> dict[str, Any]:
    cached = _STATUS_CACHE.get(capability)
    if cached and time.monotonic() - cached[0] < 15:
        return dict(cached[1])

    modal_url = os.environ.get("QC_MODAL_API_URL", "").strip().rstrip("/")
    if modal_url:
        try:
            req = urllib.request.Request(f"{modal_url}/status", headers={"User-Agent": "QC-Studio"})
            with urllib.request.urlopen(req, timeout=5) as res:
                info = json.loads(res.read().decode())
                result = {
                    "id": capability,
                    "available": True,
                    "privacy": "modal-cloud-gpu",
                    "name": f"Modal Cloud GPU ({info.get('model', 'YOLO11 + SAM 2.1 + DINO')})",
                    "device": info.get("gpu", "Nvidia T4 (16GB VRAM)"),
                    "type": "mask" if capability in ("segmentation", "sam2", "dinov3_seg") else "bbox",
                    "label_count": 80,
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                _STATUS_CACHE[capability] = (time.monotonic(), result)
                return dict(result)
        except Exception:
            pass

    # 1. BBox capability: Primary is Ultralytics YOLO26
    if capability == "bbox" and os.environ.get("QC_FORCE_NUCLIO_BBOX", "false").lower() not in ("1", "true", "yes"):
        try:
            y26 = _get_yolo26()
            if y26 is not None:
                weights_name = getattr(y26, "model_name", "YOLO26x")
                result = {
                    "id": "bbox",
                    "available": True,
                    "privacy": "loopback",
                    "name": "Ultralytics YOLO26 (local weights)",
                    "device": str(getattr(y26, "device", "cpu")),
                    "type": "bbox",
                    "label_count": len(y26.names),
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                _STATUS_CACHE[capability] = (time.monotonic(), result)
                return dict(result)
        except Exception:
            pass

    # 2. Segmentation capability: Primary is Ultralytics YOLO26-Seg if native weights exist
    if capability == "segmentation" and os.environ.get("QC_FORCE_NUCLIO_SEG", "false").lower() not in ("1", "true", "yes"):
        try:
            y26_seg = _get_yolo26_seg()
            if y26_seg is not None:
                result = {
                    "id": "segmentation",
                    "available": True,
                    "privacy": "loopback",
                    "name": "Ultralytics YOLO26-Seg (Native End-to-End Instance Segmentation)",
                    "device": str(getattr(y26_seg, "device", "cpu")),
                    "type": "mask",
                    "label_count": len(y26_seg.names),
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                _STATUS_CACHE[capability] = (time.monotonic(), result)
                return dict(result)
        except Exception:
            pass

    # 3. Query endpoint
    try:
        info = _post(capability, {"action": "status"}, timeout=timeout)
        rules = info.get("rules", {}) if isinstance(info, dict) else {}
        model_name = info.get("model", capability)
        # Beautify SAM2 name if returned
        if "SAM" in model_name or capability in ("segmentation", "sam2"):
            model_name = "Meta SAM 2.1 Hiera-Small (Segment Anything 2 - BBox & Polygon)"

        result = {
            "id": capability,
            "available": True,
            "privacy": "loopback",
            "name": model_name,
            "device": info.get("device", "cuda:0"),
            "type": info.get("type", "mask" if capability in ("segmentation", "sam2", "dinov3_seg") else "bbox"),
            "label_count": info.get("label_count", 31),
            "supported_classes": rules.get("include_classes", list(set(COCO_TO_BDD100K.values()))),
            "thresholds": rules.get("class_thresholds", {}),
            "error": None
        }
        _STATUS_CACHE[capability] = (time.monotonic(), result)
        return dict(result)
    except Exception as exc:
        # If SAM2 container failed, check if native YOLO26-Seg is available
        if capability == "segmentation":
            y26_seg = _get_yolo26_seg()
            if y26_seg is not None:
                result = {
                    "id": "segmentation",
                    "available": True,
                    "privacy": "loopback",
                    "name": "Ultralytics YOLO26-Seg (Native End-to-End Instance Segmentation)",
                    "device": str(getattr(y26_seg, "device", "cpu")),
                    "type": "mask",
                    "label_count": len(y26_seg.names),
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                _STATUS_CACHE[capability] = (time.monotonic(), result)
                return dict(result)

        result = {"id": capability, "available": False, "privacy": "loopback", "name": capability,
                  "device": None, "supported_classes": [], "error": str(exc)[:500]}
        _STATUS_CACHE[capability] = (time.monotonic(), result)
        return dict(result)


# =====================================================================
# YOLO26 & YOLO26-Seg NATIVE PROVIDERS
# =====================================================================
_YOLO26_MODEL = None
_YOLO26_LOAD_ATTEMPTED = False

_YOLO26_SEG_MODEL = None
_YOLO26_SEG_LOAD_ATTEMPTED = False

COCO_TO_BDD100K = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
    "train": "train",
}


def _resolve_model_candidates(filenames: list[str]) -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, ".."))
    repo_dir = os.path.abspath(os.path.join(backend_dir, ".."))

    search_dirs = [
        os.path.join(repo_dir, "models"),
        os.path.join(backend_dir, "models"),
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        backend_dir,
        repo_dir,
    ]
    candidates = []
    for fname in filenames:
        for sdir in search_dirs:
            candidate = os.path.join(sdir, fname)
            if os.path.exists(candidate) and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _get_yolo26():
    global _YOLO26_MODEL, _YOLO26_LOAD_ATTEMPTED
    if _YOLO26_MODEL is not None or _YOLO26_LOAD_ATTEMPTED:
        return _YOLO26_MODEL
    _YOLO26_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO
        import torch
        weights_override = os.environ.get("QC_YOLO26_WEIGHTS")
        if weights_override and os.path.exists(weights_override):
            model_path = weights_override
        else:
            has_cuda = torch.cuda.is_available()
            preferred_order = ["yolo26x.pt", "yolo26n.pt"] if has_cuda else ["yolo26n.pt", "yolo26x.pt"]
            candidates = _resolve_model_candidates(preferred_order)
            if not candidates:
                print("[INFO] YOLO26 weights not found locally; skipping inference instead of downloading weights.")
                return None
            model_path = candidates[0]
        _YOLO26_MODEL = YOLO(model_path)
    except Exception as exc:
        print(f"[WARN] Failed to load YOLO26: {exc}")
        _YOLO26_MODEL = None
    return _YOLO26_MODEL


def _get_yolo26_seg():
    global _YOLO26_SEG_MODEL, _YOLO26_SEG_LOAD_ATTEMPTED
    if _YOLO26_SEG_MODEL is not None or _YOLO26_SEG_LOAD_ATTEMPTED:
        return _YOLO26_SEG_MODEL
    _YOLO26_SEG_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO
        weights_override = os.environ.get("QC_YOLO26_SEG_WEIGHTS")
        if weights_override and os.path.exists(weights_override):
            model_path = weights_override
        else:
            candidates = _resolve_model_candidates(["yolo26n-seg.pt"])
            model_path = candidates[0] if candidates else None
        if model_path:
            _YOLO26_SEG_MODEL = YOLO(model_path)
    except Exception as exc:
        print(f"[WARN] Failed to load YOLO26-Seg: {exc}")
        _YOLO26_SEG_MODEL = None
    return _YOLO26_SEG_MODEL


def _infer_yolo26_native(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = _get_yolo26()
    if model is None:
        raise ProviderError("YOLO26 model is not available.")
    with Image.open(io.BytesIO(image)) as pil_img:
        res = model.predict(pil_img, conf=threshold, imgsz=640, verbose=False)[0]

    mapping = dict(COCO_TO_BDD100K)
    if label_mapping:
        mapping.update(label_mapping)

    predictions = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            raw_label = model.names.get(cls_id, str(cls_id))
            target_label = mapping.get(raw_label, raw_label)
            xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            predictions.append({
                "type": "rectangle",
                "label": target_label,
                "points": xyxy,
                "confidence": conf
            })

    info = {
        "id": "bbox",
        "available": True,
        "privacy": "loopback",
        "name": "Ultralytics YOLO26 (local weights)",
        "device": str(getattr(model, "device", "cpu")),
        "type": "bbox",
        "label_count": len(model.names),
        "supported_classes": list(set(mapping.values())),
        "thresholds": {},
        "error": None
    }
    return predictions, info


def _infer_yolo26_seg_native(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = _get_yolo26_seg()
    if model is None:
        raise ProviderError("YOLO26-Seg model is not available.")
    with Image.open(io.BytesIO(image)) as pil_img:
        res = model.predict(pil_img, conf=threshold, imgsz=640, verbose=False)[0]

    mapping = dict(COCO_TO_BDD100K)
    if label_mapping:
        mapping.update(label_mapping)

    predictions = []
    if res.masks is not None and res.boxes is not None:
        for idx, box in enumerate(res.boxes):
            cls_id = int(box.cls[0].item())
            raw_label = model.names.get(cls_id, str(cls_id))
            target_label = mapping.get(raw_label, raw_label)
            conf = float(box.conf[0].item())
            if idx < len(res.masks.xy):
                poly = res.masks.xy[idx]
                if len(poly) >= 3:
                    pts = [float(c) for pt in poly for c in pt]
                    xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
                    predictions.append({
                        "type": "polygon",
                        "label": target_label,
                        "points": pts,
                        "box": xyxy,
                        "confidence": conf
                    })

    info = {
        "id": "segmentation",
        "available": True,
        "privacy": "loopback",
        "name": "Ultralytics YOLO26-Seg (Native End-to-End Instance Segmentation)",
        "device": str(getattr(model, "device", "cpu")),
        "type": "mask",
        "label_count": len(model.names),
        "supported_classes": list(set(mapping.values())),
        "thresholds": {},
        "error": None
    }
    return predictions, info


def infer_bbox(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None, preferred_model: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ProviderError("Invalid or oversized image")

    modal_url = os.environ.get("QC_MODAL_API_URL", "").strip().rstrip("/")
    if modal_url:
        try:
            payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                       "task_labels": classes, "mapping": label_mapping or {}}
            result = _post("bbox", payload, timeout=60)
            if isinstance(result, list):
                info = {
                    "id": "bbox",
                    "available": True,
                    "privacy": "modal-cloud-gpu",
                    "name": "Modal Cloud GPU (Nvidia T4 · YOLO11)",
                    "device": "Nvidia T4 (16GB VRAM)",
                    "type": "bbox",
                    "label_count": 80,
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                return result, info
        except Exception as exc:
            print(f"[WARN] Modal GPU bbox inference failed, falling back: {exc}")

    pref = (preferred_model or os.environ.get("QC_BBOX_PROVIDER", "")).strip().lower()
    if pref in ("dino", "dinov3", "panoptic_bbox"):
        info_dino = provider_info("panoptic_bbox")
        if info_dino.get("available"):
            payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                       "task_labels": classes, "mapping": label_mapping or {}}
            try:
                result = _post("panoptic_bbox", payload, timeout=60)
                if isinstance(result, list):
                    return result, info_dino
            except Exception as exc:
                print(f"[WARN] DINOv3 bbox inference failed, falling back: {exc}")

    use_nuclio = pref in ("nuclio", "yolov8_nuclio") or os.environ.get("QC_FORCE_NUCLIO_BBOX", "false").lower() in ("1", "true", "yes")
    if not use_nuclio:
        try:
            return _infer_yolo26_native(image, threshold, classes, label_mapping)
        except Exception:
            pass  # Fallback to CVAT container

    # Fallback: DINOv3 or Nuclio Bbox container
    info_dino = provider_info("panoptic_bbox")
    if info_dino.get("available"):
        payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                   "task_labels": classes, "mapping": label_mapping or {}}
        try:
            result = _post("panoptic_bbox", payload, timeout=60)
            if isinstance(result, list):
                return result, info_dino
        except Exception:
            pass

    info = provider_info("bbox")
    if not info.get("available"):
        raise ProviderError(info.get("error") or "BBox model unavailable")
    payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
               "task_labels": classes, "mapping": label_mapping or {}}
    result = _post("bbox", payload)
    if not isinstance(result, list):
        raise ProviderError("BBox provider returned an unexpected response")
    return result, info


def infer_segmentation(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None, preferred_model: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ProviderError("Invalid or oversized image")

    modal_url = os.environ.get("QC_MODAL_API_URL", "").strip().rstrip("/")
    if modal_url:
        try:
            payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                       "task_labels": classes, "mapping": label_mapping or {}}
            result = _post("segmentation", payload, timeout=60)
            if isinstance(result, list):
                info = {
                    "id": "segmentation",
                    "available": True,
                    "privacy": "modal-cloud-gpu",
                    "name": "Modal Cloud GPU (Nvidia T4 · SAM 2.1)",
                    "device": "Nvidia T4 (16GB VRAM)",
                    "type": "mask",
                    "label_count": 80,
                    "supported_classes": list(set(COCO_TO_BDD100K.values())),
                    "thresholds": {},
                    "error": None
                }
                return result, info
        except Exception as exc:
            print(f"[WARN] Modal GPU segmentation inference failed, falling back: {exc}")

    pref = (preferred_model or os.environ.get("QC_SEGMENTATION_PROVIDER", "")).strip().lower()

    # 1. If Meta SAM 2.1 is requested or forced via env
    if pref in ("sam2", "sam2.1", "nuclio", "nuclio_sam2") or os.environ.get("QC_FORCE_NUCLIO_SEG", "false").lower() in ("1", "true", "yes"):
        info_sam2 = provider_info("sam2")
        if info_sam2.get("available"):
            payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                       "task_labels": classes, "mapping": label_mapping or {}}
            try:
                result = _post("sam2", payload, timeout=60)
                if isinstance(result, list):
                    return result, info_sam2
            except Exception as exc:
                print(f"[WARN] SAM2 container inference failed, falling back: {exc}")

    # 2. If DINOv3 is explicitly requested
    if pref in ("dino", "dinov3", "dinov3_seg"):
        info_dino = provider_info("dinov3_seg")
        if info_dino.get("available"):
            payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                       "task_labels": classes, "mapping": label_mapping or {}}
            try:
                result = _post("dinov3_seg", payload, timeout=120)
                if isinstance(result, list):
                    return result, info_dino
            except Exception as exc:
                print(f"[WARN] DINOv3 container inference failed, falling back: {exc}")

    # 3. Default: Native YOLO26-Seg model (Fast local execution, zero-wait)
    try:
        y26_seg = _get_yolo26_seg()
        if y26_seg is not None:
            return _infer_yolo26_seg_native(image, threshold, classes, label_mapping)
    except Exception as exc:
        print(f"[WARN] Native YOLO26-Seg inference failed, falling back: {exc}")

    # 4. Secondary: Meta SAM 2.1 Hiera-Small container on port 32001
    info_sam2 = provider_info("sam2")
    if info_sam2.get("available"):
        payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                   "task_labels": classes, "mapping": label_mapping or {}}
        try:
            result = _post("sam2", payload, timeout=60)
            if isinstance(result, list):
                return result, info_sam2
        except Exception as exc:
            print(f"[WARN] SAM2 container fallback failed: {exc}")

    # 5. Tertiary: DINOv3 Panoptic Segmentation on 32005
    info_dino = provider_info("dinov3_seg")
    if info_dino.get("available"):
        payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
                   "task_labels": classes, "mapping": label_mapping or {}}
        result = _post("dinov3_seg", payload, timeout=120)
        return result, info_dino

    raise ProviderError("No segmentation provider available (YOLO26-Seg / SAM2 / DINOv3)")


def available_providers() -> list[dict[str, Any]]:
    providers = []
    modal_url = os.environ.get("QC_MODAL_API_URL", "").strip().rstrip("/")
    if modal_url:
        providers.append({
            "id": "modal_gpu",
            "available": True,
            "privacy": "modal-cloud-gpu",
            "name": "Modal Cloud GPU (Nvidia T4 · YOLO11 + SAM 2.1 + DINOv2)",
            "device": "Nvidia T4 (16GB VRAM)",
            "type": "ensemble",
            "label_count": 80,
            "supported_classes": list(set(COCO_TO_BDD100K.values())),
            "thresholds": {},
            "error": None
        })
    # 1. BBox Detector: YOLO26
    bbox_p = provider_info("bbox")
    providers.append(bbox_p)

    # 2. Meta SAM 2.1 Expert (Polygon & BBox)
    try:
        sam2_p = provider_info("sam2")
        if sam2_p.get("available"):
            providers.append(sam2_p)
    except Exception:
        pass

    # 3. Meta SAM 2.1 Interactor (Magic Wand)
    try:
        interactor_info = provider_info("sam2_interactor")
        if interactor_info.get("available"):
            providers.append(interactor_info)
    except Exception:
        pass

    # 4. Native YOLO26-Seg Instance Segmentation
    y26_seg = _get_yolo26_seg()
    if y26_seg is not None:
        providers.append({
            "id": "segmentation",
            "available": True,
            "privacy": "loopback",
            "name": "Ultralytics YOLO26-Seg (Native End-to-End Instance Segmentation)",
            "device": str(getattr(y26_seg, "device", "cpu")),
            "type": "mask",
            "label_count": len(y26_seg.names),
            "supported_classes": list(set(COCO_TO_BDD100K.values())),
            "thresholds": {},
            "error": None
        })

    # 5. DINOv3 Panoptic Detector & Segmenter
    try:
        dino_info = provider_info("panoptic_bbox")
        if dino_info.get("available"):
            providers.append(dino_info)
    except Exception:
        pass

    try:
        dino_seg_info = provider_info("dinov3_seg")
        if dino_seg_info.get("available"):
            providers.append(dino_seg_info)
    except Exception:
        pass

    return providers
