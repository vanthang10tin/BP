"""Loopback-only adapters for the CVAT Nuclio models installed on this machine.

Images never go to a hosted model API. Provider endpoints are local by default;
remote hosts require a separate, explicit opt-in.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULTS = {
    "bbox": ("QC_BBOX_MODEL_URL", "http://127.0.0.1:32003/"),
    "segmentation": ("QC_SEGMENTATION_MODEL_URL", "http://127.0.0.1:32005/"),
    "panoptic_bbox": ("QC_DINO_BBOX_MODEL_URL", "http://127.0.0.1:32004/"),
}
MAX_IMAGE_BYTES = 31 * 1024 * 1024  # CVAT functions cap JSON request bodies at 32 MiB.
_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class ProviderError(RuntimeError):
    pass


def _url(capability: str) -> str:
    env_name, default = DEFAULTS[capability]
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


def provider_info(capability: str, timeout: int = 20) -> dict[str, Any]:
    cached = _STATUS_CACHE.get(capability)
    if cached and time.monotonic() - cached[0] < 20:
        return dict(cached[1])
        
    if capability == "bbox" and os.environ.get("QC_FORCE_NUCLIO_BBOX", "false").lower() not in ("1", "true", "yes"):
        try:
            y26 = _get_yolo26()
            if y26 is not None:
                result = {
                    "id": "bbox",
                    "available": True,
                    "privacy": "loopback",
                    "name": "Ultralytics YOLO26x (Flagship SOTA - Native End-to-End NMS-Free)",
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

    try:
        info = _post(capability, {"action": "status"}, timeout=timeout)
        rules = info.get("rules", {}) if isinstance(info, dict) else {}
        result = {"id": capability, "available": True, "privacy": "loopback", "name": info.get("model", capability),
                "device": info.get("device", "unknown"), "type": info.get("type", "bbox" if capability == "bbox" else "mask"),
                "label_count": info.get("label_count", 0), "supported_classes": rules.get("include_classes", []),
                "thresholds": rules.get("class_thresholds", {}), "error": None}
        _STATUS_CACHE[capability] = (time.monotonic(), result)
        return dict(result)
    except Exception as exc:
        result = {"id": capability, "available": False, "privacy": "loopback", "name": capability,
                  "device": None, "supported_classes": [], "error": str(exc)[:500]}
        _STATUS_CACHE[capability] = (time.monotonic(), result)
        return dict(result)


# YOLO26 Native Provider Integration
_YOLO26_MODEL = None
_YOLO26_LOAD_ATTEMPTED = False

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


def _get_yolo26():
    global _YOLO26_MODEL, _YOLO26_LOAD_ATTEMPTED
    if _YOLO26_MODEL is not None or _YOLO26_LOAD_ATTEMPTED:
        return _YOLO26_MODEL
    _YOLO26_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO
        weights_override = os.environ.get("QC_YOLO26_WEIGHTS")
        if weights_override and os.path.exists(weights_override):
            model_path = weights_override
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            app_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            repo_dir = os.path.abspath(os.path.join(app_dir, "..", ".."))
            candidates = [
                "yolo26x.pt",
                os.path.join(app_dir, "yolo26x.pt"),
                os.path.join(repo_dir, "yolo26x.pt"),
                "yolo26n.pt",
                os.path.join(app_dir, "yolo26n.pt"),
                os.path.join(repo_dir, "yolo26n.pt"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), "yolo26n.pt")
        _YOLO26_MODEL = YOLO(model_path)
    except Exception:
        _YOLO26_MODEL = None
    return _YOLO26_MODEL


def _infer_yolo26_native(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import io
    from PIL import Image
    model = _get_yolo26()
    if model is None:
        raise ProviderError("YOLO26 model is not available.")
    with Image.open(io.BytesIO(image)) as pil_img:
        res = model.predict(pil_img, conf=threshold, verbose=False)[0]
    
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
        "name": f"Ultralytics YOLO26 (Native End-to-End NMS-Free)",
        "device": str(getattr(model, "device", "cpu")),
        "type": "bbox",
        "label_count": len(model.names),
        "supported_classes": list(set(mapping.values())),
        "thresholds": {},
        "error": None
    }
    return predictions, info


def infer_bbox(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ProviderError("Invalid or oversized image")
        
    use_nuclio = os.environ.get("QC_FORCE_NUCLIO_BBOX", "false").lower() in ("1", "true", "yes")
    if not use_nuclio:
        try:
            return _infer_yolo26_native(image, threshold, classes, label_mapping)
        except Exception:
            pass  # Fallback to Nuclio container 32003

    info = provider_info("bbox")
    if not info["available"]:
        raise ProviderError(info["error"] or "BBox model unavailable")
    payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
               "task_labels": classes, "mapping": label_mapping or {}}
    result = _post("bbox", payload)
    if not isinstance(result, list):
        raise ProviderError("BBox provider returned an unexpected response (expected a CVAT prediction list)")
    return result, info


def infer_segmentation(image: bytes, threshold: float, classes: list[str], label_mapping: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ProviderError("Invalid or oversized image")
    info = provider_info("segmentation")
    if not info["available"]:
        raise ProviderError(info["error"] or "Segmentation model unavailable")
    payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold,
               "task_labels": classes, "mapping": label_mapping or {}}
    result = _post("segmentation", payload, timeout=180)
    if not isinstance(result, list):
        raise ProviderError("Segmentation provider returned an unexpected response (expected CVAT mask items)")
    return result, info


def available_providers() -> list[dict[str, Any]]:
    return [provider_info(name) for name in ("bbox", "segmentation", "panoptic_bbox")]
