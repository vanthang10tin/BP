"""Opt-in Gemini vision proposals. Credentials stay in the backend environment."""
import base64
import json
import os
import urllib.error
import urllib.request


KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEMINI_API_KEY.txt")


def _api_key() -> str:
    # One-line local key file keeps setup simple; env vars remain supported for deployment.
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    try:
        with open(KEY_FILE, "r", encoding="utf-8-sig") as key_file:
            return key_file.read().strip().strip('"').strip("'")
    except OSError:
        return ""


def configured() -> bool:
    return bool(_api_key())


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"


def inspect_frame(image_data: bytes, guideline: str, labels: list[dict], classes: list[str]) -> list[dict]:
    key = _api_key()
    if not key:
        raise ValueError("Gemini chưa có key. Dán key vào backend/GEMINI_API_KEY.txt rồi khởi động lại backend.")
    # Candidate-only instruction: Gemini never edits ground truth or decides policy compliance.
    prompt = (
        "Inspect this image for objects required by the team's guideline. Return only visible, "
        "well-supported object candidates. Compare against the uploaded annotations and include "
        "only likely missing or class-disagreed objects. Do not infer objects outside the image. "
        "Boxes use normalized coordinates 0..1000 in [ymin,xmin,ymax,xmax] order. "
        "For each candidate provide class_name, box_2d, confidence (0..1), and short evidence. "
        "If uncertain, omit it. Guideline:\n" + guideline[:20000] +
        "\nKnown ontology/classes: " + json.dumps(classes[:200], ensure_ascii=False) +
        "\nUploaded annotations (class and box): " + json.dumps(labels[:300], ensure_ascii=False)
    )
    mime = "image/png" if image_data.startswith(b"\x89PNG") else "image/jpeg"
    body = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {
            "mime_type": mime, "data": base64.b64encode(image_data).decode("ascii")
        }}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT", "properties": {"candidates": {"type": "ARRAY", "items": {
                    "type": "OBJECT", "properties": {
                        "class_name": {"type": "STRING"},
                        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "confidence": {"type": "NUMBER"}, "evidence": {"type": "STRING"}
                    }, "required": ["class_name", "box_2d", "confidence", "evidence"]
                }}}, "required": ["candidates"]
            }
        }
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name()}:generateContent"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "x-goog-api-key": key
    }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"Gemini request failed: {exc}") from exc
    text = "".join(p.get("text", "") for p in result.get("candidates", [{}])[0].get("content", {}).get("parts", []))
    try:
        output = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid structured JSON") from exc
    candidates = []
    for row in output.get("candidates", []):
        box = row.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box]
            confidence = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if ymax <= ymin or xmax <= xmin or confidence < 0.25:
            continue
        candidates.append({"category": str(row.get("class_name", "unknown")),
                           "box_normalized": [xmin, ymin, xmax, ymax],
                           "confidence": max(0.0, min(1.0, confidence)),
                           "evidence": str(row.get("evidence", ""))[:500]})
    return candidates
