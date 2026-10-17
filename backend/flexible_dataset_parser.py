"""Flexible dataset parser supporting JSON, Pascal VOC XML, CVAT XML, YOLO TXT, CSV, and ZIP archives."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2048
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_IMAGE_BYTES = 20 * 1024 * 1024


def parse_flexible_dataset(
    file_bytes: bytes,
    filename: str,
    profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Parse dataset annotations and optional images from various formats.
    
    Supported formats:
    - .zip: Archive containing annotations (json/xml/txt/csv) and images
    - .json: BDD100K, COCO, YOLO-JSON, LabelMe
    - .xml: CVAT for images XML (<annotations>), Pascal VOC XML (<annotation>)
    - .csv: Tabular bounding box annotations
    - .txt: YOLO normalized bounding boxes
    """
    profile = profile or {}
    lower_name = filename.lower()

    if lower_name.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        return _parse_zip_archive(file_bytes, profile)

    if lower_name.endswith(".json"):
        return _parse_json_content(file_bytes, filename, profile)

    if lower_name.endswith(".xml"):
        return _parse_xml_content(file_bytes, filename, profile)

    if lower_name.endswith(".csv"):
        return _parse_csv_content(file_bytes, filename, profile)

    if lower_name.endswith(".txt"):
        return _parse_txt_content(file_bytes, filename, profile)

    # Fallback: attempt JSON first, then XML, then CSV
    try:
        return _parse_json_content(file_bytes, filename, profile)
    except Exception:
        pass

    try:
        return _parse_xml_content(file_bytes, filename, profile)
    except Exception:
        pass

    try:
        return _parse_csv_content(file_bytes, filename, profile)
    except Exception:
        pass

    raise ValueError(f"Không thể tự động nhận diện định dạng dữ liệu của file '{filename}'. Hỗ trợ .json, .xml, .csv, .txt, .zip")


def _image_to_data_uri(img_bytes: bytes, ext: str) -> str:
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png" if ext == ".png" else f"image/{ext.lstrip('.')}"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _read_zip_member_limited(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    """Read an archive member incrementally so dishonest ZIP headers cannot inflate RAM use."""
    chunks: list[bytes] = []
    total = 0
    with zf.open(info, "r") as member:
        while True:
            chunk = member.read(min(1024 * 1024, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("Nội dung giải nén ZIP vượt giới hạn dung lượng cho phép.")
            chunks.append(chunk)
    return b"".join(chunks)


def _parse_zip_archive(zip_bytes: bytes, profile: dict[str, Any]) -> dict[str, Any]:
    images: dict[str, str] = {}
    anno_files: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        members = [info for info in zf.infolist() if not info.is_dir() and not info.filename.startswith("__MACOSX")]
        if len(members) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"ZIP chứa quá nhiều file (giới hạn {MAX_ARCHIVE_ENTRIES}).")
        total_uncompressed = sum(info.file_size for info in members)
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("Tổng dung lượng sau giải nén ZIP vượt giới hạn 64 MiB.")

        actual_uncompressed = 0
        for info in members:
            fname = Path(info.filename).name
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTS:
                if info.file_size > MAX_ARCHIVE_IMAGE_BYTES:
                    raise ValueError(f"Ảnh trong ZIP vượt giới hạn 20 MiB: {fname}")
                data = _read_zip_member_limited(zf, info, min(MAX_ARCHIVE_IMAGE_BYTES, MAX_ARCHIVE_UNCOMPRESSED_BYTES - actual_uncompressed))
                actual_uncompressed += len(data)
                images[fname] = _image_to_data_uri(data, ext)
            elif ext in {".json", ".xml", ".csv", ".txt"}:
                data = _read_zip_member_limited(zf, info, MAX_ARCHIVE_UNCOMPRESSED_BYTES - actual_uncompressed)
                actual_uncompressed += len(data)
                anno_files.append((info.filename, data))

    if not anno_files:
        if images:
            # ZIP only has images: return frames with empty labels so user can review them
            frames = [{"name": name, "labels": []} for name in sorted(images.keys())]
            return {
                "status": "success",
                "format": "zip_images_only",
                "frame_count": len(frames),
                "frames": frames,
                "images": images,
                "extracted_image_count": len(images)
            }
        raise ValueError("File ZIP không chứa bất kỳ file nhãn (.json, .xml, .csv, .txt) hoặc ảnh hợp lệ nào.")

    # Try parsing annotations in priority order: json > xml > csv > txt
    # Sort files so top-level annotations or well-known names come first
    def anno_priority(item: tuple[str, bytes]) -> int:
        fn = item[0].lower()
        if "annotation" in fn or "label" in fn or "instances" in fn:
            return 0
        if fn.endswith(".json"):
            return 1
        if fn.endswith(".xml"):
            return 2
        if fn.endswith(".csv"):
            return 3
        return 4

    anno_files.sort(key=anno_priority)

    # If multiple YOLO .txt files exist in a directory
    txt_files = [f for f in anno_files if f[0].lower().endswith(".txt") and not Path(f[0]).name.lower().startswith("readme")]
    if len(txt_files) > 1:
        # Multiple YOLO txt files matching images
        frames = []
        for path_str, data in txt_files:
            stem = Path(path_str).stem
            matched_img_name = next((im for im in images if Path(im).stem == stem), f"{stem}.jpg")
            res = _parse_txt_content(data, matched_img_name, profile)
            if res.get("frames"):
                frames.extend(res["frames"])
        if frames:
            return {
                "status": "success",
                "format": "zip_yolo_multitxt",
                "frame_count": len(frames),
                "frames": frames,
                "images": images,
                "extracted_image_count": len(images)
            }

    # Otherwise, try the primary annotation file
    last_error = None
    for path_str, data in anno_files:
        try:
            res = parse_flexible_dataset(data, Path(path_str).name, profile)
            if res.get("frames"):
                # Merge extracted images with any images from the zip
                res_images = res.get("images", {})
                res_images.update(images)
                res["images"] = res_images
                res["extracted_image_count"] = len(res_images)
                res["format"] = f"zip_{res['format']}"
                return res
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(f"Không thể trích xuất nhãn từ các file trong ZIP: {last_error}")


def _parse_json_content(content_bytes: bytes, filename: str, profile: dict[str, Any]) -> dict[str, Any]:
    text = content_bytes.decode("utf-8-sig", errors="replace")
    data = json.loads(text)

    # 1. COCO Format
    if isinstance(data, dict) and "annotations" in data and "images" in data and "categories" in data:
        from qc_pipeline import normalize_bbox_dataset
        frames = normalize_bbox_dataset(data, "coco", profile)
        return {"status": "success", "format": "coco_json", "frame_count": len(frames), "frames": frames, "images": {}}

    # 2. YOLO JSON Format
    if isinstance(data, dict) and "images" in data and "categories" in data:
        from qc_pipeline import normalize_bbox_dataset
        frames = normalize_bbox_dataset(data, "yolo_json", profile)
        return {"status": "success", "format": "yolo_json", "frame_count": len(frames), "frames": frames, "images": {}}

    # 3. LabelMe JSON format (single image)
    if isinstance(data, dict) and "shapes" in data:
        img_name = Path(data.get("imagePath", filename)).name
        w = int(data.get("imageWidth", 1280))
        h = int(data.get("imageHeight", 720))
        labels = []
        for idx, s in enumerate(data.get("shapes", [])):
            cat = s.get("label", "object")
            pts = s.get("points", [])
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                labels.append({
                    "id": str(idx),
                    "category": str(cat),
                    "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "attributes": {"occluded": False, "truncated": False}
                })
        frames = [{"name": img_name, "width": w, "height": h, "labels": labels}]
        images = {}
        if data.get("imageData"):
            images[img_name] = f"data:image/jpeg;base64,{data['imageData']}"
        return {"status": "success", "format": "labelme_json", "frame_count": 1, "frames": frames, "images": images}

    # 4. Standard BDD100K or Frame list
    from qc_pipeline import normalize_bbox_dataset
    frames = normalize_bbox_dataset(data, "bdd100k", profile)
    return {"status": "success", "format": "bdd100k_json", "frame_count": len(frames), "frames": frames, "images": {}}


def _parse_xml_content(content_bytes: bytes, filename: str, profile: dict[str, Any]) -> dict[str, Any]:
    text = content_bytes.decode("utf-8-sig", errors="replace")
    root = ET.fromstring(text)

    # 1. CVAT for images 1.1 format: <annotations><image ...><box ...></image></annotations>
    if root.tag in {"annotations", "cvat"}:
        frames = []
        for img in root.findall("image"):
            name = Path(img.get("name", "frame.jpg")).name
            w = int(float(img.get("width", 1280)))
            h = int(float(img.get("height", 720)))
            labels = []
            for idx, box in enumerate(img.findall("box")):
                cat = box.get("label", "object")
                occluded = box.get("occluded") in {"1", "true", "True"}
                x1 = float(box.get("xtl", 0))
                y1 = float(box.get("ytl", 0))
                x2 = float(box.get("xbr", 0))
                y2 = float(box.get("ybr", 0))
                labels.append({
                    "id": str(box.get("id", idx)),
                    "category": str(cat),
                    "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "attributes": {"occluded": occluded, "truncated": False}
                })
            # Also support polygon elements
            for idx, poly in enumerate(img.findall("polygon")):
                cat = poly.get("label", "object")
                pts_str = poly.get("points", "")
                pts = [float(p) for p in pts_str.replace(";", " ").replace(",", " ").split() if p.strip()]
                if len(pts) >= 4:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    labels.append({
                        "id": str(poly.get("id", f"poly_{idx}")),
                        "category": str(cat),
                        "box2d": {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)},
                        "poly2d": [{"vertices": list(zip(xs, ys)), "types": "LL", "closed": True}],
                        "attributes": {"occluded": False, "truncated": False}
                    })
            frames.append({"name": name, "width": w, "height": h, "labels": labels})
        if frames:
            return {"status": "success", "format": "cvat_xml", "frame_count": len(frames), "frames": frames, "images": {}}

    # 2. Pascal VOC format: <annotation><filename>...<object><bndbox>...
    if root.tag == "annotation":
        fn = root.findtext("filename") or Path(filename).stem + ".jpg"
        w = int(float(root.findtext("size/width", "1280") or 1280))
        h = int(float(root.findtext("size/height", "720") or 720))
        labels = []
        for idx, obj in enumerate(root.findall("object")):
            cat = obj.findtext("name", "object")
            box_elem = obj.find("bndbox")
            if box_elem is not None:
                x1 = float(box_elem.findtext("xmin", 0))
                y1 = float(box_elem.findtext("ymin", 0))
                x2 = float(box_elem.findtext("xmax", 0))
                y2 = float(box_elem.findtext("ymax", 0))
                occluded = (obj.findtext("occluded", "0") or "0") in {"1", "true"}
                truncated = (obj.findtext("truncated", "0") or "0") in {"1", "true"}
                labels.append({
                    "id": str(idx),
                    "category": str(cat),
                    "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "attributes": {"occluded": occluded, "truncated": truncated}
                })
        frames = [{"name": Path(fn).name, "width": w, "height": h, "labels": labels}]
        return {"status": "success", "format": "pascal_voc_xml", "frame_count": 1, "frames": frames, "images": {}}

    raise ValueError(f"Không thể phân tích file XML root <{root.tag}> theo chuẩn CVAT hoặc Pascal VOC.")


def _parse_csv_content(content_bytes: bytes, filename: str, profile: dict[str, Any]) -> dict[str, Any]:
    text = content_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("File CSV trống hoặc không có dòng tiêu đề.")

    header_lower = {k.strip().lower(): k for k in reader.fieldnames if k}
    fn_col = header_lower.get("filename") or header_lower.get("image") or header_lower.get("frame") or header_lower.get("file_name") or header_lower.get("image_name")
    cat_col = header_lower.get("category") or header_lower.get("class") or header_lower.get("label") or header_lower.get("name")
    x1_col = header_lower.get("x1") or header_lower.get("xmin") or header_lower.get("left")
    y1_col = header_lower.get("y1") or header_lower.get("ymin") or header_lower.get("top")
    x2_col = header_lower.get("x2") or header_lower.get("xmax") or header_lower.get("right")
    y2_col = header_lower.get("y2") or header_lower.get("ymax") or header_lower.get("bottom")

    if not fn_col or not cat_col or not x1_col or not y1_col or not x2_col or not y2_col:
        # Check if COCO bbox format [x, y, w, h]
        w_col = header_lower.get("width") or header_lower.get("w")
        h_col = header_lower.get("height") or header_lower.get("h")
        if fn_col and cat_col and x1_col and y1_col and w_col and h_col:
            is_xywh = True
        else:
            raise ValueError(f"File CSV cần các cột filename, category và tọa độ (x1,y1,x2,y2 hoặc xmin,ymin,xmax,ymax). Các cột hiện có: {list(reader.fieldnames)}")
    else:
        is_xywh = False

    by_image: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(reader):
        name = Path(row[fn_col].strip()).name
        cat = row[cat_col].strip()
        try:
            x1 = float(row[x1_col])
            y1 = float(row[y1_col])
            if is_xywh:
                w = float(row[w_col])
                h = float(row[h_col])
                x2 = x1 + w
                y2 = y1 + h
            else:
                x2 = float(row[x2_col])
                y2 = float(row[y2_col])
        except (ValueError, TypeError):
            continue
        by_image.setdefault(name, []).append({
            "id": str(idx),
            "category": cat,
            "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "attributes": {"occluded": False, "truncated": False}
        })

    frames = [{"name": name, "labels": labels} for name, labels in by_image.items()]
    return {"status": "success", "format": "csv_table", "frame_count": len(frames), "frames": frames, "images": {}}


def _parse_txt_content(content_bytes: bytes, filename: str, profile: dict[str, Any]) -> dict[str, Any]:
    text = content_bytes.decode("utf-8-sig", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    frame_name = Path(filename).stem + ".jpg"
    W = float(profile.get("image_width", 1280))
    H = float(profile.get("image_height", 720))

    classes = profile.get("classes", [])
    class_map = {idx: name for idx, name in enumerate(classes)} if isinstance(classes, list) else {}

    labels = []
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 5:
            # YOLO format: <class_id> <cx> <cy> <w> <h>
            try:
                cls_token = parts[0]
                try:
                    cls_id = int(cls_token)
                    cat = class_map.get(cls_id, f"class_{cls_id}")
                except ValueError:
                    cat = cls_token
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                # Convert normalized YOLO to pixel coordinates
                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H
                labels.append({
                    "id": str(idx),
                    "category": cat,
                    "box2d": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
                    "attributes": {"occluded": False, "truncated": False}
                })
            except (ValueError, IndexError):
                continue

    frames = [{"name": frame_name, "width": int(W), "height": int(H), "labels": labels}]
    return {"status": "success", "format": "yolo_txt", "frame_count": 1, "frames": frames, "images": {}}
