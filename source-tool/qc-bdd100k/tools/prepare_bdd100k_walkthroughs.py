#!/usr/bin/env python3
"""Build four disjoint, real-data BDD100K walkthroughs: two BBox and two semantic segmentation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
parents = PROJECT.resolve().parents
REPO_ROOT = parents[1] if len(parents) > 1 else PROJECT
import os
ARCHIVE = Path(os.environ.get("BDD100K_ARCHIVE", REPO_ROOT / "archive"))
OUT = REPO_ROOT / "test-data" / "bdd100k-walkthroughs"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(PROJECT / "backend"))
from prepare_bdd100k_val_pack import first_json_array_items, sha256  # noqa: E402
from qc_engine import BDDQualityChecker  # noqa: E402

from PIL import Image  # noqa: E402

BBOX_JSON = ARCHIVE / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
BBOX_IMAGES = ARCHIVE / "bdd100k/bdd100k/images/100k/val"
SEG = ARCHIVE / "bdd100k_seg/bdd100k/seg"
PACKS = ("01-bbox-workflow-50", "02-bbox-rule-review-candidates-50",
         "03-segmentation-val-workflow-50", "04-segmentation-class-coverage-50")
CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
               "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(name: str) -> str:
    if Path(name).name != name or not name.lower().endswith(".jpg"):
        raise ValueError(f"Unexpected source filename: {name!r}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-frames", type=int, default=1200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not BBOX_JSON.is_file() or not BBOX_IMAGES.is_dir() or not (SEG / "images/val").is_dir():
        raise SystemExit("Required BDD100K detection or semantic segmentation source is missing.")
    if any((OUT / p).exists() for p in PACKS) and not args.force:
        raise SystemExit("Walkthrough packs exist already; pass --force to replace these generated packs.")

    source_frames = first_json_array_items(BBOX_JSON, args.scan_frames)
    qc = BDDQualityChecker().check_dataset_advanced(source_frames)
    flagged = {f.get("frame_name") for f in qc.get("sorted_frames", []) if f.get("errors")}
    normal_bbox, flagged_bbox = [], []
    for frame in source_frames:
        name = frame.get("name")
        if not isinstance(name, str) or not frame.get("labels") or not (BBOX_IMAGES / name).is_file():
            continue
        target = flagged_bbox if name in flagged else normal_bbox
        if len(target) < 50:
            target.append(frame)
    if len(normal_bbox) < 50 or len(flagged_bbox) < 50:
        raise SystemExit(f"Bounded BBox scan yielded {len(normal_bbox)} ordinary and {len(flagged_bbox)} flagged candidates; increase --scan-frames.")

    # Use paired segmentation images and masks directly; choose first 50 plus a greedy
    # second set maximizing semantic-ID coverage. Mask pixels are never edited.
    seg_candidates = []
    for image_path in sorted((SEG / "images/val").glob("*.jpg")):
        stem = image_path.stem
        index_mask, color_mask = SEG / "labels/val" / f"{stem}_train_id.png", SEG / "color_labels/val" / f"{stem}_train_color.png"
        if index_mask.is_file() and color_mask.is_file():
            with Image.open(index_mask) as mask:
                ids = set(mask.getdata()) - {255}
                dimensions = list(mask.size)
            seg_candidates.append((image_path, index_mask, color_mask, ids, dimensions))
        if len(seg_candidates) == 300:
            break
    if len(seg_candidates) < 100:
        raise SystemExit(f"Need at least 100 paired segmentation frames; found {len(seg_candidates)}.")
    first_seg = seg_candidates[:50]
    remaining = seg_candidates[50:]
    selected_second, covered = [], set()
    while len(selected_second) < 50:
        best = max(remaining, key=lambda row: (len(row[3] - covered), -remaining.index(row)))
        selected_second.append(best)
        covered |= best[3]
        remaining.remove(best)

    bbox_names = {f["name"] for f in normal_bbox + flagged_bbox}
    seg_names = {row[0].name for row in first_seg + selected_second}
    if len(bbox_names) != 100 or len(seg_names) != 100 or bbox_names & seg_names:
        raise SystemExit("Selection is not 200-frame disjoint; refusing to write packs.")

    if OUT.exists() and not (OUT / "01-bbox-workflow-50").exists() and not args.force:
        raise SystemExit(f"Output directory exists and is not a generated pack: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    bbox_groups = [(PACKS[0], normal_bbox, "BBox review walkthrough", False),
                   (PACKS[1], flagged_bbox, "BBox rule-review candidates", True)]
    for pack, frames, title, expect_findings in bbox_groups:
        target = OUT / pack
        if target.exists():
            shutil.rmtree(target)
        (target / "images").mkdir(parents=True)
        for frame in frames:
            name = safe_name(frame["name"])
            shutil.copy2(BBOX_IMAGES / name, target / "images" / name)
        (target / "annotations.json").write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps({"task": "bbox", "dataset": "BDD100K detection val", "frames": 50,
            "rule_candidate_set": expect_findings, "source_annotation": "BDD100K val detection JSON; entries copied verbatim",
            "annotation_sha256": sha256(target / "annotations.json"),
            "images": [{"name": f["name"], "sha256": sha256(target / "images" / f["name"])} for f in frames]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "README.md").write_text(f"# {title} — 50 frames\n\nReal BDD100K detection validation images and original annotations. Upload `annotations.json`, then all JPGs. Choose **BBox**. {('The existing deterministic QC rules flag these as unconfirmed review candidates; they are not verified defects.' if expect_findings else 'These frames were selected without current rule findings; this is not a correctness guarantee.')}\n", encoding="utf-8")

    seg_groups = [(PACKS[2], first_seg, "Semantic segmentation validation walkthrough"),
                  (PACKS[3], selected_second, "Semantic segmentation class-coverage walkthrough")]
    for pack, rows, title in seg_groups:
        target = OUT / pack
        if target.exists():
            shutil.rmtree(target)
        for folder in ("images", "masks/index", "masks/color"):
            (target / folder).mkdir(parents=True)
        records, manifest_rows = [], []
        for i, (image, index_mask, color_mask, ids, dims) in enumerate(rows):
            shutil.copy2(image, target / "images" / image.name)
            shutil.copy2(index_mask, target / "masks/index" / index_mask.name)
            shutil.copy2(color_mask, target / "masks/color" / color_mask.name)
            color_name = color_mask.name
            ids_sorted = sorted(ids)
            records.append({"name": image.name, "width": dims[0], "height": dims[1],
                            "mask_index": f"masks/index/{index_mask.name}", "mask_color": f"masks/color/{color_name}",
                            "classes_present": [{"id": c, "name": CLASS_NAMES[c]} for c in ids_sorted if c < len(CLASS_NAMES)]})
            manifest_rows.append({"name": image.name, "mask_index": index_mask.name, "mask_color": color_name,
                "classes_present": ids_sorted, "sha256_image": digest(target / "images" / image.name),
                "sha256_mask_index": digest(target / "masks/index" / index_mask.name),
                "sha256_mask_color": digest(target / "masks/color" / color_name)})
        (target / "annotations.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps({"task": "semantic_segmentation", "dataset": "BDD100K semantic segmentation val",
            "frames": 50, "selection": "First 50 paired val frames" if pack == PACKS[2] else "Greedy semantic-class coverage selection from next 250 paired val frames",
            "mask_semantics": "index PNG uses BDD train IDs 0-18; 255 is ignore/unknown. color PNG is visualization only.",
            "class_names_train_id_order": CLASS_NAMES, "frames_manifest": manifest_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "README.md").write_text(f"# {title} — 50 real frames\n\nUpload `annotations.json`, all JPGs, and all PNGs under `masks/color/` in the new **Segmentation** mode. The color mask overlays the image for human visual review. `masks/index/` contains the unchanged train-ID masks for pixel-level processing; class IDs are in `manifest.json` (255 means ignore). This UI preview does not claim to run segmentation QC/model inference.\n", encoding="utf-8")

    (OUT / "INDEX.json").write_text(json.dumps({"dataset": "BDD100K", "frames_per_pack": 50, "unique_frames": 200,
        "packs": {p: {"task": "bbox" if p in PACKS[:2] else "semantic_segmentation", "frame_names": [f["name"] for f in (normal_bbox if p == PACKS[0] else flagged_bbox)] if p in PACKS[:2] else [r[0].name for r in (first_seg if p == PACKS[2] else selected_second)]} for p in PACKS}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text("# BDD100K workflow walkthroughs\n\nFour disjoint, authentic 50-frame packs: two BBox detection packs and two semantic-segmentation mask packs. Each child README has exact upload steps and limitations. BBox `poly2d` reference examples are separate under `test-data/bdd100k-poly2d-reference`; they are not segmentation. BDD100K media/annotations remain subject to upstream terms.\n", encoding="utf-8")
    print(f"Created {len(PACKS)} disjoint packs x 50 frames ({len(bbox_names)} BBox, {len(seg_names)} segmentation) in {OUT}")


if __name__ == "__main__":
    main()
