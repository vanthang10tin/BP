#!/usr/bin/env python3
"""Create a bounded, reproducible QC pack from the local official BDD100K val set."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def first_json_array_items(path: Path | str, count: int) -> list[dict]:
    """Decode only the first count objects from a large top-level JSON array."""
    path = Path(path)
    decoder = json.JSONDecoder()
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        eof = False
        started = False
        while len(items) < count:
            if not buffer and not eof:
                buffer = stream.read(1024 * 1024)
                eof = not buffer
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    break
                continue
            if not started:
                if buffer[0] != "[":
                    raise ValueError("Expected a top-level JSON array")
                buffer = buffer[1:]
                started = True
                continue
            if buffer[0] in ",]":
                if buffer[0] == "]":
                    break
                buffer = buffer[1:]
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer += stream.read(1024 * 1024)
                eof = stream.tell() == path.stat().st_size
                continue
            if not isinstance(value, dict):
                raise ValueError("Each BDD100K array item must be an object")
            items.append(value)
            buffer = buffer[end:]
    return items


def sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import os
    parents = Path(__file__).resolve().parents
    project = parents[3] if len(parents) > 3 else parents[2]
    archive = Path(os.environ.get("BDD100K_ARCHIVE", project.parent.parent / "archive" if len(parents) > 3 else project / "archive"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40, choices=range(30, 51))
    parser.add_argument("--labels", type=Path, default=archive / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json")
    parser.add_argument("--images", type=Path, default=archive / "bdd100k/bdd100k/images/100k/val")
    parser.add_argument("--output", type=Path, default=project / "test-data/bdd100k-val-40")
    args = parser.parse_args()
    if not args.labels.is_file() or not args.images.is_dir():
        raise SystemExit(f"BDD100K validation source not found. Expected labels at {args.labels} and images at {args.images}.")
    frames = first_json_array_items(args.labels, args.count)
    if len(frames) != args.count:
        raise SystemExit(f"Requested {args.count} frames but found {len(frames)}.")
    args.output.mkdir(parents=True, exist_ok=True)
    image_out = args.output / "images"
    image_out.mkdir(exist_ok=True)
    copied = []
    for frame in frames:
        name = frame.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Invalid frame name: {name!r}")
        source = args.images / name
        if not source.is_file():
            raise FileNotFoundError(f"Image referenced by validation labels is missing: {source}")
        target = image_out / name
        shutil.copy2(source, target)
        copied.append({"name": name, "sha256": sha256(target), "objects": len(frame.get("labels", []))})
    labels_path = args.output / "annotations.json"
    labels_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "BDD100K detection validation split",
        "source_labels": "archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
        "source_images": "archive/bdd100k/bdd100k/images/100k/val",
        "selection": f"first {len(frames)} entries in original validation annotation order; no labels modified",
        "frames": len(frames),
        "objects": sum(item["objects"] for item in copied),
        "annotations_file": "annotations.json",
        "images_directory": "images/",
        "images": copied,
        "license_note": "Use subject to the BDD100K dataset terms. The upstream code/toolkit BSD license does not by itself license the images or annotation data. Preserve the applicable dataset terms when sharing this pack."
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        f"# BDD100K validation QC pack ({len(frames)} frames)\n\n"
        "This pack contains actual frames and original labels from the BDD100K validation set already present in the workspace archive. It is a deterministic slice (the first entries in original annotation order), not synthetic/demo data. Annotation coordinates and categories are preserved.\n\n"
        "- `annotations.json`: BDD100K JSON array for the included frames.\n"
        "- `images/`: matching source images; select all for browser preview or place them in the configured BDD100K val image folder.\n"
        "- `manifest.json`: provenance, object counts, and image checksums.\n\n"
        "Respect the upstream BDD100K dataset terms when using or sharing this pack. The toolkit repository's BSD license does not automatically license the dataset.\n",
        encoding="utf-8")
    print(f"Created {len(frames)} real BDD100K validation frames, {manifest['objects']} labeled objects, at {args.output}")


if __name__ == "__main__":
    main()
