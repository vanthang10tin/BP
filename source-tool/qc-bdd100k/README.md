# QC Studio 2D · BDD100K

Local-first workflow for reviewing BDD100K-style 2D annotations. The repository contains a Python QC API and the newer Blazor review interface.

## Current capability

- Load BDD100K annotation JSON, optional guideline `.md`/`.txt`, and matching local images.
- Run structural geometry/ontology checks and local model comparisons; review prioritized frames in BBox or semantic Segmentation mode.
- Edit a guideline in the current API process and export reviewer Markdown, CSV issues, or the guideline.
- Optionally create frame-linked CVAT issues when a backend-only CVAT URL/token and job mapping are configured.
- Inspect real BDD100K validation data in `../../test-data/bdd100k-val-40/`.

**Model status:** the QC backend calls the local CVAT Nuclio BBox detector and DINOv3 semantic-mask provider through loopback adapters. It does not call Gemini or another hosted VLM. Detector/mask disagreements are unverified review candidates; no label is auto-corrected. The DINO/SAM components available on this machine are not all interchangeable: SAM2 is a prompt/refinement tool, not the image-level semantic model. See [AI_RECOMMENDATIONS.md](../../AI_RECOMMENDATIONS.md).

Model availability and candidate counts are host-specific: a machine with local YOLO26 weights or a reachable CVAT Nuclio endpoint can produce candidates while a machine without either will report model skips. Check `http://127.0.0.1:8000/api/models` on each host before comparing runs. Model disagreements are review candidates, not confirmed annotation errors; use the same model, weights, threshold, labels, and images when comparing results.

## Run the new Blazor workflow

Use the two-terminal setup in [`../../RUN_GUIDE.md`](../../RUN_GUIDE.md), or:

1. In this folder: `python -m pip install -r requirements.txt` then `python backend/app.py`.
2. In `frontend/BlazorApp`: `dotnet run --no-launch-profile --urls http://127.0.0.1:5000`.
3. Open <http://localhost:5000>. The API and legacy HTML dashboard remain on port 8000.

Requires Python 3.10+ and .NET 10 SDK.

## Dataset smoke run

```powershell
python qc_checker.py --input ../../test-data/bdd100k-val-40/annotations.json
```

The 40-frame pack uses original validation images/labels; its manifest includes hashes and provenance. Current output reflects deterministic rule checks, not human-adjudicated truth. Follow upstream BDD100K terms when using or sharing the pack.

`real_data_qc_report.csv` is a legacy report from the earlier simulated-model implementation. Its model claims and scores are not valid evidence; do not use it for evaluation. `backend/test_real_dataset.py` now reads only the requested number of frames and labels its CSV as rule-only.

## CVAT setup

Set `CVAT_URL=http://127.0.0.1:8080` and a scoped, expiring `CVAT_ACCESS_TOKEN` only in the Python backend process. In Export, load the exact job frame map from CVAT metadata or upload a map; otherwise explicitly confirm the frame order and offset. Only reviewer-confirmed issues are created, at the evidence box/mask-disagreement location. This does not overwrite labels or return/reopen a job. See [LOCAL_CVAT_ACCESS.md](../../LOCAL_CVAT_ACCESS.md) and [AI_RECOMMENDATIONS.md](../../AI_RECOMMENDATIONS.md).

## Other datasets

Use `config/dataset_profile.template.json` to map the source ontology to model labels, coordinates and thresholds. For semantic masks, provide `class_names` in train-ID order or an explicit `mask_id_to_class` mapping plus ignore IDs. `config/bdd100k_segmentation_profile.json` is the exact BDD100K 0–18 mapping. Never infer mask IDs by sorting only the classes present in a sample. The current adapter accepts BDD100K/normalized rows, COCO detection JSON, and the documented workspace YOLO-JSON form; YOLO `.txt` labels need a dataset manifest/parser before upload.
