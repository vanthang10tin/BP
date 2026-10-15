# BP implementation log — BBox and semantic segmentation

**Scope:** `source-tool/qc-bdd100k` and its BP-local test packs only. The sibling P307/P-037 project is out of scope. Updated 2026-10-14.

## Correct data/task split

- The product has two modes: **BBox detection** and **semantic segmentation**. BDD `poly2d` vector annotations are not semantic masks; the old polygon test material remains isolated as a reference.
- Four authentic BDD100K walkthrough folders contain 50 unique frames each: two BBox packs and two segmentation packs. Segmentation pairs JPG + train-ID index PNG + color PNG. IDs 0–18 are fixed BDD class IDs; 255 is ignore. Manifests and SHA-256 provenance travel with the packs.

## End-to-end work implemented

1. Local workspace registry stores task/profile and versioned guideline text/hash in ignored `.qc-data/` SQLite. Draft guidelines can be reviewed; activation pins the revision used for subsequent runs. A run response carries workspace and guideline provenance.
2. `/api/qc/run` is the unified path. BBox input normalizes BDD100K, COCO, or the documented structured YOLO-JSON; it compares source boxes with a real local detector. Semantic segmentation compares train-ID masks with the machine's local DINOv3 mask provider, calculating per-class IoU and candidate issues. A fixed BDD train-ID map avoids the dangerous “sort observed classes and assume IDs” bug. Other taxonomies require an explicit profile/mask map.
3. Local-model adapters use loopback providers by default and reject remote hosts without explicit opt-in. No Gemini/public VLM is called. Model differences remain unverified candidates; the application does not auto-edit labels.
4. Review UI shows GT labels, model proposals, class IoU, issue evidence and per-issue Confirm/Dismiss controls. Queue priority is driven by emitted findings. Segmentation can switch between uploaded color mask and the model mask.
5. CVAT integration creates `POST /api/issues` for confirmed candidates only, with evidence position, job/frame mapping, explanation and guideline citation. It prefers exact filename/frame metadata and requires a deliberate order confirmation for offset fallback. This is not annotation import and does not reopen/return a job. Token stays in backend environment.
6. Export supports Markdown feedback, issue CSV, segmentation per-class IoU CSV and guideline text. CVAT is exposed for either task, including masks via mismatch-region issue position.
7. API binds to `127.0.0.1`; docs describe dataset profile/template, model/privacy choice, local CVAT, and actual limitations.

## Model and research decision

- Use the installed CVAT BBox detector for proposals now; use the locally available DINOv3 segmentation endpoint for semantic masks. SAM2 is a prompt/refinement component rather than stand-alone open-world semantic discovery. Do not replace working local providers with SAM3 until checkpoint access, license, VRAM, and gold-set performance are checked.
- Keep Gemini disabled for private data. Local inference avoids image egress; private VLMs such as Qwen3-VL are a future controlled deployment candidate only after hardware and model checkpoint licensing review. See [AI_RECOMMENDATIONS.md](AI_RECOMMENDATIONS.md) for research sources and a benchmark plan.
- Model score versus existing annotations is disagreement analysis, not truth. Establish independently adjudicated, scene/video-separated gold data before reporting precision, recall, or a benchmark target.

## Validation completed

- Prior work validated the four packs: 50 frames each, 200 unique image names, paired segmentation images/index/color masks, authentic labels, and checksums.
- Python `py_compile` succeeds; 3 new unit tests pass for BDD, COCO xywh, and normalized YOLO-JSON adapters. .NET build succeeds with 0 warnings and 0 errors.
- Local API health reports all three loopback providers ready on CUDA. CVAT status is not configured because no backend access token is present.
- One real frame per task was passed through the end-to-end workspace/guideline/model/issue path. BBox produced real detector proposals. Segmentation produced model masks, class IoU, candidate issues and evidence boxes.
- Browser-level BBox smoke test completed on the local UI: uploaded authentic 50-frame BDD annotations, the explicit template-only guideline, and a matching source image; the UI completed all 50 frames and showed 568 candidates, the ranked frame queue, annotation/model overlays, issue evidence, and per-issue decisions. Confirm and Dismiss states appeared on the selected issue cards. The UI CSV download completed (272,622 bytes) and its rows retained confirmed/dismissed/pending decisions plus guideline version/hash. The currently served review/export pages were also inspected.
- UI verification caught a Razor interpolation defect that rendered `v@GetInt(...)` literally in the review/export guideline-version labels. Fixed both labels in `Home.razor`; the updated frontend builds successfully to an isolated output directory with 0 warnings and 0 errors. The already-running frontend process holds its original DLL open, so this verification build did not replace the live server binary; restart the frontend process to see the label fix in the current browser session.
- The smoke UI used the codebase's built-in guideline text, which is explicitly marked template-only and not approved. Its policy-dependent findings are therefore not a policy-compliance result. No BDD team-approved guideline was present in the workspace.
- Full 50-frame BBox pack completed in 3.9s: 238 raw model proposals, 568 candidate comparisons/issues, 0 model skips. Full 50-frame segmentation pack completed in 35.7s: 475 class/frame comparisons, mean frame macro-IoU 0.46075, 198 candidate issues, and prediction overlays on all 50 frames. CSV exports succeeded. These are disagreement smoke metrics, not model accuracy claims.
- Results are saved under [`test-data/bdd100k-walkthroughs/local-smoke-results/`](test-data/bdd100k-walkthroughs/local-smoke-results/). The updated local app is running at `http://127.0.0.1:5000/`; API binds to `127.0.0.1:8000`.
- No CVAT write was attempted. The issue API and exact job frame-map path are implemented, but the backend needs a scoped `CVAT_ACCESS_TOKEN` and an explicit target job before a real issue handoff can be tested safely.

## Known boundaries

- COCO/normalized/structured YOLO JSON are accepted; arbitrary dataset formats and YOLO text directories need a parser/manifest, not guessed conversion. Semantic profile must explicitly define class names in train-ID order or `mask_id_to_class` and ignore IDs.
- CVAT issue API needs a backend-scoped access token. Creating issues is reversible only by deleting them in CVAT and is therefore not part of smoke testing.
- Local BDD provider class coverage may not match every project ontology. Unmapped model classes are ignored/reported; calibrate thresholds and aliases on adjudicated examples before production use.
- Fifty-frame segmentation output contains full-size model mask previews and may use substantial server memory. Run batches when scaling beyond the walkthrough size.
- Review pages can display many candidates per frame (31 on the highest-priority BBox frame in this run). Before production, add issue grouping/filtering and pagination/virtualization so a single crowded frame does not require a long scroll.
