# Audit of the workflow slides in `flow/`

**Scope:** the 11 screenshots in `flow/`, the current BP implementation, the existing local CVAT/Nuclio stack, and primary documentation checked on 2026-10-14. Dataset contents were not scanned for this review.

## Verdict

The direction is sound: cheap deterministic checks first, model-generated candidates second, human review prioritized by expected value, then a measured gold/honeypot loop. That is a reasonable human-in-the-loop QC product. The slides are a research/architecture overview, though, not yet a build-ready spec. The main risk is presenting plausible model output and metric targets as if the implementation had already demonstrated them.

## What is strong

- The four-stage QA ladder avoids sending every frame directly to an expensive model or a person.
- The slides explicitly identify hallucination, missed detections, coordinate-convention differences, and the need for out-of-sample predictions.
- The reviewer workflow, issue explanations, rule citations, and a real adjudicated gold set fit the team's product goal.
- Keeping images and inference local is appropriate for private project data. The host already has local CVAT functions for YOLO26, DINOv3 detection/segmentation, and SAM2; the BP service has not connected to them yet.
- The existing 40 authentic BDD100K frames are useful to prove parsing, image matching, overlays, issue ordering, and export wiring. They are not a model benchmark.

## Corrections before treating the slides as a product spec

| Priority | Finding | Why it matters | Required correction |
|---|---|---|---|
| P0 | The implementation had claims of `98.5%` agreement and `All-Agreed` in CSV exports, while real model inference is not connected. | This gave an invented QC pass confidence and could mislead annotators or supervisors. | Corrected in `app.py` and the legacy report generator: pass rows now state that active rules found no issue and that no model/gold set independently confirmed the frame. |
| P0 | The old `PROJECT_FLOW.md` reported simulated findings and a queue-derived Recall@20 value. | Those results were stale or circular and conflict with the rules-only default. | Updated the flow document and changed Recall@20 to unavailable without independent gold labels. |
| P0 | Critical candidate findings were turned into automatic `REJECT` recommendations, and the feedback export applied a hardcoded 10% return-job threshold attributed to P-037. | A structural rule flag is not a confirmed annotation defect; this could falsely reject a job under an unapproved policy. | Corrected: findings now require reviewer confirmation; exports no longer issue accept/reject decisions or name P-037. The built-in guideline is explicitly a template and no longer embeds that return-job rule. |
| P0 | BDD100K's `labels` array contains two different geometric families: road objects use `box2d`; lane markings and drivable areas use `poly2d`. | A BBox-vs-Polygon choice cannot define the whole task. A frame may contain both, and a lane is a path (`closed: false`) while a drivable area is an area. | Key rendering, validation, model mapping, and exports off category + geometry type. Treat `lane` and `drivable area` as semantic-region classes, not extra object-detector classes. |
| P1 | Generic rules such as “one class per pixel / no mask overlap” and fixed aspect-ratio or tightness thresholds are presented as universal. | Overlap can be valid in instance labels; BDD100K's lane/drivable geometry has different semantics from object boxes. Thresholds can reject valid source annotations. | Put policy thresholds in a versioned guideline, use shape-specific rules, and validate every heuristic against adjudicated examples before it creates a high-severity issue. |
| P1 | The slides mention cross-validation / out-of-sample predictions, but the product has no training provenance or gold labels yet. | In-sample detections can simply echo training labels. A disagreement may be model error, annotation-policy difference, or an unannotated/ignored object—not proof the human label is wrong. | Store prediction provenance and label every model finding as a candidate. Use out-of-sample predictions and human adjudication before calculating error recall. |
| P1 | `Recall@20% >= 70%` appears as a target and the earlier smoke report showed a queue-derived value. | Without independent gold truth, the metric only measures how many rule-flagged frames the queue re-selects; it is not recall of actual defects. With 40 frames, the top 20% is only 8 frames. | The code now reports Recall@20 as unavailable. Define a confirmed-error frame, freeze the gold set, compare ranked vs random and severity-only baselines at the same review budget, and report confidence intervals. Also report fixed-time results because frame review effort varies. |
| P1 | The slides treat Ground Truth jobs, honeypots, review, and quality analytics as one CVAT capability. | CVAT docs distinguish Community from Online/Enterprise; current docs list immediate feedback and quality analytics for Online/Enterprise. A local installation must be checked against its edition/version. | Keep issue creation through the API as the MVP integration. Treat CVAT-native quality analytics as optional until edition support is confirmed. Ground truth itself must be adjudicated and maintained. |
| P2 | Five-model ensemble + VLM arbitration is too much for the first demo and the 4 GB laptop GPU. | More models do not automatically improve quality; latency, VRAM contention, label-map mismatch, and correlated errors increase. Existing Nuclio containers reporting healthy does not prove weights execute correctly on a real frame. | Start with one local detector. Add one local segmentation path only for polygon work. Run one model at a time, capture latency/VRAM, and add another provider only when it measurably improves confirmed-error precision or recall. |
| P2 | A “new guideline” exported from a VLM or QC run could look authoritative. | Free-form VLM text can contradict the actual project policy. The current guideline manager is in-process memory, not a controlled version history. | Have AI draft proposed changes only. Persist guideline versions, source, approver, effective date, and rule IDs; human approval promotes a proposal. |

## Recommended plan and acceptance checks

### 0. Establish a truthful baseline — in progress

- Keep simulated model output disabled and label the provider as rules-only until a real adapter is verified.
- Remove fabricated `98.5%` / `All-Agreed` pass language and stale model lists from API/exports.
- Make the 40-frame result reproducible and explicitly non-benchmark.

**Accept when:** health/API/export never name a model as configured unless an inference provider is actually configured; all pass rows say only that enabled checks found no issue.

**Additional correction:** priority labels and reviewer exports now describe candidate findings and never auto-reject a job. The default guideline is labeled as an unapproved template; policy thresholds need an uploaded/approved project guideline.

### 1. Freeze the MVP task contract

- Target only BDD100K 2D first: 10 road-object box classes, lane paths, and drivable-area polygons.
- Define category-to-shape rules, allowed attributes, coordinate origin/units, `poly2d` curve/closed semantics, image dimensions, and frame identity.
- Upload should preserve the original JSON and image hashes; validation should show actionable row/frame errors before analysis.

**Accept when:** all 40 test frames parse, every label maps to one documented shape family, overlays align to the original image, and invalid input is rejected with a frame/label location.

### 2. Stabilize deterministic QC and guideline handling

- Keep rules that can be proven from the file itself: malformed fields, out-of-bounds or degenerate geometry, invalid class/shape pairs, and exact duplicates.
- Put subjective checks—tightness, visibility, truncation, minimum object size, whether an object must be labeled—behind explicit guideline IDs and configurable thresholds.
- Give each run the guideline version/hash and each finding a rule ID, severity, evidence, and “candidate/confirmed/rejected” status.

**Accept when:** reviewers can trace every rule finding to the exact guideline version and no unapproved guideline edit silently changes prior runs.

### 3. Connect one local detector to BP

- Use an adapter contract such as `predict(image) -> [{class, box, score}]` with model ID, weight checksum, preprocessing, device, duration, and failure reason.
- Start with the local BDD-oriented YOLO function if its actual weights and class IDs match the 10 BDD object classes. The container/image name and declared checkpoint naming should be reconciled before claiming which model runs.
- Send one real frame to the function, validate coordinate orientation/scale and label mapping, then compare with its human annotations. Use class-aware matching and tuned thresholds.
- Keep Grounding DINO/open-vocabulary output as a discovery candidate for guideline classes outside the fixed ontology; never let text prompts redefine the task ontology on their own.

**Accept when:** a clean one-image run returns valid boxes on the expected device, emits recorded provenance, maps every class or explicitly marks it unmapped, and a human has inspected the overlay. No batch issue push during this step.

### 4. Add polygon support after detector validation

- Prefer the existing DINOv3 segmentation function only if its segmentation labels/output semantics match the task; otherwise prompt SAM2 with detector boxes for mask refinement.
- Do not use SAM2 as the missing-object detector. Keep lane polylines, drivable polygons, and object masks as distinct output types.
- Evaluate contour/mask overlap against reviewed masks; track boundary failures separately from box misses.

**Accept when:** at least one representative example per polygon family renders and round-trips without coordinate/curve loss, and each model-derived mask is clearly a proposal.

### 5. Validate review ranking with independent truth

- Use the 40 frames to validate mechanics only. Then sample a larger, stratified set across classes, lighting, scene, and source clips; have two reviewers independently adjudicate and resolve disagreements.
- Prevent leakage by splitting or sampling by source video/sequence where possible.
- Report issue precision, recall, false positives per 100 frames, Recall@20% of reviewed frames, a random baseline, severity-only baseline, review time, and latency/VRAM. Keep frame-level and object-level metrics separate.

**Accept when:** no metric is computed against the system's own rule flags; evaluation artifacts identify the gold set and adjudication protocol; target thresholds are selected from observed utility, not coded in advance as a guaranteed pass.

### 6. Complete CVAT handoff safely

- Keep `CVAT_URL` and a scoped token in the backend. Resolve source filenames to CVAT task/job frame IDs—do not rely on a fixed offset unless ordering is proven identical.
- Preview the exact frame, position, issue text, and guideline citation. Create issues first; they attach reviewer findings but do not themselves return/reassign a job or change annotations.
- Treat corrected annotation import as a separate reviewed action with explicit append/replace choice.

**Accept when:** a disposable task receives one issue on the expected frame, duplicate retries are idempotent, and failure/partial state is visible. Confirm installed edition supports any CVAT-native honeypot/analytics feature before making it a dependency.

### 7. Persist and polish the three-page product

- Prepare: upload data/guideline, validate schema and model/class compatibility.
- Review: prioritized queue, image with both relevant geometries, issue evidence, guideline citation, reviewer decision, and CVAT issue handoff.
- Export: findings, reviewed annotation changes, guideline proposal/approved version, and run manifest.
- Add storage for run IDs, source hashes, guideline versions, model provenance, and reviewer decisions before multi-user use.

**Accept when:** a new browser session can reopen a run and reproduce its report, with originals unchanged and every exported change traceable to a human decision.

## Model and source notes

- BDD100K's official format describes 10 road-object categories with `box2d`; `lane` and `drivable area` use `poly2d`. [BDD100K format](https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md)
- Cleanlab's object-detection API says predictions should be out-of-sample (for example cross-validation) and flags image-level candidates for missing boxes, wrong classes, or poor locations. [Cleanlab object detection](https://docs.cleanlab.ai/stable/cleanlab/object_detection/filter.html)
- CVAT has a self-hosted Community edition, while current documentation marks immediate job feedback and quality analytics as Online/Enterprise features. [CVAT Community](https://docs.cvat.ai/docs/products/community/), [quality-control availability](https://docs.cvat.ai/docs/qa-analytics/quality-control/)
- Rex-Omni is a real CVPR 2026 object-perception model, but its official repository declares IDEA License 1.0 and a Qwen research-license dependency; check permitted use before considering it for a private product. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Detect_Anything_via_Next_Point_Prediction_CVPR_2026_paper.html), [official repository/license](https://github.com/IDEA-Research/Rex-Omni)
- Gemma 4 is real and Google announced it under Apache 2.0, but multimodal VLM use is unnecessary for the first QC MVP and must still fit the GPU. [Google Gemma 4 release](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- SAM 3 is not a free drop-in replacement: use the official SAM license/checkpoint terms and benchmark it against the local SAM2 path before adopting it. [SAM 3 repository](https://github.com/facebookresearch/sam3)
