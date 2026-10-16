# AI, privacy, and CVAT implementation recommendations

**Research checked:** 2026-10-15. Treat model suitability as a hypothesis to measure on BDD100K and on the team's own labeled data; published benchmark scores do not establish QC accuracy on this task.

## Current implementation status

- The legacy Python `DualBboxEnsemble` and `VLMArbiter` remain placeholders. The new `/api/qc/run` path bypasses them and calls the machine's loopback CVAT Nuclio providers: the configured BBox function and DINOv3 semantic-mask function. Provider readiness is checked through those local endpoints and reported by `/api/models`. No cloud VLM is called by this build. Issues remain candidates requiring human confirmation.
- The old prioritization code manufactured a 75% Recall@20 result and marked the target met unconditionally. A subsequent audit found the calculated queue value was also tautological because the engine used its own rule flags as truth. The API now reports Recall@20 and its baselines as unavailable until independent adjudicated frame-level labels are supplied.
- The present 40-frame run is a **QC workflow smoke run**, not evidence of model recall/precision or proof the BDD100K labels are defective. It flags 16 structural-rule issues across 9 frames; 31 frames have no finding from current heuristics. The earlier 88.9% queue value was invalid as Recall@20 and is no longer presented as a benchmark. Some current heuristics (e.g. `BOX_NEAR_BOUNDARY_UNMARKED`) can flag valid labels because this JSON does not always carry the team's expected attributes.

## Recommended detection and segmentation design

1. Keep ontology, coordinate bounds, degeneracy, duplicate, and polygon validity checks deterministic and fast.
2. **First convert guideline to a reviewer-approved target ontology.** Extract canonical class, aliases, visual description, exclusions, attributes, desired annotation type (box, instance mask/polygon, semantic mask), and clause provenance. Merge these with the dataset profile and labels, but retain classes mentioned in the guideline even if they are absent from the uploaded labels. Surface ambiguous extraction/mapping for review rather than silently omitting it.
3. **Route by each model's capability and output type.** The local BDD detector is the efficient baseline for classes it supports. For a guideline class outside that list, evaluate local Grounding DINO with the approved class phrase/description as text prompt; the official implementation accepts image/text pairs and returns boxes/scores. This addresses open-vocabulary box proposals, but its code license does not automatically establish the checkpoint's license, and zero-shot false positives require per-class measurement.
4. **For segmentation targets, choose the right mask semantics.** Keep the current local semantic segmentation provider for the fixed BDD train-ID ontology only where its mapping is verified. Evaluate SAM 3/SAM 3.1 for instance concept segmentation: Meta's repo supports text prompts and returns masks, boxes, and scores. The official repo currently lists Python 3.12+, PyTorch 2.7+, CUDA 12.6+ and requires access approval for checkpoints; its code/checkpoint use the SAM License. This makes it a candidate, not an automatic replacement or a guaranteed semantic-segmentation model.
5. **Keep a private VLM in a different role.** A locally served Qwen3-VL can help structure guideline text or explain ambiguous crops with JSON/schema-constrained output. It does not replace a detector/segmenter, should not decide absence/presence by itself, and must not produce ground truth. The official Qwen repo documents local file/base64 inference and vLLM serving; inspect the specific checkpoint's license and hardware requirements. Require reviewer acceptance.

### Practical starting stack

- **BBox first:** a locally served, BDD100K-trained detector (evaluate a permissively licensed RT-DETR/D-FINE implementation and YOLO26 only after license review). Keep the ensemble provider behind a stable adapter (`predict(image) -> boxes, classes, scores, model/version`).
- **BBox/open classes:** current local detector + Grounding DINO fallback, with a capability table showing supported, promptable and unsupported classes before inference.
- **Polygon/instance segmentation:** compare current model + detector-seeded SAM 2 and SAM 3 concept masks on the same adjudicated subset. SAM can return useful mask geometry; class correctness still comes from its concept prompt and review/validation, not from mask shape alone.
- **Private guideline/VLM assist:** run Qwen3-VL inside controlled, self-hosted inference if hardware and the selected checkpoint's terms allow. Keep endpoints private, disable public egress, and retain human review. The repository itself indicates Apache-2.0, but verify checkpoint/model card terms separately.
- **Benchmark protocol:** freeze a gold subset, split by source video/scene to avoid adjacent-frame leakage, compare each provider against human adjudication, and publish per-class precision, recall, false-positive rate, missed-object rate, mask IoU, latency, VRAM, and reviewer correction rate. Do not compare only a single aggregate score or set a target in code.

## Gemini privacy decision

Gemini **an provider in this workflow** if you want it. Do not wire it into the local-only path implicitly. On an explicit Gemini run, the guideline text and selected images/crops go to Google's API. Google's current terms say unpaid Gemini API content can be used to improve products and may be reviewed by humans; do not use unpaid service for sensitive/confidential data. Paid API content is not used to improve products, but abuse monitoring/retention still needs review; ZDR is a separate eligibility/terms question, not an automatic property of paid use. Keep the API key in the backend environment and make mode, model, frame count, payload and cost visible before starting.

**Code status (BP):** Gemini Vision candidate integration is implemented behind a per-run checkbox. It receives guideline text plus chosen frame images and that frame's known annotations; outputs normalized box/class/confidence/evidence candidates, shown separately as a Gemini issue. Enter the key in the **Configure Gemini** card; it is stored as `GEMINI_API_KEY` in the backend `.env` (or configure the environment variable for deployment). Model defaults to `gemini-3.8-flash`. Frame cap defaults to 10 and is limited to 50. The guideline-to-ontology structured extraction/preflight remains a future step and is not implied by this image-analysis integration.

Never let Gemini silently replace annotations or make the final reject decision. For detection/masks with exact geometry, specialist local detectors/segmenters remain the quality baseline; Gemini boxes/masks are proposals to validate. The actual remote API response still needs a small pilot using the owner's key.

For a cost-controlled pilot, use Gemini Flash and start with the requested 50-frame test pack, a per-run frame cap, and billing spend cap. Google prices image input by image-token resolution and model/output tokens; cost must be calculated from the selected model and actual resolution instead of hard-coded as free. As of this research date, Gemini 3.8 Flash is listed at $0.75/1M input tokens and $3.75/1M output tokens through 2026-12-31 on paid standard tier; free tier exists but has different data terms. Newer/high-capability models (for example Gemini 4 Argon) are paid-only and cost more. Recheck before implementation/use because model IDs and rates change.

For the current app, no VLM service is called. The interface identifies local detector/mask disagreement as an unverified candidate. Do not put an API key in frontend code.

## CVAT issue handoff

The Review page calls CVAT `POST /api/issues` with frame, job, evidence location, issue type, explanation, and guideline citation; identical messages are skipped. It does not modify/replace annotations. The return-job action can also call `PATCH /api/jobs/{id}` to set stage/state; pending candidates and `annotation/rejected` are currently defaults. Check the exact CVAT Job and filename-to-frame map before submitting. The integration has not yet been write-tested against a disposable job.

Before enabling it:

1. Set `CVAT_URL` and `CVAT_ACCESS_TOKEN` only in the backend environment (see `.env.example`). Create a scoped, expiring CVAT Personal Access Token; never put it in the Blazor app or browser.
2. Prefer `Load job metadata` for an exact filename-to-frame map. The offset fallback only applies when a reviewer explicitly confirms order; it maps `frame_index` to `frame_offset + frame_index` and is unsafe for reordered/subset uploads.
3. Test against a disposable CVAT job first. Issues are external writes and stay in CVAT; identical existing issues are skipped, but partial failures can require retry.
4. If corrected annotations need to be pushed, use CVAT's task/job annotation import endpoint with an explicit reviewed file and `append`/`replace` mode decision. That is a separate operation; this UI deliberately does not overwrite annotations.

## Test pack provenance

`test-data/bdd100k-val-40/` contains 40 actual images and their unchanged labels selected from the original validation JSON in its original order. `manifest.json` records per-image checksums and object counts. The data is not synthetic. The upstream BDD100K code/toolkit license does not by itself license the image/annotation data; respect BDD100K dataset terms when using or sharing it.

## Sources

- Google, [Gemini API additional terms](https://ai.google.dev/gemini-api/terms?hl=en) and [zero data retention guidance](https://ai.google.dev/gemini-api/docs/zdr?hl=en).
- IDEA Research, [Grounding DINO official repository](https://github.com/IDEA-Research/GroundingDINO) and [Grounding DINO paper](https://arxiv.org/abs/2303.05499).
- Meta, [SAM 3/SAM 3.1 official repository](https://github.com/facebookresearch/sam3) and [SAM 3 research description](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).
- Qwen team, [Qwen3-VL official repository and deployment guide](https://github.com/QwenLM/Qwen3-VL).
- CVAT, [Issues API](https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/issues-api/), [IssueWriteRequest fields](https://docs.cvat.ai/docs/api_sdk/sdk/reference/models/issue-write-request/), [task annotation import API](https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/tasks-api/), and [Personal Access Token guidance](https://docs.cvat.ai/docs/api_sdk/access_tokens/).
- Ultralytics, [license guidance](https://www.ultralytics.com/license). Its current licensing page states private deployment requires an Enterprise License unless the project is released under AGPL-3.0; review this before selecting YOLO for a closed/private app.
- Berkeley DeepDrive, [BDD100K toolkit license](https://github.com/ucbdrive/bdd100k/blob/master/LICENSE). This is the code license, not a substitute for the dataset's own terms.
