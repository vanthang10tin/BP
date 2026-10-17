# QC Studio implementation log

## 2026-10-14 correction: BBox + semantic Segmentation

The intended workspace modes are BBox and semantic Segmentation. BDD100K detection `poly2d` is not a semantic raster mask. The old polygon walkthroughs are retained under `test-data/bdd100k-poly2d-reference/`; corrected real-data walkthroughs (2 BBox + 2 segmentation, 50 frames each) are in `test-data/bdd100k-walkthroughs/`. The Blazor UI previews paired segmentation color masks and clearly states that segmentation QC/model inference is not implemented yet. See [WORKLOG_BBOX_SEGMENTATION.md](WORKLOG_BBOX_SEGMENTATION.md) for exact changes and checks.

**Scope:** repository `New folder/BP` only. The sibling `P-037` folder is out of scope. Keep dataset inspection bounded; do not dump image/label data into chat or documentation.

## User outcome

Build a usable BBox/semantic Segmentation QC Studio with dataset and guideline upload, frame review, image/label/issue inspection, guideline lookup/editing, and report export. Keep the recently committed Blazor frontend as the presentation layer. Provide reproducible test sets made from real BDD100K data already in `archive/`, never invented examples. Investigate CVAT handoff and private model options against current official documentation.

## Progress

- [x] Locate the BP repository and inspect recent commits. Current head is `6753c5b`; commits `1a5c53c` and `a7bd6be` add a stock Blazor/MudBlazor scaffold, not a finished QC experience.
- [x] Inspect data format using the existing 20-frame sample and dataset manifest: BDD100K is a JSON array of frames (`name`, metadata, `labels`), with object `category`, `attributes`, `box2d`, and optional `poly2d`. The real train/val archives sit outside the BP repo under `archive/`; they are not to be read wholesale into the conversation.
- [x] Trace the current API: `/api/qc/check` accepts JSON or multipart; `/api/guideline` reads/writes rules; `/api/export/feedback` and `/api/export/csv` produce reports. There is no CVAT integration endpoint today.
- [x] Audit model claims: the current `DualBboxEnsemble`, segmentation ensemble and VLM arbiter use deterministic/hard-coded simulated proposals, not actual YOLO, RT-DETR, SAM, or Gemini inference. The prioritization metric is also forced to a 75% floor. Do not label current findings as model inference or benchmark evidence.
- [x] Replace the stock Blazor home with a three-workspace QC interface and connect upload, review, guideline edit, and exports to the Python API. **Superseded:** the first iteration exposed BBox/Polygon; the requested modes are BBox/semantic Segmentation.
- [x] Create an explicitly scoped, source-attributed 40-frame BDD100K validation pack (real images and matching original labels), with a manifest and usage notes.
- [x] Document private inference/CVAT integration architecture and current limitations with official-source references.
- [x] Update run instructions, fix measured Recall@20 padding, disable simulated model proposals by default, and perform Python syntax/data-flow checks.
- [x] Add four disjoint, provenance-tracked BDD100K walkthroughs with 50 real images each. **Superseded:** the first iteration used detection `poly2d` examples; corrected packs now contain two BBox and two true semantic segmentation workflows with PNG masks.
- [x] Download the official .NET 10 SDK as a portable workspace tool, repair Razor compile errors found by the first actual build, then publish and run the Blazor app locally at `127.0.0.1:5000`.

## Research conclusions (official sources, 2026-10-14)

- Gemini API unpaid service inputs may be used to improve Google products and can be reviewed by people; do not send sensitive data to the unpaid service. Paid Gemini API inputs are not used to improve products, but abuse-monitoring retention still applies. This is not equivalent to self-hosting or zero retention by default.
- Grounding DINO is an open-set, text-promptable detector, useful for selecting classes from a guideline/ontology and proposing boxes. It still needs local deployment, model/version/license review, and measured precision/recall on the target domain.
- SAM 2 is a prompt-driven image/video segmentation model, not a text-based object discovery model. Pair it with a detector (e.g. Grounding DINO) to propose objects, then SAM 2 to refine masks; do not describe SAM as a drop-in replacement for a detector.
- CVAT officially supports issue creation (`POST /api/issues`) and task annotation import (`POST /api/tasks/{id}/annotations/`). A reliable QC handoff should attach issues to task/frame with issue text and coordinates, and optionally prepare an annotation import file; it needs configured CVAT URL/auth/task mapping and should not silently overwrite task annotations.

## Next implementation decisions

- UI must clearly label the active provider state. Currently it runs **rule checks only**; the historical simulated model path is opt-in and visibly marked. Prefer local/self-hosted model providers for confidential data.
- Keep geometry/rule checks deterministic and auditable. Real models should emit predictions, confidence, provider/model version, and provenance; the QC engine should compare these with annotations and create candidate issues for human confirmation.
- The 40-frame test set is a real, deterministic slice of the official BDD100K validation labels and corresponding images. Preserve source names and original annotation values. Do not alter it to make a preferred result appear.
- Four additional walkthrough folders live under `test-data/bdd100k-walkthroughs/`. They are disjoint (200 unique images total), include both source geometry types when present, and have per-image checksums plus exact selection indices. The fourth pack intentionally contains rule-flagged review candidates; they are not verified defects.
- Simulated model proposals are now disabled by default and excluded from CVAT issue pushes. Rule-based issue reports remain available; validate the current heuristics against gold labels before treating them as defects.

## Verification snapshot

- Python syntax compilation succeeded for the backend, ensemble, and sample-pack builder.
- Existing Python QC checks pass after removing the fabricated Recall@20 assertion. The benchmark is now left unavailable without an independent gold set; the earlier queue-derived 88.9% was invalid because it treated the system's own rule flags as truth.
- The real 40-frame QC CLI run processed 40 frames and 1,081 labeled objects; it found 16 structural-rule issues across 9 frames and no model-generated findings in the default configuration. Treat the run as a wiring smoke check only.
- Live API check succeeded for health and 40-frame `/api/qc/check`; the CVAT endpoint correctly returned 503 while server-side CVAT credentials were absent.
- Portable .NET SDK 10.0.401 restored the Blazor packages and built/published with zero warnings and errors after Razor fixes. The published UI is running locally; its API health and guideline requests return HTTP 200.
- Walkthrough-pack validation confirmed exactly 50 annotations and 50 matching JPGs in each folder, 200 unique frames with no cross-pack duplicates, and all 200 image SHA-256 values match their manifests.

## Local machine follow-up (2026-10-14)

- Confirmed the host has an NVIDIA GeForce RTX 3050 Laptop GPU with 4,096 MiB VRAM. `nvidia-smi` reported 1,695 MiB in use and 0% GPU utilization at inspection time, leaving roughly 2.4 GiB free; SAM 3 should not be assumed to fit. Current local stack already has YOLO26 BBox, DINOv3 detect/seg, and SAM 2.1 Nuclio functions, so no model download is justified before checking their actual weights, memory use, and inference endpoints.
- Docker Desktop is running the CVAT Compose project from the local CVAT directory; 17 containers were up. Nuclio YOLOv8, DINOv3 detection/segmentation, SAM2 and SAM2 interactor containers reported healthy. Their host ports 32001–32005 respond on localhost (HTTP 400 for GET `/`, consistent with functions expecting an inference request rather than a browser page).
- CVAT UI/API containers (`cvat_ui`, `cvat_server`) are running but `docker inspect` shows their 8080 ports have no host binding; localhost ports 80, 8080, 8081 refused connections. So this stack is not currently reachable via the usual Windows localhost URL. Need identify the user's access path or publish a host port through Compose before BP can call the CVAT API. Do not recreate/restart the stack until its Compose config and persistent volumes are reviewed.
- Host CUDA driver is visible, but GPU allocation inside Nuclio was not validated. The current BP API still has no adapter to these Nuclio endpoints. A production-quality inference integration requires translating BDD images/labels and the CVAT function schema, matching the CVAT label IDs by task, then comparing predictions with existing annotations. Do not run large-batch inference until this mapping and a one-image smoke request are checked.
- Follow-up detail: Docker Compose reports source files under the local CVAT project directory. The active container configuration can be inspected via Docker, but the source Compose file is not available at that host path to safely diagnose or change the missing CVAT host port.
- Recovered local CVAT access without recreating any existing CVAT container or touching annotation/database volumes. Added a read-only mounted Nginx gateway, `cvat-local-gateway`, bound only to `127.0.0.1:8080`; it routes UI to `cvat_ui:8000` and API/static/data to `cvat_server:8080`. See `LOCAL_CVAT_ACCESS.md`.
- Verified the CVAT login UI, API endpoints (`/api/server/about`, `/api/auth/rules`), UI JS bundles, and local overlay JS return HTTP 200. The protected overlay config returns HTTP 401 when unauthenticated as expected. The local browser now reaches the CVAT sign-in page.

## System redesign and local run reliability (2026-10-15)

- Re-read the BP workflow, current Blazor/API/model/CVAT paths and existing design/research notes without scanning full dataset archives. The repository already contains four disjoint, authentic 50-frame packs: two BBox and two semantic segmentation.
- Added `SYSTEM_DESIGN.md` with current-vs-target architecture, format/task contracts, local/private deployment topology, model selection guardrails, security risks and prioritized acceptance plan. A key gap: SQLite persists workspace/guideline only; run results and reviewer decisions still disappear on refresh.
- Replaced stale top-level quick start with `INSTALL_AND_USE.md` and `RUN_GUIDE.md`: one-command Windows startup, BBox/segmentation pack walkthroughs, local provider setup, CVAT issue flow and troubleshooting. Added PowerShell start/stop scripts with health checks, app-owned PID files and logs; stop script verifies command lines and never stops CVAT, Docker or model containers.
- Changed fallback image preview from a browser-hardcoded `localhost:8000` URL to a same-origin Blazor proxy. This supports a separately hosted frontend configured to reach an API endpoint without relying on browser localhost/CORS assumptions.
- Python logic suite passed 8/8; dataset adapter unit tests passed 3/3. The initial .NET build failed because the default CLI attempted to create `%USERPROFILE%\.dotnet`, outside workspace permissions. Re-run using repo-local `DOTNET_CLI_HOME` and `NUGET_PACKAGES` before recording build status.
- End-to-end local launcher verification passed after two runtime fixes: Windows argument quoting around the workspace's spaced path, and .NET static-web-assets activation. UI/API now start together; browser renders the dark UI with CSS; `/api/health` and frontend are HTTP 200; one real archive JPG returns HTTP 200 through `/qc-image/{filename}`. The stop script verified process command lines, stopped only QC Studio API/frontend, and explicitly left CVAT/Docker/model services untouched; a second start verified repeatability.
- Current live provider probe says BBox, segmentation and panoptic endpoints are all unavailable; Docker CLI cannot connect to `dockerDesktopLinuxEngine`; the CVAT gateway port and CVAT API credentials are also unavailable/unconfigured in this turn. This is the current service state, despite the earlier 2026-10-14 model smoke-run notes. No model inference is claimed from this run, no CVAT write was made, and no image was sent to Gemini. Start Docker Desktop and verify CVAT/Nuclio containers before another model smoke run.
- A browser DOM check confirms the Prepare workspace loads with BBox/Segmentation mode switch, Prepare/Review/Export navigation, label/profile/guideline/image uploads, local privacy notice and disabled Run button until required inputs are selected. Existing previous worklog contains the authentic 50-frame inference/UI smoke run from 2026-10-14; it is not evidence that today's providers are currently running.
- Python logic suite passed 8/8; dataset adapter unit tests passed 3/3. Blazor Release build passed with 0 warnings/errors after redirecting NuGet/.NET state to `.tools`. The default .NET home and user NuGet config are not readable in this managed environment.
- Remaining: restore/check the host Nuclio model containers and CVAT token before a fresh current 50-frame model run; add persistent run/reviewer decision storage, server-side chunked upload, production authentication, and package local Bootstrap Icons to eliminate the UI CDN dependency. Gemini opt-in integration is documented below; no Gemini request was made during this turn.

## Gemini optional integration (2026-10-15)

- Added backend `gemini_provider.py` using the Gemini REST `generateContent` endpoint, structured JSON response and server-only `GEMINI_API_KEY` / configurable `GEMINI_MODEL` (default `gemini-3.8-flash`). No Gemini dependency or secret is embedded in the browser.
- Added `/api/models` Gemini configuration status and a UI checkbox with a 10-frame default / 50-frame maximum. When a key is detected, the checkbox is selected by default so Gemini joins the QC pipeline; uncheck for local-only. Gemini receives the guideline and up to that many prioritized frame images; returned object boxes are separate, unconfirmed review candidates.
- Added `backend/GEMINI_API_KEY.txt` as the one-line paste-only key file and ignored it in Git. Environment variables remain supported for deployments. Updated key setup/privacy docs.
- The empty ignored `GEMINI_API_KEY.txt` is created and ready. Paste only the key into it, then restart the backend. Never paste the key into chat or commit it.
- Blazor build succeeded: 0 warnings, 0 errors. Live Gemini inference remains unverified until a valid key is configured; Python runtime validation was unavailable in this environment, so pilot with 1–2 non-sensitive images before a larger run.
