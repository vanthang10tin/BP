# Local model walkthrough smoke results

Generated 2026-10-14 by running the real local API on one complete 50-frame pack per task. Images and masks stayed on this machine and were sent only to the loopback CVAT Nuclio model functions. No CVAT issue was created.

| Task | Frames | Provider | Time | Result |
|---|---:|---|---:|---|
| BBox | 50/50 | local CVAT BBox detector | 3.9 s | 238 raw model detections; 568 candidate comparisons/issues; 0 model skips |
| Semantic segmentation | 50/50 | local EoMT/DINOv3 mask provider | 35.7 s | 475 class/frame IoU comparisons; mean frame macro-IoU 0.46075; 198 review candidates; 50 prediction overlays |

Every candidate needs a reviewer decision. These metrics compare a model with the uploaded labels; they are not accuracy claims, and the BDD examples are not an independently adjudicated gold set.

- `bbox-50-summary.json` and `bbox-50-issues.csv`
- `segmentation-50-summary.json` and `segmentation-50-iou.csv`

Each CSV records the active guideline version and SHA-256. Reviewer decisions are blank/pending in this run because the smoke test validates inference and export; it does not pretend a human reviewed the findings.
