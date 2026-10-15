# Rule-review candidate walkthrough (50 real frames)

Real frames where the current deterministic QC rules produced candidate findings.

## Run this walkthrough in QC Studio

1. Choose **BBox or Polygon** in the right rail.
2. In Prepare, upload `annotations.json`.
3. Add all 50 files in `images/` for local preview (multi-select the JPGs).
4. Upload the project's approved guideline if available. The built-in guideline is an unapproved template.
5. Analyze, inspect the prioritized queue and issue evidence, then export reviewer Markdown/CSV.

The source annotations are preserved in full, including any other geometry attached to the same frame. The selected workspace mode controls which geometry the canvas displays; it does not rewrite the annotations.

Pack summary: 50 frames, 1223 box labels, 476 polygon shapes, 50 frames with current rule candidate findings. Candidate findings are not adjudicated defects.

`manifest.json` records original source-order indices and SHA-256 checksums. The four packs are disjoint. Observe the BDD100K dataset terms when using or sharing these files.
