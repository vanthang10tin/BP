# BBox review walkthrough (50 real frames)

A real BDD100K sample for the BBox workspace mode.

## Run this walkthrough in QC Studio

1. Choose **BBox** in the right rail.
2. In Prepare, upload `annotations.json`.
3. Add all 50 files in `images/` for local preview (multi-select the JPGs).
4. Upload the project's approved guideline if available. The built-in guideline is an unapproved template.
5. Analyze, inspect the prioritized queue and issue evidence, then export reviewer Markdown/CSV.

The source annotations are preserved in full, including any other geometry attached to the same frame. The selected workspace mode controls which geometry the canvas displays; it does not rewrite the annotations.

Pack summary: 50 frames, 901 box labels, 423 polygon shapes, 11 frames with current rule candidate findings. Candidate findings are not adjudicated defects.

`manifest.json` records original source-order indices and SHA-256 checksums. The four packs are disjoint. Observe the BDD100K dataset terms when using or sharing these files.
