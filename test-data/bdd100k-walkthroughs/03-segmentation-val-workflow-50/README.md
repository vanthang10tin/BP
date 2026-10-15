# Semantic segmentation validation walkthrough — 50 real frames

Choose **Segmentation**; upload `annotations.json`, an approved guideline, all JPGs, all PNGs under `masks/color/`, and all PNGs under `masks/index/`. Color masks display the source labels; train-ID masks are compared with the local DINOv3 model output. IDs 0–18 follow the fixed BDD ontology and 255 is ignored. Every disagreement requires human review and is not proof of a labeling error. See `../local-smoke-results/` for the 50-frame model comparison and CSV.
