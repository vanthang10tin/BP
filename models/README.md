# Model files and provider status

Model files and large checkpoints are excluded from Git by `.gitignore`. A file visible in one developer's local `models/` directory is not automatically available to a clean clone or cloud deployment.

## What the application can use

- The backend contains an optional Ultralytics YOLO26 BBox adapter and a YOLO26 segmentation adapter. `requirements.txt` does not install PyTorch/Ultralytics, and the adapters need compatible local weights. If these prerequisites are missing, the pipeline records model skips; rule checks can still run.
- The provider module also supports configured loopback HTTP endpoints for BBox and segmentation capabilities. It does not provision CVAT/Nuclio containers. No checked-in deployment configuration starts the SAM2/DINOv3 services described in older notes.
- Gemini is a separate optional cloud API. When enabled locally, selected images and annotation/guideline context leave the machine. It is disabled on the configured public demo.

## COCO to BDD100K mapping

The code maps a limited set of COCO detector classes (`person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `traffic light`, `stop sign`, `train`) to BDD100K labels. This is a naming alignment only: the weights are not fine-tuned on BDD100K, `stop sign` does not cover all traffic signs, and classes without a valid source class are not reliably detected. A mapped prediction is a review candidate, not a label verdict.

## Reproducing model checks

1. Install a PyTorch/Ultralytics combination supported by the chosen weights and hardware.
2. Obtain weights from their official source and review licensing/usage terms.
3. Set `QC_YOLO26_WEIGHTS` (and, for segmentation, `QC_YOLO26_SEG_WEIGHTS`) to local paths.
4. Check `/api/models` and the run output for provider availability, skips, model identity, device and thresholds before interpreting findings.
5. Record weight SHA-256, dataset/split, config, hardware and command. Do not report accuracy without independent adjudicated ground truth.

This repository does not claim SAM2, DINOv3, CVAT Nuclio, or Gemini have been deployed or benchmarked as part of a fresh clone.
