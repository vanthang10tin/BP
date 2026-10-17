# Dataset setup and reproducible measurements

## Bundled sample

`demo_data/sample_test_labels.json` and `demo_data/real_val_sample_20.json` are small fixtures included for local demonstration. The matching image directory is `demo_data/images/`. The sample data is not an independent adjudicated gold set and must not be used to report detector accuracy or annotator quality.

Run the bundled-data QC script from the repository root:

```powershell
python backend/test_real_dataset.py --limit 20
```

It reads the bundled sample by default and writes generated files under `.qc-data/reports/`, leaving committed fixtures unchanged. For a larger labels file, pass `--input <path> --limit <n>`. The script processes leading records in source order; it is not a random or stratified sample. The script does not establish ground-truth correctness of its findings.

## Obtain full BDD100K

Use the [official BDD100K portal](https://bdd-data.berkeley.edu/portal.html#download), review its terms, and download only the modalities/splits needed for the project. Do not commit images or labels unless redistribution is allowed. To reproduce a run, record the exact label file, split, record selection/order, SHA-256, configuration, model/weight hash, software versions, machine and command.

The CLI run above accepts a labels JSON path. For model inference, upload matching images through the UI or configure their local path/provider. The application does not automatically download BDD100K data or model weights.

## Model weights

Weights are excluded from Git. YOLO inference also requires separately installed PyTorch/Ultralytics; these are not in the default `requirements.txt`. See [model notes](../models/README.md). Record weight provenance and checksum, and verify `/api/models` plus per-run skip counts before reporting model outputs.

## Evaluation protocol

Input labels are not ground truth for errors in those same labels. For precision/recall, Recall@20 or reviewer-time savings, create an independent set adjudicated by qualified reviewers, define error classes and object/frame matching thresholds in advance, freeze the evaluation split, and compare against baselines at the same review budget. No such evaluation result is currently claimed by this repository.
