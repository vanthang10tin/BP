# BDD100K validation QC pack (40 frames)

This pack contains actual frames and original labels from the BDD100K validation set already present in the workspace archive. It is a deterministic slice (the first entries in original annotation order), not synthetic/demo data. Annotation coordinates and categories are preserved.

- `annotations.json`: BDD100K JSON array for the included frames.
- `images/`: matching source images; select all for browser preview or place them in the configured BDD100K val image folder.
- `manifest.json`: provenance, object counts, and image checksums.

Respect the upstream BDD100K dataset terms when using or sharing this pack. The toolkit repository's BSD license does not automatically license the dataset.
