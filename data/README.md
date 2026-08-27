# Data

This repository does not include clinical or research image datasets.

## BUS-UCLM
Place the locally obtained BUS-UCLM files under `data/raw/` after confirming the dataset licence and source documentation.

## Rules
- Preserve patient identifiers/pseudonymous patient IDs needed for clustered analysis.
- Do not commit raw ultrasound images to Git by default.
- Record provenance, licensing, class mappings, and any exclusions.
- Generate a structured `dataset_manifest.csv` before model evaluation.
