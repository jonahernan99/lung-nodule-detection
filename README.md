# Lung Nodule Detection

[![CI](https://github.com/jonahernan99/lung-nodule-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/jonahernan99/lung-nodule-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

**An independent reimplementation of my B.Eng. thesis on detecting lung nodules
in CT scans, rebuilt with a stricter, patient-level evaluation.**

> **Status: work in progress.** The evaluation and data tooling come first, so
> the model is judged correctly from its first training run. No results from
> this repository are reported yet.

## Background

Lung nodules are the main early indicator of lung cancer, and radiologists can
miss around a quarter of them in CT scans. For my thesis at the
[BIVL²ab lab](https://noesis.uis.edu.co/entities/publication/4b44ab5b-d4e8-49c6-8d12-509ff69b15f4),
Universidad Industrial de Santander (2023), I adapted YOLOv5s to localize
nodules on 2D CT slices from LIDC-IDRI. Dylan Villalba led the Eigen-CAM
explainability work, and the project was advised by Prof. Fabio Martínez and
Prof. Franklin Sepúlveda. The original code stayed with the lab; this
repository is new code written from scratch.

### What this reimplementation changes

| | Thesis (2023) | This repository |
| --- | --- | --- |
| Data split | Mostly by 2D slice | By scan, using LUNA16's ten official subsets |
| False-positive rate in FROC/CPM | Per slice | Per scan, as in the LUNA16 challenge |
| Code | Kept by the lab | Open source, tested, with CI |

Splitting by slice lets nearly identical neighbouring slices of the same scan
land in both training and test sets, which inflates results. Rates per slice
are also far more lenient than per scan, since one CT scan has around 200
slices. The thesis reported **88.0% CPM** under those conditions. Expect this
repository's numbers to come out lower and more realistic, and both will be
reported side by side with their evaluation protocols.

## What is implemented

- **`lungnodule.froc`**: scan-level FROC curve and CPM (mean sensitivity at
  1/8 to 8 false positives per scan) using the LUNA16 hit criterion, with
  support for excluded findings.
- **`lungnodule.annotations`**: readers for LUNA16 annotation and candidate CSV
  files.
- **`lungnodule.geometry`**: conversion between world coordinates (mm) and
  voxel indices, including scans with flipped axes.

```python
from lungnodule.annotations import load_annotations, load_candidates
from lungnodule.froc import compute_froc

result = compute_froc(
    load_candidates("predictions.csv"),
    load_annotations("annotations.csv"),
    scan_ids=test_scan_ids,
)
print(f"CPM: {result.cpm:.3f}", dict(zip(result.fp_rates, result.sensitivities)))
```

## Roadmap

- [x] Scan-level FROC/CPM evaluation
- [x] LUNA16 annotation readers and coordinate conversion
- [ ] Preprocessing: read `.mhd` scans, apply a lung window and export slices with YOLO labels
- [ ] Fold assignment from LUNA16 subsets, checking that no patient appears in two folds
- [ ] Training notebook for a free cloud GPU (Kaggle / Colab)
- [ ] Merge 2D slice detections into 3D candidates per scan
- [ ] Results: FROC curve and CPM, compared with the thesis
- [ ] Eigen-CAM explainability maps

## Data

The model will be trained on [LUNA16](https://luna16.grand-challenge.org/Data/),
a curated subset of LIDC-IDRI with 888 CT scans and 1,186 nodules accepted by at
least three of four radiologists. Medical images are never committed to this
repository. Download them from the official source and follow their terms of
use.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest --cov=lungnodule
```

## License

[AGPL-3.0](LICENSE), because the detector will be built on Ultralytics YOLO,
which uses this license.
