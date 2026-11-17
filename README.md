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
- **`lungnodule.preprocess`**: reads `.mhd` scans, applies a lung window and
  exports 2D slices with YOLO labels. Each box is sized to the nodule's
  cross-section on that slice. Background slices are sampled only from slices
  that do not touch any nodule, so a barely visible nodule is never taught as
  background.
- **`lungnodule.folds`**: splits scans by LUNA16 subset and checks, using
  LIDC-IDRI metadata, that no patient has scans in more than one split.
- **`lungnodule.dataset`** and the **`lungnodule build-dataset`** command:
  export a ready-to-train YOLO dataset with an Ultralytics `data.yaml`. Test
  scans are only listed in `splits.csv`, because scan-level evaluation runs
  the detector on every slice of each test scan.
- **`lungnodule.inference`**: runs any 2D detector over every slice of a scan,
  with an adapter for trained Ultralytics YOLO weights.
- **`lungnodule.candidates`**: merges detections of the same nodule on
  neighbouring slices into one 3D candidate in world coordinates, so a nodule
  seen on five slices is not counted as four false positives.
- **`lungnodule.evaluation`** and the **`lungnodule evaluate`** command:
  scan-level FROC/CPM on whole test scans, writing `candidates.csv` in the
  LUNA16 submission format and `froc.json`.

### Matching the official LUNA16 evaluation

The metric is validated against the challenge's own evaluation code,
`noduleCADEvaluationLUNA16.py`, in three ways:

1. **Rules checked line by line:** a candidate hits a nodule only when it is
   strictly within the nodule's radius, findings without a recorded size count
   as 10 mm wide, a nodule found by several candidates counts once, candidates
   on excluded findings are ignored, and sensitivities are interpolated along
   the curve.
2. **Same numbers on the official example.** The evaluation package ships a
   sample submission and the FROC curve the official script produced for it.
   [`scripts/validate_official.py`](scripts/validate_official.py) evaluates the
   same 551,065 candidates over 888 scans:

   | FP/scan | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | Official curve | 0.000000 | 0.000843 | 0.000843 | 0.001686 | 0.004216 | 0.007589 | 0.016020 |
   | This code | 0.000000 | 0.000843 | 0.000843 | 0.001686 | 0.004216 | 0.007589 | 0.016020 |

   The largest difference is 4e-10, and the found nodules (1,120) and
   duplicate detections (231) match the official report exactly. The sample
   is a deliberately weak submission, so its CPM is near zero.
3. **Same results as a port of the official algorithm** on 20 random cases
   with tied probabilities, excluded findings and negative diameters
   ([`tests/test_froc_official.py`](tests/test_froc_official.py)).

One documented discrepancy: the official example report ignores 1,294
candidates on excluded findings, while the script shipped next to it, and this
code, ignore 6,555. The difference is fully explained by the 10 mm rule for
findings without a size: with negative diameters taken literally, this code
reproduces the report's counts exactly (1,294 ignored, 548,420 false
positives), so the report was produced by an earlier version of the script.
The official script is Python 2 and does not run as is on Python 3.

### Train and evaluate on Kaggle

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) runs the whole
pipeline on a free Kaggle GPU with the `vafaeii/luna16` dataset attached: build
the dataset, train YOLOv5s and evaluate per scan. It starts in `QUICK_RUN`
mode, a short smoke test, before a full run. The model is Ultralytics'
YOLOv5s variant with an anchor-free head, so it differs from the thesis, which
used the original anchor-based YOLOv5s with tuned anchors.

### Build the training data

```bash
pip install -e ".[data]"
lungnodule build-dataset \
  --luna-root path/to/luna16 \          # folder containing subset0 ... subset9
  --annotations path/to/annotations.csv \
  --out dataset \
  --test-subset 9 --val-subset 8 \
  --patients path/to/lidc_metadata.csv  # optional: stop if a patient leaks
```

Add `--max-scans 20` for a quick first run on a free GPU.

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

```python
from lungnodule.annotations import load_annotations
from lungnodule.preprocess import export_slices, load_scan

nodules = load_annotations("annotations.csv")
scan = load_scan("subset0/<series_uid>.mhd")
export_slices(scan, nodules.get("<series_uid>", []), "<series_uid>", "dataset/train")
```

## Roadmap

- [x] Scan-level FROC/CPM evaluation
- [x] LUNA16 annotation readers and coordinate conversion
- [x] Preprocessing: read `.mhd` scans, apply a lung window and export slices with YOLO labels
- [x] Fold assignment from LUNA16 subsets, checking that no patient appears in two folds
- [x] `build-dataset` command producing an Ultralytics-ready dataset
- [x] Training notebook for a free cloud GPU (Kaggle)
- [x] Merge 2D slice detections into 3D candidates per scan
- [x] Hit criterion checked against the official LUNA16 evaluation script
- [x] FROC validated numerically against the official example evaluation
- [ ] Results: FROC curve and CPM, compared with the thesis
- [ ] Eigen-CAM explainability maps
- [ ] Experiment: stack neighbouring slices as color channels (2.5D input)

## Data

The model will be trained on [LUNA16](https://luna16.grand-challenge.org/Data/),
a curated subset of LIDC-IDRI with 888 CT scans and 1,186 nodules accepted by at
least three of four radiologists. Medical images are never committed to this
repository. Download them from the official source and follow their terms of
use.

## Development

```bash
pip install -e ".[dev]"   # includes SimpleITK and Pillow for the data tools
ruff check . && ruff format --check .
pytest --cov=lungnodule
```

## License

[AGPL-3.0](LICENSE), because the detector will be built on Ultralytics YOLO,
which uses this license.
