# Lung Nodule Detection

[![CI](https://github.com/jonahernan99/lung-nodule-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/jonahernan99/lung-nodule-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

**An independent reimplementation of my B.Eng. thesis on detecting lung nodules
in CT scans, rebuilt with a stricter, scan-level evaluation.**

> **Status:** two full training runs done. With a [2.5D input](#25d-input-experiment)
> that stacks neighbouring slices, the detector reaches a **CPM of 0.861**
> (95% CI 0.802–0.920) on 88 held-out LUNA16 scans with false positives counted
> per scan, up from 0.747 with single slices (paired difference +0.114, 95% CI
> +0.060 to +0.176). A control run rules out early stopping as the cause. See
> [Results](#results). Next: more training seeds and all ten LUNA16 folds.

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

## Results

Two full runs on a free Kaggle GPU with
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), on the same
split: trained on LUNA16 subsets 0–7, model selection on subset 8, tested on
subset 9 (88 scans, 105 nodules). Both use Ultralytics YOLOv5s (`yolov5su`,
anchor-free head) on 512 px slices and are evaluated with the best validation
checkpoint.

| Input (sensitivity at FP per scan) | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | **CPM** (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single slice (v0.4.0, 100 epochs) | 0.486 | 0.590 | 0.724 | 0.800 | 0.867 | 0.876 | 0.886 | 0.747 (0.690–0.801) |
| 2.5D stack (v0.8.0, early stopping) | 0.714 | 0.781 | 0.838 | 0.876 | 0.933 | 0.943 | 0.943 | **0.861** (0.802–0.920) |

- **The 2.5D input raises the CPM by 0.114**, with a paired 95% confidence
  interval of +0.060 to +0.176 (`lungnodule compare`, 1,000 resamples of the
  same 88 scans). The 2.5D run scored higher in every draw, and 10,000 draws
  give the same interval. The two separate intervals almost touch, which
  understates the evidence: a paired comparison removes the variation that
  comes from which scans are in the test set.
- The gain is largest where few false positives are allowed: at 1/8 per scan,
  sensitivity goes from 0.486 to 0.714. Both runs find 101 of the 105 nodules at
  some confidence; the 2.5D run ranks them above its false positives more often.
- **Control run.** The 2.5D run also used early stopping (it stopped at epoch
  81, best epoch 61), where the first run trained 100 epochs. A third run with
  single slices and the same early stopping stopped at epoch 56 and produced
  candidates byte-identical to the first run's: training is deterministic with
  seed 0, and the first run's best checkpoint was also epoch 32
  ([`summary.json`](results/luna16-subset9-gray-control/summary.json)). Early
  stopping therefore changed nothing, and the gain comes from the input.
- **Remaining caveats.** Each input was trained with one seed, so the spread
  between training runs is not measured, and all runs are tested on one fold.
  The 2.5D input was chosen from the size analysis below before the second run,
  not by comparing several models on the test set.
- **Checked independently:** both runs' candidates were re-scored offline with
  `compute_froc` and with the port of the official LUNA16 algorithm, giving the
  same seven sensitivities as Kaggle. Numbers, settings and candidates are in
  [`results/luna16-subset9/`](results/luna16-subset9/) (single slice) and
  [`results/luna16-subset9-stack/`](results/luna16-subset9-stack/) (2.5D), so
  anyone can re-score them with the official script or with `lungnodule score`
  (see [Score a candidates file](#score-a-candidates-file)).

![Scan-level FROC curve of the 2.5D run on 88 test scans, CPM 0.861](results/luna16-subset9-stack/froc.png)

### Detection by nodule size

The CPM averages over every nodule. Broken down by size
([single slice](results/luna16-subset9/by_size.json),
[2.5D](results/luna16-subset9-stack/by_size.json)), the first run's weak spot
was clear, and it is where the 2.5D input helped most:

| Nodule size | Nodules | 1/8 FP per scan: single slice | 1/8 FP: 2.5D | 1 FP per scan: single slice | 1 FP: 2.5D |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under 6 mm | 45 | 16 (36%) | **25 (56%)** | 31 (69%) | **36 (80%)** |
| 6–10 mm | 34 | 18 (53%) | **24 (71%)** | 29 (85%) | 30 (88%) |
| 10–20 mm | 20 | 11 (55%) | **20 (100%)** | 18 (90%) | **20 (100%)** |
| 20 mm or more | 6 | 6 (100%) | 6 (100%) | 6 (100%) | 6 (100%) |

Small nodules are 43% of the test set. With single slices they accounted for
most of the misses: at one false positive per scan, 14 of the 21 missed nodules
were under 6 mm. On a 512 px slice such a nodule spans only a few pixels and
appears on few slices, so a 2D detector has little to go on. With the 2.5D
input the misses at that rate fall to 13, of which 9 are under 6 mm. The groups
above 6 mm are small (6 to 34 nodules), so their percentages are rough.

### Compared with the thesis

| | Thesis (2023) | This repository |
| --- | --- | --- |
| Data split | Mostly by 2D slice (LIDC-IDRI) | By scan: LUNA16 subset 9 held out |
| False positives counted | Per slice | Per scan |
| Model | YOLOv5s, anchor-based, tuned anchors | YOLOv5s, Ultralytics anchor-free head |
| CPM | 88.0% | **86.1%** with the 2.5D input (95% CI 80.2–92.0%); 74.7% with single slices |

The two numbers are not directly comparable, and the drop is expected. A scan
has around 200 slices, so "1 false positive per slice" allows about 200 per
scan; here it allows one. The held-out scans also share no slices with
training. The 74.7% is the honest estimate of how the single-slice approach
performs on unseen scans; the 2.5D input brings the stricter, per-scan figure
to 86.1%. (This run split by scan without the optional patient check;
LIDC-IDRI has a few patients with more than one scan.) It is still well below
leading LUNA16 systems, which use 3D context and report averages over all ten
folds.

### Training curves

![Training and validation losses, precision, recall and mAP over 100 epochs](results/luna16-subset9/training_curves.png)

Training losses fall steadily, but validation losses reach their minimum
around epochs 25–45 and then rise, while validation mAP peaks near epoch 35 and
slowly declines: the model starts to overfit. Because `best.pt` keeps the best
validation epoch, the evaluated model is not the overfit final one, but the
second half of training spent GPU time without improving it. The notebook now
stops training 20 epochs after the best validation fitness (`PATIENCE`): the
2.5D run stopped at epoch 81, with its best epoch at 61
([curves](results/luna16-subset9-stack/training_curves.png)).

### 2.5D input (experiment)

A 2D detector sees one slice at a time. A small nodule and a vessel can look
the same there: the difference is in depth, where a vessel continues through
neighbouring slices and a nodule appears and fades within a few millimetres.
`--input stack` puts the slices about 2 mm on either side in the red and blue
channels, with the labeled slice in green, so the same YOLO network gets that
context without any change to its architecture.

- The distance is set in millimetres (`--stack-mm`) and converted per scan,
  since LUNA16 slice spacing ranges from about 0.6 to 2.5 mm. At 2 mm, a 6 mm
  nodule still crosses both neighbours.
- The choice is saved in `dataset/preprocessing.json` and read back by
  `evaluate`, so a model is always evaluated on the kind of image it was
  trained on.
- Ultralytics reads training images in BGR order and treats arrays passed to
  `predict` as BGR too, so the detector adapter reverses the stacked channels
  before predicting; a test checks that the two paths give the same array.
- Hue and saturation augmentation would mix the channels, so the notebook turns
  them off in `stack` mode.

**Result:** CPM 0.861 against 0.747 with single slices on the same 88 test
scans, a paired difference of +0.114 (95% CI +0.060 to +0.176), with the
largest gains on small nodules and at low false-positive rates. A single-slice
control run with the same early stopping reproduced the single-slice result
exactly, so the gain is due to the input (see [Results](#results)).

### Eigen-CAM explanations

The thesis used Eigen-CAM to show where the detector looks and to filter
candidates. Eigen-CAM projects a layer's activations onto their first
principal component, so it needs no gradients and works with a detector.
`lungnodule.explain` computes it for the layer that feeds the finest detection
scale (stride 8, a 64 x 64 map for a 512 px slice), the one that matters for
small nodules.

Looking at maps is not evidence on its own, so each map is also turned into a
number: the share of the map's weight inside the candidate's box. For the
highest-scoring candidates of an evaluation, `lungnodule explain` labels each
one with the LUNA16 hit criterion and reports the median box energy of real
nodules and of false positives, and the AUC of box energy as a separator. An
AUC near 0.5 would mean the maps cannot tell the two apart, which would argue
against using them as a filter.

```bash
lungnodule explain --luna-root path/to/luna16 --annotations annotations.csv \
  --excluded annotations_excluded.csv --candidates results/candidates.csv \
  --dataset dataset --weights best.pt --out results --top 200
```

It writes `eigencam.csv` (one row per candidate), `eigencam.json` (the
summary) and `eigencam.png` (maps of the top nodules and false positives).
Two implementation details are fixed explicitly: the arbitrary sign of the
principal component is oriented to follow the mean activation, and the
Ultralytics adapter checks that its hook captures every new image.

**Result on the 2.5D run** ([`eigencam.json`](results/luna16-subset9-stack/eigencam.json)):
among the top 200 candidates, 85 are nodules, 29 false positives and 86 lie on
excluded findings. The map is exactly zero inside 50 of the 85 nodule boxes,
even for detections with confidence near 0.9, and box energy does not separate
nodules from false positives (AUC 0.42). The single-slice control run gives the
same picture (AUC 0.50,
[`summary.json`](results/luna16-subset9-gray-control/summary.json)). The example maps show why: the first
principal component of this layer follows large structures such as the body
outline and lung borders, not the detected objects. In this setup, Eigen-CAM
cannot filter candidates. That is a result about this configuration (first
component, finest scale, sign following the mean activation), not about
saliency maps in general. A per-candidate map computed on a crop around each
detection is the natural next test; it needs the scans, so it would run on
Kaggle.

![Eigen-CAM maps for the top nodules and false positives of the 2.5D run](results/luna16-subset9-stack/eigencam.png)

## What is implemented

- **`lungnodule.froc`**: scan-level FROC curve and CPM (mean sensitivity at
  1/8 to 8 false positives per scan) using the LUNA16 hit criterion, with
  support for excluded findings.
- **`lungnodule.annotations`**: readers for LUNA16 annotation and candidate CSV
  files.
- **`lungnodule.geometry`**: conversion between world coordinates (mm) and
  voxel indices, including scans with flipped axes.
- **`lungnodule.preprocess`**: reads `.mhd` scans, applies a lung window and
  exports 2D slices with YOLO labels, as gray images or as 2.5D stacks of
  neighbouring slices. Each box is sized to the nodule's
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
- **`lungnodule.breakdown`** and the **`lungnodule score`** and
  **`lungnodule breakdown`** commands: score any candidates file and report
  detected nodules by size at chosen false-positive rates.
- **`lungnodule.froc.compare_cpm`** and the **`lungnodule compare`** command:
  a paired bootstrap of the CPM difference between two models evaluated on the
  same scans.
- **`lungnodule.evaluation`** and the **`lungnodule evaluate`** command:
  scan-level FROC/CPM on whole test scans, writing `candidates.csv` in the
  LUNA16 submission format and `froc.json`, including a bootstrap confidence
  interval for the CPM.
- **`lungnodule.explain`**, **`lungnodule.explanations`** and the
  **`lungnodule explain`** command: Eigen-CAM maps for a trained detector and
  the box energy of the top candidates, compared between nodules and false
  positives.

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
mode, a short smoke test, before a full run. `INPUT_MODE` chooses the gray or
the 2.5D input, and `PATIENCE` sets early stopping. The model is Ultralytics'
YOLOv5s variant with an anchor-free head, so it differs from the thesis, which
used the original anchor-based YOLOv5s with tuned anchors.

### Score a candidates file

Any CSV in the LUNA16 submission format can be scored without images or a GPU:

```bash
lungnodule score --candidates results/luna16-subset9/candidates.csv \
  --annotations annotations.csv --excluded annotations_excluded.csv \
  --scans results/luna16-subset9/test_scans.txt      # CPM 0.7469 (95% CI 0.690-0.801)
lungnodule breakdown --candidates results/luna16-subset9/candidates.csv \
  --annotations annotations.csv --excluded annotations_excluded.csv \
  --scans results/luna16-subset9/test_scans.txt --fp-rate 0.125 1 8 --out by_size.json
lungnodule compare --baseline results/luna16-subset9/candidates.csv \
  --candidates results/luna16-subset9-stack/candidates.csv \
  --annotations annotations.csv --excluded annotations_excluded.csv \
  --scans results/luna16-subset9/test_scans.txt     # +0.114 (paired 95% CI +0.060 to +0.176)
```

`--scans` lists every evaluated scan, so scans without any candidate still
count towards false positives per scan. `compare` scores two models on the
same scans with a paired bootstrap.

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

Add `--max-scans 20` for a quick first run on a free GPU, and `--input stack`
(optionally `--stack-mm 2`) for the [2.5D input](#25d-input-experiment).

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
- [x] Results: FROC curve and CPM with a confidence interval, compared with the thesis
- [x] Sensitivity by nodule size (small nodules are the main weakness)
- [x] Early stopping in the training notebook
- [ ] Augmentation study (validation loss rises after epoch ~35)
- [x] Eigen-CAM maps with a box-energy check of nodules against false positives
- [x] Eigen-CAM results from the 2.5D run (no separation in this setup)
- [ ] Per-candidate Eigen-CAM on crops around each detection
- [x] 2.5D input option: neighbouring slices as color channels
- [x] Kaggle run with the 2.5D input, compared with the single-slice run
- [x] Single-slice control run with the same early stopping (identical to the first run)
- [ ] More training seeds for both inputs
- [ ] All ten LUNA16 folds, to report a cross-validated CPM

## Data

The model is trained on [LUNA16](https://luna16.grand-challenge.org/Data/),
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

[AGPL-3.0](LICENSE), because the detector is built on Ultralytics YOLO, which
uses this license.
