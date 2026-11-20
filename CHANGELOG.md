# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-11-20

### Added

- `lungnodule.breakdown.sensitivity_by_size` and the `lungnodule breakdown`
  command: detected nodules by size at chosen false-positive rates. On subset 9,
  at 1 FP per scan: 69% of nodules under 6 mm, 85% of 6-10 mm, 90% of 10-20 mm
  and 100% of 20 mm or more.
- `lungnodule score`: FROC/CPM with a bootstrap interval for any candidates
  file, without images or a GPU.
- `results/luna16-subset9/`: the run's `candidates.csv`, the list of test
  scans and `by_size.json`.

### Fixed

- Bootstrap intervals are reproducible across Python processes. Scans were
  processed in set order, which depends on the string hash seed, so the same
  seed could give different draws. The subset 9 interval, recomputed from the
  full-precision candidates in sorted order, is 0.690-0.801. Earlier figures
  (0.685-0.801 from a copy with rounded probabilities, then 0.688-0.805) are
  superseded.

## [0.5.0] - 2026-11-18

### Added

- First full results: CPM 0.747 (95% CI 0.690–0.801, see 0.6.0) on the 88 scans of LUNA16
  subset 9, with the FROC curve, training curves and a summary in
  `results/luna16-subset9/`. The candidates were re-scored offline with the
  port of the official algorithm and give identical sensitivities.
- `lungnodule.froc.bootstrap_cpm`: percentile bootstrap confidence interval for
  the CPM, resampling scans with replacement.
- `froc.json` from `lungnodule evaluate` and the notebook now includes
  `cpm_interval`, and the CLI prints the interval next to the CPM.
- The Kaggle notebook zips every result into `lung_nodule_results.zip` and turns
  off Ultralytics' Ray Tune callback, which printed warnings on Kaggle.

### Changed

- The README describes the evaluation as scan-level: the run split by scan
  without the optional patient check.

## [0.4.0] - 2026-11-16

### Changed

- Sensitivities are interpolated along the FROC curve by default, as in the
  official LUNA16 script; `interpolate=False` keeps the previous mode.

### Added

- Candidate counts in `FrocResult`, `lungnodule evaluate` and the notebook:
  found nodules, false positives, ignored duplicates and candidates on
  excluded findings.
- `scripts/validate_official.py`: on the official sample submission, the
  seven sensitivities match the official FROC curve to within 4e-10.
- Tests against a Python 3 port of the official evaluation algorithm.

## [0.3.0] - 2026-11-14

### Added

- `lungnodule.inference`: run a detector over every slice of a scan, with an
  adapter for trained Ultralytics YOLO weights (`[train]` extra).
- `lungnodule.candidates`: merge slice detections into 3D candidates.
- `lungnodule.evaluation` and `lungnodule evaluate`: scan-level FROC/CPM on
  whole test scans, writing `candidates.csv` and `froc.json`.
- Kaggle notebook for training and evaluation, with a quick smoke-test mode.

### Changed

- The hit criterion now matches the official LUNA16 script exactly: strictly
  inside the radius, and 10 mm for findings without a recorded diameter.
- `.mhd` files outside subset folders, such as LUNA16's lung masks, are
  ignored instead of stopping the dataset build.

## [0.2.0] - 2026-11-13

### Added

- Train, validation and test split by LUNA16 subset, so every slice of a scan
  stays in one split.
- Patient leak check against LIDC-IDRI metadata, reporting both leaking
  patients and scans that could not be checked.
- `lungnodule build-dataset` command that exports an Ultralytics-ready YOLO
  dataset (`data.yaml`, `splits.csv`, train and validation slices).
- `--max-scans` option for quick experiments on a free GPU.

## [0.1.1] - 2026-11-11

### Added

- CT preprocessing: read `.mhd` scans with SimpleITK, apply a lung window and
  export PNG slices with YOLO labels sized to each nodule's cross-section.
- Background slices sampled only from slices that do not touch any nodule.

## [0.1.0] - 2026-11-10

### Added

- Scan-level FROC analysis and CPM metric with the LUNA16 hit criterion.
- Readers for LUNA16 annotation and candidate CSV files.
- World-to-voxel coordinate conversion, including flipped axes.

[0.6.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jonahernan99/lung-nodule-detection/releases/tag/v0.1.0
