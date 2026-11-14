# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/).

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

[0.3.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jonahernan99/lung-nodule-detection/releases/tag/v0.1.0
