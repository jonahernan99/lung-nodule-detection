# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/).

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

[0.2.0]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jonahernan99/lung-nodule-detection/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jonahernan99/lung-nodule-detection/releases/tag/v0.1.0
