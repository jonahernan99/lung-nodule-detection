"""Build a YOLO training dataset from a LUNA16 download.

Output layout::

    out_dir/
      data.yaml              Ultralytics dataset config
      splits.csv             series UID, subset and split of every scan
      train/images/*.png     train/labels/*.txt
      val/images/*.png       val/labels/*.txt

Test scans are listed in ``splits.csv`` but not exported. Scan-level FROC
evaluation runs the detector on every slice of each test scan, so sampled test
slices would be of no use and would tempt slice-level evaluation.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lungnodule.annotations import load_annotations
from lungnodule.folds import find_scans, split_by_subset
from lungnodule.preprocess import export_slices, load_scan

EXPORTED_SPLITS = ("train", "val")


@dataclass(frozen=True)
class DatasetSummary:
    scans: dict[str, int]
    """Number of scans per split, including the unexported test split."""
    slices: dict[str, int]
    """Number of exported slices per exported split."""


def _write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "names:",
        "  0: nodule",
    ]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    luna_root: str | Path,
    annotations_csv: str | Path,
    out_dir: str | Path,
    *,
    test_subset: int = 9,
    val_subset: int = 8,
    negatives: int = 4,
    max_scans_per_split: int | None = None,
    seed: int = 0,
    progress: Callable[[str, str, int], None] | None = None,
) -> DatasetSummary:
    """Export train and validation slices with YOLO labels, split by LUNA16 subset.

    ``max_scans_per_split`` limits the export for quick experiments, such as a
    first run on a free GPU. ``progress`` is called with ``(uid, split, slices)``
    after each scan.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scans = find_scans(luna_root)
    split_of = split_by_subset(
        {uid: subset for uid, (subset, _) in scans.items()},
        test_subset=test_subset,
        val_subset=val_subset,
    )
    nodules = load_annotations(annotations_csv)

    exported: Counter[str] = Counter()
    slices: Counter[str] = Counter()
    with (out_dir / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seriesuid", "subset", "split"])
        for uid in sorted(scans):
            subset, path = scans[uid]
            split = split_of[uid]
            writer.writerow([uid, subset, split])
            if split not in EXPORTED_SPLITS:
                continue
            if max_scans_per_split is not None and exported[split] >= max_scans_per_split:
                continue
            count = export_slices(
                load_scan(path),
                nodules.get(uid, []),
                uid,
                out_dir / split,
                negatives=negatives,
                seed=seed,
            )
            exported[split] += 1
            slices[split] += count
            if progress is not None:
                progress(uid, split, count)

    _write_data_yaml(out_dir)
    scan_counts = Counter(split_of.values())
    return DatasetSummary(
        scans={split: scan_counts.get(split, 0) for split in ("train", "val", "test")},
        slices={split: slices.get(split, 0) for split in EXPORTED_SPLITS},
    )
