"""Evaluate a detector on whole test scans at the scan level.

For every scan in the chosen split, the detector runs on all slices, slice
detections are merged into 3D candidates, and the candidates are scored with
scan-level FROC/CPM. Two files are written:

- ``candidates.csv`` in the LUNA16 submission format
  (``seriesuid, coordX, coordY, coordZ, probability``), which the official
  evaluation script can also read;
- ``froc.json`` with the CPM, the sensitivity at each false-positive rate and
  the full FROC curve.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

from lungnodule.annotations import Candidate, load_annotations
from lungnodule.candidates import merge_detections
from lungnodule.folds import find_scans
from lungnodule.froc import FrocResult, compute_froc
from lungnodule.inference import Detector, detect_scan
from lungnodule.preprocess import load_scan


def read_splits(path: str | Path) -> dict[str, str]:
    """Read the ``splits.csv`` written by ``build_dataset``."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["seriesuid"]: row["split"] for row in csv.DictReader(handle)}


def write_candidates(candidates: Iterable[Candidate], path: str | Path) -> None:
    """Write candidates in the LUNA16 submission format."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        for c in candidates:
            writer.writerow([c.series_uid, *(f"{v:.4f}" for v in c.center), f"{c.probability:.6f}"])


def evaluate_split(
    luna_root: str | Path,
    annotations_csv: str | Path,
    splits_csv: str | Path,
    detector: Detector,
    out_dir: str | Path,
    *,
    split: str = "test",
    excluded_csv: str | Path | None = None,
    max_scans: int | None = None,
    batch_size: int = 16,
    progress: Callable[[str, int, int], None] | None = None,
) -> FrocResult:
    """Run ``detector`` on every scan of ``split`` and compute scan-level FROC.

    ``excluded_csv`` is LUNA16's ``annotations_excluded.csv``: candidates on
    those findings are ignored rather than counted as false positives, as in
    the official evaluation. ``progress`` receives ``(uid, detections,
    candidates)`` after each scan.
    """
    scans = find_scans(luna_root)
    uids = sorted(uid for uid, name in read_splits(splits_csv).items() if name == split)
    if max_scans is not None:
        uids = uids[:max_scans]
    if not uids:
        raise ValueError(f"No scans in the {split!r} split")
    missing = [uid for uid in uids if uid not in scans]
    if missing:
        raise ValueError(f"{len(missing)} {split} scans are not under {luna_root}: {missing[:3]}")

    nodules = load_annotations(annotations_csv)
    excluded = load_annotations(excluded_csv) if excluded_csv is not None else {}

    candidates: list[Candidate] = []
    for uid in uids:
        scan = load_scan(scans[uid][1])
        detections = detect_scan(scan, detector, batch_size=batch_size)
        found = merge_detections(detections, scan, uid)
        candidates.extend(found)
        if progress is not None:
            progress(uid, len(detections), len(found))

    result = compute_froc(
        candidates,
        {uid: nodules.get(uid, []) for uid in uids},
        scan_ids=uids,
        excluded_by_scan=excluded,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_candidates(candidates, out_dir / "candidates.csv")
    report = {"split": split, "candidates": len(candidates), **asdict(result)}
    (out_dir / "froc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result
