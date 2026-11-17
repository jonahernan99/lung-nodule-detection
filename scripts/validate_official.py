"""Check compute_froc against the official LUNA16 example evaluation.

The LUNA16 evaluation package ships a sample submission together with the
FROC curve the official script produced for it. This script evaluates the
sample submission with ``lungnodule`` and compares the sensitivities at the
seven false-positive rates with the official curve.

Usage (files from the challenge's ``evaluationScript`` folder)::

    python scripts/validate_official.py path/to/evaluationScript

The folder must contain ``annotations/annotations.csv``,
``annotations/annotations_excluded.csv``, ``annotations/seriesuids.csv``,
``exampleFiles/submission/sampleSubmission.csv`` and
``exampleFiles/evaluation/froc_sampleSubmission.txt``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lungnodule.annotations import load_annotations, load_candidates
from lungnodule.froc import FP_RATES, compute_froc

TOLERANCE = 1e-6


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="The official evaluationScript folder")
    folder = parser.parse_args(argv).folder

    annotations = folder / "annotations"
    with (annotations / "seriesuids.csv").open(encoding="utf-8") as handle:
        scans = [row[0] for row in csv.reader(handle) if row]
    result = compute_froc(
        load_candidates(folder / "exampleFiles" / "submission" / "sampleSubmission.csv"),
        load_annotations(annotations / "annotations.csv"),
        scan_ids=scans,
        excluded_by_scan=load_annotations(annotations / "annotations_excluded.csv"),
    )

    official = np.loadtxt(
        folder / "exampleFiles" / "evaluation" / "froc_sampleSubmission.txt", delimiter=","
    )
    expected = np.interp(FP_RATES, official[:, 0], official[:, 1])

    print(f"Scans: {result.total_scans}, nodules: {result.total_nodules}")
    print(
        f"Found nodules: {result.found_nodules}, false positives: {result.false_positives}, "
        f"duplicates: {result.duplicate_candidates}, on excluded findings: "
        f"{result.excluded_candidates}"
    )
    print(f"{'FP/scan':>8}  {'lungnodule':>11}  {'official':>11}  {'difference':>10}")
    for rate, ours, theirs in zip(FP_RATES, result.sensitivities, expected, strict=True):
        print(f"{rate:>8g}  {ours:>11.9f}  {theirs:>11.9f}  {abs(ours - theirs):>10.2e}")
    print(f"CPM: {result.cpm:.9f} (official curve: {expected.mean():.9f})")

    worst = float(np.max(np.abs(np.asarray(result.sensitivities) - expected)))
    if worst > TOLERANCE:
        print(f"MISMATCH: largest difference {worst:.2e} exceeds {TOLERANCE:g}", file=sys.stderr)
        return 1
    print(f"OK: all seven sensitivities match within {TOLERANCE:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
