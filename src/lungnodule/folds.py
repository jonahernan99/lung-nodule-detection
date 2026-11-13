"""Split scans into train, validation and test sets by LUNA16 subset.

LUNA16 distributes its 888 scans in ten folders, ``subset0`` to ``subset9``,
intended for 10-fold cross-validation. Splitting by subset keeps every slice
of a scan in a single split, unlike the slice-level split used in the original
thesis, where neighbouring slices of one scan could land in both training and
test data.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
_SUBSET_RE = re.compile(r"^subset(\d+)$")

#: Column names commonly used for series and patient IDs in LIDC-IDRI metadata.
SERIES_COLUMNS = ("Series UID", "SeriesInstanceUID", "seriesuid", "series_uid")
PATIENT_COLUMNS = ("Subject ID", "PatientID", "patient_id")


def _subset_of(path: Path) -> int | None:
    for parent in path.parents:
        found = _SUBSET_RE.match(parent.name)
        if found:
            return int(found[1])
    return None


def find_scans(root: str | Path) -> dict[str, tuple[int, Path]]:
    """Map each series UID to its LUNA16 subset and ``.mhd`` path.

    Scans may sit directly in ``subsetN`` folders or deeper inside them, as in
    some re-uploads of the dataset.
    """
    scans: dict[str, tuple[int, Path]] = {}
    outside: list[str] = []
    for path in sorted(Path(root).rglob("*.mhd")):
        subset = _subset_of(path.relative_to(root))
        if subset is None:
            outside.append(path.name)
            continue
        uid = path.stem
        if uid in scans and scans[uid][0] != subset:
            raise ValueError(f"Scan {uid} appears in subset{scans[uid][0]} and subset{subset}")
        scans[uid] = (subset, path)
    if outside:
        raise ValueError(f"Scans outside any subsetN folder: {', '.join(outside[:5])}")
    if not scans:
        raise ValueError(f"No .mhd scans found under {root}")
    return scans


def split_by_subset(
    subset_of: Mapping[str, int], *, test_subset: int, val_subset: int
) -> dict[str, str]:
    """Assign each scan to ``train``, ``val`` or ``test`` from its subset number."""
    if test_subset == val_subset:
        raise ValueError("The test and validation subsets must differ")
    present = set(subset_of.values())
    for name, subset in (("test", test_subset), ("validation", val_subset)):
        if subset not in present:
            raise ValueError(f"No scans in subset{subset}, chosen as the {name} subset")
    return {
        uid: "test" if subset == test_subset else "val" if subset == val_subset else "train"
        for uid, subset in subset_of.items()
    }


@dataclass(frozen=True)
class LeakReport:
    """Patients whose scans fall in more than one split."""

    leaks: dict[str, tuple[str, ...]]
    """Patient ID -> the splits their scans appear in."""
    unmatched_scans: tuple[str, ...]
    """Scans with no patient ID in the metadata, which could not be checked."""

    @property
    def ok(self) -> bool:
        return not self.leaks


def _pick_column(available: Sequence[str], choices: Sequence[str], path: Path) -> str:
    for column in choices:
        if column in available:
            return column
    raise ValueError(f"{path} has none of the columns: {', '.join(choices)}")


def load_patient_ids(
    path: str | Path,
    *,
    series_column: str | None = None,
    patient_column: str | None = None,
) -> dict[str, str]:
    """Read a series UID -> patient ID mapping from a metadata CSV.

    LUNA16 files carry no patient ID, so this comes from LIDC-IDRI metadata.
    Common column names are detected; pass them explicitly for other files.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        series = series_column or _pick_column(fields, SERIES_COLUMNS, path)
        patient = patient_column or _pick_column(fields, PATIENT_COLUMNS, path)
        for column in (series, patient):
            if column not in fields:
                raise ValueError(f"{path} has no column {column!r}")
        return {row[series]: row[patient] for row in reader if row[series] and row[patient]}


def check_patient_leaks(split_of: Mapping[str, str], patient_of: Mapping[str, str]) -> LeakReport:
    """Find patients with scans in more than one split."""
    splits_by_patient: dict[str, set[str]] = {}
    unmatched = []
    for uid, split in split_of.items():
        patient = patient_of.get(uid)
        if patient is None:
            unmatched.append(uid)
        else:
            splits_by_patient.setdefault(patient, set()).add(split)
    leaks = {
        patient: tuple(sorted(splits))
        for patient, splits in sorted(splits_by_patient.items())
        if len(splits) > 1
    }
    return LeakReport(leaks=leaks, unmatched_scans=tuple(sorted(unmatched)))
