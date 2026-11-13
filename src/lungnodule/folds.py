"""Split scans into train, validation and test sets by LUNA16 subset.

LUNA16 distributes its 888 scans in ten folders, ``subset0`` to ``subset9``,
intended for 10-fold cross-validation. Splitting by subset keeps every slice
of a scan in a single split, unlike the slice-level split used in the original
thesis, where neighbouring slices of one scan could land in both training and
test data.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

SPLITS = ("train", "val", "test")
_SUBSET_RE = re.compile(r"^subset(\d+)$")


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
