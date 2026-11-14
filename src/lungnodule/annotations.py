"""Read nodule annotations and detection candidates in LUNA16 CSV format.

Annotations use the columns ``seriesuid, coordX, coordY, coordZ, diameter_mm``
and candidates use ``seriesuid, coordX, coordY, coordZ, probability``, with
coordinates in world millimetres.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

Point = tuple[float, float, float]
_COORDINATES = ("coordX", "coordY", "coordZ")

#: Diameter assumed for findings recorded with a negative diameter (no size),
#: as in the official LUNA16 evaluation script (noduleCADEvaluationLUNA16.py).
UNKNOWN_DIAMETER_MM = 10.0


@dataclass(frozen=True)
class Nodule:
    """A nodule (or other finding) annotated in one CT scan."""

    series_uid: str
    center: Point
    diameter_mm: float

    @property
    def radius_mm(self) -> float:
        diameter = UNKNOWN_DIAMETER_MM if self.diameter_mm < 0 else self.diameter_mm
        return diameter / 2

    def contains(self, point: Iterable[float]) -> bool:
        """Whether ``point`` lies strictly inside the nodule's radius.

        The strict comparison matches the official LUNA16 evaluation script.
        """
        return math.dist(self.center, tuple(point)) < self.radius_mm


@dataclass(frozen=True)
class Candidate:
    """A detection proposed by a model, with its confidence."""

    series_uid: str
    center: Point
    probability: float


T = TypeVar("T")


def _read_rows(
    path: str | Path, value_column: str, build: Callable[[str, Point, float], T]
) -> list[T]:
    required = ("seriesuid", *_COORDINATES, value_column)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

        records = []
        for line, row in enumerate(reader, start=2):
            try:
                center = tuple(float(row[column]) for column in _COORDINATES)
                value = float(row[value_column])
            except ValueError as exc:
                raise ValueError(f"{path}, line {line}: {exc}") from exc
            records.append(build(row["seriesuid"], center, value))
        return records


def load_annotations(path: str | Path) -> dict[str, list[Nodule]]:
    """Load annotations grouped by series UID."""
    grouped: dict[str, list[Nodule]] = {}
    for nodule in _read_rows(path, "diameter_mm", Nodule):
        grouped.setdefault(nodule.series_uid, []).append(nodule)
    return grouped


def load_candidates(path: str | Path) -> list[Candidate]:
    """Load detection candidates in file order."""
    return _read_rows(path, "probability", Candidate)
