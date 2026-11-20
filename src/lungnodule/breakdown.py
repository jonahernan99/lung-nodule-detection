"""Sensitivity by nodule size at a fixed false-positive rate.

The CPM averages over all nodules, which hides where a detector fails. This
module reports, for one operating point (for example 1 false positive per
scan), how many nodules of each size are detected.

The operating point uses the same matching as :func:`lungnodule.froc.compute_froc`.
Its threshold is the score of the first false positive that would exceed the
allowed number (``fp_rate`` times the number of scans): a nodule counts as
detected when its best candidate scores strictly above it. Summed over all
sizes, this gives the non-interpolated sensitivity at that rate
(``compute_froc(..., interpolate=False)``), which can differ slightly from the
interpolated value used for the CPM.

Nodules with a negative (unknown) diameter fall into no size bin and are left
out of the breakdown. LUNA16's ``annotations.csv`` has none.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lungnodule.annotations import Candidate, Nodule
from lungnodule.froc import _match_all

#: Size bins in millimetres, as (lower, upper): lower bound included, upper excluded.
SIZE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 6.0),
    (6.0, 10.0),
    (10.0, 20.0),
    (20.0, math.inf),
)


def _label(lower: float, upper: float) -> str:
    if lower == 0:
        return f"under {upper:g} mm"
    if math.isinf(upper):
        return f"{lower:g} mm or more"
    return f"{lower:g}-{upper:g} mm"


@dataclass(frozen=True)
class SizeRow:
    """Detection counts for nodules of one size range."""

    size: str
    min_mm: float
    max_mm: float
    nodules: int
    detected: int
    """Nodules found at the operating point."""
    found_at_any_score: int
    """Nodules hit by some candidate, however low its score."""

    @property
    def sensitivity(self) -> float | None:
        return self.detected / self.nodules if self.nodules else None


@dataclass(frozen=True)
class SizeBreakdown:
    """Sensitivity by size at one false-positive rate per scan."""

    fp_rate: float
    threshold: float
    """Candidates must score strictly above this to count."""
    rows: tuple[SizeRow, ...]

    @property
    def sensitivity(self) -> float:
        """Overall sensitivity at the operating point, over the binned nodules."""
        nodules = sum(row.nodules for row in self.rows)
        return sum(row.detected for row in self.rows) / nodules if nodules else 0.0

    def as_dict(self) -> dict:
        return {
            "fp_rate": self.fp_rate,
            "threshold": self.threshold,
            "sensitivity": self.sensitivity,
            "rows": [
                {
                    "size": row.size,
                    "min_mm": row.min_mm,
                    "max_mm": None if math.isinf(row.max_mm) else row.max_mm,
                    "nodules": row.nodules,
                    "detected": row.detected,
                    "sensitivity": row.sensitivity,
                    "found_at_any_score": row.found_at_any_score,
                }
                for row in self.rows
            ],
        }


def operating_threshold(fp_scores: Sequence[float], scan_count: int, fp_rate: float) -> float:
    """Score a candidate must exceed so that false positives stay within ``fp_rate`` per scan.

    Returns ``-inf`` when all false positives fit within the allowance.
    """
    if scan_count < 1:
        raise ValueError("scan_count must be at least 1")
    if fp_rate < 0:
        raise ValueError("fp_rate must not be negative")
    allowed = math.floor(fp_rate * scan_count + 1e-9)
    ranked = np.sort(np.asarray(fp_scores, dtype=float))[::-1]
    return float(ranked[allowed]) if allowed < len(ranked) else -math.inf


def sensitivity_by_size(
    candidates: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    *,
    scan_ids: Iterable[str] | None = None,
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None = None,
    fp_rates: Sequence[float] = (1.0,),
    bins: Sequence[tuple[float, float]] = SIZE_BINS,
) -> list[SizeBreakdown]:
    """Detected nodules per size bin at each of ``fp_rates`` false positives per scan.

    ``scan_ids`` should list every evaluated scan, as for ``compute_froc``.
    """
    matches = _match_all(candidates, nodules_by_scan, scan_ids, excluded_by_scan)
    if not any(match.best for match in matches.values()):
        raise ValueError("No annotated nodules in the evaluated scans")
    fp_scores = [score for match in matches.values() for score in match.false_positives]
    scored = [
        (nodule.diameter_mm, best)
        for scan, match in matches.items()
        for nodule, best in zip(nodules_by_scan.get(scan, []), match.best, strict=True)
    ]

    breakdowns = []
    for rate in fp_rates:
        threshold = operating_threshold(fp_scores, len(matches), rate)
        rows = []
        for lower, upper in bins:
            in_bin = [best for diameter, best in scored if lower <= diameter < upper]
            rows.append(
                SizeRow(
                    size=_label(lower, upper),
                    min_mm=lower,
                    max_mm=upper,
                    nodules=len(in_bin),
                    detected=sum(best > threshold for best in in_bin),
                    found_at_any_score=sum(math.isfinite(best) for best in in_bin),
                )
            )
        breakdowns.append(SizeBreakdown(fp_rate=float(rate), threshold=threshold, rows=tuple(rows)))
    return breakdowns
