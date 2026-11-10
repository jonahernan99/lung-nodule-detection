"""Scan-level FROC analysis and the competition performance metric (CPM).

The CPM is the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives
per scan. Matching follows the LUNA16 hit criterion:

- A candidate is a true positive when it lies within a nodule's radius. A
  nodule hit by several candidates counts once, with the highest probability.
  The extra candidates are neither true nor false positives.
- A candidate that misses every nodule but lies within an excluded finding
  (a lesion LUNA16 does not count as a nodule) is ignored.
- Any other candidate is a false positive.

The sensitivity reported for a false-positive rate is the highest sensitivity
reached without exceeding that rate. The official LUNA16 script interpolates
along the curve instead, so its values can differ slightly.

Rates are per scan, not per 2D slice. A per-slice rate allows hundreds of false
positives in a single CT scan, so the two are not comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lungnodule.annotations import Candidate, Nodule

FP_RATES: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class FrocResult:
    """Sensitivities at fixed false-positive rates and the full FROC curve."""

    fp_rates: tuple[float, ...]
    sensitivities: tuple[float, ...]
    cpm: float
    total_nodules: int
    total_scans: int
    curve_fps_per_scan: tuple[float, ...]
    curve_sensitivity: tuple[float, ...]


def _match_scan(
    candidates: Sequence[Candidate], nodules: Sequence[Nodule], excluded: Sequence[Nodule]
) -> tuple[list[float], list[float]]:
    """Return the best probability per nodule (``-inf`` if missed) and FP probabilities."""
    best = [-np.inf] * len(nodules)
    false_positives: list[float] = []
    for candidate in candidates:
        hits = [i for i, nodule in enumerate(nodules) if nodule.contains(candidate.center)]
        if hits:
            for i in hits:
                best[i] = max(best[i], candidate.probability)
        elif not any(finding.contains(candidate.center) for finding in excluded):
            false_positives.append(candidate.probability)
    return best, false_positives


def compute_froc(
    candidates: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    *,
    scan_ids: Iterable[str] | None = None,
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None = None,
    fp_rates: Sequence[float] = FP_RATES,
) -> FrocResult:
    """Evaluate ``candidates`` against annotated nodules over a set of scans.

    ``scan_ids`` should list every evaluated scan, including scans without
    nodules or detections, since all of them count towards false positives
    per scan. When omitted, scans are taken from annotations and candidates.
    """
    candidates = list(candidates)
    excluded_by_scan = excluded_by_scan or {}
    scans = (
        set(scan_ids)
        if scan_ids is not None
        else set(nodules_by_scan) | {c.series_uid for c in candidates}
    )
    unknown = {c.series_uid for c in candidates} - scans
    if unknown:
        raise ValueError(
            f"Candidates reference scans outside the evaluation set: {sorted(unknown)}"
        )

    by_scan: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_scan.setdefault(candidate.series_uid, []).append(candidate)

    nodule_scores: list[float] = []
    fp_scores: list[float] = []
    for scan in scans:
        best, false_positives = _match_scan(
            by_scan.get(scan, []), nodules_by_scan.get(scan, []), excluded_by_scan.get(scan, [])
        )
        nodule_scores += best
        fp_scores += false_positives

    if not nodule_scores:
        raise ValueError("No annotated nodules in the evaluated scans")

    nodules = np.sort(np.asarray(nodule_scores))
    fps = np.sort(np.asarray(fp_scores))
    finite = nodules[np.isfinite(nodules)]
    thresholds = np.unique(np.concatenate([finite, fps]))[::-1]

    # Number of scores >= each threshold, from the sorted arrays.
    found = len(nodules) - np.searchsorted(nodules, thresholds, side="left")
    false = len(fps) - np.searchsorted(fps, thresholds, side="left")
    curve_sensitivity = found / len(nodules)
    curve_fps = false / len(scans)

    sensitivities = []
    for rate in fp_rates:
        allowed = curve_fps <= rate
        sensitivities.append(float(curve_sensitivity[allowed].max()) if allowed.any() else 0.0)

    return FrocResult(
        fp_rates=tuple(float(rate) for rate in fp_rates),
        sensitivities=tuple(sensitivities),
        cpm=float(np.mean(sensitivities)),
        total_nodules=len(nodules),
        total_scans=len(scans),
        curve_fps_per_scan=tuple(curve_fps.tolist()),
        curve_sensitivity=tuple(curve_sensitivity.tolist()),
    )
