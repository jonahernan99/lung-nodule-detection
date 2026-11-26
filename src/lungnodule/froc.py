"""Scan-level FROC analysis and the competition performance metric (CPM).

The CPM is the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false positives
per scan. Matching follows the LUNA16 hit criterion:

- A candidate is a true positive when it lies strictly within a nodule's
  radius. A nodule hit by several candidates counts once, with the highest
  probability. The extra candidates are neither true nor false positives.
- A candidate that misses every nodule but lies within an excluded finding
  (a lesion LUNA16 does not count as a nodule) is ignored.
- Any other candidate is a false positive.

By default the sensitivity at each false-positive rate is interpolated
linearly along the curve from (0, 0), as the official LUNA16 evaluation script
does. On the official sample submission the seven sensitivities match the
official FROC curve to within 1e-9. ``interpolate=False`` instead reports the
highest sensitivity reached without exceeding each rate.

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
    """Sensitivities at fixed false-positive rates, candidate counts and the FROC curve."""

    fp_rates: tuple[float, ...]
    sensitivities: tuple[float, ...]
    cpm: float
    total_nodules: int
    total_scans: int
    found_nodules: int
    """Nodules hit by at least one candidate, at any probability."""
    false_positives: int
    duplicate_candidates: int
    """Extra candidates on an already-hit nodule, ignored as in the official script."""
    excluded_candidates: int
    """Candidates on excluded findings, ignored rather than counted as false positives."""
    curve_fps_per_scan: tuple[float, ...]
    curve_sensitivity: tuple[float, ...]


@dataclass
class _ScanMatch:
    best: list[float]
    false_positives: list[float]
    duplicates: int = 0
    excluded: int = 0


def _match_scan(
    candidates: Sequence[Candidate], nodules: Sequence[Nodule], excluded: Sequence[Nodule]
) -> _ScanMatch:
    """Best probability per nodule (``-inf`` if missed), FP probabilities and ignored counts."""
    match = _ScanMatch(best=[-np.inf] * len(nodules), false_positives=[])
    hits_per_nodule = [0] * len(nodules)
    for candidate in candidates:
        hits = [i for i, nodule in enumerate(nodules) if nodule.contains(candidate.center)]
        if hits:
            for i in hits:
                hits_per_nodule[i] += 1
                match.best[i] = max(match.best[i], candidate.probability)
        elif any(finding.contains(candidate.center) for finding in excluded):
            match.excluded += 1
        else:
            match.false_positives.append(candidate.probability)
    match.duplicates = sum(max(0, count - 1) for count in hits_per_nodule)
    return match


@dataclass(frozen=True)
class CpmDifference:
    """Paired bootstrap comparison of two candidate sets on the same scans."""

    cpm_baseline: float
    cpm_candidate: float
    difference: float
    """``cpm_candidate - cpm_baseline``."""
    lower: float
    upper: float
    confidence: float
    samples: int
    share_not_better: float
    """Share of bootstrap draws in which the candidate set's CPM is not above the baseline's."""


@dataclass(frozen=True)
class CpmInterval:
    """CPM with a percentile bootstrap confidence interval over scans."""

    cpm: float
    lower: float
    upper: float
    confidence: float
    samples: int


def _match_all(
    candidates: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    scan_ids: Iterable[str] | None,
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None,
) -> dict[str, _ScanMatch]:
    """Match the candidates of every evaluated scan, keyed by scan in sorted order.

    The order is fixed (sorted series UIDs, not set order, which changes between
    Python processes) so that bootstrap draws with a given seed are reproducible.
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
    return {
        scan: _match_scan(
            by_scan.get(scan, []), nodules_by_scan.get(scan, []), excluded_by_scan.get(scan, [])
        )
        for scan in sorted(scans)
    }


def _curve(
    nodule_scores: np.ndarray,
    fp_scores: np.ndarray,
    scan_count: int,
    fp_rates: Sequence[float],
    interpolate: bool,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """FROC curve (FPs per scan, sensitivity) and the sensitivities at ``fp_rates``."""
    nodules = np.sort(nodule_scores)
    fps = np.sort(fp_scores)
    finite = nodules[np.isfinite(nodules)]
    thresholds = np.unique(np.concatenate([finite, fps]))[::-1]

    # Number of scores >= each threshold, from the sorted arrays.
    found = len(nodules) - np.searchsorted(nodules, thresholds, side="left")
    false = len(fps) - np.searchsorted(fps, thresholds, side="left")
    curve_sensitivity = found / len(nodules)
    curve_fps = false / scan_count

    if interpolate:
        sensitivities = np.interp(
            fp_rates, np.concatenate([[0.0], curve_fps]), np.concatenate([[0.0], curve_sensitivity])
        ).tolist()
    else:
        sensitivities = []
        for rate in fp_rates:
            allowed = curve_fps <= rate
            sensitivities.append(float(curve_sensitivity[allowed].max()) if allowed.any() else 0.0)
    return curve_fps, curve_sensitivity, sensitivities


def compute_froc(
    candidates: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    *,
    scan_ids: Iterable[str] | None = None,
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None = None,
    fp_rates: Sequence[float] = FP_RATES,
    interpolate: bool = True,
) -> FrocResult:
    """Evaluate ``candidates`` against annotated nodules over a set of scans.

    ``scan_ids`` should list every evaluated scan, including scans without
    nodules or detections, since all of them count towards false positives
    per scan. When omitted, scans are taken from annotations and candidates.
    """
    matches = _match_all(candidates, nodules_by_scan, scan_ids, excluded_by_scan)
    nodule_scores = [score for match in matches.values() for score in match.best]
    fp_scores = [score for match in matches.values() for score in match.false_positives]
    if not nodule_scores:
        raise ValueError("No annotated nodules in the evaluated scans")

    nodules = np.asarray(nodule_scores, dtype=float)
    curve_fps, curve_sensitivity, sensitivities = _curve(
        nodules, np.asarray(fp_scores, dtype=float), len(matches), fp_rates, interpolate
    )

    return FrocResult(
        fp_rates=tuple(float(rate) for rate in fp_rates),
        sensitivities=tuple(float(s) for s in sensitivities),
        cpm=float(np.mean(sensitivities)),
        total_nodules=len(nodules),
        total_scans=len(matches),
        found_nodules=int(np.isfinite(nodules).sum()),
        false_positives=len(fp_scores),
        duplicate_candidates=sum(match.duplicates for match in matches.values()),
        excluded_candidates=sum(match.excluded for match in matches.values()),
        curve_fps_per_scan=tuple(curve_fps.tolist()),
        curve_sensitivity=tuple(curve_sensitivity.tolist()),
    )


def bootstrap_cpm(
    candidates: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    *,
    scan_ids: Iterable[str] | None = None,
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None = None,
    fp_rates: Sequence[float] = FP_RATES,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CpmInterval:
    """CPM with a percentile bootstrap interval, resampling scans with replacement.

    Scans are the independent units: a test set of a few dozen scans and about a
    hundred nodules gives a wide interval, which should be reported with the CPM.
    Draws that contain no annotated nodule are redrawn.
    """
    _check_bootstrap(samples, confidence)
    cpm_of = _cpm_function(candidates, nodules_by_scan, scan_ids, excluded_by_scan, fp_rates)
    draws = _draws(cpm_of.has_nodules, samples, seed)
    values = [cpm_of(draw) for draw in draws]
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return CpmInterval(
        cpm=cpm_of(range(len(cpm_of.has_nodules))),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def _check_bootstrap(samples: int, confidence: float) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


def _cpm_function(candidates, nodules_by_scan, scan_ids, excluded_by_scan, fp_rates):
    """``cpm_of(indices)``: the CPM of a multiset of scans, by position in sorted scan order."""
    matches = list(_match_all(candidates, nodules_by_scan, scan_ids, excluded_by_scan).values())
    if not any(match.best for match in matches):
        raise ValueError("No annotated nodules in the evaluated scans")
    best = [np.asarray(match.best, dtype=float) for match in matches]
    false = [np.asarray(match.false_positives, dtype=float) for match in matches]

    def cpm_of(indices: Sequence[int]) -> float:
        nodules = np.concatenate([best[i] for i in indices])
        fps = np.concatenate([false[i] for i in indices])
        _, _, sensitivities = _curve(nodules, fps, len(indices), fp_rates, interpolate=True)
        return float(np.mean(sensitivities))

    cpm_of.has_nodules = [len(b) > 0 for b in best]
    return cpm_of


def _draws(has_nodules: Sequence[bool], samples: int, seed: int) -> list[np.ndarray]:
    """Resamples of scan positions with replacement; draws without any nodule are redrawn."""
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < samples:
        draw = rng.integers(0, len(has_nodules), size=len(has_nodules))
        if any(has_nodules[i] for i in draw):
            draws.append(draw)
    return draws


def compare_cpm(
    baseline: Iterable[Candidate],
    candidate: Iterable[Candidate],
    nodules_by_scan: Mapping[str, Sequence[Nodule]],
    *,
    scan_ids: Iterable[str],
    excluded_by_scan: Mapping[str, Sequence[Nodule]] | None = None,
    fp_rates: Sequence[float] = FP_RATES,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CpmDifference:
    """Compare two models evaluated on the same scans with a paired bootstrap.

    Each draw resamples the scans once and scores both candidate sets on that
    same resample, so the variation that comes from which scans happen to be
    in the test set cancels out. Two separate intervals that overlap can still
    hide a clear paired difference, and the reverse. ``scan_ids`` is required:
    both sets must be scored on exactly the same scans.
    """
    _check_bootstrap(samples, confidence)
    scans = sorted(set(scan_ids))
    cpm_a = _cpm_function(baseline, nodules_by_scan, scans, excluded_by_scan, fp_rates)
    cpm_b = _cpm_function(candidate, nodules_by_scan, scans, excluded_by_scan, fp_rates)
    differences = np.array(
        [cpm_b(draw) - cpm_a(draw) for draw in _draws(cpm_a.has_nodules, samples, seed)]
    )
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(differences, [alpha, 1 - alpha])
    everything = range(len(scans))
    return CpmDifference(
        cpm_baseline=cpm_a(everything),
        cpm_candidate=cpm_b(everything),
        difference=cpm_b(everything) - cpm_a(everything),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
        share_not_better=float(np.mean(differences <= 0)),
    )
