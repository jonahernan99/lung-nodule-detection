"""Compare compute_froc with a Python 3 port of the official LUNA16 evaluation.

The reference below follows ``evaluateCAD`` and ``computeFROC`` from the
challenge's ``noduleCADEvaluationLUNA16.py`` (Python 2): included nodules are
matched first, then excluded findings, a missed nodule enters the ROC analysis
with the lowest probability and is excluded from it, and the FROC curve comes
from scikit-learn's ``roc_curve``.
"""

import numpy as np
import pytest

from lungnodule.annotations import Candidate, Nodule
from lungnodule.froc import FP_RATES, compute_froc

roc_curve = pytest.importorskip("sklearn.metrics").roc_curve

MIN_PROBABILITY = -1_000_000_000.0


def official_sensitivities(candidates, nodules_by_scan, excluded_by_scan, scans):
    gt, prob, exclude = [], [], []
    for scan in scans:
        cands = [c for c in candidates if c.series_uid == scan]
        remaining = set(range(len(cands)))
        annotations = [(n, True) for n in nodules_by_scan.get(scan, [])]
        annotations += [(e, False) for e in excluded_by_scan.get(scan, [])]
        for annotation, included in annotations:
            diameter = annotation.diameter_mm if annotation.diameter_mm >= 0 else 10.0
            radius_squared = (diameter / 2.0) ** 2
            matches = []
            for key, c in enumerate(cands):
                distance = sum(
                    (a - b) ** 2 for a, b in zip(annotation.center, c.center, strict=True)
                )
                if distance < radius_squared:
                    if included:
                        matches.append(c)
                        remaining.discard(key)
                    else:
                        remaining.discard(key)
            if included:
                gt.append(1.0)
                if matches:
                    prob.append(max(c.probability for c in matches))
                    exclude.append(False)
                else:
                    prob.append(MIN_PROBABILITY)
                    exclude.append(True)
        for key in remaining:
            gt.append(0.0)
            prob.append(cands[key].probability)
            exclude.append(False)

    local_gt = [g for g, e in zip(gt, exclude, strict=True) if not e]
    local_prob = [p for p, e in zip(prob, exclude, strict=True) if not e]
    detected = sum(local_gt)
    fpr, tpr, _ = roc_curve(local_gt, local_prob)
    if sum(gt) == len(gt):
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (len(local_prob) - detected) / len(scans)
    sens = tpr * detected / sum(gt)
    return np.interp(FP_RATES, fps, sens)


def random_case(seed):
    rng = np.random.default_rng(seed)
    scans = [f"scan-{i}" for i in range(6)]
    nodules, excluded, candidates = {}, {}, []
    for scan in scans:
        centers = rng.uniform(0, 100, size=(int(rng.integers(0, 4)), 3))
        nodules[scan] = [Nodule(scan, tuple(c), float(rng.uniform(4, 15))) for c in centers]
        excluded[scan] = [
            Nodule(scan, tuple(rng.uniform(0, 100, 3)), float(rng.choice([-1.0, 6.0, 12.0])))
            for _ in range(int(rng.integers(0, 4)))
        ]
        points = list(rng.uniform(0, 100, size=(30, 3)))
        for nodule in nodules[scan] + excluded[scan]:  # some candidates near each finding
            points += list(np.asarray(nodule.center) + rng.normal(0, 3, size=(2, 3)))
        for point in points:
            # Rounding creates ties, the case where interpolation matters most.
            probability = round(float(rng.uniform(0, 1)), 2)
            candidates.append(Candidate(scan, tuple(point), probability))
    if not any(nodules.values()):
        nodules[scans[0]] = [Nodule(scans[0], (50.0, 50.0, 50.0), 10.0)]
    return candidates, nodules, excluded, scans


@pytest.mark.parametrize("seed", range(20))
def test_matches_the_official_evaluation(seed):
    candidates, nodules, excluded, scans = random_case(seed)
    expected = official_sensitivities(candidates, nodules, excluded, scans)
    result = compute_froc(candidates, nodules, scan_ids=scans, excluded_by_scan=excluded)
    np.testing.assert_allclose(result.sensitivities, expected, atol=1e-9)
