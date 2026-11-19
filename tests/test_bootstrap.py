import numpy as np
import pytest

from lungnodule.annotations import Candidate, Nodule
from lungnodule.froc import bootstrap_cpm, compute_froc


def nodule(scan):
    return Nodule(scan, (0.0, 0.0, 0.0), diameter_mm=10.0)


def mixed_case(scan_count=30, seed=0):
    """Scans with one nodule each, found with random confidence among random false positives."""
    rng = np.random.default_rng(seed)
    nodules, candidates = {}, []
    for index in range(scan_count):
        scan = f"scan-{index}"
        nodules[scan] = [nodule(scan)]
        if rng.random() < 0.8:
            candidates.append(Candidate(scan, (1.0, 0.0, 0.0), float(rng.random())))
        for _ in range(rng.integers(0, 4)):
            candidates.append(Candidate(scan, (60.0, 60.0, 60.0), float(rng.random())))
    return candidates, nodules


def test_point_estimate_matches_compute_froc():
    candidates, nodules = mixed_case()
    interval = bootstrap_cpm(candidates, nodules, samples=50)
    assert interval.cpm == pytest.approx(compute_froc(candidates, nodules).cpm)


def test_interval_contains_the_estimate():
    candidates, nodules = mixed_case()
    interval = bootstrap_cpm(candidates, nodules, samples=300)
    assert interval.lower <= interval.cpm <= interval.upper
    assert interval.upper - interval.lower > 0
    assert (interval.samples, interval.confidence) == (300, 0.95)


def test_perfect_detector_has_a_degenerate_interval():
    nodules = {f"scan-{i}": [nodule(f"scan-{i}")] for i in range(10)}
    candidates = [Candidate(scan, (1.0, 0.0, 0.0), 0.9) for scan in nodules]
    interval = bootstrap_cpm(candidates, nodules, samples=100)
    assert (interval.lower, interval.cpm, interval.upper) == (1.0, 1.0, 1.0)


def test_same_seed_gives_the_same_interval():
    candidates, nodules = mixed_case()
    first = bootstrap_cpm(candidates, nodules, samples=100, seed=3)
    second = bootstrap_cpm(candidates, nodules, samples=100, seed=3)
    assert first == second


def test_more_scans_narrow_the_interval():
    small = bootstrap_cpm(*mixed_case(scan_count=20), samples=400)
    large = bootstrap_cpm(*mixed_case(scan_count=400), samples=400)
    assert large.upper - large.lower < small.upper - small.lower


def test_scans_without_nodules_are_resampled_too():
    candidates, nodules = mixed_case()
    scans = list(nodules) + ["empty-1", "empty-2"]
    candidates.append(Candidate("empty-1", (5.0, 5.0, 5.0), 0.99))
    interval = bootstrap_cpm(candidates, nodules, scan_ids=scans, samples=100)
    assert interval.cpm == pytest.approx(compute_froc(candidates, nodules, scan_ids=scans).cpm)


@pytest.mark.parametrize(("samples", "confidence"), [(0, 0.95), (10, 0.0), (10, 1.0)])
def test_invalid_arguments(samples, confidence):
    candidates, nodules = mixed_case()
    with pytest.raises(ValueError):
        bootstrap_cpm(candidates, nodules, samples=samples, confidence=confidence)


def test_no_nodules_is_an_error():
    with pytest.raises(ValueError, match="No annotated nodules"):
        bootstrap_cpm([Candidate("scan-a", (0.0, 0.0, 0.0), 0.5)], {}, samples=10)
