import math

import numpy as np
import pytest

from lungnodule.annotations import Candidate, Nodule
from lungnodule.breakdown import SIZE_BINS, operating_threshold, sensitivity_by_size
from lungnodule.froc import compute_froc


def nodule(scan, diameter, x=0.0):
    return Nodule(scan, (x, 0.0, 0.0), diameter_mm=diameter)


def hit(scan, probability, x=0.0):
    return Candidate(scan, (x + 0.5, 0.0, 0.0), probability)


def miss(scan, probability):
    return Candidate(scan, (500.0, 500.0, 500.0), probability)


def test_operating_threshold_allows_the_rate_times_the_scans():
    # 2 scans at 1 FP per scan: the two best false positives are allowed.
    assert operating_threshold([0.9, 0.8, 0.7, 0.1], 2, 1.0) == 0.7
    assert operating_threshold([0.9, 0.8], 2, 1.0) == -math.inf
    assert operating_threshold([0.9, 0.8], 8, 0.125) == 0.8  # one FP allowed


@pytest.mark.parametrize(("scans", "rate"), [(0, 1.0), (3, -1.0)])
def test_operating_threshold_rejects_bad_arguments(scans, rate):
    with pytest.raises(ValueError):
        operating_threshold([0.5], scans, rate)


def test_counts_per_size_bin():
    nodules = {
        "a": [nodule("a", 4.0), nodule("a", 8.0, x=100.0)],
        "b": [nodule("b", 25.0)],
    }
    candidates = [
        hit("a", 0.9),  # small nodule, confident
        hit("a", 0.2, x=100.0),  # medium nodule, below the operating point
        hit("b", 0.6),  # large nodule
        miss("a", 0.5),
        miss("b", 0.4),
    ]
    (breakdown,) = sensitivity_by_size(candidates, nodules, fp_rates=[0.5])  # one FP allowed
    assert breakdown.threshold == 0.4
    rows = {row.size: row for row in breakdown.rows}
    assert rows["under 6 mm"].detected == 1
    assert rows["6–10 mm"].detected == 0
    assert rows["6–10 mm"].found_at_any_score == 1
    assert rows["20 mm or more"].detected == 1
    assert rows["10–20 mm"].nodules == 0 and rows["10–20 mm"].sensitivity is None
    assert breakdown.sensitivity == pytest.approx(2 / 3)


def test_bin_edges_include_the_lower_bound():
    nodules = {"a": [nodule("a", 6.0)]}
    (breakdown,) = sensitivity_by_size([hit("a", 0.9)], nodules)
    assert [row.nodules for row in breakdown.rows] == [0, 1, 0, 0]


def test_negative_diameters_are_left_out():
    nodules = {"a": [nodule("a", -1.0), nodule("a", 5.0, x=100.0)]}
    (breakdown,) = sensitivity_by_size([hit("a", 0.9, x=100.0)], nodules)
    assert sum(row.nodules for row in breakdown.rows) == 1


def random_case(seed):
    rng = np.random.default_rng(seed)
    nodules, candidates = {}, []
    for index in range(25):
        scan = f"scan-{index}"
        nodules[scan] = [
            nodule(scan, float(rng.uniform(3, 30)), x=100.0 * k) for k in range(rng.integers(0, 3))
        ]
        for k, _ in enumerate(nodules[scan]):
            if rng.random() < 0.85:
                # Rounded scores create ties between nodules and false positives.
                candidates.append(hit(scan, round(float(rng.random()), 1), x=100.0 * k))
        candidates += [miss(scan, round(float(rng.random()), 1)) for _ in range(rng.integers(0, 6))]
    return candidates, nodules


@pytest.mark.parametrize("seed", range(10))
def test_totals_match_the_non_interpolated_froc(seed):
    candidates, nodules = random_case(seed)
    if not any(nodules.values()):
        pytest.skip("no nodules drawn")
    rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    expected = compute_froc(candidates, nodules, fp_rates=rates, interpolate=False)
    breakdowns = sensitivity_by_size(candidates, nodules, fp_rates=rates)
    for breakdown, sensitivity in zip(breakdowns, expected.sensitivities, strict=True):
        assert breakdown.sensitivity == pytest.approx(sensitivity)


def test_as_dict_is_json_ready():
    (breakdown,) = sensitivity_by_size([hit("a", 0.9)], {"a": [nodule("a", 30.0)]})
    data = breakdown.as_dict()
    assert data["rows"][-1]["max_mm"] is None
    assert data["rows"][-1]["sensitivity"] == 1.0
    assert len(data["rows"]) == len(SIZE_BINS)


def test_no_nodules_is_an_error():
    with pytest.raises(ValueError, match="No annotated nodules"):
        sensitivity_by_size([miss("a", 0.5)], {})
