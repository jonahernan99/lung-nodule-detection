import pytest

from lungnodule.annotations import Candidate, Nodule
from lungnodule.froc import FP_RATES, compute_froc

NODULE = Nodule("scan-a", (0.0, 0.0, 0.0), diameter_mm=10.0)
INSIDE = (1.0, 1.0, 1.0)
OUTSIDE = (50.0, 50.0, 50.0)


def candidate(scan, center, probability):
    return Candidate(scan, center, probability)


def test_single_hit_gives_perfect_cpm():
    result = compute_froc([candidate("scan-a", INSIDE, 0.9)], {"scan-a": [NODULE]})
    assert result.sensitivities == (1.0,) * len(FP_RATES)
    assert result.cpm == 1.0
    assert (result.total_nodules, result.total_scans) == (1, 1)


def test_missed_nodule_and_false_positive_give_zero_cpm():
    result = compute_froc([candidate("scan-a", OUTSIDE, 0.9)], {"scan-a": [NODULE]})
    assert result.cpm == 0.0


def test_duplicate_hits_on_one_nodule_are_not_false_positives():
    candidates = [candidate("scan-a", INSIDE, 0.9), candidate("scan-a", (2.0, 0.0, 0.0), 0.8)]
    result = compute_froc(candidates, {"scan-a": [NODULE]})
    assert result.cpm == 1.0
    assert result.curve_fps_per_scan[-1] == 0.0


def test_false_positives_are_counted_per_scan():
    # Each scan has one confident false positive; the nodule is found at a lower
    # probability, so reaching it costs 2 FPs over 2 scans = 1 FP per scan.
    candidates = [
        candidate("scan-a", OUTSIDE, 0.9),
        candidate("scan-b", OUTSIDE, 0.9),
        candidate("scan-a", INSIDE, 0.5),
    ]
    result = compute_froc(candidates, {"scan-a": [NODULE]}, scan_ids=["scan-a", "scan-b"])
    assert result.sensitivities == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert result.cpm == pytest.approx(4 / 7)


def test_scans_without_candidates_still_count_towards_the_rate():
    candidates = [candidate("scan-a", OUTSIDE, 0.9), candidate("scan-a", INSIDE, 0.5)]
    one_scan = compute_froc(candidates, {"scan-a": [NODULE]})
    eight_scans = compute_froc(
        candidates, {"scan-a": [NODULE]}, scan_ids=["scan-a"] + [f"empty-{i}" for i in range(7)]
    )
    assert one_scan.sensitivities[0] == 0.0
    assert eight_scans.sensitivities[0] == 1.0  # 1 FP over 8 scans = 1/8 per scan


def test_candidates_on_excluded_findings_are_ignored():
    excluded = {"scan-a": [Nodule("scan-a", OUTSIDE, diameter_mm=6.0)]}
    candidates = [candidate("scan-a", OUTSIDE, 0.9), candidate("scan-a", INSIDE, 0.5)]
    result = compute_froc(candidates, {"scan-a": [NODULE]}, excluded_by_scan=excluded)
    assert result.cpm == 1.0


def test_true_nodule_takes_priority_over_excluded_finding():
    excluded = {"scan-a": [Nodule("scan-a", INSIDE, diameter_mm=6.0)]}
    result = compute_froc(
        [candidate("scan-a", INSIDE, 0.9)], {"scan-a": [NODULE]}, excluded_by_scan=excluded
    )
    assert result.cpm == 1.0


def test_partial_sensitivity_with_two_nodules():
    second = Nodule("scan-a", (30.0, 0.0, 0.0), diameter_mm=4.0)
    result = compute_froc([candidate("scan-a", INSIDE, 0.7)], {"scan-a": [NODULE, second]})
    assert result.cpm == 0.5


def test_candidates_outside_evaluation_set_are_rejected():
    with pytest.raises(ValueError, match="outside the evaluation set"):
        compute_froc([candidate("scan-z", INSIDE, 0.9)], {"scan-a": [NODULE]}, scan_ids=["scan-a"])


def test_no_nodules_is_an_error():
    with pytest.raises(ValueError, match="No annotated nodules"):
        compute_froc([candidate("scan-a", INSIDE, 0.9)], {})


def test_custom_fp_rates():
    result = compute_froc([candidate("scan-a", INSIDE, 0.9)], {"scan-a": [NODULE]}, fp_rates=[1])
    assert result.fp_rates == (1.0,)
    assert result.cpm == 1.0
