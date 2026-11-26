import numpy as np
import pytest
from test_bootstrap import mixed_case

from lungnodule.annotations import Candidate
from lungnodule.froc import bootstrap_cpm, compare_cpm, compute_froc


def improved(candidates, nodules, seed=1):
    """The same detections plus a confident hit on every nodule the first set missed."""
    hit = {c.series_uid for c in candidates if c.center == (1.0, 0.0, 0.0)}
    extra = [Candidate(scan, (1.0, 0.0, 0.0), 0.99) for scan in nodules if scan not in hit]
    return candidates + extra


def test_identical_sets_have_no_difference():
    candidates, nodules = mixed_case()
    result = compare_cpm(candidates, candidates, nodules, scan_ids=nodules, samples=200)
    assert result.difference == 0.0
    assert (result.lower, result.upper) == (0.0, 0.0)
    assert result.share_not_better == 1.0


def test_better_set_has_a_positive_difference():
    candidates, nodules = mixed_case()
    better = improved(candidates, nodules)
    result = compare_cpm(candidates, better, nodules, scan_ids=nodules, samples=300)
    assert result.cpm_baseline == pytest.approx(compute_froc(candidates, nodules).cpm)
    assert result.cpm_candidate == pytest.approx(compute_froc(better, nodules).cpm)
    assert result.difference == pytest.approx(result.cpm_candidate - result.cpm_baseline)
    assert 0 < result.lower <= result.difference <= result.upper
    assert result.share_not_better == 0.0
    reverse = compare_cpm(better, candidates, nodules, scan_ids=nodules, samples=300)
    assert reverse.difference == pytest.approx(-result.difference)
    assert reverse.share_not_better == 1.0


def test_paired_interval_is_narrower_than_the_separate_intervals():
    candidates, nodules = mixed_case(scan_count=40, seed=3)
    rng = np.random.default_rng(0)
    # A second detector that agrees closely with the first: small, consistent gain.
    shifted = [Candidate(c.series_uid, c.center, min(1.0, c.probability + 0.05 * rng.random()))
               for c in candidates]  # fmt: skip
    paired = compare_cpm(candidates, shifted, nodules, scan_ids=nodules, samples=400)
    separate = bootstrap_cpm(candidates, nodules, samples=400)
    assert paired.upper - paired.lower < separate.upper - separate.lower


def test_comparison_is_reproducible_and_validated():
    candidates, nodules = mixed_case()
    better = improved(candidates, nodules)
    first = compare_cpm(candidates, better, nodules, scan_ids=nodules, samples=100, seed=5)
    again = compare_cpm(candidates, better, nodules, scan_ids=list(reversed(list(nodules))),
                        samples=100, seed=5)  # fmt: skip
    assert first == again
    with pytest.raises(ValueError, match="samples"):
        compare_cpm(candidates, better, nodules, scan_ids=nodules, samples=0)
    with pytest.raises(ValueError, match="outside the evaluation set"):
        compare_cpm(candidates, better, nodules, scan_ids=["scan-0"], samples=10)
    with pytest.raises(ValueError, match="No annotated nodules"):
        compare_cpm([], [], {}, scan_ids=["empty"], samples=10)
