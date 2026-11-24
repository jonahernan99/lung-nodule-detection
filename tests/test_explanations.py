import csv
import json

import numpy as np
import pytest

from lungnodule.annotations import Candidate, Nodule
from lungnodule.explain import (
    CandidateExplanation,
    energy_summary,
    explain_candidate,
    label_candidate,
)
from lungnodule.explanations import explain_candidates
from lungnodule.preprocess import CTScan, InputConfig

# In the mini LUNA16 fixture, scan-c has an 8 mm nodule at voxel (32, 32, 6),
# world (22.4, 22.4, 15.0); scan-d has none.
CANDIDATES = (
    "seriesuid,coordX,coordY,coordZ,probability\n"
    "scan-c,22.4,22.4,15.0,0.9\n"  # on the nodule
    "scan-c,4.2,4.2,15.0,0.8\n"  # near the corner of scan-c
    "scan-d,22.4,22.4,15.0,0.7\n"  # a scan without nodules
    "scan-d,30.0,30.0,15.0,0.1\n"  # below the top-3 cut
)


def focused_explainer(center=(32, 32)):
    """A fake detector whose map and only box sit on ``center``; records image shapes."""
    shapes = []

    def explain(image):
        shapes.append(image.shape)
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        cam = np.exp(-((yy - center[1]) ** 2 + (xx - center[0]) ** 2) / 8.0)
        return [(center[0] + 0.5, center[1] + 0.5, 10.0, 10.0, 0.9)], cam / cam.max()

    explain.shapes = shapes
    return explain


def test_labels_follow_the_luna16_hit_criterion():
    nodule = Nodule("s", (0.0, 0.0, 0.0), diameter_mm=10.0)
    excluded = Nodule("s", (50.0, 0.0, 0.0), diameter_mm=10.0)
    assert label_candidate(Candidate("s", (1.0, 1.0, 1.0), 0.9), [nodule], [excluded]) == "nodule"
    assert label_candidate(Candidate("s", (51.0, 0, 0), 0.9), [nodule], [excluded]) == "excluded"
    assert label_candidate(Candidate("s", (5.0, 0, 0), 0.9), [nodule], [excluded]) == (
        "false positive"
    )  # exactly on the radius is a miss, as in the official script


def test_candidate_uses_the_matching_detection_box_or_a_fixed_one():
    scan = CTScan(np.full((5, 64, 64), -800, dtype=np.int16), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    explain = focused_explainer(center=(32, 32))
    on, _, cam = explain_candidate(scan, Candidate("s", (32.0, 32.0, 2.0), 0.9), explain, label="x")
    off, _, _ = explain_candidate(scan, Candidate("s", (5.0, 5.0, 2.0), 0.5), explain, label="x")
    assert cam.shape == (64, 64)
    assert (on.z, on.box_source, on.box) == (2, "detection", (32.5, 32.5, 10.0, 10.0))
    assert off.box_source == "fixed" and off.box == (5.5, 5.5, 16.0, 16.0)
    assert on.energy > 0.5 > off.energy


def test_energy_summary_reports_separation():
    def item(label, energy):
        return CandidateExplanation("s", 0.5, label, 0, 0.0, 0.0, (0, 0, 1, 1), "fixed", energy)

    summary = energy_summary(
        [item("nodule", 0.6), item("nodule", 0.4), item("false positive", 0.4),
         item("false positive", 0.1), item("excluded", 0.3)]
    )  # fmt: skip
    assert summary["candidates"] == 5
    assert summary["by_label"]["nodule"] == {"count": 2, "median_energy": 0.5}
    assert summary["energy_auc"] == pytest.approx((1 + 1 + 0.5 + 1) / 4)
    assert energy_summary([item("nodule", 0.5)])["energy_auc"] is None


def test_explain_candidates_writes_reports(mini_luna, tmp_path):
    root, annotations = mini_luna
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(CANDIDATES, encoding="utf-8")
    explain = focused_explainer(center=(32, 32))
    seen = []

    summary = explain_candidates(
        root, annotations, candidates, explain, tmp_path / "out", top=3,
        input_config=InputConfig(mode="stack"), progress=lambda done, total: seen.append(done),
    )  # fmt: skip

    assert seen == [1, 2, 3]
    assert set(explain.shapes) == {(64, 64, 3)}
    with (tmp_path / "out" / "eigencam.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["seriesuid"], r["label"]) for r in rows] == [
        ("scan-c", "nodule"),
        ("scan-c", "false positive"),
        ("scan-d", "false positive"),
    ]
    assert summary["by_label"]["nodule"]["count"] == 1
    # The fake map is centred where scan-c's nodule is, and also where scan-d's
    # false positive is, so the separation is partial.
    assert summary["energy_auc"] == pytest.approx(0.75)
    assert json.loads((tmp_path / "out" / "eigencam.json").read_text())["input"]["mode"] == "stack"
    from PIL import Image

    assert Image.open(tmp_path / "out" / "eigencam.png").size[0] > 0


def test_explain_candidates_checks_its_inputs(mini_luna, tmp_path):
    root, annotations = mini_luna
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "seriesuid,coordX,coordY,coordZ,probability\nscan-zz,0,0,0,0.5\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not under"):
        explain_candidates(root, annotations, candidates, focused_explainer(), tmp_path)
    with pytest.raises(ValueError, match="top"):
        explain_candidates(root, annotations, candidates, focused_explainer(), tmp_path, top=0)
