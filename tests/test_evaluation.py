import csv
import json

import pytest

from lungnodule.annotations import load_annotations
from lungnodule.dataset import build_dataset
from lungnodule.evaluation import evaluate_split, read_splits, write_candidates
from lungnodule.preprocess import InputConfig, nodule_boxes


def oracle_detector(annotations_csv, uid_holder):
    """Detects exactly the annotated nodules of the scan currently being evaluated."""
    nodules = load_annotations(annotations_csv)

    def detect(images):
        scan, start = uid_holder["scan"], uid_holder["next_z"]
        uid_holder["next_z"] += len(images)
        uid_holder.setdefault("shapes", set()).update(image.shape for image in images)
        height, width = images[0].shape[:2]
        results = []
        for offset in range(len(images)):
            boxes = nodule_boxes(scan, nodules.get(uid_holder["uid"], []), start + offset)
            results.append(
                [
                    (
                        b.x_center * width,
                        b.y_center * height,
                        b.width * width,
                        b.height * height,
                        0.9,
                    )
                    for b in boxes
                ]
            )
        return results

    return detect


@pytest.fixture
def built(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "dataset"
    # scan-c (with a nodule) goes to test; scan-d (no nodule) to validation.
    build_dataset(root, annotations, out, test_subset=8, val_subset=9, negatives=1)
    return root, annotations, out / "splits.csv"


def test_read_splits(built):
    _, _, splits = built
    assert read_splits(splits) == {
        "scan-a": "train",
        "scan-b": "train",
        "scan-c": "test",
        "scan-d": "val",
    }


def test_write_candidates_uses_luna16_format(tmp_path):
    from lungnodule.annotations import Candidate, load_candidates

    path = tmp_path / "candidates.csv"
    write_candidates([Candidate("uid", (1.0, 2.5, -3.25), 0.75)], path)
    assert load_candidates(path) == [Candidate("uid", (1.0, 2.5, -3.25), 0.75)]


def test_oracle_detector_scores_perfect_cpm_and_writes_reports(built, tmp_path, monkeypatch):
    root, annotations, splits = built
    holder = {}
    import lungnodule.evaluation as evaluation

    real_load_scan = evaluation.load_scan

    def tracking_load_scan(path):
        scan = real_load_scan(path)
        holder.update(scan=scan, uid=path.stem, next_z=0)
        return scan

    monkeypatch.setattr(evaluation, "load_scan", tracking_load_scan)
    out = tmp_path / "results"
    seen = []

    result = evaluate_split(
        root,
        annotations,
        splits,
        oracle_detector(annotations, holder),
        out,
        batch_size=4,
        progress=lambda uid, detections, candidates: seen.append((uid, candidates)),
    )

    assert result.cpm == 1.0
    assert seen == [("scan-c", 1)]
    report = json.loads((out / "froc.json").read_text(encoding="utf-8"))
    assert report["cpm"] == 1.0
    assert report["split"] == "test"
    assert report["input"]["mode"] == "gray"
    assert report["cpm_interval"]["lower"] <= report["cpm"] <= report["cpm_interval"]["upper"]
    assert report["cpm_interval"]["samples"] == 1000
    with (out / "candidates.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["seriesuid"] for row in rows] == ["scan-c"]


def test_detector_that_finds_nothing_scores_zero(built, tmp_path):
    root, annotations, splits = built
    result = evaluate_split(
        root, annotations, splits, lambda images: [[] for _ in images], tmp_path
    )
    assert result.cpm == 0.0


def test_split_without_nodules_is_reported(built, tmp_path):
    root, annotations, splits = built
    with pytest.raises(ValueError, match="No annotated nodules"):
        evaluate_split(
            root, annotations, splits, lambda images: [[] for _ in images], tmp_path, split="val"
        )


def test_unknown_split_is_reported(built, tmp_path):
    root, annotations, splits = built
    with pytest.raises(ValueError, match="No scans in the 'holdout' split"):
        evaluate_split(root, annotations, splits, lambda images: [], tmp_path, split="holdout")


def test_scans_missing_from_disk_are_reported(built, tmp_path):
    _, annotations, splits = built
    empty_root = tmp_path / "other"
    (empty_root / "subset0").mkdir(parents=True)
    (empty_root / "subset0" / "scan-z.mhd").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="test scans are not under"):
        evaluate_split(empty_root, annotations, splits, lambda images: [], tmp_path)


def test_bootstrap_can_be_skipped(built, tmp_path):
    root, annotations, splits = built
    out = tmp_path / "results"
    no_detections = lambda images: [[] for _ in images]  # noqa: E731
    evaluate_split(root, annotations, splits, no_detections, out, bootstrap_samples=0)
    report = json.loads((out / "froc.json").read_text(encoding="utf-8"))
    assert "cpm_interval" not in report


def test_stacked_input_reaches_the_detector_and_the_report(built, tmp_path, monkeypatch):
    root, annotations, splits = built
    holder = {}
    import lungnodule.evaluation as evaluation

    real_load_scan = evaluation.load_scan

    def tracking_load_scan(path):
        scan = real_load_scan(path)
        holder.update(scan=scan, uid=path.stem, next_z=0)
        return scan

    monkeypatch.setattr(evaluation, "load_scan", tracking_load_scan)
    out = tmp_path / "results"
    config = InputConfig(mode="stack", stack_mm=5.0)

    result = evaluate_split(
        root,
        annotations,
        splits,
        oracle_detector(annotations, holder),
        out,
        input_config=config,
        bootstrap_samples=0,
    )

    assert result.cpm == 1.0
    assert holder["shapes"] == {(64, 64, 3)}
    report = json.loads((out / "froc.json").read_text(encoding="utf-8"))
    assert report["input"] == {"mode": "stack", "stack_mm": 5.0, "window": [-600.0, 1500.0]}
