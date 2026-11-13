import csv

import pytest

from lungnodule.dataset import build_dataset


def test_builds_yolo_layout_and_skips_test_scans(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "dataset"

    summary = build_dataset(root, annotations, out, test_subset=9, val_subset=8, negatives=2)

    assert summary.scans == {"train": 2, "val": 1, "test": 1}
    assert (out / "train" / "images").is_dir()
    assert (out / "val" / "labels").is_dir()
    assert not (out / "test").exists()  # test scans are evaluated on full volumes later
    assert summary.slices["train"] == len(list((out / "train" / "images").glob("*.png")))
    assert summary.slices["val"] == len(list((out / "val" / "images").glob("*.png")))


def test_labels_exist_only_for_scans_with_nodules(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "dataset"
    build_dataset(root, annotations, out, negatives=2)

    labeled = {
        p.name.split("_")[0] for p in (out / "train" / "labels").glob("*.txt") if p.read_text()
    }
    assert labeled == {"scan-a"}
    assert len(list((out / "train" / "images").glob("scan-b_*.png"))) == 2  # negatives only


def test_splits_csv_lists_every_scan(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "dataset"
    build_dataset(root, annotations, out)

    with (out / "splits.csv").open(encoding="utf-8") as handle:
        rows = {row["seriesuid"]: (row["subset"], row["split"]) for row in csv.DictReader(handle)}
    assert rows == {
        "scan-a": ("0", "train"),
        "scan-b": ("0", "train"),
        "scan-c": ("8", "val"),
        "scan-d": ("9", "test"),
    }


def test_data_yaml_points_to_exported_splits(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "dataset"
    build_dataset(root, annotations, out)

    lines = (out / "data.yaml").read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"path: {out.resolve().as_posix()}"
    assert lines[1:] == ["train: train/images", "val: val/images", "names:", "  0: nodule"]


def test_max_scans_per_split_limits_the_export(mini_luna, tmp_path):
    root, annotations = mini_luna
    seen = []
    summary = build_dataset(
        root,
        annotations,
        tmp_path / "dataset",
        max_scans_per_split=1,
        progress=lambda uid, split, count: seen.append((uid, split)),
    )
    assert seen == [("scan-a", "train"), ("scan-c", "val")]
    assert summary.scans["train"] == 2  # counts still describe the full split


def test_invalid_split_choice_is_reported(mini_luna, tmp_path):
    root, annotations = mini_luna
    with pytest.raises(ValueError, match="No scans in subset5"):
        build_dataset(root, annotations, tmp_path / "dataset", test_subset=5)
