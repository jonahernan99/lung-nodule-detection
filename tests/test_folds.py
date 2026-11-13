import pytest

from lungnodule.folds import (
    check_patient_leaks,
    find_scans,
    load_patient_ids,
    split_by_subset,
)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("header", encoding="utf-8")
    return path


def test_find_scans_reads_subset_from_folder_names(tmp_path):
    touch(tmp_path / "subset0" / "1.2.3.mhd")
    touch(tmp_path / "subset9" / "nested" / "4.5.6.mhd")
    touch(tmp_path / "subset0" / "1.2.3.raw")  # image data files are ignored

    scans = find_scans(tmp_path)

    assert {uid: subset for uid, (subset, _) in scans.items()} == {"1.2.3": 0, "4.5.6": 9}
    assert scans["4.5.6"][1].name == "4.5.6.mhd"


def test_root_folder_name_is_not_mistaken_for_a_subset(tmp_path):
    root = tmp_path / "subset5"
    touch(root / "subset1" / "1.2.3.mhd")
    assert find_scans(root)["1.2.3"][0] == 1


def test_scans_outside_subsets_are_rejected(tmp_path):
    touch(tmp_path / "loose" / "1.2.3.mhd")
    with pytest.raises(ValueError, match="outside any subsetN"):
        find_scans(tmp_path)


def test_scan_in_two_subsets_is_rejected(tmp_path):
    touch(tmp_path / "subset0" / "1.2.3.mhd")
    touch(tmp_path / "subset1" / "1.2.3.mhd")
    with pytest.raises(ValueError, match="appears in subset0 and subset1"):
        find_scans(tmp_path)


def test_empty_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="No .mhd scans"):
        find_scans(tmp_path)


def test_split_by_subset():
    subsets = {"a": 0, "b": 1, "c": 8, "d": 9, "e": 9}
    assert split_by_subset(subsets, test_subset=9, val_subset=8) == {
        "a": "train",
        "b": "train",
        "c": "val",
        "d": "test",
        "e": "test",
    }


def test_split_rejects_same_test_and_val_subset():
    with pytest.raises(ValueError, match="must differ"):
        split_by_subset({"a": 0}, test_subset=0, val_subset=0)


def test_split_rejects_empty_subsets():
    with pytest.raises(ValueError, match="No scans in subset7"):
        split_by_subset({"a": 0, "b": 9}, test_subset=9, val_subset=7)


def write_metadata(path, header, rows):
    path.write_text("\n".join([",".join(header), *(",".join(r) for r in rows)]) + "\n", "utf-8")
    return path


def test_load_patient_ids_detects_common_columns(tmp_path):
    path = write_metadata(
        tmp_path / "metadata.csv",
        ["Series UID", "Subject ID", "Modality"],
        [["1.2.3", "LIDC-IDRI-0001", "CT"], ["4.5.6", "LIDC-IDRI-0002", "CT"]],
    )
    assert load_patient_ids(path) == {"1.2.3": "LIDC-IDRI-0001", "4.5.6": "LIDC-IDRI-0002"}


def test_load_patient_ids_with_explicit_columns(tmp_path):
    path = write_metadata(tmp_path / "m.csv", ["uid", "person"], [["1.2.3", "P1"]])
    assert load_patient_ids(path, series_column="uid", patient_column="person") == {"1.2.3": "P1"}


def test_load_patient_ids_reports_unknown_columns(tmp_path):
    path = write_metadata(tmp_path / "m.csv", ["uid", "person"], [["1.2.3", "P1"]])
    with pytest.raises(ValueError, match="none of the columns"):
        load_patient_ids(path)
    with pytest.raises(ValueError, match="no column 'nope'"):
        load_patient_ids(path, series_column="nope", patient_column="person")


def test_patient_with_scans_in_two_splits_is_a_leak():
    split_of = {"a": "train", "b": "test", "c": "train", "d": "val"}
    patient_of = {"a": "P1", "b": "P1", "c": "P2"}

    report = check_patient_leaks(split_of, patient_of)

    assert report.leaks == {"P1": ("test", "train")}
    assert report.unmatched_scans == ("d",)
    assert not report.ok


def test_patient_with_scans_in_one_split_is_fine():
    report = check_patient_leaks({"a": "train", "b": "train"}, {"a": "P1", "b": "P1"})
    assert report.ok
    assert report.unmatched_scans == ()
