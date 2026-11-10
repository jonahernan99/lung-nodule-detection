import pytest

from lungnodule.annotations import Candidate, Nodule, load_annotations, load_candidates


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_annotations_groups_by_scan(tmp_path):
    path = write(
        tmp_path,
        "annotations.csv",
        "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
        "scan-a,-128.7,-175.3,-298.4,5.65\n"
        "scan-a,103.8,-211.9,-227.1,4.22\n"
        "scan-b,69.6,-140.9,876.4,5.79\n",
    )
    grouped = load_annotations(path)
    assert sorted(grouped) == ["scan-a", "scan-b"]
    assert len(grouped["scan-a"]) == 2
    assert grouped["scan-b"][0] == Nodule("scan-b", (69.6, -140.9, 876.4), 5.79)


def test_load_candidates_keeps_file_order(tmp_path):
    path = write(
        tmp_path,
        "candidates.csv",
        "seriesuid,coordX,coordY,coordZ,probability\nscan-a,1,2,3,0.9\nscan-a,4,5,6,0.1\n",
    )
    assert load_candidates(path) == [
        Candidate("scan-a", (1.0, 2.0, 3.0), 0.9),
        Candidate("scan-a", (4.0, 5.0, 6.0), 0.1),
    ]


def test_missing_columns_are_reported(tmp_path):
    path = write(tmp_path, "bad.csv", "seriesuid,coordX,coordY\nscan-a,1,2\n")
    with pytest.raises(ValueError, match="missing columns: coordZ, diameter_mm"):
        load_annotations(path)


def test_invalid_number_reports_line(tmp_path):
    path = write(
        tmp_path,
        "bad.csv",
        "seriesuid,coordX,coordY,coordZ,diameter_mm\nscan-a,1,2,3,4\nscan-a,1,oops,3,4\n",
    )
    with pytest.raises(ValueError, match="line 3"):
        load_annotations(path)


def test_nodule_contains_points_within_its_radius():
    nodule = Nodule("scan-a", (0.0, 0.0, 0.0), diameter_mm=10.0)
    assert nodule.radius_mm == 5.0
    assert nodule.contains((3.0, 4.0, 0.0))  # distance 5: on the boundary
    assert not nodule.contains((3.0, 4.0, 0.1))
