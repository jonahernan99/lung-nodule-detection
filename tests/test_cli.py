import pytest

from lungnodule.cli import EXIT_ERROR, EXIT_OK, main


def args(root, annotations, out, *extra):
    return [
        "build-dataset",
        "--luna-root",
        str(root),
        "--annotations",
        str(annotations),
        "--out",
        str(out),
        *extra,
    ]


def test_build_dataset_prints_summary(mini_luna, tmp_path, capsys):
    root, annotations = mini_luna
    assert main(args(root, annotations, tmp_path / "ds", "--negatives", "1")) == EXIT_OK
    out = capsys.readouterr().out
    assert "Scans: 2 train, 1 val, 1 test" in out
    assert (tmp_path / "ds" / "data.yaml").exists()


def test_patient_leak_stops_the_build(mini_luna, tmp_path, capsys):
    root, annotations = mini_luna
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "Series UID,Subject ID\nscan-a,P1\nscan-b,P2\nscan-c,P3\nscan-d,P1\n", encoding="utf-8"
    )
    code = main(args(root, annotations, tmp_path / "ds", "--patients", str(metadata)))
    assert code == EXIT_ERROR
    assert "patient P1 has scans in test, train" in capsys.readouterr().err
    assert not (tmp_path / "ds").exists()


def test_clean_patient_split_builds_and_warns_about_unmatched(mini_luna, tmp_path, capsys):
    root, annotations = mini_luna
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("Series UID,Subject ID\nscan-a,P1\nscan-b,P1\n", encoding="utf-8")
    assert main(args(root, annotations, tmp_path / "ds", "--patients", str(metadata))) == EXIT_OK
    assert "2 scans have no patient ID" in capsys.readouterr().err


def test_missing_luna_folder_is_reported(tmp_path, capsys):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("seriesuid,coordX,coordY,coordZ,diameter_mm\n", encoding="utf-8")
    assert main(args(tmp_path / "missing", annotations, tmp_path / "ds")) == EXIT_ERROR
    assert "No .mhd scans" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (["--test-subset", "10"], "numbered 0 to 9"),
        (["--max-scans", "0"], "must be at least 1"),
    ],
)
def test_invalid_arguments_are_rejected(tmp_path, capsys, bad, message):
    with pytest.raises(SystemExit):
        main(args(tmp_path, tmp_path / "a.csv", tmp_path / "ds", *bad))
    assert message in capsys.readouterr().err
