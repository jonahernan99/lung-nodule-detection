import json

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


def test_build_dataset_with_stacked_input(mini_luna, tmp_path):
    root, annotations = mini_luna
    out = tmp_path / "ds"
    extra = ("--negatives", "1", "--input", "stack", "--stack-mm", "5")
    assert main(args(root, annotations, out, *extra)) == EXIT_OK
    saved = json.loads((out / "preprocessing.json").read_text(encoding="utf-8"))
    assert saved == {"mode": "stack", "stack_mm": 5.0, "window": [-600.0, 1500.0]}


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
        (["--input", "rgb"], "invalid choice"),
        (["--stack-mm", "0"], "must be positive"),
    ],
)
def test_invalid_arguments_are_rejected(tmp_path, capsys, bad, message):
    with pytest.raises(SystemExit):
        main(args(tmp_path, tmp_path / "a.csv", tmp_path / "ds", *bad))
    assert message in capsys.readouterr().err


def test_evaluate_prints_sensitivities_and_cpm(mini_luna, tmp_path, capsys, monkeypatch):
    from lungnodule import inference

    root, annotations = mini_luna
    dataset = tmp_path / "ds"
    assert main(args(root, annotations, dataset, "--test-subset", "8", "--val-subset", "9")) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        inference, "ultralytics_detector", lambda *a, **k: lambda images: [[] for _ in images]
    )
    # Datasets built before preprocessing.json existed are read as gray.
    (dataset / "preprocessing.json").unlink()
    code = main(
        [
            "evaluate",
            "--luna-root",
            str(root),
            "--annotations",
            str(annotations),
            "--splits",
            str(dataset / "splits.csv"),
            "--weights",
            "best.pt",
            "--out",
            str(tmp_path / "results"),
        ]
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Input: gray slices (from the dataset)" in out
    assert "Scans: 1, nodules: 1" in out
    assert "Found nodules: 0, false positives: 0" in out
    assert "CPM: 0.0000  (95% CI 0.000-0.000)" in out
    assert (tmp_path / "results" / "froc.json").exists()


def test_evaluate_uses_the_dataset_input(mini_luna, tmp_path, capsys, monkeypatch):
    from lungnodule import inference

    root, annotations = mini_luna
    dataset = tmp_path / "ds"
    build = args(root, annotations, dataset, "--test-subset", "8", "--val-subset", "9")
    assert main([*build, "--input", "stack"]) == EXIT_OK
    capsys.readouterr()
    shapes = set()

    def detector(images):
        shapes.update(image.shape for image in images)
        return [[] for _ in images]

    monkeypatch.setattr(inference, "ultralytics_detector", lambda *a, **k: detector)
    code = main(
        [
            "evaluate",
            "--luna-root",
            str(root),
            "--annotations",
            str(annotations),
            "--splits",
            str(dataset / "splits.csv"),
            "--weights",
            "best.pt",
            "--out",
            str(tmp_path / "results"),
        ]
    )
    assert code == EXIT_OK
    assert "Input: stacked slices 2 mm apart (from the dataset)" in capsys.readouterr().out
    assert shapes == {(64, 64, 3)}


def write_scored_files(tmp_path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
        "scan-a,0,0,0,4.0\n"
        "scan-a,100,0,0,12.0\n"
        "scan-b,0,0,0,25.0\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "seriesuid,coordX,coordY,coordZ,probability\n"
        "scan-a,0.5,0,0,0.9\n"
        "scan-a,100.5,0,0,0.2\n"
        "scan-b,0.5,0,0,0.7\n"
        "scan-b,300,300,300,0.5\n"
        "scan-a,-300,300,300,0.3\n",
        encoding="utf-8",
    )
    return annotations, candidates


def test_score_reports_cpm_with_an_interval(tmp_path, capsys):
    annotations, candidates = write_scored_files(tmp_path)
    out = tmp_path / "score.json"
    code = main(
        [
            "score",
            "--candidates",
            str(candidates),
            "--annotations",
            str(annotations),
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "Scans: 2, nodules: 3" in captured.out
    assert "95% CI" in captured.out
    assert "evaluating the 2 scans in the candidates file" in captured.err
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["cpm_interval"]["lower"] <= report["cpm"] <= report["cpm_interval"]["upper"]


def test_breakdown_lists_detections_by_size(tmp_path, capsys):
    annotations, candidates = write_scored_files(tmp_path)
    scans = tmp_path / "scans.csv"
    scans.write_text("seriesuid\nscan-a\nscan-b\nscan-c\n", encoding="utf-8")
    out = tmp_path / "by_size.json"
    args = ["breakdown", "--candidates", str(candidates), "--annotations", str(annotations)]
    args += ["--scans", str(scans), "--fp-rate", "0.34", "--out", str(out)]
    code = main(args)
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert captured.err == ""  # scans were given explicitly
    # 3 scans at 0.34 FP/scan allow one of the two false positives (0.5), so the
    # threshold is 0.3 and only the hit at 0.2 falls below it.
    assert "under 6 mm        1 of 1    detected   100%" in captured.out
    assert "10-20 mm          0 of 1    detected     0%" in captured.out
    (report,) = json.loads(out.read_text(encoding="utf-8"))
    assert report["fp_rate"] == 0.34
    assert report["sensitivity"] == pytest.approx(2 / 3)


def test_score_with_a_missing_file_is_an_error(tmp_path, capsys):
    annotations, _ = write_scored_files(tmp_path)
    missing = str(tmp_path / "nope.csv")
    code = main(["score", "--candidates", missing, "--annotations", str(annotations)])
    assert code == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_explain_reports_box_energy_with_the_dataset_input(
    mini_luna, tmp_path, capsys, monkeypatch
):
    import numpy as np

    from lungnodule import explain as explain_module

    root, annotations = mini_luna
    dataset = tmp_path / "ds"
    assert main([*args(root, annotations, dataset, "--negatives", "1"), "--input", "stack"]) == 0
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "seriesuid,coordX,coordY,coordZ,probability\n"
        "scan-c,22.4,22.4,15.0,0.9\n"
        "scan-c,4.2,4.2,15.0,0.8\n",
        encoding="utf-8",
    )
    shapes = []

    def fake_explainer(weights, **options):
        def explain(image):
            shapes.append(image.shape)
            cam = np.zeros(image.shape[:2])
            cam[28:37, 28:37] = 1.0  # around scan-c's nodule at pixel (32, 32)
            return [], cam

        return explain

    monkeypatch.setattr(explain_module, "ultralytics_explainer", fake_explainer)
    capsys.readouterr()
    code = main(
        [
            "explain",
            "--luna-root",
            str(root),
            "--annotations",
            str(annotations),
            "--candidates",
            str(candidates),
            "--dataset",
            str(dataset),
            "--weights",
            "best.pt",
            "--out",
            str(tmp_path / "cam"),
            "--top",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert set(shapes) == {(64, 64, 3)}
    assert "nodule" in out and "false positive" in out
    assert "AUC 1.000" in out
    assert (tmp_path / "cam" / "eigencam.csv").exists()
