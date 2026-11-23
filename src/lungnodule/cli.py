"""Command-line interface: ``build-dataset``, ``evaluate``, ``score`` and ``breakdown``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lungnodule import inference
from lungnodule.annotations import load_annotations, load_candidates
from lungnodule.breakdown import sensitivity_by_size
from lungnodule.dataset import build_dataset
from lungnodule.evaluation import evaluate_split
from lungnodule.folds import check_patient_leaks, find_scans, load_patient_ids, split_by_subset
from lungnodule.froc import FP_RATES, bootstrap_cpm, compute_froc
from lungnodule.preprocess import DEFAULT_STACK_MM, INPUT_MODES, InputConfig

EXIT_OK = 0
EXIT_ERROR = 2


def _subset(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 9:
        raise argparse.ArgumentTypeError("LUNA16 subsets are numbered 0 to 9")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lungnodule", description="Lung nodule detection tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-dataset", help="Export YOLO training data from LUNA16")
    build.add_argument("--luna-root", type=Path, required=True, help="Folder with subset0..9")
    build.add_argument("--annotations", type=Path, required=True, help="LUNA16 annotations.csv")
    build.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    build.add_argument("--test-subset", type=_subset, default=9)
    build.add_argument("--val-subset", type=_subset, default=8)
    build.add_argument(
        "--negatives", type=int, default=4, help="Background slices sampled per scan"
    )
    build.add_argument(
        "--max-scans", type=_positive_int, metavar="N", help="Export at most N scans per split"
    )
    build.add_argument(
        "--patients",
        type=Path,
        metavar="CSV",
        help="LIDC-IDRI metadata mapping series UIDs to patients; stops if a patient leaks",
    )
    build.add_argument(
        "--input",
        choices=INPUT_MODES,
        default="gray",
        help="gray: one slice; stack: neighbouring slices as color channels (2.5D)",
    )
    build.add_argument(
        "--stack-mm",
        type=_positive_float,
        default=DEFAULT_STACK_MM,
        metavar="MM",
        help=f"Distance to each neighbouring slice in stack mode (default: {DEFAULT_STACK_MM})",
    )

    evaluate = commands.add_parser("evaluate", help="Scan-level FROC/CPM on whole test scans")
    evaluate.add_argument("--luna-root", type=Path, required=True, help="Folder with subset0..9")
    evaluate.add_argument("--annotations", type=Path, required=True, help="LUNA16 annotations.csv")
    evaluate.add_argument(
        "--splits", type=Path, required=True, help="splits.csv from build-dataset"
    )
    evaluate.add_argument("--weights", type=Path, required=True, help="Trained YOLO weights (.pt)")
    evaluate.add_argument("--out", type=Path, required=True, help="Folder for the reports")
    evaluate.add_argument("--excluded", type=Path, help="LUNA16 annotations_excluded.csv")
    evaluate.add_argument("--split", default="test", help="Split to evaluate (default: test)")
    evaluate.add_argument("--max-scans", type=_positive_int, metavar="N")
    evaluate.add_argument(
        "--confidence", type=float, default=0.01, help="Lowest detection confidence kept"
    )
    evaluate.add_argument("--image-size", type=_positive_int, default=512)

    for name, help_text in (
        ("score", "FROC/CPM with a confidence interval for a candidates CSV"),
        ("breakdown", "Sensitivity by nodule size for a candidates CSV"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--candidates", type=Path, required=True, help="CSV in the LUNA16 submission format"
        )
        command.add_argument(
            "--annotations", type=Path, required=True, help="LUNA16 annotations.csv"
        )
        command.add_argument("--excluded", type=Path, help="LUNA16 annotations_excluded.csv")
        command.add_argument(
            "--scans",
            type=Path,
            metavar="FILE",
            help="Evaluated series UIDs, one per line (first CSV column); "
            "default: the scans in the candidates file",
        )
        command.add_argument("--out", type=Path, help="Write the results as JSON")
    commands.choices["breakdown"].add_argument(
        "--fp-rate",
        type=float,
        nargs="+",
        default=[0.125, 1.0, 8.0],
        help="False positives per scan at which to report (default: 0.125 1 8)",
    )
    return parser


def _check_patients(args: argparse.Namespace) -> bool:
    scans = find_scans(args.luna_root)
    split_of = split_by_subset(
        {uid: subset for uid, (subset, _) in scans.items()},
        test_subset=args.test_subset,
        val_subset=args.val_subset,
    )
    report = check_patient_leaks(split_of, load_patient_ids(args.patients))
    if report.unmatched_scans:
        print(f"warning: {len(report.unmatched_scans)} scans have no patient ID", file=sys.stderr)
    for patient, splits in report.leaks.items():
        print(f"error: patient {patient} has scans in {', '.join(splits)}", file=sys.stderr)
    return report.ok


def _run_build(args: argparse.Namespace) -> int:
    if args.patients is not None and not _check_patients(args):
        return EXIT_ERROR

    def progress(uid: str, split: str, count: int) -> None:
        print(f"{split:<5} {count:>4} slices  {uid}")

    summary = build_dataset(
        args.luna_root,
        args.annotations,
        args.out,
        test_subset=args.test_subset,
        val_subset=args.val_subset,
        negatives=args.negatives,
        max_scans_per_split=args.max_scans,
        input_config=InputConfig(mode=args.input, stack_mm=args.stack_mm),
        progress=progress,
    )
    print(
        f"Scans: {summary.scans['train']} train, {summary.scans['val']} val, "
        f"{summary.scans['test']} test (test scans are evaluated on full volumes)"
    )
    print(f"Slices: {summary.slices['train']} train, {summary.slices['val']} val")
    print(f"Dataset config: {args.out / 'data.yaml'}")
    return EXIT_OK


def _run_evaluate(args: argparse.Namespace) -> int:
    detector = inference.ultralytics_detector(
        str(args.weights), confidence=args.confidence, image_size=args.image_size
    )

    def progress(uid: str, detections: int, candidates: int) -> None:
        print(f"{detections:>5} detections -> {candidates:>4} candidates  {uid}")

    result = evaluate_split(
        args.luna_root,
        args.annotations,
        args.splits,
        detector,
        args.out,
        split=args.split,
        excluded_csv=args.excluded,
        max_scans=args.max_scans,
        progress=progress,
    )
    print(f"\nScans: {result.total_scans}, nodules: {result.total_nodules}")
    print(
        f"Found nodules: {result.found_nodules}, false positives: {result.false_positives}, "
        f"duplicates ignored: {result.duplicate_candidates}, "
        f"on excluded findings: {result.excluded_candidates}"
    )
    for rate, sensitivity in zip(result.fp_rates, result.sensitivities, strict=True):
        print(f"  {rate:>5g} FP/scan  sensitivity {sensitivity:.3f}")
    report = json.loads((args.out / "froc.json").read_text(encoding="utf-8"))
    interval = report.get("cpm_interval")
    ci = f"  (95% CI {interval['lower']:.3f}-{interval['upper']:.3f})" if interval else ""
    print(f"CPM: {result.cpm:.4f}{ci}  (reports in {args.out})")
    return EXIT_OK


def _read_scan_ids(path: Path) -> list[str]:
    """Series UIDs from the first column of a text or CSV file, skipping a header row."""
    with path.open(encoding="utf-8", newline="") as handle:
        ids = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
    return [uid for uid in ids if uid.lower() != "seriesuid"]


def _load_scored(args: argparse.Namespace):
    candidates = load_candidates(args.candidates)
    annotations = load_annotations(args.annotations)
    excluded = load_annotations(args.excluded) if args.excluded is not None else {}
    if args.scans is not None:
        scans = _read_scan_ids(args.scans)
    else:
        scans = sorted({candidate.series_uid for candidate in candidates})
        print(
            f"note: evaluating the {len(scans)} scans in the candidates file; pass --scans to "
            "include scans without candidates",
            file=sys.stderr,
        )
    return candidates, {uid: annotations.get(uid, []) for uid in scans}, scans, excluded


def _write_json(path: Path | None, data: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _run_score(args: argparse.Namespace) -> int:
    candidates, nodules, scans, excluded = _load_scored(args)
    result = compute_froc(candidates, nodules, scan_ids=scans, excluded_by_scan=excluded)
    interval = bootstrap_cpm(candidates, nodules, scan_ids=scans, excluded_by_scan=excluded)
    print(f"Scans: {result.total_scans}, nodules: {result.total_nodules}")
    for rate, sensitivity in zip(FP_RATES, result.sensitivities, strict=True):
        print(f"  {rate:>5g} FP/scan  sensitivity {sensitivity:.3f}")
    print(f"CPM: {result.cpm:.4f}  (95% CI {interval.lower:.3f}-{interval.upper:.3f})")
    _write_json(
        args.out,
        {
            "cpm": result.cpm,
            "cpm_interval": {"lower": interval.lower, "upper": interval.upper},
            "fp_rates": list(result.fp_rates),
            "sensitivities": list(result.sensitivities),
            "total_scans": result.total_scans,
            "total_nodules": result.total_nodules,
        },
    )
    return EXIT_OK


def _run_breakdown(args: argparse.Namespace) -> int:
    candidates, nodules, scans, excluded = _load_scored(args)
    breakdowns = sensitivity_by_size(
        candidates, nodules, scan_ids=scans, excluded_by_scan=excluded, fp_rates=args.fp_rate
    )
    for breakdown in breakdowns:
        print(f"\nAt {breakdown.fp_rate:g} FP/scan (sensitivity {breakdown.sensitivity:.3f}):")
        for row in breakdown.rows:
            share = "-" if row.sensitivity is None else f"{row.sensitivity:.0%}"
            print(f"  {row.size:<14} {row.detected:>4} of {row.nodules:<4} detected  {share:>5}")
    _write_json(args.out, [breakdown.as_dict() for breakdown in breakdowns])
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "build-dataset": _run_build,
        "evaluate": _run_evaluate,
        "score": _run_score,
        "breakdown": _run_breakdown,
    }
    try:
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
