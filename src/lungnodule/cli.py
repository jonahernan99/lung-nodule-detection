"""Command-line interface: ``lungnodule build-dataset`` and ``lungnodule evaluate``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lungnodule import inference
from lungnodule.dataset import build_dataset
from lungnodule.evaluation import evaluate_split
from lungnodule.folds import check_patient_leaks, find_scans, load_patient_ids, split_by_subset

EXIT_OK = 0
EXIT_ERROR = 2


def _subset(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 9:
        raise argparse.ArgumentTypeError("LUNA16 subsets are numbered 0 to 9")
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
    print(f"CPM: {result.cpm:.4f}  (reports in {args.out})")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"build-dataset": _run_build, "evaluate": _run_evaluate}
    try:
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
