"""Command-line interface: ``lungnodule build-dataset``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lungnodule.dataset import build_dataset
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_build(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
