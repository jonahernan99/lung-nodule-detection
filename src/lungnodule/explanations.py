"""Explain the highest-scoring candidates of an evaluation with Eigen-CAM.

For the ``top`` candidates by probability, the detector's Eigen-CAM map is
computed on each candidate's center slice and the share of the map inside the
candidate's box is recorded. Three files are written:

- ``eigencam.csv``: one row per candidate, with its label and box energy;
- ``eigencam.json``: counts and median energy per label, and the AUC of energy
  as a separator of nodules from false positives;
- ``eigencam.png``: the slices of the top nodules and false positives with
  their maps, for a visual check.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from lungnodule.annotations import load_annotations, load_candidates
from lungnodule.explain import (
    CandidateExplanation,
    energy_summary,
    explain_candidate,
    label_candidate,
    overlay,
)
from lungnodule.folds import find_scans
from lungnodule.preprocess import InputConfig, load_scan

CROP_PX = 64
GRID_PER_LABEL = 4


def _crop(image: np.ndarray, x: float, y: float, size: int = CROP_PX) -> np.ndarray:
    """A ``size`` square around pixel ``(x, y)``, edge-padded at the image border."""
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="edge")
    cx, cy = int(x) + half, int(y) + half
    return padded[cy - half : cy + half, cx - half : cx + half]


def explanation_grid(items: Sequence[tuple[CandidateExplanation, np.ndarray]], scale: int = 3):
    """Tiles of overlays (a PIL image), with the candidate box and a caption under each."""
    from PIL import Image, ImageDraw

    tile, caption = CROP_PX * scale, 30
    columns = max(1, min(GRID_PER_LABEL, len(items)))
    rows = max(1, -(-len(items) // columns))
    grid = Image.new("RGB", (columns * tile, rows * (tile + caption)), (252, 252, 251))
    draw = ImageDraw.Draw(grid)
    for index, (item, blended) in enumerate(items):
        left, top = (index % columns) * tile, (index // columns) * (tile + caption)
        crop = Image.fromarray(_crop(blended, item.x, item.y)).resize((tile, tile), Image.NEAREST)
        grid.paste(crop, (left, top))
        x, y, w, h = item.box
        # The crop starts CROP_PX / 2 pixels left of and above the candidate's pixel.
        ox, oy = int(item.x) - CROP_PX // 2, int(item.y) - CROP_PX // 2
        x0, y0 = (x - w / 2 - ox) * scale + left, (y - h / 2 - oy) * scale + top
        draw.rectangle([x0, y0, x0 + w * scale, y0 + h * scale], outline=(42, 120, 214), width=2)
        text = f"{item.label}  p={item.probability:.2f}  E={item.energy:.2f}"
        draw.text((left + 4, top + tile + 8), text, fill=(11, 11, 11))
    return grid


def explain_candidates(
    luna_root: str | Path,
    annotations_csv: str | Path,
    candidates_csv: str | Path,
    explain: Callable,
    out_dir: str | Path,
    *,
    excluded_csv: str | Path | None = None,
    input_config: InputConfig | None = None,
    top: int = 200,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Explain the ``top`` candidates by probability and write the three report files.

    ``explain`` takes a slice image and returns ``(boxes, cam)``, as
    ``lungnodule.explain.ultralytics_explainer`` does. ``input_config`` must
    match the training data. Returns the summary written to ``eigencam.json``.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    candidates = sorted(
        load_candidates(candidates_csv), key=lambda c: (-c.probability, c.series_uid, c.center)
    )[:top]
    if not candidates:
        raise ValueError(f"No candidates in {candidates_csv}")
    scans = find_scans(luna_root)
    missing = sorted({c.series_uid for c in candidates} - set(scans))
    if missing:
        raise ValueError(f"{len(missing)} candidate scans are not under {luna_root}: {missing[:3]}")
    nodules = load_annotations(annotations_csv)
    excluded = load_annotations(excluded_csv) if excluded_csv is not None else {}

    results: list[CandidateExplanation] = []
    examples: dict[str, list[tuple[CandidateExplanation, np.ndarray]]] = {}
    by_scan: dict[str, list] = {}
    for candidate in candidates:
        by_scan.setdefault(candidate.series_uid, []).append(candidate)
    done = 0
    for uid in sorted(by_scan):
        scan = load_scan(scans[uid][1])
        for candidate in by_scan[uid]:
            label = label_candidate(candidate, nodules.get(uid, []), excluded.get(uid, []))
            result, image, cam = explain_candidate(
                scan, candidate, explain, label=label, config=input_config
            )
            results.append(result)
            shown = examples.setdefault(label, [])
            if label != "excluded" and len(shown) < GRID_PER_LABEL:
                shown.append((result, overlay(image, cam)))
            done += 1
            if progress is not None:
                progress(done, len(candidates))

    results.sort(key=lambda r: (-r.probability, r.series_uid, r.z))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eigencam.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seriesuid", "probability", "label", "z", "x", "y", "box_source", "energy"]
        )
        for r in results:
            writer.writerow(
                [r.series_uid, f"{r.probability:.6f}", r.label, r.z, f"{r.x:.1f}", f"{r.y:.1f}",
                 r.box_source, f"{r.energy:.4f}"]
            )  # fmt: skip
    summary = {"top": top, **energy_summary(results)}
    if input_config is not None:
        summary["input"] = input_config.to_dict()
    (out_dir / "eigencam.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shown = examples.get("nodule", []) + examples.get("false positive", [])
    if shown:
        explanation_grid(shown).save(out_dir / "eigencam.png")
    return summary
