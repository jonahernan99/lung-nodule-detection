"""A miniature LUNA16 download: real .mhd files in subset folders, plus annotations."""

from __future__ import annotations

import numpy as np
import pytest

SPACING = (0.7, 0.7, 2.5)


def _write_scan(path, sitk):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.full((12, 64, 64), -800, dtype=np.int16))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing(SPACING)
    sitk.WriteImage(image, str(path))


@pytest.fixture
def mini_luna(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    root = tmp_path / "luna16"
    layout = {"scan-a": 0, "scan-b": 0, "scan-c": 8, "scan-d": 9}
    for uid, subset in layout.items():
        _write_scan(root / f"subset{subset}" / f"{uid}.mhd", sitk)

    # One 8 mm nodule at the center of scan-a and of scan-c (voxel 32, 32, 6).
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "seriesuid,coordX,coordY,coordZ,diameter_mm\n"
        "scan-a,22.4,22.4,15.0,8.0\n"
        "scan-c,22.4,22.4,15.0,8.0\n",
        encoding="utf-8",
    )
    return root, annotations
