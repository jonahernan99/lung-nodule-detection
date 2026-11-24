"""Eigen-CAM with a real (untrained) Ultralytics model. Skipped without ultralytics."""

import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")

from lungnodule.explain import ultralytics_explainer  # noqa: E402


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """An untrained YOLOv8n saved to disk, built from its config (no download)."""
    path = tmp_path_factory.mktemp("weights") / "untrained.pt"
    model = ultralytics.YOLO("yolov8n.yaml")
    model.save(str(path))
    return str(path)


@pytest.mark.parametrize("scale", ["p3", "p5"])
def test_explainer_returns_a_map_the_size_of_the_slice(weights, scale):
    explain = ultralytics_explainer(weights, scale=scale, image_size=64, device="cpu")
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:30, 30:40] = 200
    boxes, cam = explain(image)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0 and cam.max() <= 1
    assert all(len(box) == 5 for box in boxes)


def test_explainer_accepts_stacked_slices_and_runs_repeatedly(weights):
    explain = ultralytics_explainer(weights, image_size=64, device="cpu")
    rng = np.random.default_rng(0)
    first = explain(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))[1]
    second = explain(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))[1]
    assert first.shape == second.shape == (64, 64)
    assert not np.allclose(first, second)  # the hook captures each new image


def test_explainer_rejects_non_square_slices_and_unknown_scales(weights):
    with pytest.raises(ValueError, match="scale"):
        ultralytics_explainer(weights, scale="p6")
    explain = ultralytics_explainer(weights, image_size=64, device="cpu")
    with pytest.raises(ValueError, match="square"):
        explain(np.zeros((64, 32), dtype=np.uint8))
