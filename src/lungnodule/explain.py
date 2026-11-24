"""Eigen-CAM saliency maps for the nodule detector.

Eigen-CAM (Muhammad and Yeasin, 2020) projects a layer's activations onto
their first principal component: the direction along which the feature maps
vary most. It needs no gradients and no class score, so it works with any
detector, and the thesis used it to explain detections and to filter
candidates.

Two details are fixed here that implementations often leave open:

- The sign of a principal component is arbitrary. The map is oriented so that
  it correlates positively with the mean activation, then negative values are
  clipped and the map is scaled to [0, 1].
- ``box_energy`` turns a map into a number per detection, the share of the
  map's total weight inside the box, so explanations can be compared between
  true and false positives rather than only looked at.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lungnodule.inference import Box, to_model_input

#: Overlay color (categorical orange), blended into the slice where the map is high.
OVERLAY_COLOR = (235, 104, 52)


def eigen_cam(activations: np.ndarray) -> np.ndarray:
    """Eigen-CAM map of ``(C, H, W)`` activations, as an ``(H, W)`` array in [0, 1].

    Activations are treated as ``H * W`` samples of ``C`` features, centered
    per channel, and projected onto their first right singular vector.
    """
    activations = np.asarray(activations, dtype=np.float64)
    if activations.ndim != 3:
        raise ValueError(f"Expected (C, H, W) activations, got shape {activations.shape}")
    channels, height, width = activations.shape
    samples = activations.reshape(channels, -1).T
    centered = samples - samples.mean(axis=0)
    if not np.any(centered):
        return np.zeros((height, width))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[0]
    mean_activation = samples.mean(axis=1)
    if np.dot(projection - projection.mean(), mean_activation - mean_activation.mean()) < 0:
        projection = -projection
    cam = np.maximum(projection, 0).reshape(height, width)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def resize_map(cam: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a map to ``(height, width)``, with pixel centers aligned."""
    cam = np.asarray(cam, dtype=np.float64)
    height, width = shape
    rows = (np.arange(height) + 0.5) * cam.shape[0] / height - 0.5
    cols = (np.arange(width) + 0.5) * cam.shape[1] / width - 0.5
    by_row = np.stack(
        [np.interp(cols, np.arange(cam.shape[1]), cam[i]) for i in range(cam.shape[0])]
    )
    return np.stack(
        [np.interp(rows, np.arange(cam.shape[0]), by_row[:, j]) for j in range(width)], axis=1
    )


def box_energy(cam: np.ndarray, box: Sequence[float]) -> float:
    """Share of the map's total weight inside a pixel box ``(x_center, y_center, width, height)``.

    The map must already have the image's size (see ``resize_map``). A box that
    holds a quarter of the image and a uniform map gives 0.25; a map focused on
    the box gives values near 1.
    """
    cam = np.asarray(cam, dtype=np.float64)
    total = cam.sum()
    if total <= 0:
        return 0.0
    x, y, w, h = box[:4]
    height, width = cam.shape
    x0, x1 = max(int(np.floor(x - w / 2)), 0), min(int(np.ceil(x + w / 2)), width)
    y0, y1 = max(int(np.floor(y - h / 2)), 0), min(int(np.ceil(y + h / 2)), height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(cam[y0:y1, x0:x1].sum() / total)


def overlay(image: np.ndarray, cam: np.ndarray, *, alpha: float = 0.6) -> np.ndarray:
    """Blend a map onto a slice image as an RGB ``uint8`` array.

    ``image`` is ``(H, W)`` gray or ``(H, W, 3)``; for a stacked image the
    center channel (the slice itself) is shown. Where the map is 0 the slice is
    unchanged; where it is 1 the color takes ``alpha`` of the pixel.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    gray = image if image.ndim == 2 else image[:, :, 1]
    if cam.shape != gray.shape:
        cam = resize_map(cam, gray.shape)
    weight = alpha * np.clip(cam, 0, 1)[:, :, None]
    base = np.repeat(gray[:, :, None].astype(np.float64), 3, axis=2)
    blended = base * (1 - weight) + np.asarray(OVERLAY_COLOR, dtype=np.float64) * weight
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


#: Detection scales of an Ultralytics head, finest first. P3 (stride 8) gives a
#: 64 x 64 map for a 512 px slice, fine enough for nodules a few pixels wide.
SCALES: tuple[str, ...] = ("p3", "p4", "p5")


def ultralytics_explainer(
    weights: str,
    *,
    scale: str = "p3",
    confidence: float = 0.01,
    image_size: int = 512,
    device: str | None = None,
):
    """Return ``explain(image) -> (boxes, cam)`` for trained Ultralytics YOLO weights.

    ``image`` is a square slice image from ``slice_image``. ``boxes`` are
    ``(x_center, y_center, width, height, confidence)`` in pixels, as from the
    detector, and ``cam`` is the Eigen-CAM map of the layer that feeds the
    chosen detection scale, resized to the image.
    """
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; choose from {SCALES}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Explaining a trained model requires: pip install 'lung-nodule-detection[train]'"
        ) from exc

    model = YOLO(weights)
    network = model.model.model
    layer = network[network[-1].f[SCALES.index(scale)]]
    captured: dict[str, object] = {}
    layer.register_forward_hook(lambda module, inputs, output: captured.update(output=output))

    def explain(image: np.ndarray) -> tuple[list[Box], np.ndarray]:
        height, width = image.shape[:2]
        if height != width:
            raise ValueError(f"Expected a square slice, got {height} x {width}")
        (result,) = model.predict(
            [to_model_input(image)], conf=confidence, imgsz=image_size, device=device, verbose=False
        )
        activations = captured["output"][0].detach().float().cpu().numpy()
        cam = resize_map(eigen_cam(activations), (height, width))
        xywh = result.boxes.xywh.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        boxes = [(*map(float, box), float(c)) for box, c in zip(xywh, conf, strict=True)]
        return boxes, cam

    return explain
