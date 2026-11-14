"""Run a 2D detector over every slice of a CT scan.

The detector is any callable that takes a batch of 8-bit grayscale slices and
returns, for each slice, a list of ``(x_center, y_center, width, height,
confidence)`` boxes in pixels. Keeping that interface small means the
pipeline can be tested with a fake detector and used with any model;
``ultralytics_detector`` adapts a trained Ultralytics YOLO model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lungnodule.candidates import SliceDetection
from lungnodule.preprocess import LUNG_WINDOW, CTScan, apply_window

Box = tuple[float, float, float, float, float]
Detector = Callable[[Sequence[np.ndarray]], Sequence[Sequence[Box]]]


def detect_scan(
    scan: CTScan,
    detector: Detector,
    *,
    window: tuple[float, float] = LUNG_WINDOW,
    batch_size: int = 16,
) -> list[SliceDetection]:
    """Return the detector's boxes on every slice, windowed like the training data."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    detections = []
    depth = scan.hu.shape[0]
    for start in range(0, depth, batch_size):
        indices = range(start, min(start + batch_size, depth))
        images = [apply_window(scan.hu[z], *window) for z in indices]
        results = detector(images)
        if len(results) != len(images):
            raise ValueError(f"Detector returned {len(results)} results for {len(images)} slices")
        for z, boxes in zip(indices, results, strict=True):
            detections.extend(SliceDetection(z, *box) for box in boxes)
    return detections


def ultralytics_detector(
    weights: str, *, confidence: float = 0.01, image_size: int = 512, device: str | None = None
) -> Detector:
    """Wrap a trained Ultralytics YOLO model as a ``Detector``.

    A low ``confidence`` threshold keeps weak detections, which the FROC
    analysis needs to trace the curve at higher false-positive rates.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Running a trained model requires: pip install 'lung-nodule-detection[train]'"
        ) from exc

    model = YOLO(weights)

    def detect(images: Sequence[np.ndarray]) -> list[list[Box]]:  # pragma: no cover - needs model
        rgb = [np.repeat(image[:, :, None], 3, axis=2) for image in images]
        results = model.predict(
            rgb, conf=confidence, imgsz=image_size, device=device, verbose=False
        )
        boxes = []
        for result in results:
            xywh = result.boxes.xywh.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            boxes.append([(*map(float, box), float(c)) for box, c in zip(xywh, conf, strict=True)])
        return boxes

    return detect
