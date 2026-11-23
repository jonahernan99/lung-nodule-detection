"""Run a 2D detector over every slice of a CT scan.

The detector is any callable that takes a batch of 8-bit slice images, built
by ``slice_image`` (``(H, W)`` gray or ``(H, W, 3)`` stacked, channels in RGB
order), and returns, for each slice, a list of ``(x_center, y_center, width,
height, confidence)`` boxes in pixels. Keeping that interface small means the
pipeline can be tested with a fake detector and used with any model;
``ultralytics_detector`` adapts a trained Ultralytics YOLO model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lungnodule.candidates import SliceDetection
from lungnodule.preprocess import CTScan, InputConfig, slice_image

Box = tuple[float, float, float, float, float]
Detector = Callable[[Sequence[np.ndarray]], Sequence[Sequence[Box]]]


def detect_scan(
    scan: CTScan,
    detector: Detector,
    *,
    config: InputConfig | None = None,
    batch_size: int = 16,
) -> list[SliceDetection]:
    """Return the detector's boxes on every slice, prepared like the training data.

    ``config`` must match the one the training data was built with (gray by
    default).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    detections = []
    depth = scan.hu.shape[0]
    for start in range(0, depth, batch_size):
        indices = range(start, min(start + batch_size, depth))
        images = [slice_image(scan, z, config) for z in indices]
        results = detector(images)
        if len(results) != len(images):
            raise ValueError(f"Detector returned {len(results)} results for {len(images)} slices")
        for z, boxes in zip(indices, results, strict=True):
            detections.extend(SliceDetection(z, *box) for box in boxes)
    return detections


def to_model_input(image: np.ndarray) -> np.ndarray:
    """Convert a slice image to the ``(H, W, 3)`` BGR array Ultralytics expects.

    Ultralytics reads training PNGs with OpenCV, which returns BGR, and flips
    them to RGB; ``predict`` likewise treats numpy arrays as BGR. A stacked
    RGB image (``z - offset``, ``z``, ``z + offset``) is therefore reversed
    here, so the model sees the channels in the order it was trained on.
    """
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image[:, :, ::-1])
    raise ValueError(f"Expected an (H, W) or (H, W, 3) image, got shape {image.shape}")


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
        results = model.predict(
            [to_model_input(image) for image in images],
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        boxes = []
        for result in results:
            xywh = result.boxes.xywh.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            boxes.append([(*map(float, box), float(c)) for box, c in zip(xywh, conf, strict=True)])
        return boxes

    return detect
