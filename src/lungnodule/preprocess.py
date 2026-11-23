"""Turn CT scans into 2D slices with YOLO labels for nodule detection.

Two input representations are supported (see ``InputConfig``):

- ``gray``: the windowed slice alone, repeated in every color channel by the
  detector, as in the thesis;
- ``stack`` (2.5D): the slices about ``stack_mm`` below and above the labeled
  slice fill the first and last color channels, so a 2D detector sees some 3D
  context. A vessel runs through neighbouring slices while a small nodule
  appears and disappears, which a single slice cannot show.

Each nodule is treated as a sphere. A slice receives a box where the slice
cuts that sphere, with the box sized to the circular cross-section rather than
the nodule's full diameter. Slices that only graze the edge of a nodule are
left unlabeled by default, since the nodule is barely visible there, and are
never sampled as negatives either.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lungnodule.annotations import Nodule
from lungnodule.geometry import world_to_voxel

#: Common lung window in Hounsfield units: level -600, width 1500 (-1350 to 150 HU).
LUNG_WINDOW: tuple[float, float] = (-600.0, 1500.0)

#: Label a slice only where the nodule's cross-section radius is at least this
#: fraction of its full radius.
DEFAULT_MIN_FRACTION = 0.5

INPUT_MODES: tuple[str, ...] = ("gray", "stack")

#: Distance between the labeled slice and each neighbour in ``stack`` mode. A
#: 6 mm nodule (radius 3 mm) still crosses both neighbours, so the channels
#: show it shrinking rather than vanishing.
DEFAULT_STACK_MM = 2.0


@dataclass(frozen=True)
class CTScan:
    """A CT volume in Hounsfield units, indexed ``[z, y, x]``.

    ``origin``, ``spacing`` and ``direction`` follow SimpleITK's (x, y, z) order.
    """

    hu: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...] | None = None


@dataclass(frozen=True)
class InputConfig:
    """How a CT slice becomes a detector input image.

    The same configuration must be used to build the training data and to run
    the trained detector; ``build_dataset`` stores it next to the dataset.
    """

    mode: str = "gray"
    stack_mm: float = DEFAULT_STACK_MM
    window: tuple[float, float] = LUNG_WINDOW

    def __post_init__(self) -> None:
        if self.mode not in INPUT_MODES:
            raise ValueError(f"Unknown input mode {self.mode!r}; choose from {INPUT_MODES}")
        if self.stack_mm <= 0:
            raise ValueError("stack_mm must be positive")
        object.__setattr__(self, "window", tuple(float(v) for v in self.window))

    def to_dict(self) -> dict:
        return {"mode": self.mode, "stack_mm": self.stack_mm, "window": list(self.window)}

    @classmethod
    def from_dict(cls, data: dict) -> InputConfig:
        return cls(
            mode=data.get("mode", "gray"),
            stack_mm=float(data.get("stack_mm", DEFAULT_STACK_MM)),
            window=tuple(data.get("window", LUNG_WINDOW)),
        )


def neighbour_offset(spacing_z: float, stack_mm: float = DEFAULT_STACK_MM) -> int:
    """Number of slices between the center and each neighbour in ``stack`` mode.

    Slice spacing in LUNA16 ranges from about 0.6 to 2.5 mm, so a fixed
    distance in millimetres keeps the context comparable across scans. The
    offset is at least one slice.
    """
    if spacing_z <= 0:
        raise ValueError("Slice spacing must be positive")
    return max(1, round(stack_mm / spacing_z))


def slice_image(scan: CTScan, z: int, config: InputConfig | None = None) -> np.ndarray:
    """The detector input for slice ``z``: ``(H, W)`` for ``gray``, ``(H, W, 3)`` for ``stack``.

    In ``stack`` mode the channels are, in RGB order, the slices at
    ``z - offset``, ``z`` and ``z + offset``, with indices clamped to the
    volume so edge slices repeat themselves. ``config`` defaults to ``gray``.
    """
    config = config or InputConfig()
    depth = scan.hu.shape[0]
    if not 0 <= z < depth:
        raise IndexError(f"Slice {z} is outside a volume of {depth} slices")
    if config.mode == "gray":
        return apply_window(scan.hu[z], *config.window)
    offset = neighbour_offset(scan.spacing[2], config.stack_mm)
    indices = (max(z - offset, 0), z, min(z + offset, depth - 1))
    return np.stack([apply_window(scan.hu[i], *config.window) for i in indices], axis=-1)


@dataclass(frozen=True)
class YoloBox:
    """A bounding box in YOLO format: center and size normalized to [0, 1]."""

    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int = 0

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def apply_window(
    hu: np.ndarray, level: float = LUNG_WINDOW[0], width: float = LUNG_WINDOW[1]
) -> np.ndarray:
    """Map Hounsfield units to 8-bit gray levels, clipping outside the window."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    low = level - width / 2
    scaled = (np.asarray(hu, dtype=float) - low) / width
    return np.clip(np.round(scaled * 255), 0, 255).astype(np.uint8)


def nodule_boxes(
    scan: CTScan,
    nodules: Sequence[Nodule],
    z: int,
    *,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> list[YoloBox]:
    """Return YOLO boxes for the nodules visible on slice ``z``.

    Voxel index ``i`` is the center of pixel ``i``, which spans ``[i, i + 1)``
    in image coordinates, hence the half-pixel offset below.
    """
    depth, height, width = scan.hu.shape
    spacing_x, spacing_y, spacing_z = scan.spacing
    boxes = []
    for nodule in nodules:
        cx, cy, cz = world_to_voxel(nodule.center, scan.origin, scan.spacing, scan.direction)
        radius = nodule.radius_mm
        dz_mm = abs(z - cz) * spacing_z
        if dz_mm > radius:
            continue
        slice_radius = math.sqrt(radius**2 - dz_mm**2)
        if slice_radius < min_fraction * radius:
            continue

        half_w, half_h = slice_radius / spacing_x, slice_radius / spacing_y
        x0, x1 = max(cx + 0.5 - half_w, 0.0), min(cx + 0.5 + half_w, float(width))
        y0, y1 = max(cy + 0.5 - half_h, 0.0), min(cy + 0.5 + half_h, float(height))
        if x1 <= x0 or y1 <= y0:  # entirely outside the image
            continue
        boxes.append(
            YoloBox(
                x_center=(x0 + x1) / 2 / width,
                y_center=(y0 + y1) / 2 / height,
                width=(x1 - x0) / width,
                height=(y1 - y0) / height,
            )
        )
    return boxes


def select_slices(
    scan: CTScan,
    nodules: Sequence[Nodule],
    *,
    negatives: int = 4,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Pick slices to export: every labeled slice plus random nodule-free slices.

    Negative slices are drawn only from slices that do not touch any nodule,
    so a barely visible nodule is never presented to the model as background.
    """
    depth = scan.hu.shape[0]
    positives = [
        z for z in range(depth) if nodule_boxes(scan, nodules, z, min_fraction=min_fraction)
    ]
    touched = {z for z in range(depth) if nodule_boxes(scan, nodules, z, min_fraction=0.0)}
    candidates = [z for z in range(depth) if z not in touched]

    count = min(negatives, len(candidates))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=count, replace=False) if count else []
    return positives, sorted(int(z) for z in chosen)


def export_slices(
    scan: CTScan,
    nodules: Sequence[Nodule],
    series_uid: str,
    out_dir: str | Path,
    *,
    negatives: int = 4,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    config: InputConfig | None = None,
    seed: int = 0,
) -> int:
    """Write PNG slices to ``out_dir/images`` and YOLO labels to ``out_dir/labels``.

    Images are grayscale in ``gray`` mode and RGB in ``stack`` mode (see
    ``slice_image``). Negative slices get an empty label file, which YOLO reads
    as background. Returns the number of slices written.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Exporting slices requires: pip install 'lung-nodule-detection[data]'"
        ) from exc

    images_dir = Path(out_dir) / "images"
    labels_dir = Path(out_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    positives, background = select_slices(
        scan, nodules, negatives=negatives, min_fraction=min_fraction, seed=seed
    )
    slices = sorted(positives + background)
    for z in slices:
        name = f"{series_uid}_{z:04d}"
        Image.fromarray(slice_image(scan, z, config)).save(images_dir / f"{name}.png")
        lines = [box.to_line() for box in nodule_boxes(scan, nodules, z, min_fraction=min_fraction)]
        (labels_dir / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines))
    return len(slices)


def load_scan(path: str | Path) -> CTScan:
    """Read a CT scan (such as a LUNA16 ``.mhd`` file) with SimpleITK."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading CT scans requires: pip install 'lung-nodule-detection[data]'"
        ) from exc

    image = sitk.ReadImage(str(path))
    return CTScan(
        hu=sitk.GetArrayFromImage(image),
        origin=tuple(image.GetOrigin()),
        spacing=tuple(image.GetSpacing()),
        direction=tuple(image.GetDirection()),
    )
