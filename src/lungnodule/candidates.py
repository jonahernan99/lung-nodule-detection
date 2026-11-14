"""Merge 2D slice detections into 3D nodule candidates in world coordinates.

A 2D detector usually finds one nodule on several neighbouring slices. Scan-
level evaluation expects one candidate per nodule, so detections are linked
when they lie on nearby slices and their centers are closer than the larger
of their in-plane radii. Each group of linked detections becomes one
candidate, placed at the confidence-weighted mean of its detections and
scored with their highest confidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lungnodule.annotations import Candidate
from lungnodule.geometry import voxel_to_world
from lungnodule.preprocess import CTScan


@dataclass(frozen=True)
class SliceDetection:
    """A box found on one slice, in pixel coordinates (pixel ``i`` spans ``[i, i + 1)``)."""

    z: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def merge_detections(
    detections: Sequence[SliceDetection],
    scan: CTScan,
    series_uid: str,
    *,
    max_slice_gap: int = 1,
) -> list[Candidate]:
    """Group detections of the same nodule and return one candidate per group.

    Two detections are linked when they are at most ``max_slice_gap`` slices
    apart (0 means the same slice) and their centers are within the larger of
    their in-plane radii. Candidates are returned from most to least confident.
    """
    if not detections:
        return []
    spacing_x, spacing_y, _ = scan.spacing
    # Pixel center -> voxel index: pixel i spans [i, i + 1), voxel i is its center.
    voxels = np.array([(d.x_center - 0.5, d.y_center - 0.5, d.z) for d in detections])
    radii_mm = np.array([max(d.width * spacing_x, d.height * spacing_y) / 2 for d in detections])
    confidences = np.array([d.confidence for d in detections])

    groups = _DisjointSet(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if abs(voxels[i, 2] - voxels[j, 2]) > max_slice_gap:
                continue
            dx_mm = (voxels[i, 0] - voxels[j, 0]) * spacing_x
            dy_mm = (voxels[i, 1] - voxels[j, 1]) * spacing_y
            if np.hypot(dx_mm, dy_mm) <= max(radii_mm[i], radii_mm[j]):
                groups.union(i, j)

    members: dict[int, list[int]] = {}
    for i in range(len(detections)):
        members.setdefault(groups.find(i), []).append(i)

    candidates = []
    for indices in members.values():
        weights = confidences[indices]
        center_voxel = (voxels[indices] * weights[:, None]).sum(axis=0) / weights.sum()
        center = voxel_to_world(center_voxel, scan.origin, scan.spacing, scan.direction)
        candidates.append(
            Candidate(
                series_uid=series_uid,
                center=tuple(float(c) for c in center),
                probability=float(weights.max()),
            )
        )
    return sorted(candidates, key=lambda c: -c.probability)
