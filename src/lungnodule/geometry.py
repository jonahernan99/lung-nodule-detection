"""Convert between CT world coordinates (millimetres) and voxel indices.

LUNA16 annotations are given in world coordinates, while CT volumes are
indexed by voxel. SimpleITK reports origin, spacing and direction in
(x, y, z) order, but ``GetArrayFromImage`` returns arrays indexed as
``[z, y, x]``. Every function here uses (x, y, z) order.

All functions accept a single point of shape ``(3,)`` or a batch of shape
``(N, 3)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _index_to_world_matrix(spacing: ArrayLike, direction: ArrayLike | None) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive values")
    rotation = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    return rotation.reshape(3, 3) @ np.diag(spacing)


def _as_points(points: ArrayLike) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3 or points.ndim not in (1, 2):
        raise ValueError("points must have shape (3,) or (N, 3)")
    return points


def voxel_to_world(
    index_xyz: ArrayLike,
    origin: ArrayLike,
    spacing: ArrayLike,
    direction: ArrayLike | None = None,
) -> np.ndarray:
    """Map voxel indices to world coordinates: ``origin + D @ diag(spacing) @ index``."""
    matrix = _index_to_world_matrix(spacing, direction)
    return _as_points(index_xyz) @ matrix.T + np.asarray(origin, dtype=float)


def world_to_voxel(
    world_xyz: ArrayLike,
    origin: ArrayLike,
    spacing: ArrayLike,
    direction: ArrayLike | None = None,
) -> np.ndarray:
    """Map world coordinates to (fractional) voxel indices; inverse of ``voxel_to_world``."""
    inverse = np.linalg.inv(_index_to_world_matrix(spacing, direction))
    return (_as_points(world_xyz) - np.asarray(origin, dtype=float)) @ inverse.T
