import numpy as np
import pytest

from lungnodule.geometry import voxel_to_world, world_to_voxel

ORIGIN = (-195.0, -180.5, -330.0)
SPACING = (0.7, 0.7, 2.5)


def test_identity_direction_scales_and_offsets():
    world = voxel_to_world((10, 20, 4), ORIGIN, SPACING)
    np.testing.assert_allclose(world, (-188.0, -166.5, -320.0))


def test_world_to_voxel_inverts_voxel_to_world():
    index = np.array([[0, 0, 0], [256.5, 100.25, 63]])
    world = voxel_to_world(index, ORIGIN, SPACING)
    np.testing.assert_allclose(world_to_voxel(world, ORIGIN, SPACING), index)


def test_flipped_direction_is_respected():
    direction = np.diag([-1.0, -1.0, 1.0])
    world = voxel_to_world((10, 10, 0), ORIGIN, SPACING, direction)
    np.testing.assert_allclose(world, (-202.0, -187.5, -330.0))
    np.testing.assert_allclose(world_to_voxel(world, ORIGIN, SPACING, direction), (10, 10, 0))


def test_direction_accepts_flat_sequence_like_simpleitk():
    flat = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(
        voxel_to_world((1, 2, 3), ORIGIN, SPACING, flat),
        voxel_to_world((1, 2, 3), ORIGIN, SPACING),
    )


@pytest.mark.parametrize("spacing", [(0.7, 0.7), (0.7, 0.0, 2.5), (0.7, -1.0, 2.5)])
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match="spacing"):
        world_to_voxel((0, 0, 0), ORIGIN, spacing)


def test_invalid_point_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        voxel_to_world((1, 2), ORIGIN, SPACING)
