"""The gray and stacked (2.5D) input representations."""

import numpy as np
import pytest

from lungnodule.preprocess import (
    LUNG_WINDOW,
    CTScan,
    InputConfig,
    apply_window,
    neighbour_offset,
    slice_image,
)


def ramp_scan(depth=12, spacing_z=1.0):
    """Slice z holds -1300 + 100 z HU everywhere, so a channel shows which slice it came from."""
    hu = np.stack([np.full((4, 5), -1300 + 100 * z, dtype=np.int16) for z in range(depth)])
    return CTScan(hu=hu, origin=(0.0, 0.0, 0.0), spacing=(0.7, 0.7, spacing_z))


def slice_value(z):
    return int(apply_window(np.array([-1300 + 100 * z]))[0])


def channels(image):
    return [int(image[0, 0, c]) for c in range(3)]


@pytest.mark.parametrize(
    ("spacing_z", "stack_mm", "expected"),
    [(1.0, 2.0, 2), (0.625, 2.0, 3), (2.5, 2.0, 1), (5.0, 2.0, 1), (1.25, 5.0, 4)],
)
def test_neighbour_offset_converts_millimetres_to_slices(spacing_z, stack_mm, expected):
    assert neighbour_offset(spacing_z, stack_mm) == expected


def test_neighbour_offset_rejects_bad_spacing():
    with pytest.raises(ValueError):
        neighbour_offset(0.0)


def test_gray_input_is_the_windowed_slice():
    scan = ramp_scan()
    image = slice_image(scan, 5)
    assert image.shape == (4, 5) and image.dtype == np.uint8
    assert (image == slice_value(5)).all()


def test_stack_input_puts_neighbours_in_the_outer_channels():
    scan = ramp_scan(spacing_z=1.0)  # 2 mm is 2 slices
    image = slice_image(scan, 5, InputConfig(mode="stack", stack_mm=2.0))
    assert image.shape == (4, 5, 3) and image.dtype == np.uint8
    assert channels(image) == [slice_value(3), slice_value(5), slice_value(7)]


def test_stack_input_clamps_at_the_volume_edges():
    scan = ramp_scan(depth=6, spacing_z=1.0)
    config = InputConfig(mode="stack", stack_mm=2.0)
    first, last = slice_image(scan, 0, config), slice_image(scan, 5, config)
    assert channels(first) == [slice_value(0), slice_value(0), slice_value(2)]
    assert channels(last) == [slice_value(3), slice_value(5), slice_value(5)]


def test_stack_center_channel_matches_gray_input():
    scan = ramp_scan(spacing_z=0.625)
    gray = slice_image(scan, 6)
    stacked = slice_image(scan, 6, InputConfig(mode="stack"))
    assert (stacked[:, :, 1] == gray).all()


def test_slice_outside_the_volume_is_an_error():
    with pytest.raises(IndexError):
        slice_image(ramp_scan(depth=3), 3)


def test_config_validates_and_round_trips():
    config = InputConfig(mode="stack", stack_mm=3.0, window=(-500, 1400))
    assert InputConfig.from_dict(config.to_dict()) == config
    assert config.window == (-500.0, 1400.0)
    assert InputConfig.from_dict({}) == InputConfig()
    assert InputConfig().window == LUNG_WINDOW
    with pytest.raises(ValueError, match="Unknown input mode"):
        InputConfig(mode="rgb")
    with pytest.raises(ValueError):
        InputConfig(mode="stack", stack_mm=0)
