import numpy as np
import pytest

from lungnodule.explain import OVERLAY_COLOR, box_energy, eigen_cam, overlay, resize_map


def blob_activations(center=(4, 10), channels=8, shape=(16, 16), seed=0):
    """Weak noise everywhere, plus a strong pattern shared by all channels around ``center``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    blob = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / 4.0)
    weights = rng.uniform(0.5, 1.5, size=channels)
    return weights[:, None, None] * blob + 0.05 * rng.standard_normal((channels, *shape))


def test_map_peaks_where_the_activations_vary_most():
    cam = eigen_cam(blob_activations(center=(4, 10)))
    assert cam.shape == (16, 16)
    assert np.unravel_index(cam.argmax(), cam.shape) == (4, 10)
    assert cam.min() >= 0 and cam.max() == pytest.approx(1.0)


def test_map_orientation_does_not_depend_on_the_svd_sign():
    activations = blob_activations()
    # Negating every channel flips the principal component; a strong positive
    # blob stays the hot spot because the map follows the mean activation.
    inverted = -activations + 2 * activations.mean()
    assert np.unravel_index(eigen_cam(activations).argmax(), (16, 16)) == (4, 10)
    assert eigen_cam(inverted)[4, 10] < 0.5


def test_constant_activations_give_an_empty_map():
    assert not eigen_cam(np.ones((4, 5, 6))).any()


def test_activations_must_be_three_dimensional():
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        eigen_cam(np.zeros((5, 6)))


def test_resize_keeps_values_and_centers():
    cam = np.array([[0.0, 1.0], [1.0, 0.0]])
    big = resize_map(cam, (4, 4))
    assert big.shape == (4, 4)
    assert big[0, 0] == pytest.approx(0.0) and big[0, 3] == pytest.approx(1.0)
    assert big.min() >= 0 and big.max() <= 1
    assert resize_map(cam, (2, 2)) == pytest.approx(cam)


def test_box_energy_measures_the_share_inside_the_box():
    uniform = np.ones((100, 100))
    assert box_energy(uniform, (25, 25, 50, 50)) == pytest.approx(0.25)
    focused = np.zeros((100, 100))
    focused[40:50, 60:70] = 1
    assert box_energy(focused, (65, 45, 10, 10)) == pytest.approx(1.0)
    assert box_energy(focused, (10, 10, 10, 10)) == 0.0
    assert box_energy(focused, (200, 200, 10, 10)) == 0.0  # outside the image
    assert box_energy(np.zeros((10, 10)), (5, 5, 4, 4)) == 0.0


def test_overlay_blends_color_where_the_map_is_high():
    gray = np.full((4, 4), 100, dtype=np.uint8)
    cam = np.zeros((4, 4))
    cam[0, 0] = 1.0
    out = overlay(gray, cam, alpha=0.5)
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    assert tuple(out[3, 3]) == (100, 100, 100)
    assert tuple(out[0, 0]) == tuple(round(100 * 0.5 + c * 0.5) for c in OVERLAY_COLOR)


def test_overlay_uses_the_center_slice_and_resizes_the_map():
    stacked = np.stack([np.full((8, 8), v, dtype=np.uint8) for v in (10, 50, 90)], axis=-1)
    out = overlay(stacked, np.zeros((2, 2)))
    assert out.shape == (8, 8, 3)
    assert (out == 50).all()
    with pytest.raises(ValueError, match="alpha"):
        overlay(stacked, np.zeros((2, 2)), alpha=2)
