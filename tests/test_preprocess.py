import math

import numpy as np
import pytest

from lungnodule.annotations import Nodule
from lungnodule.preprocess import (
    CTScan,
    YoloBox,
    apply_window,
    export_slices,
    load_scan,
    nodule_boxes,
    select_slices,
)

SPACING = (0.5, 0.5, 2.0)  # x, y, z in mm
SHAPE = (10, 100, 200)  # z, y, x


def make_scan(shape=SHAPE, value=-800):
    return CTScan(hu=np.full(shape, value, dtype=np.int16), origin=(0.0, 0.0, 0.0), spacing=SPACING)


# World (50, 25, 10) mm is voxel (100, 50, 5); diameter 10 mm means radius 5 mm.
NODULE = Nodule("scan", (50.0, 25.0, 10.0), diameter_mm=10.0)


def test_apply_window_maps_lung_window_to_8_bits():
    gray = apply_window(np.array([-2000, -1350, -600, 150, 3000]))
    assert gray.dtype == np.uint8
    assert gray.tolist() == [0, 0, 128, 255, 255]


def test_apply_window_rejects_non_positive_width():
    with pytest.raises(ValueError, match="positive"):
        apply_window(np.zeros(3), level=0, width=0)


def test_box_at_nodule_center_uses_full_radius_and_half_pixel_offset():
    (box,) = nodule_boxes(make_scan(), [NODULE], z=5)
    # Radius 5 mm = 10 px; pixel center 100.5 -> spans 90.5..110.5 of 200 px.
    assert box.x_center == pytest.approx(100.5 / 200)
    assert box.width == pytest.approx(20 / 200)
    assert box.y_center == pytest.approx(50.5 / 100)
    assert box.height == pytest.approx(20 / 100)


def test_box_shrinks_away_from_the_center_slice():
    (box,) = nodule_boxes(make_scan(), [NODULE], z=7)  # 4 mm from center
    assert box.width == pytest.approx(2 * math.sqrt(5**2 - 4**2) / 0.5 / 200)


def test_min_fraction_drops_slices_that_graze_the_nodule():
    scan = make_scan()
    assert nodule_boxes(scan, [NODULE], z=7)  # cross-section radius 3 mm >= 2.5 mm
    assert not nodule_boxes(scan, [NODULE], z=7, min_fraction=0.7)  # needs 3.5 mm
    assert not nodule_boxes(scan, [NODULE], z=8)  # 6 mm away: outside the sphere


def test_box_is_clipped_at_the_image_border():
    near_edge = Nodule("scan", (1.0, 25.0, 10.0), diameter_mm=10.0)  # voxel x = 2
    (box,) = nodule_boxes(make_scan(), [near_edge], z=5)
    assert box.x_center - box.width / 2 == pytest.approx(0.0)
    assert box.width == pytest.approx((2.5 + 10) / 200)


def test_nodule_outside_the_image_gives_no_box():
    outside = Nodule("scan", (500.0, 25.0, 10.0), diameter_mm=10.0)
    assert nodule_boxes(make_scan(), [outside], z=5) == []


def test_yolo_line_format():
    assert YoloBox(0.5, 0.25, 0.1, 0.2).to_line() == "0 0.500000 0.250000 0.100000 0.200000"


def test_select_slices_keeps_negatives_away_from_any_nodule():
    positives, negatives = select_slices(make_scan(), [NODULE], negatives=10)
    assert positives == [3, 4, 5, 6, 7]
    assert set(negatives) == {0, 1, 2, 8, 9}  # slices 3..7 touch the nodule


def test_select_slices_is_reproducible_and_limited():
    scan = make_scan(shape=(40, 100, 200))
    first = select_slices(scan, [NODULE], negatives=4, seed=1)
    assert first == select_slices(scan, [NODULE], negatives=4, seed=1)
    assert len(first[1]) == 4


def test_scan_without_nodules_still_yields_negatives():
    positives, negatives = select_slices(make_scan(), [], negatives=3)
    assert positives == []
    assert len(negatives) == 3


def test_export_writes_images_and_labels(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    count = export_slices(make_scan(), [NODULE], "uid", tmp_path, negatives=2)

    assert count == 7  # 5 labeled slices + 2 negatives
    assert len(list((tmp_path / "images").glob("*.png"))) == 7
    label = (tmp_path / "labels" / "uid_0005.txt").read_text()
    assert label.startswith("0 0.502500 0.505000 0.100000 0.200000")
    background = [p for p in (tmp_path / "labels").glob("*.txt") if not p.read_text()]
    assert len(background) == 2

    image = np.asarray(Image.open(tmp_path / "images" / "uid_0005.png"))
    assert image.shape == (100, 200)
    assert image.dtype == np.uint8


def test_load_scan_reads_geometry_from_disk(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    volume = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(volume)
    image.SetOrigin((-10.0, -20.0, -30.0))
    image.SetSpacing(SPACING)
    path = tmp_path / "scan.mhd"
    sitk.WriteImage(image, str(path))

    scan = load_scan(path)

    np.testing.assert_array_equal(scan.hu, volume)
    assert scan.origin == (-10.0, -20.0, -30.0)
    assert scan.spacing == SPACING
    assert scan.direction == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
