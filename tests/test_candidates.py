import numpy as np
import pytest

from lungnodule.annotations import Nodule
from lungnodule.candidates import SliceDetection, merge_detections
from lungnodule.froc import compute_froc
from lungnodule.preprocess import CTScan, nodule_boxes

SPACING = (0.5, 0.5, 2.0)
SCAN = CTScan(
    hu=np.zeros((10, 100, 200), dtype=np.int16), origin=(-50.0, -20.0, 100.0), spacing=SPACING
)


def detection(z, x=100.5, y=50.5, size=20.0, confidence=0.9):
    return SliceDetection(z, x, y, size, size, confidence)


def test_single_detection_maps_back_to_world_coordinates():
    (candidate,) = merge_detections([detection(5)], SCAN, "uid")
    # Pixel center (100.5, 50.5) is voxel (100, 50); world = origin + voxel * spacing.
    assert candidate.center == pytest.approx((-50 + 100 * 0.5, -20 + 50 * 0.5, 100 + 5 * 2.0))
    assert candidate.series_uid == "uid"


def test_one_nodule_seen_on_neighbouring_slices_becomes_one_candidate():
    detections = [
        detection(4, confidence=0.6),
        detection(5, confidence=0.9),
        detection(6, confidence=0.6),
    ]
    (candidate,) = merge_detections(detections, SCAN, "uid")
    assert candidate.probability == 0.9
    assert candidate.center[2] == pytest.approx(100 + 5 * 2.0)  # weighted mean stays centered


def test_distant_detections_on_one_slice_stay_separate():
    detections = [detection(5, x=40.5), detection(5, x=160.5, confidence=0.5)]
    candidates = merge_detections(detections, SCAN, "uid")
    assert [c.probability for c in candidates] == [0.9, 0.5]


def test_slice_gap_larger_than_allowed_splits_groups():
    detections = [detection(2), detection(5)]
    assert len(merge_detections(detections, SCAN, "uid")) == 2
    assert len(merge_detections(detections, SCAN, "uid", max_slice_gap=3)) == 1


def test_chained_detections_join_one_group():
    # Each detection touches the next one, so the whole chain is one nodule.
    detections = [detection(z, x=100.5 + 6 * z) for z in range(4)]
    assert len(merge_detections(detections, SCAN, "uid")) == 1


def test_no_detections_give_no_candidates():
    assert merge_detections([], SCAN, "uid") == []


def test_perfect_slice_detections_give_perfect_scan_level_cpm():
    nodule = Nodule("uid", (0.0, 5.0, 110.0), diameter_mm=8.0)
    height, width = SCAN.hu.shape[1:]
    detections = [
        SliceDetection(
            z,
            box.x_center * width,
            box.y_center * height,
            box.width * width,
            box.height * height,
            0.8,
        )
        for z in range(SCAN.hu.shape[0])
        for box in nodule_boxes(SCAN, [nodule], z)
    ]
    assert len(detections) > 1  # visible on several slices

    candidates = merge_detections(detections, SCAN, "uid")

    assert len(candidates) == 1
    assert compute_froc(candidates, {"uid": [nodule]}).cpm == 1.0
