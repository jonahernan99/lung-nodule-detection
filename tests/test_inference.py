import numpy as np
import pytest

from lungnodule.inference import detect_scan
from lungnodule.preprocess import CTScan

SCAN = CTScan(hu=np.full((5, 32, 48), -600, dtype=np.int16), origin=(0, 0, 0), spacing=(1, 1, 1))


class RecordingDetector:
    """Finds one box on slice 2 and records the batches it receives."""

    def __init__(self):
        self.batches = []

    def __call__(self, images):
        self.batches.append([(image.shape, image.dtype, int(image[0, 0])) for image in images])
        first = sum(len(batch) for batch in self.batches) - len(images)
        return [[(10.0, 12.0, 6.0, 6.0, 0.7)] if first + i == 2 else [] for i in range(len(images))]


def test_every_slice_is_windowed_and_batched():
    detector = RecordingDetector()
    detect_scan(SCAN, detector, batch_size=2)
    assert [len(batch) for batch in detector.batches] == [2, 2, 1]
    shape, dtype, value = detector.batches[0][0]
    assert shape == (32, 48)
    assert dtype == np.uint8
    assert value == 128  # -600 HU sits at the center of the lung window


def test_boxes_are_returned_with_their_slice_index():
    (found,) = detect_scan(SCAN, RecordingDetector(), batch_size=2)
    assert (found.z, found.x_center, found.y_center, found.confidence) == (2, 10.0, 12.0, 0.7)


def test_detector_returning_wrong_number_of_results_is_an_error():
    with pytest.raises(ValueError, match="returned 0 results for 5 slices"):
        detect_scan(SCAN, lambda images: [], batch_size=8)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        detect_scan(SCAN, RecordingDetector(), batch_size=0)
