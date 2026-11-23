import numpy as np
import pytest

from lungnodule.inference import detect_scan, to_model_input
from lungnodule.preprocess import CTScan, InputConfig

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


def test_stacked_input_gives_three_channel_slices():
    shapes = []

    def detector(images):
        shapes.extend(image.shape for image in images)
        return [[] for _ in images]

    detect_scan(SCAN, detector, config=InputConfig(mode="stack"), batch_size=2)
    assert shapes == [(32, 48, 3)] * 5


def test_model_input_repeats_gray_slices():
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    converted = to_model_input(image)
    assert converted.shape == (2, 3, 3)
    assert all((converted[:, :, c] == image).all() for c in range(3))


def test_model_input_reverses_stacked_channels_to_bgr():
    # Channels in RGB order: slice below, center slice, slice above.
    image = np.stack([np.full((2, 2), v, dtype=np.uint8) for v in (10, 20, 30)], axis=-1)
    converted = to_model_input(image)
    assert [int(converted[0, 0, c]) for c in range(3)] == [30, 20, 10]
    assert converted.flags["C_CONTIGUOUS"]


def test_model_input_matches_how_training_images_are_read(tmp_path):
    """A stacked PNG read back as OpenCV does (BGR) equals the array passed to predict."""
    from PIL import Image

    image = np.stack([np.full((2, 2), v, dtype=np.uint8) for v in (10, 20, 30)], axis=-1)
    Image.fromarray(image).save(tmp_path / "slice.png")
    read_rgb = np.asarray(Image.open(tmp_path / "slice.png"))
    read_bgr = read_rgb[:, :, ::-1]  # what cv2.imread returns for the same file
    assert (to_model_input(image) == read_bgr).all()


def test_model_input_rejects_other_shapes():
    with pytest.raises(ValueError, match="Expected"):
        to_model_input(np.zeros((2, 2, 4), dtype=np.uint8))
