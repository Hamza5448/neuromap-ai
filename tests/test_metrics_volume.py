import numpy as np
import pytest

from src.neuromap.metrics import dice_score, iou_score, sensitivity_score
from src.neuromap.preprocessing import normalize_modalities, prepare_mask
from src.neuromap.volume import mask_volume_ml


def test_segmentation_metrics():
    prediction = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 0]])
    assert dice_score(prediction, target) == pytest.approx(2 / 3)
    assert iou_score(prediction, target) == pytest.approx(1 / 2)
    assert sensitivity_score(prediction, target) == pytest.approx(1.0)


def test_empty_masks_are_perfect():
    empty = np.zeros((2, 2))
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0


def test_volume_uses_spacing():
    mask = np.ones((2, 2, 2))
    assert mask_volume_ml(mask, (1, 2, 5)) == pytest.approx(0.08)


def test_preprocessing_contracts():
    image = np.ones((2, 2, 1, 4), dtype=np.float32)
    normalized = normalize_modalities(image)
    assert normalized.shape == image.shape
    slice_image = np.ones((2, 2, 4), dtype=np.float32)
    assert normalize_modalities(slice_image).shape == slice_image.shape
    assert np.array_equal(prepare_mask(np.array([[0, 2], [4, 0]])), np.array([[0, 1], [1, 0]], dtype=np.float32))


def test_invalid_image_shape():
    with pytest.raises(ValueError):
        normalize_modalities(np.zeros((2, 2, 3)))
