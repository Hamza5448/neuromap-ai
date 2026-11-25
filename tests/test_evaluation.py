import numpy as np
import pytest

from evaluate import _restore_slices, _score_case
from tune_threshold import _choose_threshold


def test_restore_slices_uses_native_shape_and_slice_order():
    predictions = np.zeros((2, 2, 2), dtype=np.uint8)
    predictions[0, 0, 0] = 1
    predictions[1, 1, 1] = 1

    restored = _restore_slices(predictions, (4, 4))

    assert restored.shape == (4, 4, 2)
    assert restored[0, 0, 0]
    assert restored[3, 3, 1]
    assert not restored[3, 3, 0]


def test_score_case_reports_segmentation_and_physical_volume_error():
    target = np.ones((2, 2, 1), dtype=bool)
    prediction = np.zeros_like(target)
    prediction[0, 0, 0] = True

    scores = _score_case(prediction, target, (1.0, 1.0, 2.0))

    assert scores["dice"] == pytest.approx(0.4)
    assert scores["iou"] == pytest.approx(0.25)
    assert scores["sensitivity"] == pytest.approx(0.25)
    assert scores["predicted_volume_ml"] == pytest.approx(0.002)
    assert scores["reference_volume_ml"] == pytest.approx(0.008)
    assert scores["volumetric_error_ml"] == pytest.approx(0.006)


def test_threshold_selection_uses_validation_dice_then_volume_error():
    results = [
        {
            "threshold": 0.4,
            "mean_metrics": {"dice": 0.8, "volumetric_error_ml": 2.0},
        },
        {
            "threshold": 0.5,
            "mean_metrics": {"dice": 0.8, "volumetric_error_ml": 1.0},
        },
        {
            "threshold": 0.6,
            "mean_metrics": {"dice": 0.79, "volumetric_error_ml": 0.1},
        },
    ]

    assert _choose_threshold(results) == 0.5
