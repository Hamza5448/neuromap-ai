"""Segmentation and volumetric evaluation metrics."""

from __future__ import annotations

import numpy as np


def _binary_arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction).astype(bool)
    expected = np.asarray(target).astype(bool)
    if predicted.shape != expected.shape:
        raise ValueError("Prediction and target must have the same shape.")
    return predicted, expected


def dice_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    predicted, expected = _binary_arrays(prediction, target)
    denominator = predicted.sum() + expected.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(predicted, expected).sum() / (denominator + epsilon))


def iou_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    predicted, expected = _binary_arrays(prediction, target)
    union = np.logical_or(predicted, expected).sum()
    return 1.0 if union == 0 else float(np.logical_and(predicted, expected).sum() / (union + epsilon))


def sensitivity_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    predicted, expected = _binary_arrays(prediction, target)
    true_positive = np.logical_and(predicted, expected).sum()
    actual_positive = expected.sum()
    return 1.0 if actual_positive == 0 and true_positive == 0 else float(true_positive / (actual_positive + epsilon))
