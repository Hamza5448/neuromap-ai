"""MRI validation and normalization helpers."""

from __future__ import annotations

import numpy as np


def normalize_modalities(image: np.ndarray) -> np.ndarray:
    """Normalize each MRI modality using non-zero voxels."""
    if image.ndim not in (3, 4) or image.shape[-1] != 4:
        raise ValueError("Expected image shape (height, width, 4) or (height, width, depth, 4).")
    result = image.astype(np.float32, copy=True)
    for modality in range(result.shape[-1]):
        channel = result[..., modality]
        nonzero = channel[np.isfinite(channel) & (channel != 0)]
        if nonzero.size:
            mean = float(nonzero.mean())
            std = float(nonzero.std())
            result[..., modality] = (channel - mean) / max(std, 1e-6)
        else:
            result[..., modality] = 0
    return np.nan_to_num(result, copy=False)


def prepare_mask(label: np.ndarray) -> np.ndarray:
    """Convert MSD labels to a binary non-background mask."""
    if label.ndim not in (2, 3):
        raise ValueError("Expected a 2D slice or 3D label volume.")
    return (np.asarray(label) > 0).astype(np.float32)
