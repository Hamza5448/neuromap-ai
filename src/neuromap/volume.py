"""Physical volume calculations from masks and voxel metadata."""

from __future__ import annotations

import numpy as np


def voxel_volume_mm3(spacing_mm: tuple[float, float, float] | np.ndarray) -> float:
    """Return one voxel volume in cubic millimeters."""
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive values.")
    return float(np.prod(spacing))


def mask_volume_ml(mask: np.ndarray, spacing_mm: tuple[float, float, float] | np.ndarray) -> float:
    """Calculate binary mask volume in milliliters."""
    return float(np.asarray(mask).astype(bool).sum() * voxel_volume_mm3(spacing_mm) / 1000.0)
