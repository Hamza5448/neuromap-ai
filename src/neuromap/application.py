"""Input validation, inference, and export helpers for the Stage A application."""

from __future__ import annotations

import gzip
import zlib

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional

from .metrics import dice_score, iou_score, sensitivity_score
from .preprocessing import normalize_modalities, prepare_mask
from .volume import mask_volume_ml

_MRI_MODALITY_INDICES = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}


def load_nifti_bytes(payload: bytes) -> nib.Nifti1Image:
    """Load an uncompressed or gzip-compressed NIfTI file from uploaded bytes."""
    if not payload:
        raise ValueError("The uploaded NIfTI file is empty.")
    try:
        content = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    except (EOFError, OSError, zlib.error) as error:
        raise ValueError(f"The uploaded gzip-compressed NIfTI is invalid: {error}") from error
    return nib.Nifti1Image.from_bytes(content)


def validate_multimodal_image(
    image: nib.Nifti1Image,
) -> tuple[float, float, float]:
    """Validate a four-channel NIfTI and return its physical voxel spacing."""
    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError(
            "Expected one 4D NIfTI with channels ordered as T1, T1ce, T2, FLAIR."
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8
    ):
        raise ValueError("The NIfTI spatial affine is invalid or singular.")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("The NIfTI voxel spacing must contain three positive values.")
    return spacing


def load_mri_array(image: nib.Nifti1Image) -> np.ndarray:
    """Load and validate finite four-modality MRI voxel values."""
    validate_multimodal_image(image)
    volume = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("The MRI volume contains non-finite voxel values.")
    return volume


def load_reference_mask(
    reference: nib.Nifti1Image,
    image: nib.Nifti1Image,
) -> np.ndarray:
    """Validate reference-mask geometry and return a binary whole-tumor mask."""
    if reference.ndim != 3 or tuple(reference.shape) != tuple(image.shape[:3]):
        raise ValueError("The reference mask dimensions must match the MRI volume.")
    if not np.allclose(reference.affine, image.affine, rtol=0, atol=1e-4):
        raise ValueError("The reference mask spatial affine must match the MRI.")
    return prepare_mask(np.asarray(reference.dataobj))


@torch.inference_mode()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: tuple[int, int],
    threshold: float,
    batch_size: int,
    device: torch.device,
    modalities: tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> np.ndarray:
    """Predict and restore a binary mask on the input volume's native voxel grid."""
    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError("Expected image shape (height, width, depth, 4).")
    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("Select one or more distinct MRI modalities.")
    unknown_modalities = set(modalities) - _MRI_MODALITY_INDICES.keys()
    if unknown_modalities:
        raise ValueError(f"Unknown MRI modalities: {sorted(unknown_modalities)}")
    if not np.isfinite(image).all():
        raise ValueError("The MRI volume contains non-finite voxel values.")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    height, width, depth, _ = image.shape
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)
        normalized = np.stack(
            [
                normalize_modalities(image[:, :, index, :])
                for index in range(start, stop)
            ]
        )
        modality_indices = [_MRI_MODALITY_INDICES[name] for name in modalities]
        normalized = normalized[..., modality_indices]
        inputs = torch.from_numpy(normalized).permute(0, 3, 1, 2).contiguous()
        inputs = functional.interpolate(
            inputs,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).to(device, non_blocking=device.type == "cuda")
        logits = model(inputs)
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("The model must return one segmentation logit per slice.")
        binary = torch.sigmoid(logits) >= threshold
        native = functional.interpolate(
            binary.float(),
            size=(height, width),
            mode="nearest",
        )
        predictions.extend(native[:, 0].cpu().numpy().astype(bool))
    return np.stack(predictions, axis=2)


def summarize_prediction(
    prediction: np.ndarray,
    spacing_mm: tuple[float, float, float],
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Return predicted volume and optional reference-mask evaluation metrics."""
    summary = {"predicted_volume_ml": mask_volume_ml(prediction, spacing_mm)}
    if reference is not None:
        if prediction.shape != reference.shape:
            raise ValueError("Prediction and reference masks must have the same shape.")
        reference_volume = mask_volume_ml(reference, spacing_mm)
        summary.update(
            {
                "dice": dice_score(prediction, reference),
                "iou": iou_score(prediction, reference),
                "sensitivity": sensitivity_score(prediction, reference),
                "reference_volume_ml": reference_volume,
                "volumetric_error_ml": abs(
                    summary["predicted_volume_ml"] - reference_volume
                ),
            }
        )
    return summary


def grayscale_rgb(image_slice: np.ndarray) -> np.ndarray:
    """Convert one MRI slice to an 8-bit grayscale RGB image with robust contrast."""
    if image_slice.ndim != 2:
        raise ValueError("Expected a 2D MRI slice.")
    values = np.asarray(image_slice, dtype=np.float32)
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0]
    contrast = nonzero if nonzero.size else finite
    if contrast.size == 0:
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    low, high = np.percentile(contrast, (1, 99))
    scaled = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
    scaled = np.nan_to_num(scaled, copy=False)
    grayscale = (scaled * 255).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def segmentation_overlay(
    base_rgb: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay predictions and optional reference in distinct, readable colors."""
    if base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError("Base image must have RGB shape (height, width, 3).")
    if prediction.shape != base_rgb.shape[:2]:
        raise ValueError("Prediction and base image shapes must match.")
    output = base_rgb.copy()
    predicted = np.asarray(prediction, dtype=bool)
    if reference is None:
        output[predicted] = (
            0.45 * output[predicted] + 0.55 * np.array([255, 40, 40])
        ).astype(np.uint8)
        return output
    if reference.shape != prediction.shape:
        raise ValueError("Prediction and reference masks must have the same shape.")
    expected = np.asarray(reference, dtype=bool)
    output[predicted & expected] = (255, 210, 0)
    output[predicted & ~expected] = (255, 40, 40)
    output[~predicted & expected] = (0, 220, 255)
    return output


def mask_nifti_bytes(mask: np.ndarray, reference: nib.Nifti1Image) -> bytes:
    """Serialize a binary mask as an uncompressed NIfTI with source geometry."""
    if mask.shape != reference.shape[:3]:
        raise ValueError("Segmentation mask dimensions must match the source MRI.")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference.affine, header)
    return image.to_bytes()
