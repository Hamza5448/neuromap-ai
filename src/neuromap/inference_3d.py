"""Memory-bounded 3D inference and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .model_3d import build_unet_3d


def load_checkpoint_3d(
    checkpoint_path: str | Path,
    device: torch.device,
    in_channels: int = 4,
    out_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128),
) -> torch.nn.Module:
    """Load a compatible 3D U-Net checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "3d_unet":
        raise ValueError(f"Not a compatible 3D U-Net checkpoint: {checkpoint_path}")
    model = build_unet_3d(in_channels, out_channels, channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_volume_probabilities_3d(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    device: torch.device,
    overlap: float = 0.5,
    sliding_window_batch_size: int = 1,
) -> np.ndarray:
    """Return full-volume probabilities using overlapping, Gaussian-weighted tiles."""
    if image.ndim != 4 or image.shape[-1] < 1:
        raise ValueError("Expected image shape (height, width, depth, channels).")
    if not np.isfinite(image).all():
        raise ValueError("The MRI volume contains non-finite values.")
    if len(patch_size) != 3 or any(size < 1 for size in patch_size):
        raise ValueError("Patch size must contain three positive dimensions.")
    if not 0 <= overlap < 1:
        raise ValueError("Sliding-window overlap must be in [0, 1).")
    if sliding_window_batch_size < 1:
        raise ValueError("Sliding-window batch size must be positive.")

    volume = torch.from_numpy(
        np.ascontiguousarray(image.transpose(3, 0, 1, 2))
    ).unsqueeze(0)
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=mixed_precision,
    ):
        logits = sliding_window_inference(
            volume,
            roi_size=patch_size,
            sw_batch_size=sliding_window_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu"),
        )
    probabilities = torch.sigmoid(logits[0, 0]).numpy()
    return probabilities.astype(np.float32, copy=False)
