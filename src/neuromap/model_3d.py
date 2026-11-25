"""Three-dimensional segmentation model definitions."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet


def build_unet_3d(
    in_channels: int = 4,
    out_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128),
) -> UNet:
    """Build a patch-based 3D U-Net with logits output."""
    if in_channels < 1 or out_channels < 1:
        raise ValueError("Channel counts must be positive.")
    if len(channels) < 2 or any(channel < 1 for channel in channels):
        raise ValueError("At least two positive feature channel sizes are required.")
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=(2,) * (len(channels) - 1),
        num_res_units=1,
    )


def predict_logits_3d(model: torch.nn.Module, volumes: torch.Tensor) -> torch.Tensor:
    """Run a shape-checked 3D forward pass."""
    if volumes.ndim != 5:
        raise ValueError(
            "Expected volumes with shape (batch, channels, height, width, depth)."
        )
    return model(volumes)
