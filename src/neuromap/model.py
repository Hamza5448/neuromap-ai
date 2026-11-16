"""Segmentation model definitions."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet


def build_unet(in_channels: int = 4, out_channels: int = 1) -> UNet:
    """Build the Stage A 2D U-Net with logits output."""
    if in_channels < 1 or out_channels < 1:
        raise ValueError("Channel counts must be positive.")
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )


def predict_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run a shape-checked forward pass."""
    if images.ndim != 4:
        raise ValueError("Expected images with shape (batch, channels, height, width).")
    return model(images)
