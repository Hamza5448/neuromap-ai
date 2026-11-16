"""Model loading and MRI inference."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import build_unet


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    in_channels: int = 4,
    out_channels: int = 1,
) -> torch.nn.Module:
    """Load a model checkpoint with an explicit device."""
    model = build_unet(in_channels=in_channels, out_channels=out_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def infer(model: torch.nn.Module, images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return binary predictions from a batch of normalized image tensors."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    probabilities = torch.sigmoid(model(images))
    return probabilities >= threshold
