"""Training loops for the patch-based 3D segmentation model."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def segmentation_loss_3d(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Combine 3D voxel-wise BCE with soft Dice loss."""
    if logits.ndim != 5 or masks.shape != logits.shape:
        raise ValueError("3D logits and masks must have matching 5D shapes.")
    logits = logits.float()
    masks = masks.float()
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * masks).sum(dim=dimensions)
    dice_loss = 1 - (
        (2 * intersection + 1e-6)
        / (
            probabilities.sum(dim=dimensions)
            + masks.sum(dim=dimensions)
            + 1e-6
        )
    ).mean()
    return nn.BCEWithLogitsLoss()(logits, masks) + dice_loss


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def train_epoch_3d(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool = True,
) -> float:
    """Train one 3D epoch, using CUDA bfloat16 when supported."""
    model.train()
    total_loss = 0.0
    batches = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            logits = model(images)
            loss = segmentation_loss_3d(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        batches += 1
    if batches == 0:
        raise ValueError("3D training loader is empty.")
    return total_loss / batches


@torch.no_grad()
def evaluate_epoch_3d(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    mixed_precision: bool = True,
) -> float:
    """Evaluate the model on fixed validation patches."""
    model.eval()
    total_loss = 0.0
    batches = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)
        with _autocast(device, mixed_precision):
            logits = model(images)
            loss = segmentation_loss_3d(logits, masks)
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("3D validation loader is empty.")
    return total_loss / batches
