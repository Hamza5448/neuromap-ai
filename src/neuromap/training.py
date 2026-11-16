"""Training loops and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Combine voxel-wise BCE with soft Dice loss."""
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    dice_loss = 1 - (
        (2 * intersection + 1e-6)
        / (
            probabilities.sum(dim=(1, 2, 3))
            + masks.sum(dim=(1, 2, 3))
            + 1e-6
        )
    ).mean()
    return nn.BCEWithLogitsLoss()(logits, masks) + dice_loss


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch using Dice plus binary cross-entropy."""
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / batches


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Evaluate the segmentation loss without updating model parameters."""
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        logits = model(batch["image"].to(device))
        loss = segmentation_loss(logits, batch["mask"].to(device))
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("Evaluation loader is empty.")
    return total_loss / batches


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str | Path) -> None:
    """Save model and optimizer state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, destination)
