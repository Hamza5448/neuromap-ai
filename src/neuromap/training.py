"""Training loops and checkpointing."""

from __future__ import annotations

import csv
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
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)
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
    non_blocking = device.type == "cuda"
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=non_blocking))
        masks = batch["mask"].to(device, non_blocking=non_blocking)
        loss = segmentation_loss(logits, masks)
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("Evaluation loader is empty.")
    return total_loss / batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    best_model_state_dict: dict[str, torch.Tensor] | None = None,
    best_validation_loss: float | None = None,
) -> None:
    """Save resumable training state and the best model for inference."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": best_model_state_dict or model.state_dict(),
        "last_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best_validation_loss is not None:
        checkpoint["best_validation_loss"] = best_validation_loss
    torch.save(checkpoint, destination)


def write_epoch_metrics(
    path: str | Path,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    append: bool = False,
) -> None:
    """Persist one epoch's losses to a CSV history file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    should_append = append and destination.is_file() and destination.stat().st_size > 0
    with destination.open(
        "a" if should_append else "w",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.writer(history_file)
        if not should_append:
            writer.writerow(("epoch", "train_loss", "validation_loss"))
        writer.writerow((epoch, train_loss, validation_loss))
