"""Training entry point for NeuroMap AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.neuromap.data_loader import (
    MRISliceDataset,
    discover_cases,
    load_split_manifest,
    save_split_manifest,
    split_cases,
)
from src.neuromap.model import build_unet
from src.neuromap.training import evaluate_epoch, save_checkpoint, train_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/baseline.pt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.manifest.exists():
        cases = load_split_manifest(args.manifest)
    else:
        cases = split_cases(
            discover_cases(args.data_root),
            config["data"]["split"]["train"],
            config["data"]["split"]["validation"],
            config["project"]["seed"],
        )
        save_split_manifest(cases, args.manifest)
    if any(case.label_path is None for case in cases["train"]):
        raise ValueError("Training requires a label for every training case.")
    dataset = MRISliceDataset(cases["train"], tuple(config["data"]["image_size"]), training=True)
    validation_dataset = MRISliceDataset(
        cases["validation"], tuple(config["data"]["image_size"])
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    model = build_unet(config["model"]["in_channels"], config["model"]["out_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    best_validation_loss = float("inf")
    for epoch in range(config["training"]["epochs"]):
        loss = train_epoch(model, loader, optimizer, device)
        validation_loss = evaluate_epoch(model, validation_loader, device)
        print(
            f"epoch={epoch + 1} train_loss={loss:.5f} "
            f"validation_loss={validation_loss:.5f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, optimizer, epoch + 1, args.output)


if __name__ == "__main__":
    main()
