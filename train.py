"""Training entry point for NeuroMap AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.neuromap.data_loader import (
    MRISliceDataset,
    PatientGroupedBatchSampler,
    discover_cases,
    load_split_manifest,
    save_split_manifest,
    split_cases,
)
from src.neuromap.model import build_unet
from src.neuromap.training import (
    evaluate_epoch,
    save_checkpoint,
    train_epoch,
    write_epoch_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/baseline.pt"))
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint at --output.",
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
    modalities = tuple(config["data"]["modalities"])
    if len(modalities) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    dataset = MRISliceDataset(
        cases["train"],
        tuple(config["data"]["image_size"]),
        training=True,
        modalities=modalities,
    )
    validation_dataset = MRISliceDataset(
        cases["validation"],
        tuple(config["data"]["image_size"]),
        modalities=modalities,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=PatientGroupedBatchSampler(
            dataset,
            config["training"]["batch_size"],
        ),
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    model = build_unet(config["model"]["in_channels"], config["model"]["out_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    start_epoch = 0
    best_validation_loss = float("inf")
    best_model_state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {args.output}")
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
        last_model_state_dict = checkpoint.get(
            "last_model_state_dict", checkpoint["model_state_dict"]
        )
        model.load_state_dict(last_model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_model_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint["model_state_dict"].items()
        }
        saved_best_loss = checkpoint.get("best_validation_loss")
        if saved_best_loss is None:
            model.load_state_dict(best_model_state_dict)
            best_validation_loss = evaluate_epoch(model, validation_loader, device)
            model.load_state_dict(last_model_state_dict)
        else:
            best_validation_loss = float(saved_best_loss)

    history_path = args.history or args.output.with_name(
        f"{args.output.stem}_history.csv"
    )
    append_history = args.resume and history_path.is_file()
    for epoch in range(start_epoch, config["training"]["epochs"]):
        loss = train_epoch(model, loader, optimizer, device)
        validation_loss = evaluate_epoch(model, validation_loader, device)
        print(
            f"epoch={epoch + 1} train_loss={loss:.5f} "
            f"validation_loss={validation_loss:.5f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            args.output,
            best_model_state_dict,
            best_validation_loss,
        )
        write_epoch_metrics(
            history_path,
            epoch + 1,
            loss,
            validation_loss,
            append=append_history,
        )
        append_history = True
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
