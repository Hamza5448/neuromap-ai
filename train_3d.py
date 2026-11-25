"""Train the patch-based 3D U-Net on the frozen patient split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.neuromap.data_loader import load_split_manifest
from src.neuromap.data_loader_3d import (
    MRIVolumePatchDataset,
    PatientPatchBatchSampler,
)
from src.neuromap.model_3d import build_unet_3d
from src.neuromap.training import write_epoch_metrics
from src.neuromap.training_3d import evaluate_epoch_3d, train_epoch_3d


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task_a_3d.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour_deduplicated.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/unet_3d_deduplicated.pt"),
    )
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Patient split manifest not found: {args.manifest}")
    cases = load_split_manifest(args.manifest)
    for split_name in ("train", "validation"):
        if not cases[split_name]:
            raise ValueError(f"The {split_name} split contains no patients.")
        if any(case.label_path is None for case in cases[split_name]):
            raise ValueError(f"Every {split_name} case must have a reference label.")

    modalities = tuple(config["data"]["modalities"])
    if len(modalities) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    patches_per_case = int(config["training"]["patches_per_case"])
    batch_size = int(config["training"]["batch_size"])
    if batch_size != 1:
        raise ValueError("The initial 3D setup uses batch_size=1 to bound GPU memory.")
    if int(config["training"].get("num_workers", 0)) != 0:
        raise ValueError("The patient-volume cache requires num_workers=0.")

    train_dataset = MRIVolumePatchDataset(
        cases["train"],
        patch_size,
        patches_per_case,
        training=True,
        modalities=modalities,
    )
    validation_dataset = MRIVolumePatchDataset(
        cases["validation"],
        patch_size,
        patches_per_case=1,
        training=False,
        modalities=modalities,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PatientPatchBatchSampler(train_dataset, batch_size),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_unet_3d(
        config["model"]["in_channels"],
        config["model"]["out_channels"],
        tuple(config["model"]["channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    start_epoch = 0
    best_validation_loss = float("inf")
    best_model_state_dict = _cpu_state_dict(model)
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {args.output}")
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "3d_unet":
            raise ValueError("Checkpoint is not a compatible 3D U-Net checkpoint.")
        if tuple(checkpoint.get("patch_size", ())) != patch_size:
            raise ValueError("Checkpoint patch size differs from the active config.")
        last_state = checkpoint.get(
            "last_model_state_dict",
            checkpoint["model_state_dict"],
        )
        model.load_state_dict(last_state)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_model_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in checkpoint["model_state_dict"].items()
        }
        best_validation_loss = float(checkpoint["best_validation_loss"])

    history_path = args.history or args.output.with_name(
        f"{args.output.stem}_history.csv"
    )
    append_history = args.resume and history_path.is_file()
    print(
        f"device={device} mixed_precision_bfloat16={mixed_precision} "
        f"train_patients={len(cases['train'])} "
        f"validation_patients={len(cases['validation'])} "
        f"patch_size={patch_size} patches_per_patient={patches_per_case}"
    )
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_loss = train_epoch_3d(
            model,
            train_loader,
            optimizer,
            device,
            mixed_precision,
        )
        validation_loss = evaluate_epoch_3d(
            model,
            validation_loader,
            device,
            mixed_precision,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state_dict = _cpu_state_dict(model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": "3d_unet",
                "epoch": epoch + 1,
                "patch_size": list(patch_size),
                "model_state_dict": best_model_state_dict,
                "last_model_state_dict": _cpu_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
            },
            args.output,
        )
        write_epoch_metrics(
            history_path,
            epoch + 1,
            train_loss,
            validation_loss,
            append=append_history,
        )
        append_history = True
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.5f} "
            f"validation_patch_loss={validation_loss:.5f}"
        )
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
