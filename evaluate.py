"""Evaluate a trained model on the patient-level held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader

from src.neuromap.data_loader import MRISliceDataset, PatientCase, load_split_manifest
from src.neuromap.inference import infer, load_checkpoint
from src.neuromap.metrics import dice_score, iou_score, sensitivity_score
from src.neuromap.preprocessing import prepare_mask
from src.neuromap.volume import mask_volume_ml


def _restore_slices(
    predicted_slices: np.ndarray,
    native_shape: tuple[int, int],
) -> np.ndarray:
    """Resize model-grid slices to the native voxel grid using nearest neighbors."""
    if predicted_slices.ndim != 3:
        raise ValueError("Predictions must have shape (depth, height, width).")
    restored = functional.interpolate(
        torch.from_numpy(predicted_slices.astype(np.float32)).unsqueeze(1),
        size=native_shape,
        mode="nearest",
    )
    return restored.squeeze(1).numpy().astype(bool).transpose(1, 2, 0)


def _score_case(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> dict[str, float]:
    """Calculate segmentation scores and absolute physical volume error."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and reference masks must have the same shape.")
    predicted_volume_ml = mask_volume_ml(prediction, spacing_mm)
    reference_volume_ml = mask_volume_ml(target, spacing_mm)
    return {
        "dice": dice_score(prediction, target),
        "iou": iou_score(prediction, target),
        "sensitivity": sensitivity_score(prediction, target),
        "predicted_volume_ml": predicted_volume_ml,
        "reference_volume_ml": reference_volume_ml,
        "volumetric_error_ml": abs(predicted_volume_ml - reference_volume_ml),
    }


def _evaluate_case(
    model: torch.nn.Module,
    case: PatientCase,
    image_size: tuple[int, int],
    threshold: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    modalities: tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> dict[str, float | str]:
    if case.label_path is None:
        raise ValueError(f"Test case {case.patient_id} has no reference label.")

    image = nib.load(str(case.image_path))
    label = nib.load(str(case.label_path))
    target = prepare_mask(np.asarray(label.dataobj))
    if tuple(image.shape[:3]) != target.shape:
        raise ValueError(f"Image and reference shapes differ for {case.patient_id}.")

    dataset = MRISliceDataset(
        [case],
        image_size=image_size,
        modalities=modalities,
    )
    if len(dataset) != target.shape[2]:
        raise ValueError(f"Slice count does not match the label for {case.patient_id}.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predicted_slices: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        predictions = infer(model, images, threshold=threshold)
        predicted_slices.extend(
            predictions.squeeze(1).cpu().numpy().astype(np.uint8)
        )

    prediction = _restore_slices(
        np.stack(predicted_slices),
        (int(image.shape[0]), int(image.shape[1])),
    )
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return {"patient_id": case.patient_id, **_score_case(prediction, target, spacing)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/test_metrics.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Test split manifest not found: {args.manifest}")
    cases = load_split_manifest(args.manifest)["test"]
    if not cases:
        raise ValueError("The test split contains no patients.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(
        args.checkpoint,
        device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    )
    threshold = (
        float(config["evaluation"]["threshold"])
        if args.threshold is None
        else args.threshold
    )
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    modalities = tuple(config["data"]["modalities"])
    if len(modalities) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    image_size = tuple(config["data"]["image_size"])
    results = [
        _evaluate_case(
            model,
            case,
            image_size,
            threshold,
            config["training"]["batch_size"],
            config["training"]["num_workers"],
            device,
            modalities,
        )
        for case in cases
    ]
    metric_names = (
        "dice",
        "iou",
        "sensitivity",
        "predicted_volume_ml",
        "reference_volume_ml",
        "volumetric_error_ml",
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "threshold": threshold,
        "modalities": list(modalities),
        "model_input_size": list(image_size),
        "native_grid_resampling": "nearest",
        "test_patient_count": len(results),
        "mean_metrics": {
            name: float(np.mean([result[name] for result in results]))
            for name in metric_names
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluated {len(results)} held-out patients on {device}.")
    for name in ("dice", "iou", "sensitivity", "volumetric_error_ml"):
        print(f"mean_{name}={report['mean_metrics'][name]:.5f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
