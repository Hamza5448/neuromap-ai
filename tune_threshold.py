"""Select a segmentation threshold using validation patients only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from evaluate import _restore_slices, _score_case
from src.neuromap.data_loader import MRISliceDataset, load_split_manifest
from src.neuromap.inference import load_checkpoint
from src.neuromap.preprocessing import prepare_mask


def _choose_threshold(results: list[dict]) -> float:
    """Prefer mean patient Dice, breaking ties by lower mean volume error."""
    if not results:
        raise ValueError("No validation threshold results were provided.")
    selected = max(
        results,
        key=lambda result: (
            result["mean_metrics"]["dice"],
            -result["mean_metrics"]["volumetric_error_ml"],
        ),
    )
    return float(selected["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour_deduplicated.json"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        raise ValueError("Every threshold must be between 0 and 1.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    modalities = tuple(config["data"]["modalities"])
    if len(modalities) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    validation_cases = load_split_manifest(args.manifest)["validation"]
    if not validation_cases:
        raise ValueError("The validation split contains no patients.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(
        args.checkpoint,
        device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    )
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["training"]["batch_size"]
    per_threshold: dict[float, list[dict[str, float | str]]] = {
        threshold: [] for threshold in args.thresholds
    }

    for case in validation_cases:
        if case.label_path is None:
            raise ValueError(f"Validation case {case.patient_id} has no reference label.")
        image = nib.load(str(case.image_path))
        target_image = nib.load(str(case.label_path))
        target = prepare_mask(np.asarray(target_image.dataobj))
        if tuple(image.shape[:3]) != target.shape:
            raise ValueError(f"Image and reference shapes differ for {case.patient_id}.")
        dataset = MRISliceDataset(
            [case],
            image_size=image_size,
            modalities=modalities,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        probability_slices: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                probabilities = torch.sigmoid(model(images))
                probability_slices.extend(probabilities[:, 0].cpu().numpy())
        probabilities = np.stack(probability_slices)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        for threshold in args.thresholds:
            prediction = _restore_slices(
                probabilities >= threshold,
                (int(image.shape[0]), int(image.shape[1])),
            )
            metrics = _score_case(prediction, target, spacing)
            per_threshold[threshold].append(
                {"patient_id": case.patient_id, **metrics}
            )

    metric_names = (
        "dice",
        "iou",
        "sensitivity",
        "predicted_volume_ml",
        "reference_volume_ml",
        "volumetric_error_ml",
    )
    results = [
        {
            "threshold": threshold,
            "mean_metrics": {
                name: float(np.mean([case[name] for case in case_results]))
                for name in metric_names
            },
            "cases": case_results,
        }
        for threshold, case_results in per_threshold.items()
    ]
    selected_threshold = _choose_threshold(results)
    report = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "manifest": str(args.manifest),
        "split": "validation",
        "device": str(device),
        "selection_rule": "Highest mean patient Dice; ties use lower mean volume error.",
        "selected_threshold": selected_threshold,
        "results": results,
    }
    output = args.output or Path("reports") / f"{args.config.stem}_thresholds.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selected = next(
        result for result in results if result["threshold"] == selected_threshold
    )
    print(
        f"Selected validation threshold={selected_threshold:.2f} "
        f"mean_dice={selected['mean_metrics']['dice']:.5f} "
        f"mean_volume_error_ml={selected['mean_metrics']['volumetric_error_ml']:.5f}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
