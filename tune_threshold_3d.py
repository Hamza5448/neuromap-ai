"""Select a 3D segmentation threshold using validation patients only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from evaluate import _score_case
from src.neuromap.data_loader import load_split_manifest
from src.neuromap.inference_3d import (
    infer_volume_probabilities_3d,
    load_checkpoint_3d,
)
from src.neuromap.preprocessing import normalize_modalities, prepare_mask
from tune_threshold import _choose_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/task_a_3d.yaml"))
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/task_a_3d_thresholds.json"),
    )
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
    model = load_checkpoint_3d(
        args.checkpoint,
        device,
        config["model"]["in_channels"],
        config["model"]["out_channels"],
        tuple(config["model"]["channels"]),
    )
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    per_threshold = {threshold: [] for threshold in args.thresholds}
    for case in validation_cases:
        if case.label_path is None:
            raise ValueError(f"Validation case {case.patient_id} has no reference label.")
        image_nii = nib.load(str(case.image_path))
        label_nii = nib.load(str(case.label_path))
        if tuple(image_nii.shape[:3]) != tuple(label_nii.shape):
            raise ValueError(f"Image and label shapes differ for {case.patient_id}.")
        if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-4):
            raise ValueError(f"Image and label affines differ for {case.patient_id}.")
        image = normalize_modalities(np.asarray(image_nii.dataobj, dtype=np.float32))
        image = image[..., [("t1", "t1ce", "t2", "flair").index(name) for name in modalities]]
        target = prepare_mask(np.asarray(label_nii.dataobj)).astype(bool)
        probabilities = infer_volume_probabilities_3d(
            model,
            image,
            patch_size,
            device,
            float(config["evaluation"]["sliding_window_overlap"]),
            int(config["evaluation"]["sliding_window_batch_size"]),
        )
        spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
        for threshold in args.thresholds:
            metrics = _score_case(probabilities >= threshold, target, spacing)
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
                name: float(np.mean([case[name] for case in cases]))
                for name in metric_names
            },
            "cases": cases,
        }
        for threshold, cases in per_threshold.items()
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selected = next(
        result for result in results if result["threshold"] == selected_threshold
    )
    print(
        f"Selected 3D validation threshold={selected_threshold:.2f} "
        f"mean_dice={selected['mean_metrics']['dice']:.5f} "
        f"mean_volume_error_ml="
        f"{selected['mean_metrics']['volumetric_error_ml']:.5f}"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
