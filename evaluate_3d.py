"""Evaluate a trained 3D U-Net once on the patient-level test split."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/task_a_3d.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour_deduplicated.json"),
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/task_a_3d_test.json"),
    )
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    modalities = tuple(config["data"]["modalities"])
    if len(modalities) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    test_cases = load_split_manifest(args.manifest)["test"]
    if not test_cases:
        raise ValueError("The test split contains no patients.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_3d(
        args.checkpoint,
        device,
        config["model"]["in_channels"],
        config["model"]["out_channels"],
        tuple(config["model"]["channels"]),
    )
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    results = []
    for case in test_cases:
        if case.label_path is None:
            raise ValueError(f"Test case {case.patient_id} has no reference label.")
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
        results.append(
            {
                "patient_id": case.patient_id,
                **_score_case(probabilities >= args.threshold, target, spacing),
            }
        )
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
        "config": str(args.config),
        "manifest": str(args.manifest),
        "split": "test",
        "device": str(device),
        "threshold": args.threshold,
        "patch_size": list(patch_size),
        "sliding_window_overlap": config["evaluation"]["sliding_window_overlap"],
        "test_patient_count": len(results),
        "mean_metrics": {
            name: float(np.mean([case[name] for case in results]))
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
