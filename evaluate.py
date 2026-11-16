"""Evaluation entry point for NeuroMap AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.neuromap.data_loader import discover_cases, split_cases
from src.neuromap.inference import infer, load_checkpoint
from src.neuromap.metrics import dice_score, iou_score, sensitivity_score
from src.neuromap.volume import mask_volume_ml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cases = split_cases(discover_cases(args.data_root), config["data"]["split"]["train"], config["data"]["split"]["validation"], config["project"]["seed"])
    model = load_checkpoint(args.checkpoint, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Test cases: {len(cases['test'])}; loaded model: {type(model).__name__}")
    print("Per-case 3D evaluation is intentionally pending until the inference resampling policy is finalized.")
    for case in cases["test"][:1]:
        print(f"Ready for evaluation: {case.patient_id}")


if __name__ == "__main__":
    main()
