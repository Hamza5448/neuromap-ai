"""Create and audit a reproducible patient-level dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.neuromap.data_loader import discover_cases, save_split_manifest, split_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/task01_brain_tumour.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cases = discover_cases(args.data_root)
    unlabeled = [case.patient_id for case in cases if case.label_path is None]
    if unlabeled:
        raise ValueError(f"Missing labels for {len(unlabeled)} cases: {unlabeled[:3]}")
    splits = split_cases(
        cases,
        config["data"]["split"]["train"],
        config["data"]["split"]["validation"],
        config["project"]["seed"],
    )
    save_split_manifest(splits, args.manifest)
    print(f"Manifest: {args.manifest}")
    for split_name, split_cases_list in splits.items():
        print(f"{split_name}: {len(split_cases_list)} patients")


if __name__ == "__main__":
    main()
