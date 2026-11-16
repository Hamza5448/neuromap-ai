"""Patient-level discovery and slice loading for MSD Task01_BrainTumour."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import normalize_modalities, prepare_mask


@dataclass(frozen=True)
class PatientCase:
    """Paths for one patient, kept together during splitting."""

    patient_id: str
    image_path: Path
    label_path: Path | None


def discover_cases(dataset_root: str | Path) -> list[PatientCase]:
    """Discover labeled cases from an MSD dataset directory."""
    root = Path(dataset_root)
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    cases: list[PatientCase] = []
    for image_path in sorted(images_dir.glob("*.nii*")):
        if image_path.name.startswith("._"):
            continue
        patient_id = re.sub(r"\.nii(\.gz)?$", "", image_path.name)
        label_path = labels_dir / image_path.name
        cases.append(
            PatientCase(
                patient_id=patient_id,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
            )
        )
    if not cases:
        raise FileNotFoundError(f"No NIfTI images found in {images_dir}")
    return cases


def split_cases(
    cases: list[PatientCase],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[PatientCase]]:
    """Create a deterministic patient-level split."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave test cases.")
    if not cases:
        raise ValueError("At least one case is required.")
    shuffled = list(cases)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * train_fraction))
    validation_end = max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(shuffled) - 1)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def save_split_manifest(
    splits: dict[str, list[PatientCase]],
    path: str | Path,
) -> None:
    """Persist patient IDs and paths so later runs use the same split."""
    manifest = {
        split_name: [
            {
                "patient_id": case.patient_id,
                "image_path": str(case.image_path),
                "label_path": str(case.label_path) if case.label_path else None,
            }
            for case in split_cases_list
        ]
        for split_name, split_cases_list in splits.items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, list[PatientCase]]:
    """Load a previously persisted patient-level split."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    required_splits = {"train", "validation", "test"}
    if set(payload) != required_splits:
        raise ValueError(f"Split manifest must contain {sorted(required_splits)}.")
    splits: dict[str, list[PatientCase]] = {}
    seen: set[str] = set()
    for split_name in ("train", "validation", "test"):
        cases: list[PatientCase] = []
        for item in payload[split_name]:
            patient_id = str(item["patient_id"])
            if patient_id in seen:
                raise ValueError(f"Patient appears in multiple splits: {patient_id}")
            seen.add(patient_id)
            cases.append(
                PatientCase(
                    patient_id=patient_id,
                    image_path=Path(item["image_path"]),
                    label_path=Path(item["label_path"]) if item["label_path"] else None,
                )
            )
        splits[split_name] = cases
    return splits


class MRISliceDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy 2D slice dataset; each patient remains in one supplied split."""

    def __init__(
        self,
        cases: list[PatientCase],
        image_size: tuple[int, int] = (256, 256),
        training: bool = False,
    ) -> None:
        self.cases = cases
        self.image_size = image_size
        self.training = training
        self._slices: list[tuple[PatientCase, int]] = []
        for case in cases:
            image = nib.load(str(case.image_path))
            depth = image.shape[2]
            self._slices.extend((case, index) for index in range(depth))

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case, slice_index = self._slices[index]
        image = np.asarray(nib.load(str(case.image_path)).dataobj, dtype=np.float32)
        image = image[:, :, slice_index, :]
        image = normalize_modalities(image)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        if case.label_path is not None:
            label = np.asarray(nib.load(str(case.label_path)).dataobj)
            mask = prepare_mask(label[:, :, slice_index])
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=self.image_size,
            mode="nearest",
        ).squeeze(0)
        return {"image": image_tensor, "mask": mask_tensor, "patient_id": case.patient_id}


def load_case_arrays(
    case: PatientCase,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load a case image, binary mask, and voxel spacing for inference/evaluation."""
    image = nib.load(str(case.image_path))
    image_array = normalize_modalities(np.asarray(image.dataobj, dtype=np.float32))
    mask_array = None
    if case.label_path is not None:
        mask_array = prepare_mask(np.asarray(nib.load(str(case.label_path)).dataobj))
    return image_array, mask_array, np.asarray(image.header.get_zooms()[:3], dtype=np.float32)
