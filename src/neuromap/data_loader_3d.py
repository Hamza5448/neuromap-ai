"""Patient-grouped 3D MRI patch loading for volumetric training."""

from __future__ import annotations

from typing import Iterator

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, Sampler

from .data_loader import PatientCase
from .preprocessing import normalize_modalities, prepare_mask


_MRI_MODALITY_INDICES = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}


class MRIVolumePatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample fixed-size 3D patches while keeping scans cached patient-by-patient."""

    def __init__(
        self,
        cases: list[PatientCase],
        patch_size: tuple[int, int, int] = (96, 96, 64),
        patches_per_case: int = 2,
        training: bool = True,
        modalities: tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
    ) -> None:
        if not cases:
            raise ValueError("At least one patient case is required.")
        if len(patch_size) != 3 or any(size < 1 for size in patch_size):
            raise ValueError("Patch size must contain three positive dimensions.")
        if patches_per_case < 1:
            raise ValueError("Patches per case must be positive.")
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("Select one or more distinct MRI modalities.")
        unknown_modalities = set(modalities) - _MRI_MODALITY_INDICES.keys()
        if unknown_modalities:
            raise ValueError(f"Unknown MRI modalities: {sorted(unknown_modalities)}")
        if any(case.label_path is None for case in cases):
            raise ValueError("3D supervised training requires labels for every case.")

        self.cases = cases
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.training = training
        self.modality_indices = [_MRI_MODALITY_INDICES[name] for name in modalities]
        self._cached_case_index: int | None = None
        self._cached_image: torch.Tensor | None = None
        self._cached_mask: torch.Tensor | None = None
        self._foreground_coordinates: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.cases) * self.patches_per_case

    def _load_case(self, case_index: int) -> None:
        case = self.cases[case_index]
        if case.label_path is None:
            raise ValueError(f"3D training case {case.patient_id} has no label.")
        image_nii = nib.load(str(case.image_path))
        label_nii = nib.load(str(case.label_path))
        if image_nii.ndim != 4 or image_nii.shape[-1] != 4:
            raise ValueError(
                f"Expected four MRI modalities for patient {case.patient_id}."
            )
        if tuple(image_nii.shape[:3]) != tuple(label_nii.shape):
            raise ValueError(f"Image and label shapes differ for {case.patient_id}.")
        if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-4):
            raise ValueError(f"Image and label affines differ for {case.patient_id}.")

        image = np.asarray(image_nii.dataobj, dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"Image contains non-finite values for {case.patient_id}.")
        image = normalize_modalities(image)[..., self.modality_indices]
        mask = prepare_mask(np.asarray(label_nii.dataobj))
        self._cached_image = torch.from_numpy(
            np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        )
        self._cached_mask = torch.from_numpy(mask).unsqueeze(0)
        self._foreground_coordinates = torch.from_numpy(
            np.argwhere(mask > 0).astype(np.int64, copy=False)
        )
        self._cached_case_index = case_index

    def _ensure_patch_size(self) -> None:
        if self._cached_image is None or self._cached_mask is None:
            raise RuntimeError("Patient volume was not loaded.")
        height, width, depth = self._cached_image.shape[1:]
        patch_height, patch_width, patch_depth = self.patch_size
        pad_height = max(0, patch_height - height)
        pad_width = max(0, patch_width - width)
        pad_depth = max(0, patch_depth - depth)
        if pad_height or pad_width or pad_depth:
            padding = (0, pad_depth, 0, pad_width, 0, pad_height)
            self._cached_image = functional.pad(self._cached_image, padding)
            self._cached_mask = functional.pad(self._cached_mask, padding)

    def _crop_start(self) -> tuple[int, int, int]:
        if self._cached_image is None:
            raise RuntimeError("Patient volume was not loaded.")
        spatial_shape = self._cached_image.shape[1:]
        foreground = self._foreground_coordinates
        center: tuple[int, int, int] | None = None
        if foreground is not None and foreground.numel() > 0:
            if self.training:
                use_foreground = bool(torch.rand(()) < 0.5)
                if use_foreground:
                    point = foreground[torch.randint(len(foreground), ())]
                    center = tuple(int(value) for value in point.tolist())
            else:
                center = tuple(
                    int(value)
                    for value in foreground.float().mean(dim=0).round().tolist()
                )

        starts: list[int] = []
        for axis, (dimension, patch_dimension) in enumerate(
            zip(spatial_shape, self.patch_size)
        ):
            last_start = dimension - patch_dimension
            if center is None:
                start = (
                    int(torch.randint(last_start + 1, ()).item())
                    if self.training
                    else last_start // 2
                )
            else:
                start = center[axis] - patch_dimension // 2
                start = min(max(start, 0), last_start)
            starts.append(start)
        return tuple(starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        case_index = index // self.patches_per_case
        if case_index != self._cached_case_index:
            self._load_case(case_index)
        self._ensure_patch_size()
        if self._cached_image is None or self._cached_mask is None:
            raise RuntimeError("Patient volume was not loaded.")

        start_height, start_width, start_depth = self._crop_start()
        patch_height, patch_width, patch_depth = self.patch_size
        image = self._cached_image[
            :,
            start_height : start_height + patch_height,
            start_width : start_width + patch_width,
            start_depth : start_depth + patch_depth,
        ]
        mask = self._cached_mask[
            :,
            start_height : start_height + patch_height,
            start_width : start_width + patch_width,
            start_depth : start_depth + patch_depth,
        ]
        if self.training:
            for dimension in (1, 2):
                if torch.rand(()) < 0.5:
                    image = torch.flip(image, dims=(dimension,))
                    mask = torch.flip(mask, dims=(dimension,))
        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "patient_id": self.cases[case_index].patient_id,
        }


class PatientPatchBatchSampler(Sampler[list[int]]):
    """Shuffle patients while keeping each batch within one cached volume."""

    def __init__(
        self,
        dataset: MRIVolumePatchDataset,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        patient_order = torch.randperm(len(self.dataset.cases)).tolist()
        for patient_index in patient_order:
            start = patient_index * self.dataset.patches_per_case
            indices = list(
                range(start, start + self.dataset.patches_per_case)
            )
            patch_order = torch.randperm(len(indices)).tolist()
            shuffled_indices = [indices[index] for index in patch_order]
            for offset in range(0, len(shuffled_indices), self.batch_size):
                yield shuffled_indices[offset : offset + self.batch_size]

    def __len__(self) -> int:
        batches_per_case = (
            self.dataset.patches_per_case + self.batch_size - 1
        ) // self.batch_size
        return len(self.dataset.cases) * batches_per_case
