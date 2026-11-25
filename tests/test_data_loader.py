from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from src.neuromap.data_loader import (
    MRISliceDataset,
    PatientGroupedBatchSampler,
    deduplicate_cases_by_label,
    discover_cases,
    load_case_arrays,
    load_split_manifest,
    save_split_manifest,
    split_cases,
)


def test_load_case_arrays_returns_arrays(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    image = np.zeros((4, 5, 2, 4), dtype=np.float32)
    image[1, 2, 0, 0] = 3
    label = np.zeros((4, 5, 2), dtype=np.int16)
    label[1, 2, 0] = 1
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(image, affine), images_dir / "case_001.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), labels_dir / "case_001.nii.gz")

    case = discover_cases(tmp_path)[0]
    loaded_image, loaded_mask, spacing = load_case_arrays(case)

    assert loaded_image.shape == image.shape
    assert loaded_mask is not None
    assert loaded_mask.shape == label.shape
    assert loaded_mask[1, 2, 0] == 1
    np.testing.assert_allclose(spacing, [1.0, 2.0, 3.0])


def test_discover_cases_ignores_macos_metadata(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    image = np.zeros((2, 2, 1, 4), dtype=np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(image, affine), images_dir / "case_001.nii.gz")
    (images_dir / "._case_001.nii.gz").write_bytes(b"metadata")

    cases = discover_cases(tmp_path)

    assert [case.patient_id for case in cases] == ["case_001"]


def test_split_manifest_round_trip_preserves_patient_membership(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    affine = np.eye(4)
    for index in range(6):
        name = f"case_{index:03d}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1, 4), dtype=np.float32), affine), images_dir / name)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.int16), affine), labels_dir / name)

    splits = split_cases(discover_cases(tmp_path), 0.5, 0.25, seed=42)
    manifest = tmp_path / "split.json"
    save_split_manifest(splits, manifest)
    loaded = load_split_manifest(manifest)

    assert {
        split_name: [case.patient_id for case in cases]
        for split_name, cases in loaded.items()
    } == {
        split_name: [case.patient_id for case in cases]
        for split_name, cases in splits.items()
    }


def test_deduplicate_cases_by_label_keeps_one_deterministic_case(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    affine = np.eye(4)
    shared_mask = np.zeros((2, 2, 1), dtype=np.int16)
    shared_mask[0, 0, 0] = 1
    unique_mask = np.zeros((2, 2, 1), dtype=np.int16)
    unique_mask[1, 1, 0] = 1
    for patient_id, mask in (
        ("case_002", shared_mask),
        ("case_001", shared_mask),
        ("case_003", unique_mask),
    ):
        nib.save(
            nib.Nifti1Image(np.full((2, 2, 1, 4), len(patient_id), dtype=np.float32), affine),
            images_dir / f"{patient_id}.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(mask, affine),
            labels_dir / f"{patient_id}.nii.gz",
        )

    unique_cases, removed_cases = deduplicate_cases_by_label(discover_cases(tmp_path))

    assert [case.patient_id for case in unique_cases] == ["case_001", "case_003"]
    assert [case.patient_id for case in removed_cases] == ["case_002"]


def test_patient_batch_sampler_shuffles_slices_without_mixing_patients(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    affine = np.eye(4)
    for index in range(3):
        image_name = f"case_{index:03d}.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 8, 4), dtype=np.float32), affine),
            images_dir / image_name,
        )
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 8), dtype=np.int16), affine),
            labels_dir / image_name,
        )
    dataset = MRISliceDataset(discover_cases(tmp_path), image_size=(2, 2))
    sampler = PatientGroupedBatchSampler(dataset, batch_size=3)

    with torch.random.fork_rng():
        torch.manual_seed(42)
        first_epoch = list(sampler)
        torch.manual_seed(43)
        second_epoch = list(sampler)

    assert len(first_epoch) == len(sampler)
    assert set(index for batch in first_epoch for index in batch) == set(range(len(dataset)))
    assert all(
        len({dataset._slices[index][0].patient_id for index in batch}) == 1
        for batch in first_epoch
    )
    assert first_epoch != second_epoch


def test_slice_dataset_reuses_patient_volume(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    affine = np.eye(4)
    image_path = images_dir / "case_001.nii.gz"
    label_path = labels_dir / "case_001.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 5, 2, 4), dtype=np.float32), affine), image_path)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 2), dtype=np.int16), affine), label_path)
    case = discover_cases(tmp_path)[0]
    dataset = MRISliceDataset([case], image_size=(4, 5))

    with patch("src.neuromap.data_loader.nib.load", wraps=nib.load) as load:
        dataset[0]
        dataset[1]

    assert load.call_count == 2


def test_slice_dataset_can_select_flair_only(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    image = np.zeros((4, 5, 1, 4), dtype=np.float32)
    for channel in range(4):
        image[:, :, 0, channel] = np.arange(20, dtype=np.float32).reshape(4, 5) * (
            channel + 1
        )
    label = np.zeros((4, 5, 1), dtype=np.int16)
    image_path = images_dir / "case_001.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), labels_dir / image_path.name)
    case = discover_cases(tmp_path)[0]
    all_modalities = MRISliceDataset([case], image_size=(4, 5))[0]["image"]
    flair_only = MRISliceDataset(
        [case],
        image_size=(4, 5),
        modalities=("flair",),
    )[0]["image"]

    assert flair_only.shape == (1, 4, 5)
    torch.testing.assert_close(flair_only, all_modalities[3:4])


def test_training_augmentation_flips_image_and_mask_together(tmp_path: Path) -> None:
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    image = np.arange(4 * 5 * 4, dtype=np.float32).reshape(4, 5, 1, 4) + 1
    label = np.zeros((4, 5, 1), dtype=np.int16)
    label[1:3, 2:4, 0] = 1
    image_path = images_dir / "case_001.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), labels_dir / image_path.name)
    case = discover_cases(tmp_path)[0]
    reference = MRISliceDataset([case], image_size=(4, 5))[0]
    training = MRISliceDataset([case], image_size=(4, 5), training=True)

    with patch("src.neuromap.data_loader.torch.rand", return_value=torch.tensor(0.0)) as rand:
        augmented = training[0]

    torch.testing.assert_close(
        augmented["image"],
        torch.flip(reference["image"], dims=(1, 2)),
    )
    torch.testing.assert_close(
        augmented["mask"],
        torch.flip(reference["mask"], dims=(1, 2)),
    )
    assert rand.call_count == 2
