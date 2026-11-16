from pathlib import Path

import nibabel as nib
import numpy as np

from src.neuromap.data_loader import (
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
