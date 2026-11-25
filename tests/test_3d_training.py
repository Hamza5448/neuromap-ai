import nibabel as nib
import numpy as np
import pytest
import torch

from src.neuromap.data_loader import discover_cases
from src.neuromap.data_loader_3d import (
    MRIVolumePatchDataset,
    PatientPatchBatchSampler,
)
from src.neuromap.inference_3d import infer_volume_probabilities_3d
from src.neuromap.inference_3d import load_checkpoint_3d
from src.neuromap.model_3d import build_unet_3d, predict_logits_3d
from src.neuromap.training_3d import (
    evaluate_epoch_3d,
    segmentation_loss_3d,
    train_epoch_3d,
)


def _make_case(tmp_path, patient_id="case_001"):
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    image = np.arange(8 * 10 * 6 * 4, dtype=np.float32).reshape(8, 10, 6, 4)
    label = np.zeros((8, 10, 6), dtype=np.int16)
    label[2:5, 3:7, 1:4] = 1
    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        images_dir / f"{patient_id}.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(label, np.eye(4)),
        labels_dir / f"{patient_id}.nii.gz",
    )
    return discover_cases(tmp_path)


def test_3d_patch_dataset_pads_and_returns_aligned_channels(tmp_path):
    cases = _make_case(tmp_path)
    dataset = MRIVolumePatchDataset(
        cases,
        patch_size=(12, 12, 8),
        patches_per_case=2,
        training=False,
        modalities=("flair",),
    )

    patch = dataset[0]

    assert patch["image"].shape == (1, 12, 12, 8)
    assert patch["mask"].shape == (1, 12, 12, 8)
    assert patch["mask"].sum() > 0
    assert patch["patient_id"] == "case_001"


def test_patient_patch_sampler_keeps_each_batch_within_one_patient(tmp_path):
    _make_case(tmp_path)
    _make_case(tmp_path, "case_002")
    cases = discover_cases(tmp_path)
    dataset = MRIVolumePatchDataset(
        cases,
        patch_size=(4, 4, 4),
        patches_per_case=3,
    )
    sampler = PatientPatchBatchSampler(dataset, batch_size=2)
    batches = list(sampler)

    assert len(batches) == 4
    assert all(
        len({dataset.cases[index // 3].patient_id for index in batch}) == 1
        for batch in batches
    )
    assert sorted(index for batch in batches for index in batch) == list(range(6))


def test_3d_unet_and_loss_have_expected_contract():
    model = build_unet_3d(
        in_channels=1,
        channels=(2, 4, 8),
    )
    images = torch.randn(1, 1, 16, 16, 16)
    masks = torch.zeros(1, 1, 16, 16, 16)
    logits = predict_logits_3d(model, images)

    assert logits.shape == masks.shape
    assert torch.isfinite(segmentation_loss_3d(logits, masks))
    with pytest.raises(ValueError, match="Expected volumes"):
        predict_logits_3d(model, torch.randn(1, 1, 16, 16))


def test_3d_sliding_window_inference_reconstructs_full_probability_volume():
    class PositiveLogitModel(torch.nn.Module):
        def forward(self, images):
            return torch.ones(
                (images.shape[0], 1, *images.shape[2:]),
                device=images.device,
            )

    image = np.zeros((9, 11, 7, 1), dtype=np.float32)
    probabilities = infer_volume_probabilities_3d(
        PositiveLogitModel(),
        image,
        patch_size=(8, 8, 8),
        device=torch.device("cpu"),
        overlap=0.5,
    )

    assert probabilities.shape == image.shape[:3]
    np.testing.assert_allclose(probabilities, 1 / (1 + np.exp(-1)), rtol=1e-5)


def test_3d_train_and_validation_loops_run_with_fixed_patches():
    model = torch.nn.Conv3d(1, 1, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "image": torch.randn(1, 1, 8, 8, 8),
        "mask": torch.zeros(1, 1, 8, 8, 8),
    }

    train_loss = train_epoch_3d(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        mixed_precision=False,
    )
    validation_loss = evaluate_epoch_3d(
        model,
        [batch],
        torch.device("cpu"),
        mixed_precision=False,
    )

    assert np.isfinite(train_loss)
    assert np.isfinite(validation_loss)


def test_3d_checkpoint_loader_reconstructs_model(tmp_path):
    source = build_unet_3d(in_channels=1, channels=(2, 4))
    checkpoint_path = tmp_path / "model_3d.pt"
    torch.save(
        {
            "architecture": "3d_unet",
            "model_state_dict": source.state_dict(),
        },
        checkpoint_path,
    )

    loaded = load_checkpoint_3d(
        checkpoint_path,
        torch.device("cpu"),
        in_channels=1,
        channels=(2, 4),
    )

    assert not loaded.training
    assert loaded(torch.randn(1, 1, 8, 8, 8)).shape == (1, 1, 8, 8, 8)
