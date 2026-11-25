import gzip

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from src.neuromap.application import (
    load_mri_array,
    load_nifti_bytes,
    load_reference_mask,
    mask_nifti_bytes,
    predict_volume,
    segmentation_overlay,
    summarize_prediction,
    validate_multimodal_image,
)
from src.neuromap.preprocessing import normalize_modalities


class PositiveLogitModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            device=images.device,
        )


class SingleChannelLogitModel(PositiveLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[1] == 1
        self.inputs = images.detach().cpu()
        return super().forward(images)


def test_load_nifti_bytes_reads_gzipped_multimodal_volume():
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    source = nib.Nifti1Image(np.zeros((3, 4, 2, 4), dtype=np.float32), affine)

    loaded = load_nifti_bytes(gzip.compress(source.to_bytes()))

    assert loaded.shape == source.shape
    np.testing.assert_allclose(loaded.affine, affine)
    assert validate_multimodal_image(loaded) == (1.0, 2.0, 3.0)


def test_load_nifti_bytes_rejects_corrupt_gzip_stream():
    payload = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xffgarbage"

    with pytest.raises(ValueError, match="gzip-compressed"):
        load_nifti_bytes(payload)


def test_multimodal_input_validation_rejects_wrong_channel_count():
    image = nib.Nifti1Image(np.zeros((3, 4, 2, 3), dtype=np.float32), np.eye(4))

    with pytest.raises(ValueError, match="4D NIfTI"):
        validate_multimodal_image(image)


def test_reference_mask_must_match_source_geometry():
    image = nib.Nifti1Image(np.zeros((3, 4, 2, 4), dtype=np.float32), np.eye(4))
    reference = nib.Nifti1Image(np.zeros((3, 4, 2), dtype=np.int16), np.diag([2, 2, 2, 1]))

    with pytest.raises(ValueError, match="affine"):
        load_reference_mask(reference, image)


def test_predict_volume_restores_binary_prediction_to_native_shape():
    image = np.zeros((3, 5, 2, 4), dtype=np.float32)

    prediction = predict_volume(
        PositiveLogitModel(),
        image,
        image_size=(8, 8),
        threshold=0.5,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert prediction.shape == (3, 5, 2)
    assert prediction.dtype == np.bool_
    assert prediction.all()


def test_predict_volume_supports_flair_only_models():
    image = np.zeros((2, 2, 1, 4), dtype=np.float32)
    image[:, :, 0, 0] = [[1, 2], [3, 4]]
    image[:, :, 0, 3] = [[4, 3], [2, 1]]
    model = SingleChannelLogitModel()

    prediction = predict_volume(
        model,
        image,
        image_size=(2, 2),
        threshold=0.5,
        batch_size=1,
        device=torch.device("cpu"),
        modalities=("flair",),
    )

    assert prediction.shape == (2, 2, 1)
    assert prediction.all()
    assert model.inputs is not None
    expected_flair = normalize_modalities(image[:, :, 0, :])[:, :, 3]
    np.testing.assert_allclose(model.inputs[0, 0].numpy(), expected_flair)


def test_prediction_summary_and_nifti_export_preserve_physical_geometry():
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    image = nib.Nifti1Image(np.zeros((2, 2, 1, 4), dtype=np.float32), affine)
    reference = np.zeros((2, 2, 1), dtype=bool)
    prediction = np.zeros_like(reference)
    prediction[0, 0, 0] = True
    scores = summarize_prediction(prediction, (1.0, 2.0, 3.0), reference)

    exported = load_nifti_bytes(mask_nifti_bytes(prediction, image))

    assert scores["predicted_volume_ml"] == pytest.approx(0.006)
    assert scores["reference_volume_ml"] == 0.0
    assert scores["volumetric_error_ml"] == pytest.approx(0.006)
    assert exported.shape == reference.shape
    assert exported.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_allclose(exported.affine, affine)
    np.testing.assert_array_equal(np.asarray(exported.dataobj), prediction.astype(np.uint8))


def test_segmentation_overlay_distinguishes_false_positives_and_negatives():
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    prediction = np.array([[True, False], [False, False]])
    reference = np.array([[True, False], [False, True]])

    overlay = segmentation_overlay(base, prediction, reference)

    np.testing.assert_array_equal(overlay[0, 0], [255, 210, 0])
    np.testing.assert_array_equal(overlay[1, 1], [0, 220, 255])
    np.testing.assert_array_equal(overlay[0, 1], [0, 0, 0])
