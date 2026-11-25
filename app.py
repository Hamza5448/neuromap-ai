"""Streamlit workflow for multimodal MRI segmentation and volume export."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st
import torch
import yaml

from src.neuromap.application import (
    grayscale_rgb,
    load_mri_array,
    load_nifti_bytes,
    load_reference_mask,
    mask_nifti_bytes,
    predict_volume,
    segmentation_overlay,
    summarize_prediction,
    validate_multimodal_image,
)
from src.neuromap.data_loader import load_split_manifest
from src.neuromap.inference import load_checkpoint
from src.neuromap.inference_3d import (
    infer_volume_probabilities_3d,
    load_checkpoint_3d,
)
from src.neuromap.preprocessing import normalize_modalities

_MODEL_LABELS = {
    "baseline": "Unaugmented multimodal 2D U-Net",
    "augmented": "Augmented multimodal 2D U-Net",
    "flair": "FLAIR-only 2D U-Net",
    "3d": "3D U-Net",
}


@st.cache_resource(show_spinner=False)
def _cached_model(
    checkpoint_path: str,
    checkpoint_mtime_ns: int,
    in_channels: int,
    out_channels: int,
) -> tuple[torch.nn.Module, torch.device]:
    del checkpoint_mtime_ns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(
        checkpoint_path,
        device,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return model, device


@st.cache_resource(show_spinner=False)
def _cached_3d_model(
    checkpoint_path: str,
    checkpoint_mtime_ns: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
) -> tuple[torch.nn.Module, torch.device]:
    del checkpoint_mtime_ns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_3d(
        checkpoint_path,
        device,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
    )
    return model, device


def _completed_epochs(checkpoint_path: Path) -> int | None:
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_history.csv")
    if not history_path.is_file():
        return None
    last_epoch = 0
    with history_path.open(encoding="utf-8", newline="") as history_file:
        for row in csv.reader(history_file):
            if not row or row[0] == "epoch":
                continue
            try:
                last_epoch = int(row[0])
            except ValueError as error:
                raise ValueError(
                    f"Invalid epoch in training history: {history_path}"
                ) from error
    return last_epoch


def _default_model_choice(comparison_path: Path) -> str:
    if not comparison_path.is_file():
        return _MODEL_LABELS["baseline"]
    try:
        report = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read validation model comparison: {comparison_path}"
        ) from error
    if not isinstance(report, dict):
        raise ValueError("The validation model comparison must be a JSON object.")
    if report.get("selection_split") != "validation":
        raise ValueError("The app model default must come from validation results.")
    try:
        return _MODEL_LABELS[report["selected_model"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown selected model in comparison report: {error.args[0]}"
        ) from error


def _read_validation_threshold(report_path: Path, configured_threshold: float) -> float:
    if not report_path.is_file():
        return configured_threshold
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read validation threshold report: {report_path}"
        ) from error
    if not isinstance(report, dict):
        raise ValueError("The validation threshold report must be a JSON object.")
    if report.get("split") != "validation":
        raise ValueError("The app threshold must come from validation results.")
    try:
        threshold = float(report["selected_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not read validation threshold report: {report_path}"
        ) from error
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid selected threshold in {report_path}.")
    return threshold


def _read_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if len(config["data"]["modalities"]) != config["model"]["in_channels"]:
        raise ValueError("Configured modalities must match the model input channels.")
    modalities = tuple(config["data"]["modalities"])
    if modalities not in (
        ("t1", "t1ce", "t2", "flair"),
        ("flair",),
    ):
        raise ValueError(
            "The app supports four-modality or FLAIR-only MRI configurations."
        )
    return config


def _load_uploaded_nifti(uploaded_file) -> nib.Nifti1Image:
    try:
        return load_nifti_bytes(uploaded_file.getvalue())
    except (
        EOFError,
        OSError,
        ValueError,
        gzip.BadGzipFile,
        nib.filebasedimages.ImageFileError,
    ) as error:
        raise ValueError(f"Could not read the uploaded NIfTI file: {error}") from error


def _load_preloaded_case():
    manifest_path = Path("artifacts/splits/task01_brain_tumour_deduplicated.json")
    if not manifest_path.is_file():
        st.error(f"Preloaded sample manifest not found: {manifest_path}")
        return None
    cases = [
        case
        for case in load_split_manifest(manifest_path)["validation"]
        if case.image_path.is_file() and case.label_path is not None
    ]
    if not cases:
        st.error("No preloaded validation samples are available at the manifest paths.")
        return None
    st.caption("Preloaded samples are drawn from validation; the held-out test split is reserved.")
    selected_id = st.selectbox(
        "Preloaded validation patient",
        [case.patient_id for case in cases],
    )
    return next(case for case in cases if case.patient_id == selected_id)


def _summary_csv(
    patient_id: str,
    threshold: float,
    summary: dict[str, float],
    spacing_mm: tuple[float, float, float],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("patient_id", "threshold", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", *summary))
    writer.writerow(
        (
            patient_id,
            threshold,
            *spacing_mm,
            *summary.values(),
        )
    )
    return output.getvalue().encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="NeuroMap AI", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --nm-ink: #edf5ff;
            --nm-muted: #9db0c5;
            --nm-line: rgba(159, 189, 218, 0.16);
            --nm-cyan: #62d7cf;
            --nm-blue: #79a9ff;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(32, 113, 140, 0.20), transparent 34rem),
                linear-gradient(160deg, #0a1623 0%, #0c1928 52%, #101d2d 100%);
        }
        [data-testid="stHeader"] { background: rgba(10, 22, 35, 0.72); }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1725, #101e2d);
            border-right: 1px solid var(--nm-line);
        }
        h1, h2, h3 { color: var(--nm-ink); letter-spacing: -0.025em; }
        p, label, [data-testid="stCaptionContainer"] {
            color: var(--nm-muted);
        }
        .nm-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 2rem;
            padding: 2rem 2.2rem;
            margin: 0 0 1.4rem;
            border: 1px solid rgba(98, 215, 207, 0.24);
            border-radius: 22px;
            background:
                radial-gradient(circle at 85% 50%, rgba(98, 215, 207, 0.15), transparent 15rem),
                linear-gradient(125deg, rgba(22, 42, 59, 0.96), rgba(15, 33, 49, 0.94));
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
        }
        .nm-eyebrow {
            margin: 0 0 0.65rem;
            color: var(--nm-cyan);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.16em;
        }
        .nm-hero h1 {
            margin: 0;
            color: var(--nm-ink);
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.08;
            font-weight: 760;
        }
        .nm-subtitle {
            max-width: 46rem;
            margin: 0.8rem 0 0;
            color: #b6c6d8;
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .nm-badge {
            flex: 0 0 auto;
            padding: 0.7rem 1rem;
            border: 1px solid rgba(121, 169, 255, 0.34);
            border-radius: 999px;
            color: #cfe0ff;
            background: rgba(121, 169, 255, 0.10);
            font-size: 0.85rem;
            font-weight: 650;
        }
        [data-testid="stMetric"] {
            min-height: 7rem;
            padding: 1.05rem 1.15rem;
            border: 1px solid var(--nm-line);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(24, 43, 61, 0.92), rgba(17, 33, 49, 0.92));
        }
        [data-testid="stMetricLabel"] { color: #aabbd0; }
        [data-testid="stMetricValue"] { color: var(--nm-ink); }
        [data-testid="stAlert"] p { color: #263547 !important; font-weight: 600; }
        [data-testid="stFileUploader"] section,
        [data-testid="stImage"] img {
            border-radius: 14px;
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(121, 169, 255, 0.42);
            background: rgba(17, 34, 50, 0.64);
        }
        .stButton > button, [data-testid="stDownloadButton"] > button {
            min-height: 2.8rem;
            border: 1px solid rgba(98, 215, 207, 0.38);
            border-radius: 11px;
            color: #eafffd;
            background: linear-gradient(110deg, #147f83, #226da0);
            font-weight: 680;
            transition: filter 120ms ease, transform 120ms ease;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
            border-color: var(--nm-cyan);
            filter: brightness(1.12);
            transform: translateY(-1px);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        @media (max-width: 700px) {
            .nm-hero { align-items: flex-start; flex-direction: column; padding: 1.5rem; }
            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1.1rem 1rem 2rem;
            }
        }
        </style>
        <section class="nm-hero">
          <div>
            <p class="nm-eyebrow">NEUROMAP · RESEARCH WORKSPACE</p>
            <h1>Brain MRI, made more understandable.</h1>
            <p class="nm-subtitle">Explore multimodal scans, review segmentation overlays, and inspect volume measurements in one focused workspace.</p>
          </div>
          <div class="nm-badge">2D / 3D U-Net · Research prototype</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Not a medical device. Segmentation and measurements are research outputs, "
        "not diagnoses or treatment advice."
    )

    config_2d = Path(
        os.environ.get("NEUROMAP_CONFIG", "configs/task_a_multimodal.yaml")
    )
    checkpoint_2d = Path(
        os.environ.get(
            "NEUROMAP_CHECKPOINT",
            "artifacts/checkpoints/baseline_deduplicated.pt",
        )
    )
    config_augmented = Path(
        os.environ.get(
            "NEUROMAP_AUGMENTED_CONFIG",
            "configs/task_a_multimodal.yaml",
        )
    )
    checkpoint_augmented = Path(
        os.environ.get(
            "NEUROMAP_AUGMENTED_CHECKPOINT",
            "artifacts/checkpoints/baseline_augmented.pt",
        )
    )
    config_flair = Path(
        os.environ.get("NEUROMAP_FLAIR_CONFIG", "configs/task_a_flair.yaml")
    )
    checkpoint_flair = Path(
        os.environ.get(
            "NEUROMAP_FLAIR_CHECKPOINT",
            "artifacts/checkpoints/flair_deduplicated.pt",
        )
    )
    config_3d = Path(
        os.environ.get("NEUROMAP_3D_CONFIG", "configs/task_a_3d.yaml")
    )
    checkpoint_3d = Path(
        os.environ.get(
            "NEUROMAP_3D_CHECKPOINT",
            "artifacts/checkpoints/unet_3d_deduplicated.pt",
        )
    )
    model_specs = {
        _MODEL_LABELS["baseline"]: (
            config_2d,
            checkpoint_2d,
            Path("reports/task_a_multimodal_unaugmented_thresholds.json"),
        ),
        _MODEL_LABELS["augmented"]: (
            config_augmented,
            checkpoint_augmented,
            Path("reports/task_a_multimodal_augmented_thresholds.json"),
        ),
        _MODEL_LABELS["flair"]: (
            config_flair,
            checkpoint_flair,
            Path("reports/task_a_flair_thresholds.json"),
        ),
        _MODEL_LABELS["3d"]: (
            config_3d,
            checkpoint_3d,
            Path("reports/task_a_3d_thresholds.json"),
        ),
    }
    comparison_path = Path("reports/task_a_validation_model_comparison.json")
    try:
        default_model_choice = _default_model_choice(comparison_path)
    except ValueError as error:
        st.error(str(error))
        return
    with st.sidebar:
        st.header("Inference settings")
        if not comparison_path.is_file():
            st.info(
                "Validation comparison is unavailable; defaulting to the "
                "unaugmented multimodal model."
            )
        model_choice = st.radio(
            "Segmentation model",
            tuple(model_specs),
            index=tuple(model_specs).index(default_model_choice),
        )
        config_path, checkpoint_path, threshold_report_path = model_specs[model_choice]
        try:
            config = _read_config(config_path)
            default_threshold = _read_validation_threshold(
                threshold_report_path,
                float(config["evaluation"]["threshold"]),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            st.error(str(error))
            return
        if not threshold_report_path.is_file():
            st.info(
                "No validation threshold report is available for this model; "
                "using the configured threshold without validation confirmation."
            )
        threshold = st.slider(
            "Probability threshold",
            min_value=0.1,
            max_value=0.9,
            value=default_threshold,
            step=0.05,
            key=f"threshold_{model_choice}",
        )
        st.caption(f"Checkpoint: {checkpoint_path.name}")
        if model_choice == "3D U-Net":
            st.caption(
                f"Patch: {tuple(config['data']['patch_size'])} · "
                f"Overlap: {config['evaluation']['sliding_window_overlap']:.0%}"
            )

    st.markdown("### 01 · Choose a study")
    st.caption("Upload a four-modality NIfTI or explore a preloaded validation sample.")
    source = st.radio(
        "Study source",
        ("Upload a 4D NIfTI", "Preloaded sample"),
        horizontal=True,
    )
    image: nib.Nifti1Image | None = None
    reference_image: nib.Nifti1Image | None = None
    patient_id = "uploaded-study"
    uploaded_reference = None
    upload_fingerprint: str | None = None

    if source == "Upload a 4D NIfTI":
        uploaded_image = st.file_uploader(
            "MRI volume (.nii or .nii.gz)",
            type=("nii", "gz"),
            help="One 4D NIfTI with the last axis ordered T1, T1ce, T2, FLAIR.",
        )
        uploaded_reference = st.file_uploader(
            "Optional reference mask (.nii or .nii.gz)",
            type=("nii", "gz"),
        )
        confirmed_order = st.checkbox(
            "I confirm the input channel order is T1, T1ce, T2, FLAIR."
        )
        if uploaded_image is None:
            st.info("Upload a compatible 4D NIfTI volume to begin.")
            return
        patient_id = Path(uploaded_image.name).name
        upload_fingerprint = hashlib.sha256(uploaded_image.getvalue()).hexdigest()
        try:
            image = _load_uploaded_nifti(uploaded_image)
            if uploaded_reference is not None:
                reference_image = _load_uploaded_nifti(uploaded_reference)
        except ValueError as error:
            st.error(str(error))
            return
        if not confirmed_order:
            st.info("Confirm the modality order before running inference.")
            return
    else:
        selected_case = _load_preloaded_case()
        if selected_case is None:
            return
        patient_id = selected_case.patient_id
        image = nib.load(str(selected_case.image_path))
        reference_image = nib.load(str(selected_case.label_path))

    if image is None:
        st.error("No MRI volume is selected.")
        return
    try:
        spacing_mm = validate_multimodal_image(image)
        image_array = load_mri_array(image)
        reference_mask = (
            load_reference_mask(reference_image, image)
            if reference_image is not None
            else None
        )
    except ValueError as error:
        st.error(str(error))
        return

    st.write(
        f"Volume shape: {image_array.shape[:3]} | "
        f"Voxel spacing: {spacing_mm} mm | "
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )
    checkpoint_exists = checkpoint_path.is_file()
    try:
        completed_epochs = (
            _completed_epochs(checkpoint_path) if checkpoint_exists else None
        )
    except (OSError, ValueError) as error:
        st.error(f"Could not verify training completion: {error}")
        return
    training_complete = (
        completed_epochs is not None
        and completed_epochs >= int(config["training"]["epochs"])
    )
    if not checkpoint_exists:
        st.error(
            f"Trained checkpoint not found: {checkpoint_path}. "
            "Train this model before running inference."
        )
    elif completed_epochs is None:
        st.info(
            "The matching training history is missing, so training completion "
            "cannot be verified."
        )
    elif not training_complete:
        st.info(
            f"Training is still in progress ({completed_epochs}/"
            f"{config['training']['epochs']} epochs). Inference unlocks when "
            "the configured training run is complete."
        )

    if upload_fingerprint is not None:
        source_key = upload_fingerprint
    else:
        source_key = hashlib.sha256(
            (
                patient_id
                + str(image.shape)
                + str(tuple(np.round(image.affine.ravel(), 6)))
            ).encode("utf-8")
        ).hexdigest()
    result_key = f"result_{model_choice}_{source_key}_{threshold:.2f}"
    if st.button("Run segmentation", disabled=not training_complete):
        try:
            if model_choice != "3D U-Net":
                model, device = _cached_model(
                    str(checkpoint_path),
                    checkpoint_path.stat().st_mtime_ns,
                    config["model"]["in_channels"],
                    config["model"]["out_channels"],
                )
                with st.spinner("Running slice-based volume inference..."):
                    prediction = predict_volume(
                        model,
                        image_array,
                        tuple(config["data"]["image_size"]),
                        threshold,
                        config["training"]["batch_size"],
                        device,
                        modalities=tuple(config["data"]["modalities"]),
                    )
            else:
                model, device = _cached_3d_model(
                    str(checkpoint_path),
                    checkpoint_path.stat().st_mtime_ns,
                    config["model"]["in_channels"],
                    config["model"]["out_channels"],
                    tuple(config["model"]["channels"]),
                )
                with st.spinner("Running overlapping 3D sliding-window inference..."):
                    probabilities = infer_volume_probabilities_3d(
                        model,
                        normalize_modalities(image_array),
                        tuple(config["data"]["patch_size"]),
                        device,
                        float(config["evaluation"]["sliding_window_overlap"]),
                        int(config["evaluation"]["sliding_window_batch_size"]),
                    )
                    prediction = probabilities >= threshold
            st.session_state[result_key] = prediction
        except (RuntimeError, ValueError) as error:
            st.error(f"Inference failed: {error}")
            return

    prediction = st.session_state.get(result_key)
    if prediction is None:
        return

    summary = summarize_prediction(prediction, spacing_mm, reference_mask)
    st.markdown("### 02 · Review segmentation")
    st.caption("Navigate the scan and compare the model overlay with an optional reference mask.")
    st.subheader("Measurement")
    st.metric("Predicted whole-tumor volume", f"{summary['predicted_volume_ml']:.2f} mL")
    if reference_mask is not None:
        columns = st.columns(4)
        for column, metric in zip(
            columns,
            ("dice", "iou", "sensitivity", "volumetric_error_ml"),
        ):
            suffix = " mL" if metric == "volumetric_error_ml" else ""
            column.metric(metric.replace("_", " ").title(), f"{summary[metric]:.4f}{suffix}")

    modality_names = ("T1", "T1ce", "T2", "FLAIR")
    control_columns = st.columns((1, 2))
    modality_index = modality_names.index(
        control_columns[0].selectbox("Display modality", modality_names, index=3)
    )
    slice_index = control_columns[1].slider(
        "Axial slice",
        min_value=0,
        max_value=image_array.shape[2] - 1,
        value=image_array.shape[2] // 2,
    )
    base = grayscale_rgb(image_array[:, :, slice_index, modality_index])
    overlay = segmentation_overlay(
        base,
        prediction[:, :, slice_index],
        reference_mask[:, :, slice_index] if reference_mask is not None else None,
    )
    left, right = st.columns(2)
    left.image(base, caption=f"{modality_names[modality_index]} MRI", use_container_width=True)
    right.image(
        overlay,
        caption=(
            "Reference overlap: yellow=match, red=false positive, cyan=false negative"
            if reference_mask is not None
            else "Prediction overlay: red=predicted tumor"
        ),
        use_container_width=True,
    )

    st.markdown("### 03 · Export results")
    export_columns = st.columns(2)
    export_columns[0].download_button(
        "Download segmentation mask (.nii)",
        data=mask_nifti_bytes(prediction, image),
        file_name=f"{Path(patient_id).stem}_neuromap_mask.nii",
        mime="application/octet-stream",
        use_container_width=True,
    )
    export_columns[1].download_button(
        "Download measurement summary (.csv)",
        data=_summary_csv(patient_id, threshold, summary, spacing_mm),
        file_name=f"{Path(patient_id).stem}_neuromap_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
