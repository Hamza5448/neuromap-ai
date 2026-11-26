# NeuroMap AI

NeuroMap AI is a software-based research prototype for brain MRI tumor-region
segmentation and volumetric analysis. The project compares unaugmented and
augmented multimodal 2D U-Nets, a FLAIR-only 2D U-Net, and a patch-based
multimodal 3D U-Net through a Streamlit interface.

The project is not a medical diagnostic or treatment system.

## Project team

- Fizza Fatima (24i-3077)
- Irtaza Abid (24i-3085)
- Hamza Ahmad (24i-3104)

## Stage A scope

- Medical Segmentation Decathlon `Task01_BrainTumour`.
- Four MRI modalities in NIfTI format.
- Binary whole-tumor-region segmentation.
- Patient-level train, validation, and test splits.
- Validation-only threshold and model selection before held-out testing.
- Held-out evaluation using segmentation and volumetric metrics.
- Streamlit visualization, measurement, and export workflow.

## Project structure

```text
NeuroMap/
├── app.py                         # Streamlit entry point
├── train.py                       # Training entry point
├── train_3d.py                    # Patch-based 3D training entry point
├── prepare_data.py                # Split and dataset audit entry point
├── evaluate.py                    # Evaluation entry point
├── evaluate_3d.py                 # Sliding-window 3D evaluation
├── tune_threshold.py              # 2D validation threshold selection
├── tune_threshold_3d.py           # 3D validation threshold selection
├── compare_validation_models.py   # Validation-only model selection
├── requirements.txt               # Python dependencies
├── PROJECT_BRIEF.md               # Partner-facing project definition
├── configs/
│   ├── baseline.yaml              # Reproducible 2D baseline configuration
│   └── task_a_3d.yaml             # Patch-based 3D configuration
├── src/
│   └── neuromap/
│       ├── data_loader.py
│       ├── data_loader_3d.py
│       ├── preprocessing.py
│       ├── model.py
│       ├── model_3d.py
│       ├── training.py
│       ├── training_3d.py
│       ├── inference.py
│       ├── inference_3d.py
│       ├── metrics.py
│       ├── volume.py
│       └── visualization.py
├── tests/
├── artifacts/
│   └── checkpoints/
├── reports/
└── data/
```

Large datasets, checkpoints, and generated reports are kept out of source
control. See `.gitignore`.

## Frameworks and libraries

| Tool | Role |
|---|---|
| Python | Application and experiment language |
| PyTorch | Neural network training and inference |
| MONAI | Medical imaging datasets and transforms |
| NiBabel | NIfTI loading and spatial metadata |
| NumPy | Array operations |
| SciPy | Supporting image processing |
| Streamlit | Interactive application |
| Matplotlib and Plotly | Visualization |
| pandas | Tabular reports |
| PyYAML | Configuration loading |
| pytest | Focused tests |

## Local setup

The verified training environment uses Python 3.11, PyTorch 2.11.0 with CUDA
12.8, and MONAI 1.6.1. Dependency names are listed in `requirements.txt`; install
a PyTorch build compatible with the local GPU and driver before running the
experiments. Dataset files, checkpoints, and generated JSON reports are
machine-local and are intentionally not included in the repository.

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

To run the app, first place the dataset and trained checkpoints at the
configured local paths (or set the `NEUROMAP_*` path overrides documented
below). The split manifest is local and machine-specific; regenerate it on the
target computer rather than copying the ignored manifest.

## External resources

- [Medical Segmentation Decathlon](https://medicaldecathlon.com/)
- [Task01_BrainTumour AWS registry](https://registry.opendata.aws/msd/)
- [MONAI documentation](https://monai.readthedocs.io/)
- [PyTorch documentation](https://pytorch.org/docs/)
- [Streamlit documentation](https://docs.streamlit.io/)

The dataset must be downloaded from an official source and stored locally or
in approved cloud storage. It must not be committed to GitHub. Local datasets
and checkpoints are ignored by Git. The generated split manifest also contains
machine-specific absolute data paths and is ignored; recreate it on each
machine from the documented seed and data root.

## Planned execution order

1. Verify the environment and dataset access.
2. Audit modalities, labels, geometry, and metadata.
3. Create and freeze the patient-level split manifest.
4. Implement preprocessing and a small overfitting test.
5. Implement and train the baseline U-Net.
6. Evaluate on held-out patients.
7. Integrate inference into Streamlit.
8. Test, document, and prepare the demonstration.

Create the split manifest before training:

```powershell
python prepare_data.py --data-root data/raw/Task01_BrainTumour
```

The manifest is written to `artifacts/splits/` and is reused by `train.py` so
future runs cannot silently change patient membership.

The error analysis found exact duplicate binary labels whose MRI volumes match
after per-modality intensity normalization, including duplicates across the
original split boundaries. Prepare a deduplicated split before the next run:

```powershell
python prepare_data.py `
    --data-root data/raw/Task01_BrainTumour `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --deduplicate-identical-labels
python train.py `
    --data-root data/raw/Task01_BrainTumour `
    --config configs/task_a_multimodal.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --output artifacts/checkpoints/baseline_deduplicated.pt
```

This changes patient membership; start a fresh run rather than using `--resume`
with the old checkpoint. Training slices are shuffled each epoch and receive
paired random in-plane flips; validation remains unaugmented. Losses are saved
to a CSV beside the checkpoint (or to `--history` if supplied). See
`reports/error_analysis.md` and its overlays for the findings behind this
change. The original held-out metrics are provisional because of the
cross-split duplicates.

The FLAIR-only comparison uses the same frozen manifest and can be trained
independently:

```powershell
python train.py `
    --data-root data/raw/Task01_BrainTumour `
    --config configs/task_a_flair.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --output artifacts/checkpoints/flair_deduplicated.pt
```

For each checkpoint, choose a threshold using only its corrected validation
patients, then use that fixed threshold on test:

```powershell
python tune_threshold.py `
    --checkpoint artifacts/checkpoints/baseline_deduplicated.pt `
    --config configs/task_a_multimodal.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --output reports/task_a_multimodal_thresholds.json
python tune_threshold.py `
    --checkpoint artifacts/checkpoints/flair_deduplicated.pt `
    --config configs/task_a_flair.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --output reports/task_a_flair_thresholds.json
```

To continue an interrupted run from the checkpoint in `artifacts/checkpoints/baseline.pt`
through the configured epoch count, add `--resume`:

```powershell
python train.py --data-root data/raw/Task01_BrainTumour --resume
```

Each completed epoch is saved for resumption, while `model_state_dict` remains
the best-validation model for inference.

Evaluate the best checkpoint on the patient-level test split recorded in the
manifest. Predictions are resized back to each scan's native in-plane grid
before calculating 3D segmentation metrics and absolute volume error in mL:

```powershell
python evaluate.py `
    --checkpoint artifacts/checkpoints/baseline_deduplicated.pt `
    --config configs/task_a_multimodal.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --threshold 0.5 `
    --output reports/task_a_multimodal_test.json
```

The per-patient results and macro averages are written to
the selected output file. Replace `0.5` with the validation-selected
threshold, then run the equivalent evaluation for the FLAIR checkpoint.

## Patch-based 3D comparison

The 3D experiment uses 96 x 96 x 64 patches, a batch size of one, and two
patches per training patient per epoch. CUDA bfloat16 is used when supported.
Validation loss is measured on a fixed foreground-centered patch per patient;
threshold selection uses full-volume validation predictions. Final predictions
are reconstructed with overlapping, Gaussian-weighted sliding windows. The
3D model and its optimizer can be resumed from the checkpoint.

```powershell
python train_3d.py `
    --config configs/task_a_3d.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --output artifacts/checkpoints/unet_3d_deduplicated.pt
```

Add `--resume` to continue an interrupted 3D training run. After training,
choose a threshold on validation only, then evaluate once on the held-out test
split:

```powershell
python tune_threshold_3d.py `
    --checkpoint artifacts/checkpoints/unet_3d_deduplicated.pt `
    --config configs/task_a_3d.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json
python compare_validation_models.py `
    --candidate baseline=reports/task_a_multimodal_unaugmented_thresholds.json `
    --candidate augmented=reports/task_a_multimodal_augmented_thresholds.json `
    --candidate flair=reports/task_a_flair_thresholds.json `
    --candidate 3d=reports/task_a_3d_thresholds.json
python evaluate_3d.py `
    --checkpoint artifacts/checkpoints/unet_3d_deduplicated.pt `
    --config configs/task_a_3d.yaml `
    --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
    --threshold <validation-selected-threshold>
```

The comparison command chooses a model using validation Dice, with lower mean
volume error as the tie-breaker. Do not select a model or threshold using test
metrics. Evaluate each candidate once with its validation-selected threshold;
report test results descriptively after model selection is frozen.

## Interactive app

The sidebar lets the user choose the unaugmented or augmented multimodal 2D
model, the FLAIR-only 2D model, or the 3D model. When the validation comparison
report exists, its winning candidate becomes the default model. Each model
loads its validation-selected threshold from its corresponding threshold
report. The uploaded 4D NIfTI must contain T1, T1ce, T2, and FLAIR channels in
that order; the FLAIR-only model uses its final channel. Override model paths
with `NEUROMAP_CHECKPOINT` / `NEUROMAP_CONFIG` for unaugmented 2D,
`NEUROMAP_AUGMENTED_CHECKPOINT` / `NEUROMAP_AUGMENTED_CONFIG` for augmented 2D,
`NEUROMAP_FLAIR_CHECKPOINT` / `NEUROMAP_FLAIR_CONFIG` for FLAIR-only 2D, and
`NEUROMAP_3D_CHECKPOINT` / `NEUROMAP_3D_CONFIG` for 3D.
Inference stays disabled until the checkpoint's matching
`<checkpoint-stem>_history.csv` confirms all configured epochs have completed.

Run the interactive app after the multimodal checkpoint is present:

```powershell
streamlit run app.py
```

Set `NEUROMAP_CHECKPOINT` and `NEUROMAP_CONFIG` to use different local model
artifacts.

## GPU training in Google Colab

The baseline can run on a free Colab GPU. Upload the project source (without
the local `data/` directory) to Colab, select **Runtime > Change runtime type >
T4 GPU**, then run:

```python
!nvidia-smi
!pip install -r requirements.txt
```

Download and extract the dataset in the Colab runtime:

```bash
!mkdir -p data/raw
!wget -c https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar \
    -O data/raw/Task01_BrainTumour.tar
!tar -xf data/raw/Task01_BrainTumour.tar -C data/raw
```

Regenerate the split manifest in Colab because the local manifest contains
Windows paths:

```bash
!python prepare_data.py --data-root data/raw/Task01_BrainTumour
!python train.py \
    --data-root data/raw/Task01_BrainTumour \
    --manifest artifacts/splits/task01_brain_tumour.json \
    --output artifacts/checkpoints/baseline.pt
```

The seed remains `42`, so the patient assignment is identical to the local
manifest while using paths valid in Colab.

## Deployed Streamlit app

Open the deployed NeuroMap app at
[neuromap-ai-ykwthcitvmhafuyomhtqtx.streamlit.app](https://neuromap-ai-ykwthcitvmhafuyomhtqtx.streamlit.app/).
For a quick exhibition run:

1. Select **Upload a 4D NIfTI**.
2. Upload one Task01_BrainTumour image from `imagesTr` (`.nii` or `.nii.gz`).
3. Confirm that the last image axis is ordered T1, T1ce, T2, FLAIR.
4. Choose a segmentation model, then click **Run segmentation**.
5. Review the MRI overlay and volume/segmentation metrics.
6. Test both the NIfTI mask and CSV download buttons.

The app's validation-comparison and threshold notices explain when it falls
back to the unaugmented model or configured threshold. If either notice is
shown, confirm the matching validation JSON reports are available to the
deployed app before presenting its model-selection results. The info notices
use high-contrast text against the dark theme. The preloaded-sample option
requires the split manifest and dataset paths, which are not included in the
public repository; it currently reports a missing manifest on the deployed
app. Model checkpoints are also excluded from Git, so confirm the deployed
runtime has its checkpoint before treating successful upload as proof that
inference works.

## Final submission materials

The report, editable presentation, qualitative validation figure, and
evaluator handoff notes are in [`submission/`](./submission/). They were
updated against the AI2002 project statement and Software Category rubric.
The presentation has 13 slides and includes both aggregate metrics and an
individual validation overlay example.

The deployment URL above was provided by the team. The latest manual check
found that the hosted runtime lacks the preloaded manifest and selected model
checkpoint, so inference and exports still need verification. The final
one-ZIP upload to GCR remains outstanding. Add the actual group number before
submission if required; all three members' names and roll numbers are included
in the app header, report, and presentation. The model archive is included in
the submission package, not in Git; the public dataset must be downloaded
separately.

The portable submission bundle containing source code, report, presentation,
evaluation evidence, and the model archive is
`artifacts/NeuroMap_AI_Submission.zip`. Rebuild it after any final edits. The
report still has a group-number placeholder because the number was not
provided; fill it in before GCR upload if required by the instructor. The ZIP
does not contain the dataset.

The four trained checkpoints and their histories, configurations, and
validation reports are bundled locally as
`artifacts/submission_models_2026-11-25.zip`. This generated archive is ignored
by Git. If GCR limits require separate transfer, share it through
instructor-approved storage. The dataset is not included; download it from the
official MSD source and regenerate the machine-specific split manifest as described in
[`submission/artifact_access.md`](./submission/artifact_access.md).
