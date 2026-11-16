# NeuroMap AI

## Brain Tumor Segmentation and Volumetric MRI Analysis

**Project type:** Software-based deep learning research prototype  
**Domain:** Healthcare AI and medical image processing  
**Primary interface:** Streamlit web application  
**Primary dataset:** Medical Segmentation Decathlon, Task01_BrainTumour  
**Planned submission:** December 6, 2026

> **Important:** NeuroMap AI is an academic research prototype. It is not a
> medical device, diagnostic tool, treatment recommender, or substitute for
> clinical judgment.

## 1. Executive Summary

NeuroMap AI will use a trained deep learning model to identify glioma-related
regions in multimodal brain MRI scans. The application will generate
segmentation masks, display the predictions as overlays on MRI slices, and
calculate the physical volume of the predicted region.

The primary implementation will use a 2D U-Net with four MRI modalities and
will be delivered through an interactive Streamlit application. The initial
release will focus on a complete, reproducible, evaluated Stage A system.
Longitudinal comparison of follow-up scans is an optional extension and will
not block the primary release.

## 2. Problem and Proposed Solution

Glioma-related regions can have complex shapes and appearances across MRI
modalities. Manually outlining these regions across many slices is time
consuming and can be difficult to reproduce.

NeuroMap AI will:

1. Accept compatible multimodal MRI data.
2. Validate dimensions, modality order, and spatial metadata.
3. Preprocess the images for model inference.
4. Predict a tumor-related segmentation mask with a trained U-Net.
5. Visualize the prediction over the MRI slices.
6. Calculate the predicted region's physical volume.
7. Export the mask and a measurement summary.

The system will initially combine verified non-background tumor labels into a
single whole-tumor-region mask. Multi-region segmentation is an enhancement.

## 3. Goals and Success Criteria

### Primary goals

- Build a reproducible patient-level data pipeline.
- Train and evaluate a baseline multimodal 2D U-Net.
- Provide a working Streamlit workflow from upload through export.
- Report segmentation and volumetric results honestly on held-out patients.
- Compare four-modality input with a FLAIR-only baseline.

### Stage A success criteria

- Dataset acquisition and preprocessing are documented.
- Train, validation, and test patients are kept strictly separate.
- A trained checkpoint can perform real inference.
- The application validates MRI input and displays prediction overlays.
- Volumes are calculated from image spatial metadata.
- Evaluation results are reproducible and include the selected metrics.
- The application can export segmentation results and measurements.

No performance value will be claimed before experiments are completed.

## 4. Scope

### Included in the initial release

- NIfTI MRI input.
- Four-modality MRI processing:
  - T1-weighted.
  - Contrast-enhanced T1-weighted.
  - T2-weighted.
  - FLAIR.
- Binary whole-tumor-region segmentation.
- 2D U-Net training and inference.
- MRI slice and modality navigation.
- Segmentation overlays.
- Physical volume calculation in mL or cm³.
- Optional reference-mask evaluation.
- Pretrained checkpoint loading.
- Preloaded sample study.
- Export of masks and measurement summaries.

### Deferred or optional

- Multi-region segmentation for edema, enhancing tumor, and non-enhancing or
  necrotic tumor core.
- Longitudinal post-treatment MRI comparison.
- Additional model architectures.
- Clinical deployment or diagnostic integration.

The optional features will be considered only after the Stage A system is
complete, evaluated, and stable.

## 5. Dataset

### Primary dataset

**Medical Segmentation Decathlon — Task01_BrainTumour**

- 484 labeled training cases.
- 266 challenge test cases; publicly available ground-truth labels must not
  be assumed.
- Four MRI modalities.
- NIfTI format.
- Expert segmentation masks.
- Semantic segmentation task.
- CC BY-SA 4.0 license, subject to the dataset terms.

The project will reserve its own test set from the 484 labeled cases. All
slices from a patient must remain in the same partition.

### Planned data split

| Partition | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

### Preprocessing requirements

- Read NIfTI files using NiBabel or MONAI.
- Validate dimensions, modality order, and spatial alignment.
- Normalize MRI intensities.
- Prepare matching images and masks.
- Apply suitable augmentation during training.
- Preserve spatial metadata for volume calculations.
- Visually inspect representative images and masks.

The large dataset and generated artifacts must remain outside source control.

## 6. Machine Learning Approach

### Baseline model

| Component | Selection |
|---|---|
| Architecture | 2D U-Net |
| Framework | PyTorch |
| Medical imaging library | MONAI |
| Input channels | 4 |
| Input | Multimodal MRI slices |
| Output | Binary segmentation mask |
| Loss | Dice loss combined with binary cross-entropy |
| Optimizer | AdamW |
| Training | Supervised learning |

U-Net is appropriate because the task requires spatial predictions rather than
one label for an entire image. Its encoder extracts features, its decoder
reconstructs the prediction, and skip connections preserve spatial detail.

### Planned experiments

1. Establish a four-modality U-Net baseline.
2. Train a FLAIR-only U-Net using the same evaluation protocol.
3. Compare segmentation and volumetric performance.
4. Investigate multi-region segmentation only if time and results justify it.

Hyperparameters will be selected using validation results and available GPU
memory. Training may use a local GPU, Google Colab, or Kaggle.

## 7. Evaluation

Final evaluation will use held-out patients and, where practical, reconstructed
3D volumes rather than selected individual slices.

### Metrics

- **Dice Similarity Coefficient**
  \[
  Dice = \frac{2|P \cap G|}{|P| + |G|}
  \]
- **Intersection over Union**
  \[
  IoU = \frac{|P \cap G|}{|P \cup G|}
  \]
- **Sensitivity**
  \[
  Sensitivity = \frac{TP}{TP + FN}
  \]
- **Volumetric error**
  \[
  E_V = |V_{predicted} - V_{reference}|
  \]
- **Optional:** HD95 for boundary quality.

### Volumetric analysis

The predicted physical volume will be calculated using image metadata:

\[
V = N_{voxels} \times V_{voxel}
\]

The result will be converted from cubic millimeters to milliliters or cm³
using \(1\,mL = 1000\,mm^3\). This is a measurement of the predicted mask,
not an independent determination of tumor burden.

## 8. Application Workflow

1. Upload a compatible MRI study or select a preloaded sample.
2. Validate modalities, dimensions, geometry, and metadata.
3. Preprocess the study.
4. Load a trained model checkpoint.
5. Run U-Net inference.
6. Navigate slices and switch modalities.
7. Display the segmentation overlay.
8. Calculate the segmented-region volume.
9. Optionally compare with a reference mask.
10. Export the segmentation and measurement summary.

## 9. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Upload compatible MRI data |
| FR-02 | Validate MRI input |
| FR-03 | Preprocess MRI data |
| FR-04 | Execute U-Net segmentation |
| FR-05 | Display tumor segmentation overlays |
| FR-06 | Navigate MRI slices |
| FR-07 | Switch between MRI modalities |
| FR-08 | Calculate segmented-region volume |
| FR-09 | Evaluate against reference masks |
| FR-10 | Load a pretrained model checkpoint |
| FR-11 | Select a preloaded sample MRI |
| FR-12 | Export segmentation and results |

## 10. Optional Longitudinal Extension

An advanced extension may compare two MRI examinations from the same patient
using the UCSF Adult Longitudinal Post-Treatment Diffuse Glioma MRI Dataset.
The dataset contains follow-up examinations and expert annotations, subject to
its access and noncommercial data-use requirements.

The extension would validate scan compatibility, segment both examinations,
calculate both volumes, and report:

\[
\Delta V = V_2 - V_1
\]
\[
PercentageChange = \frac{V_2 - V_1}{V_1} \times 100
\]

This feature requires compatible segmentation definitions and a model
validated on post-treatment MRI. A measured change does not independently
establish progression or treatment response, so this extension is not a
dependency of Stage A.

## 11. Technology and Proposed Structure

### Technology stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network implementation |
| MONAI | Medical imaging and transformations |
| NiBabel | NIfTI file processing |
| NumPy | Numerical operations |
| SciPy | Image processing |
| Matplotlib or Plotly | Visualization |
| Streamlit | Interactive application |
| pandas | Evaluation reports |
| pytest | Testing |
| Git/GitHub | Version control |

### Proposed structure

```text
NeuroMap/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── configs/
│   └── baseline.yaml
├── src/
│   └── neuromap/
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── model.py
│       ├── training.py
│       ├── inference.py
│       ├── metrics.py
│       ├── volume.py
│       └── visualization.py
├── tests/
├── artifacts/
│   └── checkpoints/
├── reports/
└── data/
```

## 12. Risks, Limitations, and Controls

| Risk or limitation | Planned control |
|---|---|
| Dataset download or access problems | Use official AWS Open Data and S3 sources |
| Limited GPU memory | Start with a 2D U-Net and use cloud GPU resources |
| Class imbalance | Use Dice-based loss and appropriate sampling |
| Incorrect preprocessing | Validate metadata and inspect images and masks visually |
| Data leakage | Split by patient, never by slice |
| Poor model performance | Tune on validation data and report results honestly |
| Post-treatment domain shift | Keep longitudinal work as a separate experiment |
| Slow deployment inference | Benchmark inference before deployment |
| Clinical misinterpretation | Clearly label the system as a research prototype |

## 13. Research Questions

1. Does multimodal MRI input improve glioma segmentation compared with
   FLAIR-only input using the same U-Net architecture?
2. How well does a model trained on pre-treatment glioma MRI generalize to
   post-treatment MRI?
3. How accurately can the system estimate changes in corresponding
   glioma-related region volumes between follow-up examinations?

The first question is the primary research question. The second and third are
advanced directions.

## 14. Deliverables

- Dataset acquisition and documentation.
- Patient-level train, validation, and test split.
- MRI preprocessing pipeline.
- Implemented 2D U-Net.
- Trained model checkpoint.
- Evaluation metrics and experiment report.
- Streamlit application.
- MRI upload and validation.
- Segmentation visualization.
- Volumetric measurement.
- Results export.
- Working deployment where feasible.
- Project report and presentation.
- Live demonstration.

## 15. Proposed Schedule

| Dates | Work |
|---|---|
| Nov 13–15, 2026 | Dataset acquisition and inspection |
| Nov 16–19 | Data loader, preprocessing, and baseline model |
| Nov 20–24 | Model training and experiments |
| Nov 25–27 | Evaluation and volumetric analysis |
| Nov 28–30 | Streamlit application |
| Dec 1–3 | Integration, testing, and deployment |
| Dec 4–5 | Report and presentation |
| Dec 6 | Final submission |

The earlier October 25 project-finalization deadline has passed. If project
registration is incomplete, this should be confirmed with the instructor
before implementation begins.

## 16. Decisions Required Before Coding

The partners and project team should confirm:

1. Whether the Stage A scope and binary segmentation target are approved.
2. Dataset access, licensing, storage, and download responsibilities.
3. Available GPU or cloud-computing resources.
4. The minimum evaluation metrics and reporting format.
5. Whether the FLAIR-only comparison is required or optional.
6. The deployment target and expected demonstration environment.
7. Whether the optional longitudinal extension is excluded from the deadline.
8. Ownership of the code, reports, model artifacts, and presentation materials.

After these decisions, the team can agree on the detailed risk register and
implementation roadmap before writing production code.

## 17. References

1. [Medical Segmentation Decathlon](https://medicaldecathlon.com/)
2. [Medical Segmentation Decathlon AWS Registry](https://registry.opendata.aws/msd/)
3. [Task01_BrainTumour download](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar)
4. [Medical Segmentation Decathlon research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9287542/)
5. [MONAI documentation](https://monai.readthedocs.io/)
6. [UCSF Longitudinal Glioma Dataset](https://imagingdatasets.ucsf.edu/dataset/2)
7. [UCSF dataset research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)
8. [Streamlit documentation](https://docs.streamlit.io/)
9. Uploaded course document: `Recommendations.pdf`
