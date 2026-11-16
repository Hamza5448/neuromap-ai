# NeuroMap AI

NeuroMap AI is a software-based research prototype for brain MRI tumor-region
segmentation and volumetric analysis. The initial release uses a multimodal
2D U-Net and a Streamlit interface.

The project is not a medical diagnostic or treatment system.

## Stage A scope

- Medical Segmentation Decathlon `Task01_BrainTumour`.
- Four MRI modalities in NIfTI format.
- Binary whole-tumor-region segmentation.
- Patient-level train, validation, and test splits.
- Held-out evaluation using segmentation and volumetric metrics.
- Streamlit visualization, measurement, and export workflow.

## Project structure

```text
NeuroMap/
├── app.py                         # Streamlit entry point
├── train.py                       # Training entry point
├── prepare_data.py                # Split and dataset audit entry point
├── evaluate.py                    # Evaluation entry point
├── requirements.txt               # Python dependencies
├── PROJECT_BRIEF.md               # Partner-facing project definition
├── configs/
│   └── baseline.yaml              # Reproducible baseline configuration
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

Dependency versions will be pinned after the Python interpreter and GPU
environment are confirmed.

## External resources

- [Medical Segmentation Decathlon](https://medicaldecathlon.com/)
- [Task01_BrainTumour AWS registry](https://registry.opendata.aws/msd/)
- [MONAI documentation](https://monai.readthedocs.io/)
- [PyTorch documentation](https://pytorch.org/docs/)
- [Streamlit documentation](https://docs.streamlit.io/)

The dataset must be downloaded from an official source and stored locally or
in approved cloud storage. It must not be committed to GitHub.

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
