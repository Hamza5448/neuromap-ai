# Dataset and model artifact handoff

## Dataset

The experiment uses the labeled **Medical Segmentation Decathlon,
Task01_BrainTumour** dataset. The dataset must be obtained from the official
challenge or AWS Open Data source and stored locally; it is not included in
GitHub or the model archive.

Official sources:

- <https://medicaldecathlon.com/>
- <https://registry.opendata.aws/msd/>
- <https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar>

Extract the dataset so the project can find:

```text
data/raw/Task01_BrainTumour/imagesTr/
data/raw/Task01_BrainTumour/labelsTr/
data/raw/Task01_BrainTumour/imagesTs/
```

Recreate the deterministic deduplicated patient split on the recipient's
machine; do not copy the existing manifest because it contains absolute paths
from the training machine:

```powershell
python prepare_data.py `
  --data-root data/raw/Task01_BrainTumour `
  --manifest artifacts/splits/task01_brain_tumour_deduplicated.json `
  --deduplicate-identical-labels
```

The resulting labeled cohort contains 478 cases after six duplicate-label
cases are removed. With seed 42, the expected split sizes are 334 train, 72
validation, and 72 test. Verify the generated counts before reproducing
training or evaluation.

## Trained model archive

The local archive is:

```text
artifacts/submission_models_2026-11-25.zip
```

Size: 26,974,450 bytes. SHA-256:
`8d2162153ecbc270eab5ca1fa6b882e3669b643295dd879fb1800467d4239b9e`.

It contains the four final checkpoints, matching 50-epoch histories, model
configuration files, and validation threshold/model-selection reports. The
archive is intentionally excluded from GitHub and is included inside the
prepared single GCR submission ZIP. After extracting the outer ZIP, extract
this model archive at the project root. If GCR size or policy limits require
separate transfer, use instructor-approved storage. Do not upload the dataset
or the machine-specific split manifest.

After extracting the archive at the project root, the default app checkpoint
paths and history checks are available. The archive includes the validation
reports needed for the app to select the augmented multimodal 2D model and its
threshold of 0.75 by default.

## Runtime and demonstration

The verified environment was Python 3.11, PyTorch 2.11.0+cu128, MONAI 1.6.1,
and Streamlit 1.67.0. Install dependencies according to `README.md`, restore
the dataset and checkpoint archive, then start the app:

```powershell
streamlit run app.py
```

The team has provided a screenshot of the app running on Streamlit Community
Cloud. The latest manual check found the preloaded-sample manifest and selected
model checkpoint missing from the deployed runtime. Restore the required
artifacts and test live inference before relying on the hosted app. The app is
a research prototype, not a clinical service.
