# Task A status

Last updated: November 25, 2026; deployment status is based on the team's
Streamlit Community Cloud screenshot.

## Completed

- Removed six duplicated binary-label cases before generating the fixed
  334/72/72 train/validation/test manifest.
- Completed the fresh unaugmented multimodal run: 50/50 epochs; best validation
  loss 0.64583 at epoch 49. Its checkpoint and full CSV history are saved.
- Implemented slice-level training shuffling that retains patient-local
  volume caching, paired random in-plane image/mask flips, epoch-loss CSVs,
  FLAIR-only model input, and validation-only threshold selection.
- Implemented the Streamlit workflow for MRI upload/sample selection,
  multimodal 2D, FLAIR-only 2D, and 3D model selection, modality and slice
  navigation, overlays, reference metrics, and export.
- The full test suite passes: 40 tests, including 3D patch loading,
  sliding-window inference, checkpoint loading, validation-only model
  selection, and inference gating until all configured training epochs finish.
- Evaluated the clean unaugmented baseline once on the corrected 72-patient
  held-out test split using its validation-selected threshold of 0.65:
  mean Dice 0.83444, IoU 0.72762, sensitivity 0.81923, and absolute volume
  error 11.30874 mL. Detailed results are in
  `reports/task_a_multimodal_test.json`.
- Reviewed the original model's lowest-Dice and largest-volume-error cases;
  those original test metrics remain provisional because of duplicate cases.
- Added a patch-based 3D U-Net training/evaluation pipeline. A real-patient 3D
  patch and short mixed-precision GPU training benchmark completed successfully.
- All four experiments' training is complete: the unaugmented and augmented
  multimodal 2D, FLAIR-only 2D, and patch-based 3D models each trained 50/50
  epochs on the deduplicated manifest.
- Validation-only threshold sweeps and the four-model comparison are complete.
  Augmented multimodal 2D ranked first on validation (Dice 0.88615, volume
  error 9.79871 mL) and was selected before considering test results. Selected
  thresholds are 0.65 (unaugmented 2D), 0.75 (augmented 2D), 0.75 (FLAIR-only),
  and 0.80 (3D).
- All four models have completed exactly one evaluation on the 72-patient
  held-out test split, using their validation-selected thresholds. Test Dice:
  augmented 2D 0.84895, unaugmented 2D 0.83444, FLAIR-only 2D 0.78574, and 3D
  0.83741. Validation selection remains frozen on augmented 2D.
- Polished and visually reviewed the Streamlit interface. The local app passed
  end-to-end smoke runs on validation patient BRATS_029 with augmented 2D and
  3D inference, overlays, measurements, and CUDA. The app health endpoint
  returned HTTP 200, and inference remains gated on complete training history.
- Both browser exports were downloaded and verified: the NIfTI mask is binary,
  240 x 240 x 155 with 1 mm spacing, and the CSV contains the expected
  measurement fields and values.
- Updated the app header with both team members and improved alert contrast.
  Browser inspection measured at least 11.6:1 contrast for the info message
  styling against both ends of its gradient background.
- Final results and error-analysis docs include the 3D held-out metrics. The
  final test suite passes (40 tests), and Python compileall succeeds.

## Blocked or pending

- The code is merged into the GitHub `main` branch. Final report, presentation,
  pipeline diagram, qualitative validation figure, and artifact-access
  instructions are prepared in `submission/` and updated against the supplied
  AI2002 Project Statement and Software Category Rubric.
- The team-provided screenshot confirms that the app is running on Streamlit
  Community Cloud. It also shows that validation-comparison and threshold
  reports are unavailable; verify live inference after restoring these
  artifacts. The screenshot is not an end-to-end cloud inference test.
- The UI update is pushed to `docs/final-submission-materials` but is not yet
  merged into the deployment's branch; merge/redeploy before expecting the
  live app to show the contrast improvement.
- The report now lists Fizza Fatima (24i-3077) and Irtaza Abid (24i-3085).
  Add the group number before submission; it was not provided.
- The portable ZIP at `artifacts/NeuroMap_AI_Submission.zip` contains source,
  report, presentation, evaluation evidence, and the model archive. The report
  still has a group-number placeholder because no group number was supplied;
  replace it before GCR upload if required. The DOCX/PDF report and slide deck
  predate the deployment screenshot and need refreshed deployment wording.
- The four final checkpoints and histories are packaged locally in the
  Git-ignored `artifacts/submission_models_2026-11-25.zip` (SHA-256 is in
  `submission/artifact_access.md`). Share that archive using
  instructor-approved storage. The dataset is available from the official MSD
  source and must be downloaded by the evaluator; do not commit it or the
  machine-specific split manifest.
- Do not push the dataset or model archive. See
  [GITHUB_PUSH_REMINDER.txt](./GITHUB_PUSH_REMINDER.txt); never push the parent
  `C:\Users\Lab` repository.

These are research-prototype results and require expert review before any
clinical use.
