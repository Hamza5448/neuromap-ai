# NeuroMap baseline error analysis

## Executive summary

The original checkpoint completed 50 epochs, and the initial evaluation scored
73 test patients at threshold 0.5. Its macro means were Dice 0.8251, IoU
0.7106, sensitivity 0.8903, and absolute volume error 16.06 mL. Those numbers
are **not a clean held-out estimate**: six pairs have identical binary masks
and effectively identical MRI inputs after per-modality intensity
normalization. Three pairs crossed the original train/test boundary and two
crossed train/validation. Best-checkpoint selection and test evaluation were
therefore exposed to duplicate cases.

The known duplicate labels were removed before creating a new split. The new
manifest contains 334 train, 72 validation, and 72 test patients, with no
repeated binary-label volumes and no missing paths. A fresh training run is
required, so a new baseline was trained rather than resuming the old
checkpoint. The unaugmented baseline completed 50 epochs and has now been
evaluated on the corrected test split. The augmented multimodal and FLAIR-only
models have also completed training and test evaluation. Validation-only
model ranking is frozen with the augmented multimodal model selected. The
patch-based 3D model has also completed its one-time held-out evaluation.

## Evaluation findings

### Corrected validation baseline (not a test estimate)

The completed unaugmented 50-epoch baseline was evaluated on the corrected
validation split only. The validation sweep selected threshold 0.65 by mean
patient Dice: mean Dice 0.8709, IoU 0.7766, sensitivity 0.8737, and mean
absolute volume error 9.51 mL. Threshold performance is relatively flat from
0.50 through 0.80; 0.75 minimized mean volume error by only 0.07 mL more, while
0.65 gave the highest Dice. This threshold must not be reused for augmented
models.

The lowest-Dice validation cases include BRATS_353 and BRATS_224, with
sensitivity 0.4913 and 0.5559, while the largest volume errors include
BRATS_207, BRATS_400, and BRATS_477 (44.57–57.53 mL). The errors therefore
include both missed tumor and volume overestimation; no single threshold
adjustment fixes both. Threshold selection and these case rankings use
validation only.

### Corrected held-out test baseline

The unaugmented multimodal checkpoint was evaluated once on the 72-patient
corrected test split, using the threshold selected on validation (0.65). The
macro means were Dice 0.83444, IoU 0.72762, sensitivity 0.81923, and absolute
volume error 11.30874 mL. Mean predicted and reference volumes were 92.50 mL
and 96.08 mL, respectively. Per-patient results are in
`reports/task_a_multimodal_test.json`.

### Requirement-matched held-out comparison

Each threshold below was selected on the corrected validation patients only.
Each reported model was then evaluated once on the same 72-patient test split.
Validation-only ranking selected augmented multimodal 2D before the test
metrics were inspected.

| Model | Validation threshold | Test Dice | Test IoU | Test sensitivity | Absolute volume error (mL) |
|---|---:|---:|---:|---:|---:|
| Augmented multimodal 2D (selected by validation) | 0.75 | 0.84895 | 0.75002 | 0.83788 | 11.67610 |
| Unaugmented multimodal 2D | 0.65 | 0.83444 | 0.72762 | 0.81923 | 11.30874 |
| FLAIR-only 2D | 0.75 | 0.78574 | 0.66519 | 0.79234 | 12.77144 |
| Patch-based multimodal 3D | 0.80 | 0.83741 | 0.73782 | 0.86814 | 11.50800 |

The augmented multimodal model ranked highest on validation and was selected
before held-out results were evaluated. It also has the highest test Dice in
this descriptive comparison, but test metrics did not influence selection.
The per-patient JSON reports are generated locally and excluded from GitHub.

### Original contaminated test evaluation (provisional)

| Result | Finding |
|---|---:|
| Original test patients | 73 |
| Mean Dice / IoU / sensitivity | 0.8251 / 0.7106 / 0.8903 |
| Mean absolute volume error | 16.06 mL |
| Mean predicted vs. reference volume | 100.47 vs. 88.57 mL |
| Patients with over-predicted volume | 58 / 73 |
| Median Dice; 10th percentile; minimum | 0.8488; 0.7112; 0.5173 |

Over-segmentation is the most common volume-level error, but there is no single
failure mode. For example, BRATS_321 has near-complete sensitivity (0.9928)
but predicts 159.12 mL against 66.65 mL (92.47 mL error); BRATS_114 and
BRATS_077 under-segment, with sensitivities 0.4848 and 0.4584, respectively.
The low-Dice cases are not all explained by lesion size: the Dice/target-volume
correlation is 0.465, while Dice/volume-error correlation is -0.309.

| Case | Dice | Sensitivity | Predicted / reference volume (mL) | Volume error (mL) | FP / FN voxels |
|---|---:|---:|---:|---:|---:|
| BRATS_138 | 0.5173 | 0.8283 | 28.12 / 12.77 | 15.35 | 17,545 / 2,193 |
| BRATS_077 | 0.5860 | 0.4584 | 20.65 / 36.58 | 15.93 | 3,880 / 19,813 |
| BRATS_321 | 0.5862 | 0.9928 | 159.12 / 66.65 | 92.47 | 92,950 / 478 |
| BRATS_114 | 0.6412 | 0.4848 | 37.19 / 72.61 | 35.42 | 1,982 / 37,407 |
| BRATS_315 | 0.6438 | 0.8519 | 60.68 / 36.86 | 23.82 | 29,281 / 5,459 |
| BRATS_241 | 0.7582 | 0.9478 | 201.89 / 134.58 | 67.31 | 74,336 / 7,023 |
| BRATS_310 | 0.8318 | 0.9873 | 174.74 / 127.19 | 47.55 | 49,162 / 1,613 |
| BRATS_015 | 0.8594 | 0.9859 | 257.16 / 198.69 | 58.47 | 61,273 / 2,801 |
| BRATS_280 | 0.8594 | 0.9859 | 257.16 / 198.69 | 58.47 | 61,270 / 2,803 |

Excluding the three known train/test duplicate patients and one repeated
test/test patient leaves 69 cases with descriptive means of Dice 0.8254, IoU
0.7113, sensitivity 0.8863, and volume error 15.39 mL. This is only a
post-hoc diagnostic subset, not a replacement unbiased evaluation: the
original validation split also had duplicate patients, and the subset was
chosen after inspecting test results.

## Duplicate-data audit

All six pairs below have identical binary reference masks. In every modality,
their non-zero MRI voxel masks match and the non-zero intensity correlation
after per-modality z-score normalization is 0.999999; normalized mean absolute
difference is approximately 0.0009–0.0013.

| Original split A | Original split B | Consequence |
|---|---|---|
| BRATS_293 (train) | BRATS_002 (test) | Train/test overlap |
| BRATS_290 (train) | BRATS_007 (test) | Train/test overlap |
| BRATS_013 (train) | BRATS_288 (test) | Train/test overlap |
| BRATS_001 (train) | BRATS_275 (validation) | Train/validation overlap |
| BRATS_010 (train) | BRATS_278 (validation) | Train/validation overlap |
| BRATS_280 (test) | BRATS_015 (test) | Test set double-counted one scan |

The duplicate-label groups were removed deterministically before the new split.
The retained IDs are the lexicographically first ID in each group. This
deduplicates confirmed cases; it does not claim that every possible
near-duplicate MRI in the source dataset has been detected.

## Visual review

At the time of the initial review, the app was a stub, so the first review used
static native-grid overlays. The app now supports MRI upload, modality/slice
navigation, prediction overlays, optional reference-mask metrics, and
mask/measurement export. A local uploaded validation case was segmented in the
browser on the unaugmented baseline at threshold 0.65: Dice 0.7021, IoU 0.5410,
sensitivity 0.7513, and absolute volume error 2.6480 mL. The overlays and
measurement display rendered correctly, and the local health endpoint returned
HTTP 200. Browser download events did not complete; NIfTI serialization is
covered by unit tests. Green is reference, red is prediction; the error panel
uses red for false positives, cyan for false negatives, and yellow for overlap.

Patient-derived overlay images are generated locally under
`reports/error_analysis/overlays/` and are excluded from GitHub by `.gitignore`.

## Training readiness

- The original training `DataLoader` used `shuffle=False`, preserving
  contiguous patient/slice order in every epoch. Training now shuffles slices;
  validation remains ordered.
- Positive voxels account for about 1.19% of the original training mask volume,
  while about 45.8% of training slices contain any tumor. The training pipeline
  now adds paired random in-plane flips, preserving image/mask alignment; the
  BCE-plus-Dice loss addresses voxel imbalance.
- The previous run has no persisted epoch history, so its convergence curve
  cannot be assessed. New runs now save epoch losses to a CSV beside the
  checkpoint (or to `--history`).
- The clean unaugmented baseline completed 50 epochs from scratch on the
  corrected split; its best validation loss was 0.64583 at epoch 49. Its
  validation-selected threshold and held-out test evaluation are complete.
  The previous contaminated checkpoint was not resumed. The augmented
  multimodal and FLAIR-only runs have completed 50 epochs each. The 3D run had
  saved epoch 47/50 when its process stopped; its final epochs have been
  resumed. Threshold tuning, validation-only ranking, and held-out evaluation
  remain pending. The augmented run uses single-process data loading after a
  Windows worker memory error.

To reproduce the corrected split and begin a fresh run, see the commands in
`README.md`. These research-prototype results are not clinical validation.
