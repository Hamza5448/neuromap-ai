# NeuroMap submission package

Prepared November 25, 2026 against the supplied AI2002 Project Statement,
Software Category Evaluation Rubric, and Recommendations. The Recommendations
document contains illustrative project examples; the Project Statement and
Rubric define the mandatory deliverables and evaluation criteria.

The report and presentation explicitly identify Path A (Software-Based).
The report lists Fizza Fatima (24i-3077) and Irtaza Abid (24i-3085). Add the
actual group number before submission. The team has since provided a screenshot
of the app running on Streamlit Community Cloud. The screenshot shows missing
validation-comparison and threshold reports, so live inference still needs
verification after those artifacts are restored. The app header identifies
both members; its information alerts use high-contrast text on the dark theme.

## Deliverables

- `final_project_report.md`: editable report source.
- `final_project_report.docx`: formatted report.
- `final_project_report.pdf`: paginated, read-only report.
- `final_presentation.pptx`: 13-slide project presentation with quantitative
  metrics and a qualitative validation example.
- `system_pipeline.png`: end-to-end software architecture diagram.
- `qualitative_validation_overlay.png`: actual validation-sample application
  view; per-case values are labeled separately from aggregate test metrics.
- `artifact_access.md`: dataset and model-weight handoff instructions.

## Evidence and scope

The report uses the corrected patient-level split, validation-only model and
threshold selection, and one held-out evaluation per final model. Results are
summarized in `../reports/error_analysis.md`; patient-level JSON outputs and
the machine-specific split manifest are intentionally not included in Git.

All four final checkpoints and their 50-epoch histories are present locally.
They are assembled into `../artifacts/submission_models_2026-11-25.zip`,
which is deliberately ignored by Git because model weights are large generated
artifacts. Transfer that archive through instructor-approved storage rather
than committing it. The dataset is also not included; it is available from the
official Medical Segmentation Decathlon source described in
`artifact_access.md`.

The local Streamlit app was smoke-tested with augmented 2D and 3D inference.
The NIfTI mask and CSV downloaded from the browser were verified in the local
Downloads folder; the NIfTI dimensions, binary values, voxel spacing, and CSV
fields were checked.

The Project Statement requires one ZIP submission on GCR containing the source
code, report, presentation, evaluation evidence, and other required files.
The model archive is included in the prepared package; the large public
dataset is not. Creating the package does not submit it to GCR.
The portable bundle is at `artifacts/NeuroMap_AI_Submission.zip`. The report
still has a group-number placeholder because the number was not supplied.
This screenshot confirms the app loads, not end-to-end cloud inference.

These are research-prototype results, not clinical evidence or medical advice.
