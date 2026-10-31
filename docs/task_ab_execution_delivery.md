# Task A/B execution and delivery

This research prototype is not clinical software. Notebook 07 is the shared private
synthetic execution launcher. Notebooks 05/06 remain separate explicitly gated
verified-data workflows. No real-data training is launched by Notebook 07 or the
execution CLI. No diagnosis, healthy label, spatial annotation or data access is
inferred from computational success.

## Launch

Build the shared code-only ZIP from the exact reviewed commit, using Python 3.12:

```bash
python scripts/build_task_a_bundle.py --output /tmp/task-ab-code.zip
sha256sum /tmp/task-ab-code.zip
```

Retain the printed SHA-256 independently; upload that reviewed ZIP to private Drive.
Open notebook 07 in private Colab, disable output saving, choose Python 3.12 and a
T4 or better GPU. Set bundle path/hash in the mount cell. Choose TASKS='a', 'b', or
'both'; MODE='synthetic_gpu' is the default. Run all cells. Task A and B execute
sequentially in separate private directories. A unique timestamp/random run name
prevents overwriting prior outputs. Installation checks the shared pinned research
packages; restart session and rerun from top if loaded versions are stale. The
verified manifest and source hashes restore CODE_ROOT without manual repair.

Torch 2.6.0's standard CUDA wheel supports T4 hardware; this is a compatibility
target, not evidence of an executed T4 run. The real GPU validator must run on the
actual Colab host. Synthetic tensors are 16 cubed, batch 1; real Task B defaults
to 64-cubed patches and width 8. Measure peak memory before scaling real runs.
Never change GPU mode to CPU to make a GPU validation pass.

The same validator can run outside Colab:

```bash
python -m brainagenticthon.execution.validation --tasks both \
  --mode synthetic_gpu --output /PRIVATE/NEW-RUN
```

Explicit CPU checks use --mode synthetic. A CPU PASS is not GPU evidence. CLI exit
status is 0 for the requested successful validation and 1 for a failed run.
No-CUDA GPU mode writes FAIL, no_cpu_fallback, and gpu_executed=false.

## Evidence and inventory

Private execution_report.json and execution_report.sha256 describe the actual
host/device/package versions, requested mode, per-task losses, model updates,
finite gradients (CUDA smoke plus Task B training), AMP state, Task A accumulation,
checkpoint filenames/bytes/SHA-256, epoch completion, safe reload, resume, and final
synthetic held-out metrics. Each task trains epoch 1, resumes to epoch 2, verifies
its committed checkpoint, reloads the best weights, and then evaluates synthetic
test data. A failed task does not prevent the other selected task from being tried.
Reports omit raw patient identities/paths. The report hash detects accidental
corruption; it is not an authenticity signature. Do not publish these private run
directories, reports or checkpoints. JSON templates contain no claimed results.

## Read-only private readiness

Optional READINESS_CONFIG_A/B parameters point to private copies of the existing
configs on Drive. The launcher checks those paths and saves aggregate reports.
With no config, reports list the exact required evidence and BLOCKED status.
Standalone inspection uses:

```bash
python -m brainagenticthon.execution.readiness --task a \
  --config /PRIVATE/task-a-config.json --output /PRIVATE/NEW-readiness-a.json
python -m brainagenticthon.execution.readiness --task b \
  --config /PRIVATE/task-b-config.json --output /PRIVATE/NEW-readiness-b.json
```

The inspector delegates to unchanged Task A eligibility and BraTS2021-specific Task
B gates. It reads/hash-checks inputs without modifying them and never calls fit.
Reports are exclusive-created, outside Git, and disjoint from the configured input
files/images. Nonempty references/approved flags must reflect real independent
reviews, not be filled in just to make software pass. A gate pass is computational
eligibility under supplied evidence, not independent clinical approval.

Task A still lacks an established registry-eligible healthy-control cohort, true
MRI Normal/Abnormal evidence, globally reconciled identities, native-T1/whole-brain
coverage, independent geometry/laterality review and qualified visual QC. OASIS-2
Nondemented/CDR=0 is not MRI-normal; Demented is not automatically Alzheimer's.
Both classes in all patient splits and source/site confounding safeguards remain.
Task B needs authorized BraTS2021 glioma files, verified masks/provenance/access,
release label evidence for 0/1/2/4, aligned geometry, reviewed file hashes, and
frozen globally reconciled patient splits. GLI/MET/MEN releases are not interchangeable.

## Training, inference and output boundaries

Task A's trained classifier is reloadable through detection.predict.load_model,
which requires the same gated cohort/config and explicit trust. predict_validation
returns uncalibrated scores only for already-gated validation images; final test
inference remains the explicit evaluate_test step. Existing patient/source/site
metrics now include descriptive Brier score, ECE and reliability bins after patient
aggregation. No calibrator or threshold is fitted on held-out test data. Small or
synthetic reliability results establish no calibration or clinical usefulness.

classification_result rejects invalid scores and declares TaskAOutput incompatible:
the existing schema requires genuine predicted boxes for a positive finding.
Reviewed expert/challenge boxes remain localization training targets only, never
classifier predictions. There is no whole-image fallback or trained localizer.

Task B preserves its supplied grid, uses synchronized crops/flips, restores native
shape via sliding windows and exports release labels with the original affine.
segmentation_result derives actual mask components and millimetre voxel volumes
only from supplied validated geometry. Six-connected boxes identify predicted
mask components; empty masks yield none. Anatomy is explicitly unavailable.
BraTS2021 0/1/2/4 cannot be silently exported into the existing approved BraTS2024
registry contract. Legacy adapters fail explicitly rather than changing that policy.
Task C and its schemas/policies remain untouched. Research contract schemas are
additive and do not claim challenge-compatible A-to-B or B-to-C integration.

## Remaining submission work

Obtain independently reviewed data/access/clinical evidence, run notebook 07 on
actual CUDA hardware and retain its report privately, perform a small approved
real BraTS overfit, then freeze experiments and run eligible training/evaluation
in notebooks 05/06. A trained lesion localizer, an authorized source-policy review
for missing cohorts/release mappings, independent anatomical mapping if required,
external validation and full challenge-contract integration remain unfinished.
No synthetic result establishes that these requirements are complete.
