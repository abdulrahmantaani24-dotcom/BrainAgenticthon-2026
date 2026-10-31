# Task A research training foundation

This is a binary MRI **classification baseline**, not the complete detection and
localization system and not clinical software. It estimates an uncalibrated
abnormality score. It does not diagnose Alzheimer's, MCI or any other disease.
No real patients were trained on during implementation. GPU execution and real
cohort validity require separate verification; synthetic performance is not
medical performance.

## Scope and unchanged contracts

`detection/model.py` implements a small residual 3D CNN (8/16/32 channels,
GroupNorm for small batches, global average pooling, dropout and one logit).
`predict_scores` returns research sigmoid scores, with no calibrated-probability
claim. No `TaskAOutput` exporter is provided: that schema requires a real bounding
box for every positive detection. A class score or saliency map cannot establish
a lesion box. A negative classifier score also does not prove no finding.

`localization.py` is a separate annotation adapter for a future supervised
localizer. It requires reviewed expert/challenge spatial annotations, image hash,
case/source identity, provenance and inclusive bounds on the native voxel grid.
Positive labels without genuine boxes fail; there is no whole-image fallback.
These are **training targets**, never predicted detections or evaluation answers.
The caller must supply `native_shape` from the independently validated source
header; the adapter rejects an annotation grid that differs from it.
Training a localizer and transforming its predictions back to the native grid
remain future work. Existing Task A output schemas, registry, readiness policy,
clinical policy and all Task B/C implementations/contracts are unchanged.

## Explicit input profile and preprocessing

`task-a-native-t1-research-v1` accepts one externally verified native T1 channel.
This differs from the existing schema documentation's four-sequence
T1/T1ce/T2/FLAIR input. It is a separate research profile, not an implementation
of that multi-sequence detector or a replacement for Task C preprocessing.
Preprocessed challenge T1 images are not automatically verified native T1.

Supported training containers are Nifti1/2 single files and paired NIfTI files
(`.hdr/.img`, including `.nifti.hdr/.nifti.img`). True Analyze containers can be
computationally inspected by the shared validator but this training profile
rejects them without coded NIfTI geometry and millimetre units. Both `(X,Y,Z)`
and `(X,Y,Z,1)` are accepted; only a confirmed singleton fourth axis is selected
with `data[..., 0]`. Other 4D data, 2D, complex, constant, corrupt and nonfinite
inputs fail. Limits are 32 million voxels and 256 MiB per source file, with
bounded target grids. At most one source volume per worker is materialized;
default workers=0 and batch=1. There is no whole-cohort tensor cache.

The loader reuses `NIfTIValidator` and the corrected shared paired validator.
It checks spacing, finite/invertible affine, coded-transform agreement, units,
companion integrity and original bytes. In memory only, it permutes/flips to
header-derived RAS, clips nonzero intensity support to its 1st/99th percentiles,
standardizes that support and trilinearly interpolates the **entire field of
view** to `(96,112,96)`. Zeros remain background before interpolation. Nonzero
support is not a brain or lesion mask. No skull stripping, lesion cropping,
spatial-axis squeezing, registration or automatic clinical QC is performed.
This fixed-grid research representation changes spatial resolution and may
distort anatomy across fields of view; it must not produce physical measurements.
Header RAS does not independently verify laterality. Geometry and native-T1
evidence are required separately. No source affine/header, MRI or manifest is
written. Hashes bind each source file to eligibility review and are checked
again on every dataset read; paired payloads require their own reviewed hash.

The old `task_a_readiness` exact-3D image gate remains unchanged. The new training
gate reuses its complete metadata audit (`check_images=False`), rejects every
ineligible row and every blocker except the explicit `image_checks_not_run`
marker, then performs this profile's mandatory image checks. This is an additive
singleton-aware research input gate, not a claim that the legacy audit passed.

## Required private inputs for real training

1. Existing `ManifestCollection` JSON with explicit `task_a_label` 0/1, actual
   diagnosis/annotation provenance, source, session, site/scanner, quality and a
   frozen train/validation/test split. Raw sources stay isolated; the manifest
   is a virtual union. Do not convert the private OASIS clinical export into
   approved labels just to satisfy the schema.
2. Existing `RecordReview` JSON list, keyed by `dataset_source`, `case_id`,
   `image_path`. Access authorization, identity reconciliation, label verification,
   native-T1 and whole-brain evidence must be genuine. Normal labels additionally
   require `healthy_control_verified`. `expected_sha256` is mandatory and refers
   to the manifest image file. For pairs, prefer the `.hdr` manifest path.
3. Separate `TrainingEvidence` JSON list with those same keys and these fields:

   | Field | Requirement |
   |---|---|
   | `manual_review_status` | Explicit `approved`; default is `pending` |
   | `visual_qc_approved`, `anatomical_geometry_verified` | Actual reviewed JSON booleans |
   | `abnormal_mri_verified` | Required true for abnormal MRI labels |
   | `mri_label_evidence_reference` | Independently documented MRI label evidence |
   | `visual_qc_evidence_reference` | Qualified visual/coverage review reference |
   | `geometry_evidence_reference` | Independent geometry/laterality review reference |
   | `modality_evidence_reference` | Native-T1 acquisition/provenance review reference |
   | `companion_sha256` | Required hash of the other file for paired input |

No example contains patient records or approved clinical attestations. A reviewer
must not set flags solely because an image loaded or a clinical Group/CDR exists.
Missing diagnosis, missing masks and empty masks never imply normality.

The gate retains the existing registry/schema source restrictions: currently only
BraTS GLI/MET/MEN are schema-eligible for A. Access and native-T1 status are still
unverified by catalogue entries. OASIS-2 is absent; OASIS-3/ADNI are A-ineligible;
the generic local flag cannot override schema restrictions. **There is no
established real normal-control cohort that can presently pass this gate.**
An independently reviewed, separately authorized source-policy extension is
needed before using a new control source. This implementation does not add one.

Nondemented/CDR=0 remains a cognitive comparison candidate, not MRI-normal.
Demented is not automatically Alzheimer's; CDR=0.5 is not automatically MCI.
Converted visits retain their original clinical policy. The current readiness
policy blocks inconsistent within-patient binary labels rather than collapsing
longitudinal change into one diagnosis. A future longitudinal experiment needs
its own reviewed endpoint; do not relabel visits to make this gate pass.

## Leakage, confounding and evaluation

Reuse of `SplitManager` through the readiness audit checks subject/case leakage;
globally reconciled `RecordReview.patient_id` additionally binds cross-source
identity. All visits/images from one patient must stay in one split. Duplicate
paths and exact bytes fail. Hash checks do not detect all resaved/near-duplicate
scans, so independent identity reconciliation remains necessary.

Both classes must exist in train/validation/test. Perfect source/site confounding
fails; the additional gate also rejects any pure-class source or source/site
group, including partial confounding that the old global rule alone misses.
This does not establish absence of subtler scanner shortcuts. Review the private
readiness crosstabs and predefine source/site-held-out experiments and an
independent test cohort. A normal-OASIS versus tumor-BraTS experiment cannot pass.

Training samples patients with inverse image-count weights. Validation/test first
average image scores per globally reconciled patient, then compute AUROC,
sensitivity, specificity, precision, recall and `[[TN,FP],[FN,TP]]`. Source and
source/site reports use anonymous integer groups; they never print identifiers.
Single-class AUROC and zero-denominator metrics are null with documented meaning.
Scores are uncalibrated. Threshold defaults to a preregistered 0.5; any selection
must use validation only and be frozen before test. Best weights use validation
patient AUROC. `fit` never loads test images; final evaluation requires an explicit
call and writes a separate aggregate report. Continuing a run after its test
report exists is refused. Independent calibration, confidence intervals and
clinical usefulness remain unestablished.

## Colab execution, including unpublished code

1. Locally build a reviewed **code-only** bundle; this does not include private
   data, Visual QC work, weights or clinical manifests:

   ```bash
   .venv/bin/python scripts/build_task_a_bundle.py --output /tmp/task-a-code.zip
   ```

   Record the printed SHA-256 independently. Upload that ZIP to authorized private
   Drive. Keep the notebook itself private once used; disable saving outputs and
   never share runtime state, screenshots, logs, manifests or trained weights.
2. Open `notebooks/05_task_a_research_training.ipynb` privately in Colab. Select a
   GPU runtime for real training and Python 3.12 for the tested dependency profile.
   Set `BUNDLE_PATH` and its independently retained `BUNDLE_SHA256`. Run cells in
   order: mount Drive, verify/extract the bounded bundle, install pinned packages,
   import modules. A fresh runtime is recommended; notebook checks Torch/NumPy/
   NiBabel versions and refuses stale imports after installation.
3. Copy the example config to private Drive, choose a fresh `output_run` name,
   set `CONFIG_PATH` to that private file and first run with `mode="synthetic"`.
   It trains one CPU epoch on generated tensors only. It is an engineering check.
4. Prepare the three reviewed private JSON inputs above. Place source images in
   the separate configured `image_root`. Copy the example configuration to a
   private Drive file, update its generic paths and `output_run` to a fresh name,
   then set `CONFIG_PATH` in the notebook. No patient files go into the code tree.
5. Only after approvals and source-policy eligibility genuinely exist, set
   `mode="verified"`, `real_training_enabled=true`, `device="cuda"` in that private
   config. Use batch=1, accumulation=2, AMP=true, workers=0 initially on a T4/L4.
   Run the gate cell; only aggregate eligibility counts print. Failure stops the
   run. No class labels or splits are generated. Adjust batch size only after
   measuring GPU memory; changes require a new run.
6. Run training. Epoch checkpoints, `latest.json` and `history.json` go only to
   the configured private Drive run directory, outside Git and disjoint from
   source images/metadata. No experiment-tracking network service is used.
7. After interruption, rerun imports/config/gate, keep the same run directory,
   set `resume=true`, `trusted_resume=true`, and specify a greater total epoch
   count. Only resume checkpoints you created and trust. Completed epochs are
   not duplicated; an incomplete epoch is replayed. The atomic commit marker
   ignores an uncommitted generation after an interrupted save. Checksums detect
   accidental corruption, not authenticity. Loading always uses
   `torch.load(..., weights_only=True)` with size bounds and CPU mapping.
8. Resume requires unchanged data/evidence hashes, profile, code, dependencies,
   seed, device and optimization settings; only total epochs may increase.
   RNG, sampler, optimizer and AMP scaler state are restored. CPU equivalence is
   regression-tested; CUDA reproducibility across hardware/driver changes is not
   guaranteed. Drive disconnects can interrupt saves; retain authorized backups.
9. Freeze the experiment, then set `final_test_evaluation=true` only for its final
   held-out evaluation cell. Reports contain aggregate metrics only. This is not
   clinical approval or independent calibration validation.

## Offline checks and remaining blockers

CPU smoke command, with generated tensors and weights outside the repository:

```bash
PYTHONPATH=src .venv/bin/python -m brainagenticthon.detection.training \
  --synthetic --epochs 1 --output /tmp/task-a-synthetic-NEW-RUN
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
git diff --check
```

The tests cover shape/geometry/intensity failures, pairing and byte preservation,
metadata/evidence rejection, patient leakage and source confounding, exact metric
values, actual CPU gradient updates, trusted/integrity-bound resume, interrupted
checkpoint recovery and notebook/bundle structure. All fixtures are synthetic;
invented BraTS-like controls are not claims of real BraTS healthy subjects.

Remaining blockers for real training: an eligible authorized normal/abnormal
cohort, verified labels and review evidence, native-T1/geometry compatibility,
independent source/site evaluation design and a real Colab GPU smoke run. The
localization model, multi-sequence detector compatibility, clinical calibration
and Task A -> B/C inference integration are additional unfinished deliverables.
The successful OASIS computational audit can support inventory and safe IO; it
cannot approve these prerequisites. Notebook 04 or equivalent qualified review
can supply visual evidence but does not establish diagnoses or source eligibility.
