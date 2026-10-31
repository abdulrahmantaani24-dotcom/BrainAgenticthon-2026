# Four-source private MRI inspection and research handoff

Notebook 09 is the primary entry for this sprint. It uses one reviewed bundle for
notebooks 05–09, Python 3.12, and an optional pinned pydicom 3.0.1 dependency for
DICOM **headers only**. No DICOM pixel decoder, conversion or archive extraction
runs. No real dataset is included. Inventory mode never triggers training.

## Evidence boundary and current matrix

The user reported these private directories under
`/content/drive/MyDrive/BrainMRI/datasets/DiseaseAcquisition/`:

| Directory | User-reported file presence | Current Task A status | Current Task B status |
|---|---|---|---|
| hwang_meningioma_5945963 | 925 files, 915 NIfTI gzip; source/license and manifests | Excluded by current source registry; research candidate pending clinical evidence | Candidate pending genuine masks, disease semantics, modalities and reviewed adapter |
| isles2022 | 1,699 files, 1,000 NIfTI gzip, 250 PNG, JSON and ZIP | Excluded by current source registry; research candidate pending clinical evidence | Candidate pending verified stroke annotation/modality/access evidence and reviewed adapter |
| oasis2_legacy | 3,392 files, 1,368 header/payload pairs | Excluded by existing source policy; technical audit is not MRI-normality evidence | No verified segmentation cohort; candidate only if genuine annotations are independently established |
| proteas_23171182 | 75,908 files, 74,381 DICOM, 1,192 NIfTI gzip, 45 ZIP | Excluded by current source registry; research candidate pending clinical evidence | Candidate pending masks, disease/sequence semantics, conversion provenance and reviewed adapter |

These are **user-reported presence counts**, not observations by the cloud worker,
patient counts, clinical labels or access grants. No independent BraTS directory
was reported. The workflow performs no implicit BraTS search. None of these four
sources is enabled for current A/B training. Pretraining/auxiliary use is not
approved automatically. A task adapter alone cannot override a clinical/source
policy. Existing registry, policies, Task C and teammate work remain unchanged.

## Minimal private launch

1. From the exact published code commit, build one code-only bundle:

   ```bash
   python scripts/build_task_a_bundle.py --output /tmp/task-ab-code.zip
   sha256sum /tmp/task-ab-code.zip
   ```

   Independently retain the actual archive's SHA-256 and upload the ZIP to private
   Drive. Deterministic builds are checked on the same Python/zlib runtime; hash
   the actual reviewed archive rather than trusting a hash stored alongside it.
2. Open notebook 09 in Colab, save a **private** copy and disable output saving.
   Set bundle path/hash in the first code cell, then Run all. CONFIG_PATH=None
   uses the exact four requested roots; one optional private JSON config can
   override paths/limits and link reviewed task configurations. No per-cell path
   repair is needed. After Restart session rerun from the top with saved parameters.
3. Read summary.json for aggregate format/coverage and conservative task blockers.
   inventory.private.json includes filenames, metadata snippets, pseudonym hashes
   and supplied references which may identify patients. Never publish or share it.

The supplied request authorizes local read-only inspection of the four roots;
`inspection_authorized=true` records that scope, **not** access/license/clinical
approval for training. Set false to skip a source. Training evidence flags in the
separate review template default false/pending. Missing Drive roots produce
BLOCKED, never invented counts. Existing output directories are never replaced.

## Bounded technical and evidence inspection

Per-source defaults count up to 100,000 files/directories, sufficient for the
reported PROTEAS tree. Hard caps prevent arbitrary expansion. Header sampling is
at most 4,000 NIfTI/Analyze candidates and 256 DICOM files; metadata sampling is
64 JSON/CSV files. File reads/hashes are byte-bounded; the default MRI hashing
budget is 512 MiB per source. Archives and PNG files are counted only. Explicit
coverage fields prevent a sample from being mistaken for full-cohort verification.
Symlinks abort the affected source. Errors/limits preserve partial private evidence.

NIfTI/Analyze inspection reads headers only: shape, affine, orientation, spacing,
units and coded transform agreement where available. Analyze unit/geometry gaps
remain explicit. A header does not establish native T1, whole-brain coverage or
laterality. DICOM inspection uses stop_before_pixels and an explicit tag allowlist.
It records documented MR modality, sequence/protocol candidates and plane geometry
without reconstructing a volume affine. IDs/UIDs are hashed inside private reports;
they are not reconciliation proof and should not be treated as anonymized public data.

Source/license JSON, file/patient manifests and conversion records are bounded
metadata candidates. Strings or mere document presence never create approvals.
Common explicit manifest fields are diagnosed for case/patient conflicts,
patient split leakage, volume/hash identity conflicts and source/class confounding.
Unknown schemas and aliases require independent reconciliation. DICOM SOP/patient
conflicts and exact MRI byte duplicates are checked within sampled coverage.
Resaved, compressed, cropped or near-duplicate scans may evade exact hashing;
independent near-duplicate/identity review remains mandatory. No patient count is
extrapolated from scan, series, file or manifest samples.

For source review, privately copy disease_source_review.template.json to the
configured source's task_ab_review.json. Supply the exact source_id, independent
approved review, genuine per-field references and optional relative evidence
files with reviewed SHA-256. The software verifies referenced bytes if provided;
it cannot authenticate a reviewer or adjudicate medical truth. Document diagnosis
and mask semantics separately. Nondemented/CDR=0 never means MRI-normal; Demented
never automatically means Alzheimer's. Empty/missing masks never create normals.

Explicit reviewed_pairs may map arbitrary **documented** disease modalities to
image files and a genuine mask, with patient_id/case_id/split, approved review,
identity/modality/mask-semantics references, conversion_required (true/false),
conversion_reference when applicable, label_definitions, background_label=0, and
path/sha256 mappings. Values/semantics must be independently reviewed; no clinical
patient examples are published. Checks stream bounded reviewed masks, preserve
supplied coded mm geometry, reject noninteger/nonfinite/unknown mask labels and
misalignment, and flag repeated identities/bytes/splits. No registration or mask
interpolation occurs. If a future resampling adapter is reviewed, masks must use
nearest-neighbor interpolation and retain an invertible original-grid transform.
This is an annotation inspection adapter, **not** authorization to train a new
non-BraTS model or to relabel meningioma/stroke as glioma subregions.

## Existing gated cohort, splits and training

Optional task_configs.a/b point to private readiness configs relative to the
private project root. Notebook 09 supports explicit validate, freeze, smoke,
train, resume and evaluate actions through the existing Notebook 08 implementation.
No config is generated from filenames or sampled clinical strings. An absent or
ineligible config remains BLOCKED. Task A still requires an approved true MRI
Normal/Abnormal cohort, all clinical/native-T1/QC/identity reviews, both classes
in each patient split, and absence of source/site class confounding. Pairing an
all-normal source against an all-abnormal source does not satisfy these gates.
Task A is a one-native-T1 classifier; four-sequence detection remains unsupported.
Task B's active training profile remains reviewed BraTS2021 glioma T1/T1ce/T2/FLAIR
with genuine 0/1/2/4 masks. No availability or permission for that cohort is assumed.

Only a passing full authoritative gate can freeze existing patient assignments,
copy private metadata/evidence, record exclusions and bind hashes/provenance.
No new clinically validated manifest or split can be generated here without
actual source evidence and reconciled identities. The existing freeze routine
never invents diagnoses or randomizes splits to obtain a passing cohort. Keep
source files and clinical/audit checkpoints immutable. See task_ab_real_data_workflow.md.

Real smoke/train/resume/evaluate additionally require explicit verified mode,
real_training_enabled=true, an unchanged freeze and actual CUDA. Smoke reads one
training batch/patch; full training needs a different fresh output run. Existing
AMP, patient sampling/gradient accumulation (A), synchronized crops/flips (B),
atomic last/best checkpoints, checksums, RNG restore and trusted resume remain.
Final test evaluation is explicit and one-shot, after validation-only model
selection. No test-set tuning occurs. Synthetic overfit checks are not real-data
training. A real limited overfit/schedule cannot run until its cohort passes.

## Architecture, evaluation and interfaces

Task A: residual 3D CNN (8/16/32 channels, GroupNorm, pooled one-logit head),
uncalibrated sigmoid research scores. Existing validation/test metrics aggregate
by reconciled patient and source/site: AUROC, sensitivity, specificity, precision,
recall, confusion matrix, Brier/ECE/reliability bins. No clinical calibration claim.
Trusted inference entrypoints: detection.predict.load_model/predict_validation.
Positive TaskAOutput requires genuine predicted localization; classifier scores
cannot supply boxes. Expert annotations remain training targets, not predictions.

Task B: compact width-8 two-level 3D U-Net, four modalities/four internal classes,
CE plus soft Dice, patch learning and full supplied-grid sliding-window inference.
segmentation.predict reloads trusted weights and exports the original affine/grid
with release label 4 restored. region_dice computes WT/TC/ET for the active BraTS
profile. lesion_metrics accepts explicit binary foreground masks plus a semantics
reference for generic disease lesion Dice, greedy one-to-one component-IoU
sensitivity/false positives, exact symmetric boundary HD95/mean distance and volume
error where **verified mm geometry** is supplied. Empty/oversized surfaces produce
null boundary distances with reasons, never fabricated values or hidden subsampling.
These are research conventions, not claims of challenge leaderboard equivalence.
Anatomical atlas fields remain unsupported. Boxes come from real mask components.

execution.handoff.research_envelope provides an optional serializable research-only
payload for Rand's backend or Fadwa's agent, bound to a checkpoint SHA-256. It
retains unsupported legacy exports and never invokes Task C. It changes no teammate
routes, services, preprocessing or weights. Inputs should remain authenticated and
private; integration owners must review an actual endpoint before deployment.

Model/checkpoint inventories and evaluation templates remain under the existing
private workflows. No weights, private reports, observed diagnoses or medical
performance are bundled. Submission requires actual independent clinical/source
review, an eligible cohort, real GPU training/evaluation evidence, localization
completion, approved dataset-specific adapter/source changes and any required atlas.

## Reproducibility and Docker/integration handoff requirements

Pin the code commit and bundle hash, Python 3.12, requirements-task-a.txt and optional
requirements-inventory.txt. Record GPU/driver/CUDA, seed, optimization settings,
cohort/freeze hashes and private checkpoint hashes. Source data must be mounted
read-only and outputs separately writable. A future GPU container needs an
NVIDIA-compatible CUDA runtime, the pinned packages, no baked-in Drive credentials,
no scans/weights in its build context, and no automatic real-data action on startup.
No Docker deployment or endpoint publication is performed by this sprint.

Acceptance: full/focused offline tests, empty notebook outputs/schema validation,
deterministic allowlisted bundle, staged privacy/secrets checks and no new Ruff
findings. Preserve pre-existing lint findings separately. Do not merge/main or open
PRs. Retain independent audit artifacts and communicate all remaining blockers.
