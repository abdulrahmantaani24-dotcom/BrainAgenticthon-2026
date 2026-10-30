# Task C integration readiness

This documentation-only proposal carries forward the completed read-only audit.
It does not implement an adapter, change model behavior, or declare Task C ready
for deployment. All checklist items below remain open.

| Baseline | Revision |
|---|---|
| Main repository under review | `abdulrahmantaani24-dotcom/BrainAgenticthon-2026` at `1d235147cca72b888dc2028a0e5e3eed1711d00c` |
| Task C audited baseline | `M0maniiR0/MRI-` at `1c1c997` — full SHA `1c1c99730788a733484aa92f09057c144b37f8fb` |
| Documentation branch | `docs/task-c-integration-readiness`, created from the verified current remote `main` commit above |

The cloud checkout initially had only a local `work` branch. A read-only remote
lookup confirmed that `refs/heads/main` still points to the audited main SHA;
the feature branch was created directly from that commit without changing main.

## Documents for review

- [Complete audit and proposed file tree](task-c-audit.md)
- [All 21 risks, R01–R21](task-c-risk-register.md)
- [All 26 proposed acceptance-test groups, C01–C26](task-c-acceptance-tests.md)
- [Contract preservation, strict offline verification, and independent calibration requirements](task-c-verification-requirements.md)

## Verified test history and limits

The original audit executed the main suite: **201 passed, zero failures, zero
errors, zero skips**. One Task C calibration-parameter test **failed** because its
expected keys do not match the shipped JSON. Task C collection produced **three
dependency-related errors**: SciPy was missing for preprocessing, and torch was
missing for two inference test modules. That collection-only run exited 2; it
was not a successful suite execution. **Task C did not pass all tests.**

Five isolated source/metadata probes reproduced findings; they did not run the
complete preprocessing pipeline or checkpoint. No checkpoint inference, clinical
benchmark reproduction, strict offline model execution, or independent calibration
evaluation was performed. The 26 proposed acceptance-test groups remain unrun.
These historical tests were not rerun for this documentation-only change.

## Prioritized blocker checklist

The categories below describe work boundaries and may overlap. “Critical blockers”
means release-gating prerequisites; it does not change the original severity
rating of any risk. References retain the audit's risk and test IDs. Checking an
item requires reviewable evidence, not merely a design or a passing mocked test.

### Critical blockers

- [ ] **R01, R02 — establish a safe, reproducible runtime before loading weights.**
  Resolve the torch/MONAI pin conflict using the actual training environment and
  a supported patched runtime; verify authenticated artifact digests before
  constrained loading. No unsafe pickle fallback. Evidence: C07, C19, C20.
- [ ] **R03, R04 — separate model scores, calibration scope, and routing scores.**
  Reconcile three-class calibration/evaluation with ten-logit inference in a new
  opt-in interface; label existing QC/routing allocations as policy-only in a
  sidecar. Keep legacy JSON and calibration values unchanged. Evidence: C08–C11,
  C24 and independent evaluation requirements.
- [ ] **R05, R06 — enforce support limits and withhold unsupported claims.**
  Only Glioma, Meningioma, and Metastasis have training/evaluation support. Block
  unsupported/failed-QC projection into main. Do not claim the uncertainty router
  reliably detects unseen diseases. External generalization/calibration claims
  require C25, not just synthetic support-gate tests C04–C06.
- [ ] **R07 — clarify rights and provenance before importing code or weights.**
  Obtain owner-approved code/model licensing, source release/DUA provenance,
  trusted bundle hashes, and the authoritative competition contract. Public GitHub
  visibility is not redistribution permission. No checkpoint transfer, deletion,
  or history rewrite is authorized by this checklist.

### High-priority integration risks

- [ ] **R08, R12 — make valid input and QC requirements explicit.** Require native
  non-contrast T1, an explicit channel/axis for 4D inputs, valid affine/dimensions,
  raw finite-value checks, and typed QC outcomes. Do not silently use MSD channel
  0 or normalize malformed data into a passing result. Evidence: C11–C14.
- [ ] **R09, R10 — protect spatial meaning.** Initially reject DICOM in the new
  adapter; its legacy conversion loses geometry. Define an original/model-space
  transform trace without changing trained tensor values or claiming cropped
  tissue can be reconstructed. Evidence: C14, C15, C17, C18.
- [ ] **R11, R14 — preserve pretrained input semantics.** Keep mask, orientation,
  spacing tolerance, crop/pad, normalization, and precision behavior versioned.
  Quantify tissue/lesion loss and tensor/logit drift before considering a change;
  assess retraining/recalibration separately. Evidence: C15–C19.
- [ ] **R13, R18 — preserve true source and split identities.** Obtain frozen
  manifests, content/config fingerprints, source-scoped IDs, and cross-release
  patient deduplication. Do not label MSD or BraTS-MEN 2023 as BraTS 2024 to make
  a validator accept them. Evidence: C23–C25.
- [ ] **R15 — add an independent optional bridge.** Preserve `TaskCOutput` and
  `classification.json`; use a new result envelope and metadata sidecar. Never
  represent abstention as `no_finding` or invent A/B outputs. Evidence: C01–C06,
  C10, C26.
- [ ] **R16, R17 — establish portable tests and observable offline/error behavior.**
  Preserve existing tests, add focused regressions, record failures/skips honestly,
  and verify the real installed package with networking disabled and monitored.
  No hard-coded `offline_verified` flag counts as evidence. Evidence: C02, C11,
  C20–C24.

### Issues requiring Task C owner approval or supplied evidence

- [ ] **R07, R20 — code/model and derivative distribution.** Confirm licensing,
  attribution, exact source-data terms, public `best_model.pt` provenance, and
  medical-image PNG distribution/de-identification. No PHI finding was asserted
  from the uninspected images. Any removal/relocation or history rewrite needs a
  separate explicit decision.
- [ ] **R13, R21 — training lineage.** Supply the actual environment export,
  frozen biological-patient splits, checkpoint/config hashes, loss configuration,
  and run metadata. Resolve the class-weighted loss description and inconsistent
  historical cohort counts without inventing a replacement training history.
- [ ] **R03–R06, R17 — probability and contract decisions.** Confirm the exact
  competition specification and accept the documented support/calibration limits.
  Any new numerical output profile, thresholds, CLI failure behavior, or
  recalibration is separately versioned and approved; this task changes none.
- [ ] **R08–R12, R14 — preprocessing/precision changes.** Approve the reference
  fixtures and parity tolerances. Changes to modality, mask, crop/FOV, affine
  interpretation, resampling, normalization, or precision require a model-impact
  assessment and potentially retraining; packaging is not permission to change
  them.
- [ ] **R06, R13, R18 — independent evaluation access and protocol.** Establish
  approved cohort access, provenance, deduplication, untouched test membership,
  and predeclared acceptance thresholds with the data/evaluation owners. Do not
  download medical data or refit calibration as part of this documentation task.

### Work that can proceed independently in Task A and Task B

- [ ] **R19 / Task A:** define a genuine normal-versus-abnormal detection contract
  and independent data/evaluation plan using explicitly verified normal controls.
  Task C's conditional tumor-type scores cannot supply this detector. Existing
  `TaskAOutput` requires boxes for a positive result; do not fabricate localization
  for a scan-level binary detector. Evidence: C01, C26.
- [ ] **R10, R19 / Task B:** define native-space lesion-mask geometry, source-specific
  label conventions, localization/segmentation, atlas/laterality mapping, lesion
  counts and affine-based volumetry, with synthetic known-geometry tests. Task C's
  brain mask is not a lesion mask. Evidence: C17, C26.
- [ ] **R13, R18 / shared A/B data work:** design source/patient/session provenance
  and cross-release deduplication that also protects C's held-out split, retaining
  the existing manifests, validators and tests. Evidence: C01, C23, C25.
- [ ] **R15, R19 / composition:** keep A/B absent or optional in the new C envelope.
  A/B design and unit-test work need not wait for C weights or licensing, and C
  packaging must not require A/B models. End-to-end readiness requires real
  validated outputs from each component.

## Review boundary

Only the five Markdown documents in this directory are added. Model source,
preprocessing, checkpoint locations/bytes, calibration values, dependency
declarations, existing schema contracts, manifests and tests are unchanged.
The original audit evidence remains outside the checkout. No commit, push,
merge, pull request, model run, retraining, data acquisition or publication is
part of this change.
