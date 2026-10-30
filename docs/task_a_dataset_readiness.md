# Task A: read-only dataset readiness

Task A needs a **verified normal-versus-abnormal 3D MRI cohort**, not simply a
collection of tumor scans. The new audit reads existing manifests, review evidence
and optional local NIfTI files. It never downloads, copies, renames, transforms or
labels an image, assigns a split, trains a model, or changes any registry/schema.
It does not require Task B or Task C. Task C's audited `1c1c997` model,
preprocessing, calibration and `classification.json` contract remain untouched.

This workflow starts from main commit
`1d235147cca72b888dc2028a0e5e3eed1711d00c`. The independent documentation PR #1 is
not part of this branch and is neither modified nor merged.

## Current source catalogue and access requirements

Run the offline inventory from an installed checkout:

```bash
python -m brainagenticthon.preprocessing.task_a_readiness --list-sources
```

This reads `APPROVED_SOURCES`; it does not authenticate or check a remote grant.
Catalogue membership, approved research access, and Task A eligibility are three
different things. **No medical dataset access or healthy cohort is established by
this change.**

| Source IDs | Current Task A status | Access requirement recorded in registry |
|---|---|---|
| `brats2024_gli`, `brats2024_met`, `brats2024_men` | Registry and manifest allow Task A; source names do not supply a normal-control cohort. Every label still requires review. | Synapse account, cohort-specific granted access and accepted BraTS 2024 DUA; real entity IDs replacing placeholders; securely configured Synapse credentials. |
| `oasis3` | Registry explicitly disallows Task A. A manifest with a Task A label also fails existing schema validation. | OASIS registration and signed DUA; approved portal access. |
| `adni` | Registry explicitly disallows Task A. A manifest with a Task A label also fails existing schema validation. | ADNI/LONI application and DUA; registry additionally records IRB approval. Confirm current provider requirements. |
| `figshare_2d` | 2D, pretraining-only; excluded from Task A regardless of its tumor/no-tumor label. | Registry records public CC-BY-4.0 access; this does not authorize using 2D images as 3D controls. |
| `local` | Registry marks eligible, but existing manifest validation allows Task A labels only for the three BraTS sources. The audit respects both checks and reports the discrepancy. | Actual source, licence/DUA and access must be documented; registry licence is unknown. A local folder is not permission. |

OASIS/ADNI have potentially relevant clinical control metadata, but **are not
already Task A-approved**. Their source-wide restriction currently also excludes
normal-vs-abnormal use without segmentation. Any change needs a separate reviewed
registry/schema decision and backward-compatibility tests; the audit has no
override flag. Never alias them to BraTS or `local` to bypass the restriction.
The unmodified tests continue to enforce their current status.

## Explicit eligibility rules

Every candidate must satisfy all of the following:

1. Existing `validate_record` succeeds, the source is Task A eligible in the
   registry, and the source is not 2D/pretraining-only. Preserve original source,
   release, modality and image path; no data pooling or source relabeling.
2. `task_a_label` is an explicit integer **0** or **1**, with a non-missing
   `original_diagnosis`, `clinical_record` or `challenge_annotation` provenance,
   and a reviewer reference verifying the Task A label. Inferred labels are
   ineligible. Empty/unknown diagnoses and missing labels are reported, never
   filled from Task C, directory names, masks or segmentation absence.
3. A **normal (0)** case additionally has a recognized explicit control diagnosis
   (`Normal`, `healthy control`, `control`, `CN`, or `no finding`) **and** an
   explicit healthy-control attestation for Task A's MRI scope. Cognitive CN alone
   does not establish a neurologically normal MRI. Reviewers must examine the
   original diagnostic criteria and scan/session correspondence. Other control
   vocabularies need an explicit reviewed mapping, not automatic guessing.
4. An **abnormal (1)** case has an explicit non-control diagnosis and reviewed
   clinical/expert evidence of an abnormal brain MRI. A segmentation is neither
   required nor used as proof of a diagnosis. Negative, missing or absent masks
   are never healthy-control evidence. Mixed/ambiguous labels require review.
5. Access/licence and evidence reference are verified; the declared licence must
   match the source's registered licence/DUA requirement. A globally reconciled,
   de-identified patient ID is supplied. Review evidence records authorization,
   not credentials; no API keys, PHI or clinical notes belong in Git or reports.
6. Use native non-contrast T1 for comparability: accepted declared aliases are
   `t1`, `t1w`, `t1n` (case insensitive), plus explicit native-T1 verification.
   T1ce/T1c, FLAIR, T2, segmentation and 2D images are excluded. An arbitrary
   channel from a 4D image is not automatically T1. Whole-brain coverage and
   `quality_flag="ok"` need review; dimensionality/field size cannot prove that a
   scan covers the whole brain or lacks motion.
7. Actual local NIfTI checks must pass: exactly 3D, plausible dimensions, finite
   non-singular affine, agreement of coded qform/sform when both are present,
   plausible positive spacing, finite nonconstant intensities,
   and readable bytes with SHA-256. Reuse `NIfTIValidator.validate_all`; block its
   errors and warnings except native non-RAS orientation, which is reported
   without changing the image. Limit each scan to 64 million voxels before full
   reads; stream hashes and process sequentially. An optional expected SHA-256
   verifies byte identity. No resampling, normalization or skull stripping occurs.

Reviewer attestations are required inputs and cannot be independently proven by
software. They must be grounded in authorized evidence; never set flags to true
merely to get a ready report. Current BraTS catalogue entries do not establish
that any verified normal controls exist. Synthetic test controls are **test-only
fixtures**, not invented clinical cohort labels.

## Virtual manifests and separate evidence

Reuse `ManifestCollection` and its JSON format: a list of `ManifestRecord` objects
or the existing `{"records": [...]}` envelope. `merge_manifests` returns a virtual
union without copying raw images or removing duplicates. Keep each source/release
in its own raw directory. Relative image paths resolve from the explicitly supplied
`--image-root`; absolute paths are retained. Reports contain pseudonymous row IDs
and hashes, not raw paths or reviewer reference text. They still require secure
storage because pseudonyms are sensitive metadata.

Provide a **separate JSON list** of review entries. Each entry's
`dataset_source`, `case_id`, `image_path` must exactly match a manifest row.
Never modify a diagnosis or Task A label to make it match the review file.

| Review field | Required meaning |
|---|---|
| `patient_id` | Globally reconciled pseudonym across sources, names, sessions and releases; same person uses the same ID. |
| `evidence_reference` | Nonempty secure record of who/what verified the assertions; no embedded credentials or PHI. |
| `patient_identity_verified` | Cross-source identity resolution/overlap review completed. Different raw subject IDs may still represent the same patient. |
| `access_authorized` | Authorized use and licence/DUA requirements have been verified for the exact dataset and purpose. |
| `label_verified` | Existing explicit Task A label verified against original clinical/expert evidence for the correct scan/session. |
| `healthy_control_verified` | Additional MRI-normality verification, required for label 0; default false. |
| `native_t1_verified` | Sequence is native non-contrast T1, not inferred from filename alone. |
| `whole_brain_verified` | Reviewed whole-brain coverage, not a cropped lesion/hippocampus ROI. |
| `expected_sha256` | Optional trusted lowercase 64-character file digest. |

All verification flags default false and accept JSON booleans only. Missing
reviews, repeated review keys and malformed evidence fail conservatively. An empty
review file is `[]`; it grants no access or eligibility. No evidence field
overrides source restrictions. Retain the original frozen patient/split manifests;
the audit never calls `assign_splits` or rebalances a cohort.

## Readiness audit and interpretation

After appropriate access and verification, on the machine containing the data:

```bash
python -m brainagenticthon.preprocessing.task_a_readiness \
  --manifest /secure/manifests/source-a.json \
  --manifest /secure/manifests/source-b.json \
  --reviews /secure/manifests/task-a-reviews.json \
  --image-root /content/task_a > /secure/reports/task-a-readiness.json
```

Paths are placeholders for already authorized files, not a dataset download or a
claim that two sources are approved. The command only prints JSON to stdout; shell
redirection is an explicit caller-controlled report write. Keep that output
outside raw/source directories. Exit codes are **0** ready, **1** not ready, **2**
input/dependency/audit error. Source inventory alone exits 0 and is not readiness.
`--metadata-only` never reads images and **always reports not ready**; its counts
are labeled metadata candidates, not image-validated eligibility.

The report includes:

- Total, eligible/ineligible records, unique eligible patients per class, per-row
  eligibility reasons, aggregate missing-label warnings, actual image checks and
  image geometry/hash results. Manifest and review SHA-256 fingerprints plus the
  audit version identify inputs; preserve an environment export alongside them.
- Existing `SplitManager.check_leakage` violations and heuristic raw subject ID
  overlaps; source/site/split statistics; additional globally reconciled patient
  checks spanning train, validation, test **and holdout**. Same-source subjects
  cannot be assigned inconsistent global identities. Conflicting patient labels
  require review, including real longitudinal changes; the conservative audit
  does not decide such clinical questions.
- Duplicate resolved image paths and identical file bytes under any names, which
  block readiness. Legitimate repeat sessions in the same patient/split are
  counted once per patient/class in each table group, not mislabeled as duplicate
  patients. Cross-source raw ID equality is an overlap warning, not proof of
  identity; reviewed global IDs govern cross-source split leakage. Different
  aliases require real identity reconciliation. Byte hashes do not detect all
  re-encoded copies, so neither ID checks nor hashes prove no biological overlap.
- Declared (unverified/ineligible included) and eligible patient-count source/class
  and source-site/class crosstabs, plus eligible split/class and source-site-split
  tables. Group keys are JSON arrays (for example `["brats2024_gli", "site-1"]`)
  to avoid delimiter collisions. Repeated scans do not inflate patient counts.
- A perfect-confounding flag when both classes are present and every observed
  source (or source/site group) contains only one class. This blocks readiness
  even if each split has both classes: source recognition could solve the task.
  These checks also expose confounding among excluded candidates. A false flag
  is **not** proof of no shortcut learning; mixed-source cohorts still need
  protocol, scanner, intensity, geometry and independent-site evaluation.

`ready` applies only to the **eligible subset for offline development**. It
requires both classes in each of train/validation/test (at least one distinct
patient of each class per split), completed image checks, no detected
leakage/duplicates, complete site/scanner metadata, and no perfect source/site
confounding. Ineligible auxiliary rows remain visible. A minimum of one patient
per class is a structural check, not adequate statistical power or clinical
validation. The audit does not approve a source, train a model, verify privacy,
prove complete patient deduplication or promise clinical performance.

## Source-aware and site-aware evaluation design

Prespecify independent evaluation before any future training. Retain patient and
session grouping across **all** A/B/C releases and their historical overlaps.
Balance normal/abnormal support within acquisition source and, where possible,
within site/scanner/protocol. Use the reported tables to identify missing strata.
Do not fabricate or relabel healthy controls to fill them.

Plan patient-disjoint train/validation/test and genuinely independent source/site
holdout cohorts. For each holdout fold, check whether both classes are supported
and report missing/unsupported strata; do not quote a pooled score as source
generalization. Source-held-out folds where source perfectly matches class cannot
separate site learning from disease learning. Selecting thresholds or normality
criteria on the held-out test invalidates that test.

Later, separately authorized model evaluation should include source/site-only
baselines and acquisition/geometry/intensity shortcut probes, per-source/site and
per-class sensitivity/specificity, calibrated operating points, false positives
on verified controls, confidence intervals grouped by patient/site, and comparison
with untouched external sites. The audit reports data risk; it does not execute
these model experiments or claim reliable detection of unseen pathology.

## Authorized Colab access without a local Windows data download

The existing [cloud workflow](cloud_data_workflow.md) and
[data policy](data_policy.md) remain applicable. Use a reviewed exact code commit
in Colab, selecting the appropriate feature branch or published revision once
available. Do not assume the default main checkout contains these commands or
that copying a notebook executes the new tests.

1. Obtain the actual provider's approval and current DUA/other institutional
   requirements. Record dataset releases, permissions and actual Synapse IDs;
   the placeholder IDs cannot be used. OASIS/ADNI acquisition permission alone
   does not change their Task A registry/schema restrictions.
2. In an authorized Colab runtime, check out the reviewed exact code commit and
   create a Python environment with the project's dev extras plus `nibabel` and
   `pyyaml` (these imaging/YAML packages are needed by existing offline tests but
   are not declared as core project dependencies). Install `synapseclient` only
   if a later authorized BraTS acquisition requires it. Do not upload a token to
   chat, notebook output or version control.
3. Use Colab's secure secret mechanisms or a privately provisioned Synapse config
   appropriate to the provider's SDK. Limit its permissions, suppress credential
   output and validate actual access. For BraTS, configure the existing
   `BraTS2024Source` with the approved cohort/entity ID, credential path and an
   output directory **inside Colab**, such as `/content/task_a/raw/brats2024_gli`.
   Its `fetch(dry_run=True)` may authenticate/list remote cases; it is not an
   offline check or a grant of download permission. Review the listing before any
   separately authorized `fetch` call. No such calls are run by the audit.
4. When downloads are separately authorized, acquire **directly into Colab** or an
   approved cloud storage volume, isolated by source/release. Do not download to
   Windows first. Ephemeral `/content` data disappear on runtime reset; persist
   raw data in private encrypted/provider-approved storage only if the DUA and
   institution allow it. A private Drive mount is optional and not automatically
   approved storage. Never make folders or medical-image previews public.
5. Build manifests in Colab from actual provider metadata, independently review
   labels/controls and globally reconcile patients. Keep raw paths and original
   diagnoses intact; create secure review sidecars. Run this audit there. Export
   only approved de-identified aggregate reports/code, never raw MRI, patient
   metadata or credentials to GitHub/the local Windows machine.

The existing data-access notebook targets Colab acquisition/storage, not automatic
Task A eligibility. Do not run all cells blindly or treat its example paths and
placeholder credentials as approved configuration.

## Offline validation

```bash
python -m pytest -q
git diff --check
```

The new tests generate temporary synthetic NIfTI images and metadata only. They
cover review/source gates, no healthy-label inference, missing classes,
modality/dimensionality/QC, immutable virtual merges, duplicate patient identity,
global/holdout leakage, repeated sessions, path/byte duplicates, exact source/site
confounding, reproducibility and CLI exit behavior. No clinical data or model is
required. Registry, validators, manifest schema, split manager and all prior tests
are reused without modification.
