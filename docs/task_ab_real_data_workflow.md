# Private real-data readiness and training

Notebook 08 is a single private entry point for **inventory**, **validate**,
**freeze**, **smoke**, **train**, **resume** and **evaluate**. It defaults to
inventory; no mode automatically starts real training. Task A/B real execution
still requires actual CUDA, all original eligibility gates, a current immutable
split freeze, explicit verified mode and real_training_enabled=true.

The user reported private Colab Python 3.12.13, Torch 2.6.0+cu124, Tesla T4 and
synthetic GPU PASS for A/B. This is user-reported evidence, not a run observed by
this environment. It does not approve real cohorts. No private scans, manifests,
clinical reviews, Drive access or successful real training are included here.

## Start once, keep everything private

Build one shared reviewed code-only bundle from the published commit:

```bash
python scripts/build_task_a_bundle.py --output /tmp/task-ab-code.zip
sha256sum /tmp/task-ab-code.zip
```

Retain the actual archive's SHA-256 independently; upload only that ZIP to private
Drive. Save notebook 08 privately in Colab, disable output saving, select Python
3.12 and GPU. Set BUNDLE_PATH, BUNDLE_SHA256 and CONFIG_PATH. The same manifest
verification, pinned dependency setup and restart recovery as notebooks 05/06/07
apply. After Restart session rerun from top; no manual CODE_ROOT repair.

Copy configs/task_a_readiness.example.json or task_b_readiness.example.json to
private Drive. Templates intentionally have access flags false, empty references,
real training disabled and placeholder output/freeze paths. Never flip flags just
to make a gate pass. Configure separate dataset, metadata, freeze and output
subdirectories, e.g. raw/... versus task_a/verified, task_a/frozen, task_a/runs.
Output paths must be disjoint from all source roots and metadata, outside Git.

Each inventory.roots entry has a private-root-relative path, actual JSON boolean
access_authorized, genuine access_reference and license_reference. Access review
must cover all files under that root and the supplied private metadata. Roots may
not overlap or contain symlinks. The configured image_root must lie under an
explicitly authorized root. These added controls never override source policies.
Task B additionally requires inventory.modality_identity_verified=true and an
independent modality_identity_reference. Filenames alone cannot establish sequence
identity; the reviewed case manifest maps the four modalities and mask explicitly.

## Read-only inventory and individual blockers

Choose TASK='a' or 'b', ACTION='inventory', then run all cells. Up to max_files
(default 10000) recognized NIfTI/header/payload files are inventoried privately.
File/header limits, SHA-256, shape, spacing and exact-byte duplication are recorded;
symlinks and resource overruns stop the inventory. Headers commonly repeat and
are excluded from naive duplicate-image claims. Paired payload hashes and full
selected-cohort duplicate checks remain authoritative. No image is rewritten or
resampled on disk. Extra discovered files are inventory candidates, **not** an
approved cohort; approval applies only to the explicit reviewed manifest.

Private reports contain root-relative file paths (which may include identifiers),
per-record blockers, genuine supplied evidence references and aggregate blocker
counts. Keep the reports private; the notebook prints only status/blocker codes
and counts. ACTION='validate' skips folder discovery and checks selected records.
PASS means the original training gate also passed; diagnostics alone never grant
approval. Corrupt/absent metadata or invalid schemas remain BLOCKED. Unknown raw
exception details are redacted. References are recorded, not independently signed
or clinically adjudicated by software. Observed mask label values do not prove
that their release semantics or expert annotations are valid.

Task A diagnostic checks reuse the unchanged readiness policy and the stricter
training loader: explicit true Normal/Abnormal provenance, registry source/license,
native T1, whole-brain and healthy-control review, visual QC, anatomical geometry,
MRI-label evidence, original hashes/paired hashes, identities and patient splits.
Pure-class source/site groups are blocked. OASIS-2 Nondemented/CDR=0 is not MRI
Normal; Demented is not automatically Alzheimer's. The existing policy still has
no established eligible normal-control cohort; no source-policy extension is made.

Task A private inputs remain ManifestCollection JSON plus RecordReview and
TrainingEvidence lists keyed by dataset_source/case_id/image_path. See the complete
field contracts in task_a_research_training.md. No patient-containing examples or
approved clinical attestations are published.

Task B uses **BraTS 2021 adult glioma only**, source release
brats2021-glioma-training. Private manifest records require release, case_id,
globally reconciled patient_id, frozen split, approved case review, and exact
files/sha256 maps for t1/t1ce/t2/flair/seg. Release evidence requires approved
review, access/mask/geometry/identity booleans and genuine access/provenance/mask/
label/geometry/identity references. The existing loader checks 0/1/2/4 labels,
matching coded mm geometry, dimensions, finite intensities, masks, duplicates and
all three independent patient splits. It never assumes GLI/MET/MEN equivalence.
Missing or empty masks never create Task A normal labels. An empty valid mask is
not proof of healthy status. See task_b_research_training.md for full contracts.

## Freeze existing patient splits

After validation PASS, choose a new private config.freeze path and ACTION='freeze'.
The tool **does not invent, randomize or rewrite patient splits or labels**. It
snapshots the already verified manifest/reviews/evidence and patient split ledger,
binding them to cohort fingerprints, image/profile/root and authorization hashes.
All three splits must already satisfy the existing policy. If they are unassigned,
a qualified reviewer must supply a frozen patient assignment and rerun validation.
This avoids silently optimizing splits to force eligibility.

Freeze files include patient pseudonyms and clinical metadata and stay private.
They are exclusive-created and checksummed, not authenticity signatures. Any data,
label, review, geometry, split or authorization change invalidates the freeze.
Training epochs and resume flags may change without redefining the cohort. Never
manually edit a freeze; create a new reviewed run if inputs change.

## Explicit GPU actions after all gates pass

Set mode='verified' and real_training_enabled=true only after actual approvals.
All actions revalidate authorization, selected cohort and freeze, then require
actual CUDA. No CPU fallback is provided for real execution.

- smoke: fresh output_run, one train image/batch (A) or patch (B), finite gradients,
  parameter update and private weights-only checkpoint reload. No test images or
  claims of evaluation performance. This is a real-data operation and stays gated.
- train: different fresh output_run, resume=false. Uses existing patient-aware A
  or streaming patch-based B training and validation checkpoint selection.
- resume: same run, resume=true, trusted_resume=true, larger total epochs. Code,
  data, runtime and optimization settings must otherwise match existing signatures.
- evaluate: same trusted run, trusted_resume=true and final_test_evaluation=true.
  Reload best validation-selected weights; perform one-shot held-out evaluation.
  No fitting occurs. Existing test reports prevent further training or reevaluation.

The small smoke's checkpoint is not a resumable full experiment. Keep smoke and
full training runs separate. Failures preserve previous checkpoints; a failed new
freeze/run may leave partial private files, which are not automatically deleted.
A new destination must be selected after review. Never publish checkpoint files,
private reports, snapshots or manifests. Evaluation/report templates are unexecuted
schemas, not performance claims. Real inference uses the existing trusted
classifier/segmentation APIs on reviewed inputs; anatomy/clinical conclusions are
not added by this workflow.

Task A class scores still cannot satisfy a positive TaskAOutput without genuine
localizer predictions. Reviewed boxes remain supervised training annotations,
never fabricated detections. Task B reconstructs the supplied original shape/grid
with its original affine, maps internal 3 back to release 4, and extracts boxes
only from actual masks. BraTS2021 legacy export and anatomical atlas fields remain
explicitly unsupported. Task C, existing schemas and prior policies are unchanged.
