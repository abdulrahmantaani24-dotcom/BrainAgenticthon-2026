# OASIS-2 full-cohort MRI quality audit

This is a read-only MRI audit for an authorized **private Google Colab** session.
It processes one pair at a time and writes only aggregate reports/resume state
to a dedicated private output directory. It never changes MRI bytes, headers,
affines, clinical metadata, diagnoses, labels or splits, and never trains models.
Task A/C schema contracts, Task C, the data registry and clinical-label policy
are unchanged. OASIS-2 registration/access and final labels remain separate gates.

The owner reports 150 subjects, 373 sessions and 1,368 image-pair records, with
3/3 sampled NIfTI-1 pairs passing at shape `(256,256,128,1)`, subject-safe split
planning and preservation checks. These are supplied observations. Codex has
used synthetic files only; it has not audited the private cohort. No final
clinical diagnosis or Task A/C label is approved.

## Shared validator and bounded processing

`oasis2_mri_quality.py` contains the corrected validator extracted from notebook
02, without changing its validation/singleton semantics. Notebook 02 and
`oasis2_full_quality_audit.py` import that same function. Existing sample tests
exercise it through notebook 02, including calls to the unchanged
`NIfTIValidator` header, dimensions, spacing and orientation checks.

Supported inputs are paired Analyze/SPM Analyze and NIfTI-1/-2 `.hdr/.img` files.
Only the last suffix is replaced, so `mpr-1.nifti.hdr` locates
`mpr-1.nifti.img`. The header class and actual file map determine the format.
True 3D and verified `(X,Y,Z,1)` arrays are accepted; only the latter uses
`data[...,0]`, a 3D view in memory. No other axis is squeezed or series selected.

Default/hard upper limits for the full audit are 32,000,000 voxels per pair,
256 MiB `.img` payload and 1 MiB header; spatial axes remain 10–1024 voxels.
Full-audit callers may lower the limits, not increase them. Lowering a validation
limit changes the job binding and requires a new output directory. A float32
volume and temporary finite-value mask can consume roughly 160 MB at the voxel
limit, in addition to input mappings/library overhead. There is no parallel MRI
loading, prefetch queue or retained list of voxel arrays; metadata/state scale
with the small cohort manifest, not total MRI voxel count.

Checks retain path containment, hidden/AppleDouble/`__MACOSX` exclusion, companion
existence and payload length, real dtype, source spacing, affine validity,
paired-NIfTI qform/sform agreement, finite voxels, nonconstant intensity and
streaming SHA-256 preservation. Unsafe or unsupported records and pair files
linked to different visits prevent an all-cohort pass. `.hdr`/`.img` rows for the
same visit are deduplicated. Optional `.mat` sidecars still require separate
review. Unverified Analyze geometry, native T1 and missing visual QC remain
explicit warnings. No computational result sets `quality_flag=ok`.

## Exact Colab workflow for these unpublished changes

No branch push or commit is required. Build/use the reviewed **code-only** bundle;
an older remote checkout alone lacks the newly extracted module.

1. In the project workspace, generate the public code snapshot:

   ```bash
   python scripts/build_oasis2_audit_bundle.py \
     --output /tmp/oasis2-full-quality-audit-code.zip
   ```

   The builder includes an explicit allowlist of Python source, notebooks 02/03,
   packaging metadata and this guide, plus their SHA-256 manifest. It never
   traverses data/checkpoint/report directories or includes patient fixtures.
   The archive lives outside the repository. Review it before use. Its checksums
   detect accidental changes, not malicious replacement/authenticity.

2. Open `notebooks/03_oasis2_full_quality_audit.ipynb` in Colab using **File →
   Upload notebook**. Select a CPU runtime with Python 3.12+. Run section 1 and
   upload **only** `oasis2-full-quality-audit-code.zip` when prompted. Public
   NumPy/NiBabel pins are installed; code is validated/extracted to
   `/content/oasis2-audit-code`. No dataset is fetched. If Colab had imported an
   older NumPy, restart the runtime and rerun setup before mounting inputs.

3. In the private working copy, set the configuration to the actual authorized
   paths. The following values are placeholders, not known private locations:

   ```python
   AUTHORIZED_PRIVATE_ACCESS = True  # Only after confirming source access authorization.
   PRIVATE_ROOT = Path("/content/drive/MyDrive/PRIVATE")
   MANIFEST = PRIVATE_ROOT / "verified_clinical/oasis2_clinical_manifest.json"
   IMAGE_ROOT = PRIVATE_ROOT / "OASIS-2/raw"
   OUTPUT_DIR = PRIVATE_ROOT / "quality_audits/oasis2_run_001"
   DRY_RUN = True
   MAX_NEW_PAIRS = 3
   RECOVER_STALE_LOCK = False
   ```

   `IMAGE_ROOT` must be the exact base for existing `relative_path` values; there
   is no guessed prefix stripping. The default parser reads the existing JSON
   array fields (`patient_id`, `visit_id`, `clinical_group`, `cdr`, `relative_path`,
   `dataset`, provenance and nullable label fields). An explicit field map is
   optional. Missing MMSE/linkage fields are never fabricated. This quality tool
   does not reinterpret Group/CDR, rerun linkage or approve clinical labels.

4. Run the mount and import cells, then **Run or resume** in dry-run mode. Verify
   subject/visit/record totals, planned unique pairs and exclusion counts. Dry-run
   opens no MRI voxel data, creates no output directory or lock and writes no
   reports. For an existing run, cached progress is displayed with
   `committed_prefix_rechecked=False`; that is not fresh MRI verification.

5. Set `DRY_RUN=False`, keep `MAX_NEW_PAIRS=3` for an initial batch, rerun the
   configuration and audit cell, and inspect the aggregate results. To finish
   the cohort, set `MAX_NEW_PAIRS=None` and rerun those cells. Alternatively use
   `25` repeatedly: each invocation checks up to 25 **new** pairs from the same
   plan. No completed, committed voxel checks are repeated.

6. After disconnect/restart, upload the same bundle, mount the same authorized
   Drive and use the same paths and output directory. The audit rehashes the
   committed source prefix before resuming. If a previous hard-killed runtime
   left a lock, confirm it is stopped before setting `RECOVER_STALE_LOCK=True`
   for recovery; reset it to `False` afterward. Never delete an active lock or
   run two writers. Use a **new** output directory for changed inputs or code.

Notebook 02 now also uses the extracted code directory. To rerun it, first load
the same bundle via notebook 03's setup, then open notebook 02 in that runtime or
use its local-code setup after extracting the bundle. Its old GitHub-only setup
was replaced because the unpublished shared module does not exist at the old pin.

For a CLI inside the prepared Colab runtime (not the local Windows computer):

```bash
PYTHONPATH=/content/oasis2-audit-code/src python -m \
  brainagenticthon.preprocessing.oasis2_full_quality_audit \
  --manifest /content/drive/MyDrive/PRIVATE/verified_clinical/oasis2_clinical_manifest.json \
  --image-root /content/drive/MyDrive/PRIVATE/OASIS-2/raw \
  --output-dir /content/drive/MyDrive/PRIVATE/quality_audits/oasis2_run_001
```

The default is dry-run. Add `--run --max-pairs 25` for a resumable batch; omit
`--max-pairs` to process all remaining pairs. `--field-map`, `--records-key` and
explicit `--recover-stale-lock` are available. Exit 0 means dry-run completed or
all-cohort computational checks passed; 1 means partial processing or review
failures; 2 means an input/checkpoint/IO stop; 130 means interruption. None of
these codes approves clinical quality or labels.

## Checkpoints, privacy and restart behavior

The output directory contains only `checkpoint.json` and `aggregate_report.json`
when idle. The checkpoint is authoritative. Each pair gets one immediate atomic
checkpoint replacement containing aggregate counters, a completion cursor and
hashed job/byte-prefix bindings. The report is regenerated if interrupted between
those two writes. UTC run/checkpoint/report timestamps identify the verification
scope in time; a previous report is historical after a failed resume attempt.
Temporary JSON and `.audit.lock` files can exist during a run.
No patient identifiers, image filenames/paths, clinical rows, image arrays,
individual findings or split assignments are stored or logged.

The job binds the exact manifest bytes, canonical raw root and pair plan,
validation limits, field map, relevant source-code fingerprints and Python/
NumPy/NiBabel versions. On resume, bounded streaming hashes recheck completed
files; voxel validation is skipped for that prefix. This detects changed bytes
even with preserved size/timestamps. Rejected oversized files are not hashed
beyond the resource limit and retain size/mtime signatures instead; their prior
failure can never become an all-cohort pass without a fresh audit.

Source hashes before/after the shared validation and outer pair processing
prevent committing a pass if a concurrent source change is observed. Originals
must nevertheless remain stable; no software audit establishes immutable Drive
storage. Resuming adds substantial sequential Drive I/O, and current pairs get
multiple streaming integrity reads. An interrupted **uncommitted/in-flight**
pair may repeat after a hard kill; committed counts/checks are never duplicated.

The writer uses exclusive lock creation, checksummed JSON, local `fsync` and
atomic replacement. Google Drive/FUSE remote sync, cross-runtime lock semantics
and durability are external limitations: use one writer, keep private backups,
and investigate IO errors. Never silently repair/reset a corrupt checkpoint;
preserve it and start a new run directory. Input/manifest/code changes likewise
refuse resume without overwriting prior state. Output directories overlapping
raw data, clinical metadata or code, unrelated existing files, symlinks and
hardlinked outputs are rejected.

## Report interpretation and remaining work

The report counts records, unique subjects/visit keys/pairs, duplicates/exclusions,
completed/remaining pairs, computational pass/fail, byte preservation, singleton
views and aggregate failure categories, geometry and QC warnings. Completion means
all planned pairs were examined; it can include failures. A full computational
pass additionally requires no blocked records/failures and verified preservation
of every pair. `visual_qc_approved`, `clinical_diagnoses_approved`,
`final_task_labels_assigned` and `task_a_or_c_release_ready` always remain false.

Even aggregate rare-geometry counts and dataset fingerprints belong in private
storage. Do not commit reports, checkpoints, edited private notebook paths or
saved outputs. This tool creates no patient-level failure index; authorized
investigation of failures is separate. It cannot authenticate pre-existing MRI
bytes without trusted source checksums, approve native modality/orientation or
visual quality, reconcile multiple pseudonyms, prove cohort-wide absence of
byte duplicates across distinct filenames, validate clinical diagnoses, or
provide independent calibration/generalization evidence. Uncertainty cannot be
assumed to detect unseen diseases. Task C is unchanged.

Offline verification uses generated MRI pairs only, including interruption before
and after checkpoint commit, atomic-write failure, stale locks, changed source
bytes/semantics, malformed pairs, path/output protection and privacy assertions:

```bash
python -m pytest -q
ruff check src/brainagenticthon/preprocessing/oasis2_mri_quality.py \
  src/brainagenticthon/preprocessing/oasis2_full_quality_audit.py \
  notebooks/02_oasis2_clinical_validation.ipynb notebooks/03_oasis2_full_quality_audit.ipynb \
  tests/test_oasis2_full_quality_audit.py tests/test_oasis2_full_quality_notebook.py \
  scripts/build_oasis2_audit_bundle.py
git diff --check
```
