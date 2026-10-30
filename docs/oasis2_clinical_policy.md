# OASIS-2 clinical-label policy: metadata-only dry run

This module reviews the **existing verified linkage metadata**, not MRI pixels or
the source clinical Excel. It does not link new patients, download files, infer
diagnoses, change the private JSON, assign splits or release final model labels.
All source images and clinical Excel remain on private Google Drive. Only
synthetic fixtures have been executed in Codex.

The existing Task A readiness work remains unchanged. This policy reuses
`ManifestRecord`/`ManifestCollection` and `validate_record`, `SplitManager` leakage
checks, and `task_a_readiness.audit_readiness(check_images=False)`. OASIS-2 uses
the distinct source ID `oasis2`, which is **not currently registered**. It is never
aliased to `oasis3`, BraTS or `local` to bypass source restrictions. Therefore the
metadata audit cannot make Task A or Task C ready for release, even when a reviewer
provides visit-specific diagnosis evidence. Task C's model, preprocessing,
calibration and `classification.json` are unchanged.

## Supplied observations, not newly confirmed diagnoses

The user-provided aggregate matching report describes:

| Quantity | Supplied count |
|---|---:|
| Unique subjects | 150 |
| Clinical subject + MRI-visit keys | 373 |
| Linked image records | 1,368 |
| Unmatched / ambiguous linkage records | 0 / 0 |
| Nondemented, CDR 0 comparison-candidate visits / subjects | 188 / 72 |
| Additional Nondemented visits at CDR 0.5, requiring review | 2 |
| All Nondemented visits | 190 |
| Demented visits / subjects | 146 / 64 |
| Converted visits / subjects | 37 / 14 |
| Missing MMSE in the original clinical data | 2 visits |

The 373 visits reconcile as **190 + 146 + 37**, not 188 + 146 + 37. The supplied
Group/CDR table is preserved as evidence of possible discordance:

| Original Group | CDR 0 | CDR 0.5 | CDR 1 | CDR 2 |
|---|---:|---:|---:|---:|
| Nondemented | 188 | 2 | 0 | 0 |
| Demented | 0 | 102 | 41 | 3 |
| Converted | 18 | 19 | 0 | 0 |

These are **clinical group/score counts**, not confirmed Alzheimer's, MCI,
neurologically normal MRI, or final Task A/C class counts. A synthetic test
reproduces this count pattern with invented test-only IDs to verify aggregation;
it does not revalidate the private cohort or its linkage.

## Input schema and preserved values

`verified_clinical/oasis2_clinical_manifest.json` is a JSON **array**, with one
linked image per record and the following supplied field names. The parser uses
this exact profile by default:

| Private field | Policy field / meaning |
|---|---|
| `patient_id` | `subject_id`, retained in memory only |
| `visit_id` | `mri_visit`, retained as the original MRI visit key |
| `clinical_group` | Original `group`, without diagnostic remapping |
| `cdr` | Per-visit number or null; never filled from Group |
| `relative_path` | Original `image_path`; never opened or rewritten |
| `dataset` | Original `source_dataset`, expected `OASIS-2` |
| `diagnosis_source` | Original provenance text `OASIS-2 demographics`; not proof of etiology |
| `task_a_label`, `task_c_label` | Existing declared labels; currently null, never overwritten |
| `label_status` | Original `PENDING_CLINICAL_POLICY`, retained without claiming approval |

The supplied export has **no `mmse`, `linkage_status` or split field**. The parser
does not invent them:

- MMSE remains unavailable with `mmse_field_present=false`; the audit reports
  `mmse_not_exported` once per unique visit. If all 373 visits are present in that
  export, expect **373 MMSE-not-exported warnings**, not a claim that 373 patients
  had clinically missing scores. The two missing clinical MMSE values cannot be
  located or verified from this export. An explicitly exported `mmse: null` is
  reported separately as `missing_mmse`. No scores are imputed or thresholded
  into diagnoses.
- Stored linkage remains `unknown`. An optional separate prior-report attestation
  can establish **externally verified linkage for this audit**, without adding a
  field to each source record. An explicit unmatched/ambiguous value, if supplied
  in a different export, cannot be overridden by that attestation.
- Missing splits remain `unassigned`. The audit reports that all visits/images
  still need a subject-level split plan; it does not silently make one.

An explicit JSON field map supports other authorized metadata exports, including
dot-separated nested field names and optional MMSE/split/linkage fields. For
example `{"mmse": "clinical.MMSE", "split": "reviewed_split"}` changes only the
in-memory view. No fuzzy key matching or numeric-string conversion is done. A
JSON object envelope is supported with `--records-key`, but the actual private
array does not require that option.

All image records of the same `(subject_id, mri_visit)` must agree on clinical
Group, CDR, MMSE and MMSE availability. Conflicting versions are flagged, not
collapsed by arbitrarily taking the first value. Clinical counts are per visit,
image/linkage counts per image, and subject counts deduplicated. Repeated images
cannot multiply candidate counts or MMSE warnings. A path associated with more
than one visit is flagged as ambiguous; this metadata check cannot independently
prove the original Excel join was correct.

## Conservative candidate and diagnosis rules

- **Nondemented + CDR 0** with consistent, matched visit metadata can be a
  **cognitively normal comparison candidate**. Missing/unexported MMSE alone does
  not fabricate a score or disqualify that comparison candidate. This status
  assigns neither `task_a_label=0` nor `task_c_label=cn` and is not proof of a
  neurologically normal MRI.
- **Demented** describes the original clinical group, not confirmed Alzheimer's.
  **CDR 0.5** is a score, not an automatic MCI diagnosis. MMSE thresholds are not
  used to create either diagnosis.
- **Converted** remains the original group for its original visits. CDR 0 and
  CDR 0.5 visits are retained separately and flagged for longitudinal review.
  Neither the baseline nor later visit inherits a diagnosis from another visit.
  MRI visit identifiers are not treated as independently verified dates, so the
  report makes no unsupported temporal-onset claim.
- Nondemented with nonzero CDR, Demented with CDR 0, unknown groups, invalid/missing
  CDR, invalid MMSE and within-visit discrepancies are explicit review issues.
  Nonfinite/nonnumeric scores are invalid inputs. Source values are never
  corrected to make Group and CDR agree.

Final annotation proposals require a **separate visit-scoped evidence JSON list**,
joined by `subject_id` and `mri_visit`. It is not the linkage output and must stay
private. Fields are:

| Evidence field | Requirement |
|---|---|
| `manual_review_status` | `pending` (default), `needs_review`, `rejected` or explicitly `approved` |
| `verified_diagnosis` | Actual independently documented diagnosis text; absent by default |
| `diagnosis_provenance` | Existing `LabelProvenance.CLINICAL_RECORD` value `clinical_record`; inferred/missing evidence is not sufficient |
| `evidence_reference`, `reviewer_reference` | Nonempty references to authorized diagnosis verification; never printed |
| `conflict_resolution_reference` | Explicit reviewed resolution required for Group/CDR disagreement; does not rewrite original values |
| `task_a_label` | Explicit reviewer-requested 0/1 or null; no automatic assignment |
| `healthy_mri_verified`, `abnormal_mri_verified` | Separate Task A MRI-scope evidence; cognitive status alone cannot satisfy either |
| `task_c_label` | Explicit reviewer-requested existing `TASK_C_LABELS` key or null |

An approved generic “Demented”, “Converted”, “dementia” or “CDR 0.5” string still
cannot support a confirmed etiologic mapping. Task C `ad` requires explicit
Alzheimer's diagnosis evidence; `mci` requires explicit MCI evidence; `cn` requires
explicit cognitive-normality evidence. Labels are checked against that diagnosis,
not generated from it or the original group. Missing evidence yields null policy
labels, and unsafe labels already present in an older export are flagged rather
than erased. Review-supported annotation counts describe **reviewer attestations**,
not diagnoses independently confirmed by this software. Registry/access/MRI
validation and the main schema remain separate final-release blockers.

## Subject-level split safety

All visits and images from a subject must share one train/validation/test split.
`val` is recognized as `validation` only in the in-memory audit view; the original
split string is preserved. Existing `SplitManager` checks are reused and extended
to count holdout crossings and partially unassigned subjects. A subject with one
assigned visit and another unassigned visit is not split-ready. Converted subjects
must not have early visits in training and later visits in testing.

Do not call the existing split assigner independently on image rows with partial
assignments. A later authorized split plan must resolve fixed assignments, create
one assignment per unique subject, and broadcast it to all visits/images in a
**new** private sidecar. Re-audit that plan without replacing the verified linkage
manifest. This module assigns no splits and cannot declare sample size or
external validation adequate.

## Private Colab dry run

After this code has been reviewed and made available in an authorized Colab
checkout, run the following **there**, not against an invented local copy of the
private data. Mount/access Drive only through approved institutional procedures.
No MRI or Excel file is opened by this command.

```bash
python -m brainagenticthon.preprocessing.oasis2_clinical_policy \
  --manifest /content/drive/MyDrive/PRIVATE/verified_clinical/oasis2_clinical_manifest.json \
  --linkage-attestation /content/drive/MyDrive/PRIVATE/reviews/oasis2-linkage-attestation.json
```

The separate attestation can contain the supplied aggregate match evidence:

```json
{
  "evidence_reference": "private reference to the verified matching report",
  "matched_image_records": 1368,
  "matched_subject_visit_keys": 373,
  "unmatched_image_records": 0,
  "ambiguous_image_records": 0
}
```

The operator must confirm that the reference concerns this exact source manifest;
matching counts alone are not a new proof of linkage. Count mismatch or a
nonzero unmatched/ambiguous attestation prevents use as matched-link evidence.
No reference text is logged. The original export still has no linkage field.

Optional arguments: `--field-map /private/field-map.json`,
`--evidence /private/visit-diagnosis-review.json`, `--records-key nested.records`.
No output file option exists, and the module only reads explicitly named metadata
inputs. Do not redirect stdout over the original manifest or linkage output. If a
report is needed, write it separately in an approved private report directory.
The report includes aggregate counts, warnings, split violations, main integration
blockers, policy version and the input JSON SHA-256. No patient/visit IDs, image
paths, diagnosis strings or review references are printed; errors are generic.

Exit **0** means the supplied metadata passed the structural/linkage/split review
checks, **not** final diagnostic readiness. Exit **1** means review is required
(the supplied source has unassigned splits and two Group/CDR conflicts), and exit
**2** means an input/schema error. Every report explicitly keeps final Task A/C
release readiness false. The actual private export has not been run in Codex;
neither its counts nor any patient diagnosis has been newly confirmed here.

## Offline validation

```bash
python -m pytest -q
git diff --check
```

Synthetic tests cover the exact private array schema, field mapping, omitted vs
missing MMSE, external linkage attestations, Converted visit changes, repeated
images and visits, split leakage, conflicting Group/CDR/within-visit values,
missing/unsafe diagnosis evidence, preservation and identifier-free CLI output.
The original 274 offline tests and the existing Task A implementation are retained
unchanged.
