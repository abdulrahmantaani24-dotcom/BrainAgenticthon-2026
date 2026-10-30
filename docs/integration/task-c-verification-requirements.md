# Task C contract and verification requirements

These requirements make the existing audit gates concrete. They are design and
release criteria, **not implemented or executed checks**. Use the [full audit and
file tree](task-c-audit.md), [21 risks](task-c-risk-register.md), [26 proposed test
groups](task-c-acceptance-tests.md), and [prioritized checklist](README.md).

Baselines: main `1d235147cca72b888dc2028a0e5e3eed1711d00c`; Task C
`1c1c997` (`1c1c99730788a733484aa92f09057c144b37f8fb`). The historical evidence is
**201 main tests passed, one Task C calibration test failed, and three
dependency-related collection errors**. Task C has not passed its complete suite,
strict offline verification, or independent calibration evaluation.

## 1. Preserve both existing contracts

The initial adapter is additive and opt-in:

| Existing contract | Preservation requirement | New information belongs in |
|---|---|---|
| `TaskCOutput`, `DifferentialEntry`, `TASK_C_LABELS`, `TASK_SCHEMA_VERSION` | Keep fields, labels, validators, defaults, ranks, serialization and schema version unchanged; retain all existing tests. | A separate versioned result envelope and an optional projection bridge. |
| Upstream `classification.json` | Keep exact ten names/order, top-level fields, legacy values, rounding and sum tolerance unchanged for the compatibility profile. Do not silently replace its probability vector, entropy or fallback behavior. | Adjacent `classification.meta.json` plus the new internal result type. |
| Upstream CLI and preprocessing | Keep their existing semantics unchanged. A new package entry point can offer strict validation and error behavior without modifying the legacy entry point. | A separately documented adapter API/CLI with its own version. |
| A/B schemas and `PipelineOutput` | Keep current contracts; never instantiate dummy A/B outputs to run Task C. | Independent C results, with optional real A/B composition later. |

The new envelope records `classified`, `abstained`, `unsupported_output`,
`qc_failed`, or `error`; actual QC; score origin/domain; explicit support flags;
artifact/config/policy hashes; modality/channel; geometry provenance; and runtime
precision. The sidecar is required for safe consumption of legacy JSON. A failed
model or invalid scan has **no diagnostic model probabilities** even if the legacy
file contains its existing `.91/.01` policy allocation.

Only supported, accepted results with validated score semantics may be explicitly
projected as Glioma→`glioma`, Meningioma→`meningioma`, Metastasis→`metastasis`.
The projection remains unavailable until the required gates pass; the envelope
retains the restricted conditional score scope. No Normal→`cn`/`no_finding`,
unsupported→healthy, or uncertainty→`no_finding` mapping is permitted. No fake
clinical evidence is added to fill an existing field.

Future corrected scoring, revised thresholds, altered JSON or a new numerical
profile requires a separate owner-approved version and evaluation. It is outside
this unchanged-contract adapter plan. New adapter policy can reject a legacy
result without mutating the legacy file or claiming the two numerical contracts
are equivalent. Relevant groups: C01–C11, C26.

## 2. Strict offline verification

### Provisioned artifact and environment prerequisites

Before an offline inference run, a separately authorized provisioning step must
provide the trusted checkpoint and complete runtime locally. This documentation
does not authorize downloading, moving, deleting, converting or loading weights.

- Resolve the documented torch/MONAI conflict with a supported patched runtime.
  Capture exact Python, OS/architecture, torch/MONAI/NumPy/SciPy and transitive
  versions; CPU/GPU build, driver if applicable; wheel hashes; and a build/image
  digest. Do not bypass the resolver, TLS or checksum verification.
- Verify trusted SHA-256 digests before checkpoint deserialization. Bind the
  checkpoint to architecture/head, label order, preprocessing configuration,
  calibration and policy version. Reject a missing/mismatched artifact; no online
  fetch or unsafe pickle retry is allowed.
- Preserve `.4` temperature and all existing calibration values during this task.
  Preserve trained preprocessing semantics, and document any candidate runtime's
  numerical drift before adopting it.
- Keep dependencies and all resources local and read-only during runtime. Install
  and packaging tests must show that inference does not need Git, PyPI, model hubs,
  telemetry services, cloud storage or credentials once provisioned.

### Isolation and representative execution

An offline claim requires the **real installed package and trusted checkpoint**,
not only a fake backend or an `offline_verified=True` field:

1. Use an isolated sandbox/container with networking disabled by a verified
   OS/container control, such as a documented no-network namespace. Block DNS,
   outbound IPv4/IPv6 and proxy routes; do not rely only on an unavailable URL or
   unset proxy variables. No A/B service is needed.
2. Monitor the complete cold-start/import and inference process tree for attempted
   network operations. Record isolation configuration and attempted/denied calls
   separately. A blocked attempted download is a failure of the zero-network-call
   requirement even if local inference later succeeds. No observation is not
   proof of isolation when monitoring coverage is incomplete.
3. Execute actual CPU inference on approved local fixtures, including valid native
   T1, explicit multi-channel T1 selection, invalid/corrupt input, missing or wrong
   artifacts, failed QC, and serialization. Use synthetic inputs where possible;
   any real patient fixture requires separate approved access. A synthetic
   checkpoint run validates mechanics, not diagnostic accuracy.
4. Run the new package/API/CLI from outside repository CWD. Confirm resource/path
   resolution, lazy imports, no A/B or ground-truth-mask dependence, bounded input
   handling, explicit device/precision, and deterministic behavior within reviewed
   tolerances. Measure CPU behavior; GPU claims require their own executed run.
5. Verify ten-name legacy compatibility and the sidecar/envelope gates. New
   adapter operational failures must not become successful diagnosis or
   `no_finding`; the legacy CLI itself remains unchanged. Write new outputs
   atomically under scratch/output roots and verify input, config and checkpoint
   hashes are unchanged.

Store the revision/bundle hashes, environment and isolation manifest, sanitized
commands, current-run logs, network observations, input provenance, output hashes,
exit codes, timing/memory and test results. Logs must exclude secrets and PHI.
Relevant groups: C02, C10–C12, C19–C24. Missing trusted weights or required runtime
support means **unrun/blocked**, not a passed offline verification.

## 3. Independent calibration evaluation

### State what is being evaluated

The shipped calibration script fits temperature on logits `[1,2,3]`, conditional
on the three supported tumor labels, while inference computes a ten-way softmax.
It also applies different uncertainty rules and heuristic mass reassignment.
These distributions cannot share a calibration claim merely because their
vectors sum to one. Entropy, confidence margin, clinical probability and routing
allocation must remain separate concepts.

Freeze and identify the exact checkpoint, preprocessing, precision, class domain,
scorer, temperature and decision policy before evaluation. Evaluate the actual
candidate deployment path, retaining the unchanged legacy path as a separately
labeled comparator. The recorded temperature `.4` is not refitted by this task.
Any later recalibration needs separate authorization and validation/calibration
data; no thresholds or parameters may be fitted to the final test set.

### Independent data and leakage controls

- Obtain authorized cohorts independently of the existing training-source split,
  with reliable diagnosis provenance, verified normal controls, supported diseases
  across multiple sites/sources, and relevant unsupported pathologies. Acquisition
  rights and secure access remain prerequisites; do not infer health from absent
  diagnoses or Alzheimer's from cropped hippocampus data.
- Identify biological patients/sessions and image-content duplicates across all
  A/B/C releases, including MSD/TCGA/BraTS overlaps. Keep the author's frozen
  split membership for benchmark reproduction. A newly shuffled copy of the same
  source data is not an independent external test.
- Separate development, calibration/threshold selection and untouched external
  test cohorts. Freeze the protocol and acceptance thresholds before access to
  test outcomes, with patient/site grouping and source/diagnosis stratification.
  A leave-source-out experiment where source equals class cannot disentangle
  source generalization from removal of the disease class.
- Record counts, prevalence, scanner/protocol and modality, exclusions, failed QC,
  acquisition/source provenance and dataset versions. Secure manifests can retain
  patient-level identifiers/hashes; published reports must not expose PHI.

### Metrics and decision criteria

Report calibration for the evaluated domain using reliability plots, ECE with
declared bins, NLL/log loss, Brier score and uncertainty intervals, together with
per-class support and discrimination. Evaluate three-way conditional calibration
only on its appropriate supported population, and state that conditioning
explicitly. Do not infer calibrated ten-disease probabilities, normal detection,
or unseen-disease sensitivity from those results.

Compare forced-choice, decided-only and deployment-with-abstention results from
the exact selected policy. Report coverage, abstention/QC/error counts,
class-specific recall/precision and coverage-risk tradeoffs. On verified normals
and unsupported diseases, measure false confident supported-tumor calls and
routing failures; a low-confidence rule does not reliably recognize unseen
disease. Policy-created fallback mass is not evaluated or promoted as a calibrated
disease posterior.

Use patient/site-aware confidence intervals as supported by the sampling design,
describe small-group uncertainty, and retain counts behind perfect-looking
metrics. Predeclare numerical acceptance thresholds with the clinical/evaluation
owners for the intended restricted use; this audit did not establish acceptable
clinical thresholds. Missing independent data, undefined criteria, or incomplete
exact-path evaluation blocks new calibration/generalization claims. Relevant
groups: C07–C09, C19, C23–C25.

## 4. Evidence and reporting gate

Every verification run must distinguish passes, failures, errors, expected
failures, skips, and unrun checks. Collection-only success, missing-dependency
errors, mocked tests, file existence, or a no-network flag cannot substitute for
real checkpoint execution. Preserve the runner exit status and match result
files to the current run and selected revision.

The original audit results remain **201 main tests passed; one Task C calibration
test failed; three dependency-related collection errors**. No new model or
clinical tests are executed by saving this documentation. A subsequent release
must attach new evidence rather than relabel these results as Task C success.
