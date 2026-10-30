# Task C integration risk register

All **21 risks** from the completed read-only audit are retained below, including original priority, severity, evidence, impact, proposed owner, mitigation, and status. They are findings and proposed actions, not fixes implemented by this documentation branch.

Baselines: main `1d235147cca72b888dc2028a0e5e3eed1711d00c`; Task C `1c1c997` (`1c1c99730788a733484aa92f09057c144b37f8fb`).

P0 = prerequisite to model integration/claims; P1 = integration hardening and validation; P2 = independent evaluation and A/B development. Priority and severity are separate. Owner approval is a dependency that may overlap any priority; it does not block independent documentation or A/B design work.

Use the [blocker checklist](README.md), [full audit](task-c-audit.md), [26 proposed acceptance-test groups](task-c-acceptance-tests.md), and [verification requirements](task-c-verification-requirements.md) together. No risk is marked resolved by saving these documents.

## R01: Unsafe checkpoint loading

- **Priority:** P0
- **Severity:** Critical
- **Evidence:** torch==2.5.1; implicit torch.load; CVE-2025-32434; no trusted checkpoint SHA-256
- **Impact:** Arbitrary code execution or untrusted artifact execution
- **Proposed owner:** Model/backend owner
- **Mitigation / exit gate:** Patched compatible runtime; digest verification before explicit safe strict loading; no unsafe retry
- **Audit status:** Open; weights not downloaded/loaded

## R02: Inconsistent dependency pins

- **Priority:** P0
- **Severity:** High
- **Evidence:** MONAI1.6.1 requires torch>=2.8.0; requirements pin2.5.1
- **Impact:** Unreproducible environment; numerical and loader drift
- **Proposed owner:** Runtime owner
- **Mitigation / exit gate:** Actual training export; compatible supported lock; parity evidence
- **Audit status:** Confirmed metadata conflict

## R03: Calibration distribution mismatch

- **Priority:** P0
- **Severity:** High
- **Evidence:** Calibration/evaluation softmax logits1/2/3; inference softmax all10
- **Impact:** Incorrect calibrated-probability claims and altered routing
- **Proposed owner:** Task C owner
- **Mitigation / exit gate:** Separate score domains; exact-path evaluation; versioned policy
- **Audit status:** Confirmed source mismatch

## R04: Fabricated fallback/routing probabilities

- **Priority:** P0
- **Severity:** High
- **Evidence:** QC .91/.01; entropy .45; margin policy redistributes mass
- **Impact:** Clinical consumer misreads decision allocation as posterior
- **Proposed owner:** Task C and reporting owners
- **Mitigation / exit gate:** Typed failure/result envelope; policy score labels; no legacy clinical projection
- **Audit status:** Reproduced source characterization

## R05: Unsupported disease output accepted

- **Priority:** P0
- **Severity:** High
- **Evidence:** No support mask; synthetic high-confidence Normal accepted
- **Impact:** False reassurance or confident unsupported diagnosis
- **Proposed owner:** Task C owner
- **Mitigation / exit gate:** Support gate and consistent states; explicit coverage limits; external evaluation
- **Audit status:** Reproduced source characterization

## R06: Unproven source generalization/OOD routing

- **Priority:** P0
- **Severity:** High
- **Evidence:** Source recognition83.89%; one source per class; no unsupported positive cohort
- **Impact:** Confident tumor calls on new sources or diseases
- **Proposed owner:** Evaluation lead
- **Mitigation / exit gate:** Independent source-balanced cohort; normals/unseen diseases; exact-policy coverage/error analysis
- **Audit status:** Reported upstream; not rerun

## R07: Licensing and public artifact provenance

- **Priority:** P0
- **Severity:** High
- **Evidence:** No Task C LICENSE; API license null; best_model.pt publicly tracked45,630,165 bytes
- **Impact:** Unauthorized redistribution; uncertain integrity/privacy obligations
- **Proposed owner:** Repository/model/data owners
- **Mitigation / exit gate:** Code/model grant; authoritative source licenses/DUAs; provenance and public-artifact review
- **Audit status:** Open; no transfer/history edits

## R08: Wrong T1 channel/default modality

- **Priority:** P1
- **Severity:** High
- **Evidence:** CLI defaults0; MSD T1 is1; single-channel ignores invalid index
- **Impact:** FLAIR/contrast input accepted as native T1
- **Proposed owner:** Adapter owner
- **Mitigation / exit gate:** Explicit modality/axis/index metadata and bounds; golden fixtures
- **Audit status:** Confirmed source

## R09: DICOM geometry lost

- **Priority:** P1
- **Severity:** High
- **Evidence:** Identity affine and slice stacking; pydicom undeclared
- **Impact:** Wrong spatial interpretation and inconsistent model input
- **Proposed owner:** Imaging lead
- **Mitigation / exit gate:** NIfTI-only initial boundary; separately validated DICOM converter
- **Audit status:** Confirmed source; conversion unrun

## R10: Lost affine/transform trace

- **Priority:** P1
- **Severity:** High
- **Evidence:** Returns array plus crop/pad only; no original/transformed affine
- **Impact:** Invalid native-space evidence/localization; A/B misuse
- **Proposed owner:** Imaging/Task B owners
- **Mitigation / exit gate:** Parallel geometry provenance and landmarks; no full inverse claim
- **Audit status:** Confirmed source

## R11: Crop/mask tissue and lesion loss

- **Priority:** P1
- **Severity:** High
- **Evidence:** Center128cube; report8-16% peripheral brain loss; lesion loss unmeasured
- **Impact:** Missed peripheral tumors and shifted distribution
- **Proposed owner:** Task C owner
- **Mitigation / exit gate:** Preserve V1; quantify loss; assess retraining before mask/FOV changes
- **Audit status:** Upstream reported; not measured here

## R12: Late or weak input QC

- **Priority:** P1
- **Severity:** High
- **Evidence:** QC only after mask/crop; constant normalized values pass with large mask
- **Impact:** Malformed or uninformative input accepted; hidden raw NaNs
- **Proposed owner:** Imaging/adapter owners
- **Mitigation / exit gate:** Raw finite/affine/dimension/variance gates; explicit QC failure semantics
- **Audit status:** Partial QC weakness reproduced

## R13: Unavailable and nonreproducible splits/cache

- **Priority:** P1
- **Severity:** High
- **Evidence:** as_completed order before seeded shuffle; unversioned stale cache; no frozen manifest
- **Impact:** Different benchmark membership; hidden patient/source overlaps
- **Proposed owner:** Data/evaluation lead
- **Mitigation / exit gate:** Author frozen manifest; config/content hashes; deterministic ordering; cross-release dedup
- **Audit status:** Confirmed source; patient identity unverified

## R14: Float precision/runtime drift

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** Float16 cache vs fresh float32; inference CUDA autocast vs evaluation FP32
- **Impact:** Score/threshold changes; stale temperature
- **Proposed owner:** Runtime/Task C owners
- **Mitigation / exit gate:** Golden tensors/logits; CPU/GPU parity; explicit precision policy
- **Audit status:** Unmeasured

## R15: Legacy schema misuse

- **Priority:** P1
- **Severity:** High
- **Evidence:** Main lacks uncertainty; TaskCOutput defaults no_finding; PipelineOutput requires A/B
- **Impact:** Failed/unsupported results become healthy; invented upstream output
- **Proposed owner:** Main adapter owner
- **Mitigation / exit gate:** New envelope; supported-only explicit projection; keep old schemas/tests
- **Audit status:** Design required

## R16: Stale or impossible upstream tests

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** Calibration key failure; missing unified_inference.py; local Windows paths; t1c fixtures
- **Impact:** False readiness claims; no portable CI
- **Proposed owner:** Task C/tests owner
- **Mitigation / exit gate:** Preserve tests; new independent regression suite; repair stale tests separately
- **Audit status:** One test actually failed; full collection blocked

## R17: Offline/CLI/error contract gaps

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** offline_verified hard-coded; missing files raise; QC CLI success; non-atomic overwrite
- **Impact:** Misreported availability; partial output; sensitive path leakage
- **Proposed owner:** Backend owner
- **Mitigation / exit gate:** Network-blocked real run; typed errors; exit policy; atomic finite JSON
- **Audit status:** Offline execution unverified

## R18: Data-source and documentation contradictions

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** 2023/MSD cohorts vs main2024 IDs; old hippocampus/MS claims; differing counts/licenses
- **Impact:** Wrong labels/provenance; invalid cross-release assumptions
- **Proposed owner:** Data/docs owners
- **Mitigation / exit gate:** Versioned original provenance; authoritative license review; no misleading registry alias
- **Audit status:** Confirmed documents

## R19: Task A/B missing

- **Priority:** P2
- **Severity:** High
- **Evidence:** Only package docstrings in main; no normal detector or lesion spatial model
- **Impact:** Cannot claim end-to-end pipeline or normal detection
- **Proposed owner:** A/B leads
- **Mitigation / exit gate:** Separately validated detection/localization/segmentation; C remains standalone
- **Audit status:** Confirmed tree

## R20: Public medical-image derivatives

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** Tracked calibration/QC PNGs; contents intentionally not inspected
- **Impact:** Potential distribution/consent concerns requiring owner assessment
- **Proposed owner:** Data owner
- **Mitigation / exit gate:** Review derivative permissions/de-identification; preserve history pending explicit decision
- **Audit status:** No PHI finding asserted

## R21: Unverified training descriptions

- **Priority:** P1
- **Severity:** Medium
- **Evidence:** README says class-weighted focal loss; code supplies no explicit weights; no training seeds recorded
- **Impact:** Misstated method and inability to reproduce checkpoint
- **Proposed owner:** Task C owner
- **Mitigation / exit gate:** Confirm loss/runtime/run metadata and author checkpoint lineage; do not retrain during audit
- **Audit status:** Confirmed source discrepancy
