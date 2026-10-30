# Task C integration audit and proposed plan

Original read-only audit date: 30 October 2026. Documentation branch: `docs/task-c-integration-readiness`. Status: **research integration candidate; not ready for clinical probability claims or an A→B→C deployment**. This document is a plan, not an implementation.

The recommended path is a separately importable Task C package with an explicit input contract, an official-name compatibility serializer, and a new result envelope. Preserve the main project's schemas, labels, manifests, validators, tests, and unfinished A/B modules. Resolve the environment, artifact-safety, probability semantics, and provenance blockers before enabling model inference in the main workflow.

## 1. Scope, evidence, and preservation

| Repository | Audited revision | Evidence |
|---|---|---|
| `abdulrahmantaani24-dotcom/BrainAgenticthon-2026` | `1d235147cca72b888dc2028a0e5e3eed1711d00c` | Existing checkout; remote `refs/heads/main` returned the same SHA during audit. |
| `M0maniiR0/MRI-` | `1c1c99730788a733484aa92f09057c144b37f8fb` | Exact Task C baseline: supplied short commit `1c1c997`; complete recursive Git tree, `truncated=false`; 33 text files fetched and verified against their Git blob SHA-1s. |

The original audit evidence bundle is retained outside this checkout at `/workspace/audits/task-c-1c1c997/`. Its `source/` directory holds the Task C text snapshot, not a merged checkout. Evidence filenames below refer to that external bundle; these diagnostics, sources, images, and weights are not part of this documentation change. Checkpoint and image blobs were deliberately excluded. No medical datasets, checkpoints, or medical-image PNGs were downloaded. No checkpoint was loaded. During the original audit, no training, model calibration, branch update, merge, push, source edit, checkpoint move/deletion, or environment draft/publication operation occurred. Main Git status was clean before and after that audit; its preservation evidence is in `preservation.json`. This follow-up creates a documentation-only feature branch from the same verified current `main` commit and adds Markdown documents under `docs/integration/`; it does not change model code, preprocessing, weights, calibration values, existing schemas, or tests. Nothing is committed, pushed, merged, or published by this task.

No repository `AGENTS.md` was found. Review covered the complete trees; both READMEs; main package metadata, data policies, cloud notebook and preprocessing source/tests; all Task C Python source, tests, configurations, JSON/CSV results, report, and data-source documentation. Binary checkpoint contents and published PNG contents were not inspected. The independent competition rulebook was not supplied: “official contract” below means the ten-name contract declared in the pinned Task C repository, which still needs confirmation against the organizer's specification.

### Checks actually executed in the original audit

| Check | Actual result | Limits |
|---|---|---|
| Main `.venv/bin/python -m pytest -q -p no:cacheprovider` | **201 passed**, 0 failed, 0 errors, 0 skipped; 0.58 s reported | Offline main library/schema tests; does not establish Task C inference or A/B model readiness. See `main-tests.xml`. |
| Task C `test_task6_acceptance.py::TestTask6CalibrationAcceptance::test_calibration_parameters_valid` | **1 failed**: expected `ece_before`, shipped key is `ece_before_val` | Shipped JSON also lacks expected `ece_after` and `entropy_threshold`. No data/weights required for this test. See `calibration-test.xml`. |
| Task C full `pytest --collect-only -q -p no:cacheprovider` | **Exit 2**, three collection errors; three tests discovered before interruption | Missing `scipy` for preprocessing and `torch` for inference suites in current main venv. This is not a test-suite execution or pass. See `task-c-collection.log`. |
| Five synthetic, AST-extracted source/metadata characterization probes | **Completed; five findings reproduced** | Isolated upstream function bodies/branches and package metadata, not full module imports, MONAI transforms, checkpoint execution, or clinical validation. See `audit_probes.py`, `source-probes.json`. |
| Task C text integrity | 33 fetched text files matched pinned Git blob IDs | Git blob identity is not checkpoint trust, a signature, or an artifact SHA-256. |
| Package-release metadata | MONAI/PyTorch pin conflict confirmed | Metadata read from PyPI; no Task C dependency install or resolver success claimed. |

Not executed: complete Task C tests, actual DenseNet inference, DICOM conversion, MONAI resampling/orientation, benchmark reproduction, offline container tests, GPU tests, independent patient-data evaluation, or proposed adapter acceptance tests. The artifact-existence test expecting `reports/calibration_curve.png` was not run because this audit intentionally did not fetch image blobs; tree metadata confirms that upstream image exists.

## 2. Repository inventory and implementation maturity

Main has a `src/brainagenticthon` package, `pyproject.toml`, three test modules, a Colab data-access notebook, storage template, and data-handling/architecture documentation. `preprocessing/` implements dataset/source metadata, dataset access helpers, NIfTI validation, master manifests, leakage checks/splits, and A/B/C output dataclasses. `classification/`, `detection/`, `spatial/`, `agents/`, `verification/`, and `reporting/` contain only package docstrings; `backend/` is a placeholder. Architecture descriptions of a complete agentic MRI pipeline are aspirations, not implemented models. No CI configuration or lockfile was present in either inspected tree.

The main validators validate files and labels; they are not an established trained preprocessing pipeline to substitute for Task C's pipeline. The default case validator expects `t1,t1ce,t2,flair,seg`, so using that default as an inference gate would incorrectly require contrast, extra modalities, and a ground-truth segmentation. Use single-input header/integrity checks and a dedicated Task C gate. Main orientation mismatches are warnings, not automatic failures; inspect the report rather than assuming `is_valid` certifies every desired property.

Task C is a flat script repository, with no packaging metadata, importable package namespace, Dockerfile, or CI. Its complete text inventory is preserved in `tree.json`.

| Role | Exact upstream paths |
|---|---|
| Runtime | `inference.py`, `preprocess.py` |
| Runtime configuration | `preprocessing_config.yaml`, `calibration_params.json`, `label_map.json`, `requirements.txt` |
| Model artifact | `best_model.pt` at repository root |
| Training/cache provenance | `cache_dataset.py`, `train_native_t1_classifier.py`, `training_log.csv` |
| Calibration/evaluation | `calibrate_native_t1.py`, `evaluate_native_t1_bootstrap.py`, `evaluate_with_abstention.py` |
| Bias audits | `run_shortcut_diagnosis_native_t1.py`, `run_source_recognition_check.py`, `source_recognition_results.json` |
| Results/documentation | `README.md`, `coverage_report.md`, `data_registry.csv`, `test_evaluation_results.json`, `abstention_evaluation_results.json`, `reports/task_c_evaluation_report.md`, `DATA/README.md`, four `DATA/*/DATA_SOURCE_INFO.md` files |
| Packaging utility | `package_clean_deliverables.py` — hard-coded paths and top-level ZIP creation; do not import/run during integration. |
| Tests | `tests/test_task1_acceptance.py`, `tests/test_task2_preprocessing.py`, `tests/test_task6_acceptance.py`, `tests/test_task7_contract.py`, `tests/test_unified_inference.py` |

Do not wholesale copy this tree into main's package or tests. Archive provenance, then extract runtime functionality into a new namespace after findings review. Keep legacy tests intact; new tests should expose defects rather than overwrite previous assertions.

## 3. Exact model, dependency, artifact, and CLI contract

### Runtime model

`inference.py:59–100` constructs MONAI `DenseNet121(spatial_dims=3, in_channels=1, out_channels=10)`, chooses `cuda:0` if available or CPU, loads a checkpoint, and enters evaluation mode. It accepts `state_dict`, `model_state_dict`, or a direct state dict. `load_state_dict` is strict by default. Input is `(1,1,128,128,128)` float32; CUDA inference uses autocast. There is no Task A/B dependency.

`checkpoint_path`, `calibration_path`, and `device` can be supplied to the constructor. Defaults are files beside `inference.py`: `best_model.pt` and `calibration_params.json`. `predict(path, channel_index=0)` cannot supply a preprocessing config path; it calls `preprocess_mri` with its default sibling YAML. The convenience `run_inference` constructs a new engine every call and offers neither artifact nor device overrides.

The training script saves a dictionary with `epoch`, `state_dict`, `val_balanced_acc`, `present_classes`, and `timestamp`. This describes the writer; the public checkpoint's actual contents, match to training run, and checksum were not verified. The code passes no explicit class weights to `FocalLoss(to_onehot_y=True, gamma=2.0)`, despite README/comments describing a “class-weighted” loss. Confirm MONAI loss defaults and training environment with the author; do not change the head or loss retrospectively.

### Pinned runtime requirements

The file comments specify Python **3.11.9**:

```text
torch==2.5.1
monai==1.6.1
nibabel==5.4.2
numpy==2.4.6
scipy==1.17.1
pillow==12.3.0
pyyaml==6.0.3
```

Authoritative PyPI metadata for `monai==1.6.1` requires **`torch>=2.8.0`**. Thus the shipped pins cannot form a normally resolved environment. NumPy/SciPy pins require Python >=3.11, whereas main declares >=3.10. The restored main venv is Python **3.12.14**, NumPy **2.5.4**, nibabel **5.4.2**, PyYAML **6.0.3**; it has no torch, MONAI, SciPy, Pillow, scikit-learn, matplotlib, or pydicom. Main declares no core dependencies and dev extras omit nibabel/PyYAML even though some existing tests need them; the earlier environment setup supplied these separately.

The Task C requirements cover inference imports, not all research scripts: calibration additionally needs matplotlib; source recognition needs scikit-learn; DICOM needs pydicom and potentially pixel decoders. There is no lockfile, transitive dependency inventory, CUDA build selection, wheel hash list, or verified compatible runtime image. Obtain the actual training environment export. Choose an isolated, patched compatible environment, lock its full CPU baseline, and compare tensors/logits before adopting it. Do not solve dependency conflicts with `--no-deps` or by silently changing model semantics. An optional Task C extra must require Python >=3.11, or initially use a separate process/environment so main's >=3.10 support stays honest.

### Checkpoint and safety

GitHub metadata reports `private=false`, no detected repository license, and a normal tracked blob:

```text
best_model.pt
size: 45,630,165 bytes (~43.52 MiB)
Git blob SHA-1: 55acd14d8fbdee705c8bdb895a48ab7c90ef2658
```

This is a tracked public checkpoint, not evidence of consent/licensing, de-identification, authenticity, or safe deserialization. Upstream `.gitignore` ignores `.pth` and `models/*`, but not root `.pt` files. Main policy explicitly keeps weights out of Git and in private/local artifact storage. Do not import the weight blob into main or rewrite upstream history in this task. A separate owner decision is required for provenance, authorized redistribution, retention, and any remediation of past exposure.

`inference.py:93` uses `torch.load` without explicit `weights_only=True` or a verified SHA-256. Under pinned PyTorch 2.5.1 this exposes pickle-style loading. Moreover, GitHub's critical advisory [CVE-2025-32434 / GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6) covers torch <2.6.0 even with `weights_only=True`. A future loader needs a supported patched runtime, authenticated artifact digest before loading, explicit constrained loading to CPU first, strict key/shape checks, and bounded resource use. Never automatically retry with unrestricted pickle or add arbitrary globals. A tensor-only format is a later controlled conversion after trusted inspection, not a reason to move/delete the current artifact. Review current advisories for the ultimately selected runtime; fixing this one CVE is not a blanket security attestation.

### CLI and JSON

Existing CLI:

```text
python inference.py INPUT [--output OUTPUT|-o OUTPUT] [--channel INDEX|-c INDEX]
default OUTPUT: ./classification.json
default INDEX: 0
```

The README omits the channel switch. MSD Task01 T1 is **channel 1**; its default channel 0 is FLAIR. A syntactically successful default CLI run on that 4D input can therefore use the wrong modality. CLI offers no device/checkpoint/calibration/config switches. Artifacts resolve relative to source; output resolves relative to CWD and parent directories are created. JSON and banners are both printed to stdout, so stdout is not a JSON-only machine interface.

Returned keys: `model_name`, `predicted_class`, `top_probability`, `probabilities`, `is_uncertain_category`, `entropy`, `margin`, `qc_status`, `qc_details`, `inference_time_sec`, `offline_verified`. The probability dictionary has the ten names in the order in section 4, six-decimal values, and rounded-sum correction. Missing input raises before QC handling; missing artifacts fail during engine construction. Preprocessing exceptions/failed QC return a fallback dictionary and CLI success; model/device/forward errors are not caught by that fallback. Writes are non-atomic and overwrite an existing output path. `json.dump` permits NaN by default and no finite-output/logit gate exists.

The inference path contains no explicit network request or pretrained downloader. That supports an offline design, not an executed offline guarantee. `offline_verified=True` is hard-coded. Future offline proof must block network access and exercise actual artifacts, including package import/startup. Provision trusted dependencies before air-gapped runtime; do not clone/download at inference time.

## 4. Label-space reconciliation without changing existing labels

Main's `manifest_schema.py:104` defines seven keys: `glioma`, `metastasis`, `meningioma`, `ad`, `mci`, `cn`, `no_finding`. `task_output_schemas.py` imports this mapping; `TaskCOutput` and `DifferentialEntry` reject other labels, including pituitary, and enforce [0,1] values and consecutive differential ranks. They cannot express uncertainty, unsupported classes, or unavailable probabilities. Default `class_label='no_finding'` must never be used as an error fallback. `PipelineOutput` requires real A, B, and C outputs, so it is unsuitable for standalone Task C.

| ID | Exact external name | Model support | Main mapping / disposition |
|---:|---|---|---|
| 0 | `Normal` | No trained/evaluated examples | No current prediction mapping. `cn` describes cognitive normality and `no_finding` describes no pathological finding; neither is a safe alias for this unsupported output. |
| 1 | `Glioma` | Training/evaluation support | Explicit alias to `glioma`. |
| 2 | `Meningioma` | Training/evaluation support | Explicit alias to `meningioma`. |
| 3 | `Metastasis` | Training/evaluation support | Explicit alias to `metastasis` (main display: “Brain Metastasis”). |
| 4 | `Alzheimer's` | No support | Do not infer `ad` or `mci`. Official mapping says MCI is folded into Alzheimer's, while main preserves them separately: future evaluation crosswalk only, retaining original diagnosis. |
| 5 | `FCD epilepsy` | No support | No main counterpart; coverage metadata, no legacy classification. |
| 6 | `Post-stroke lesion` | No support | No main counterpart; coverage metadata, no legacy classification. |
| 7 | `Parkinson's` | No support | No main counterpart; coverage metadata, no legacy classification. |
| 8 | `ASD` | No support | No main counterpart; coverage metadata, no legacy classification. |
| 9 | `Uncertain/Other` | No trained disease class; routing category | A decision state in the new envelope, never `no_finding`, `cn`, or a new silent manifest label. |

Do not replace `TASK_C_LABELS` with the ten-name list, reorder logits to main dictionary insertion order, remove `mci`, or weaken pituitary/2D/missing-diagnosis tests. Validate `label_map.json`, embedded inference names, model output dimension, calibration indices, and declared support together. Preserve original IDs, versioned alias map, and source labels.

Cohort identity is another mismatch: main source IDs describe **BraTS 2024** cohorts and OASIS/ADNI/Figshare/local; Task C used **MSD Task01**, **BraTS-MEN 2023**, and a metastasis collection. Do not relabel them as BraTS 2024 to satisfy existing validators. Keep original provenance and frozen split identities in an external Task C provenance manifest. Any future registry additions must be additive, reviewed, and keep existing validation behavior/tests. Inference on a genuinely unseen source should record unknown provenance without pretending that training-source approval or disease coverage has been established.

## 5. Preprocessing compatibility and retraining assessment

Sources: `preprocess.py:46–84,87–131,134–210,213–344`; `preprocessing_config.yaml`; cache/training scripts. **Do not feed a generic main preprocessed tensor or A/B lesion ROI into this model.** First supply a known native non-contrast T1 image/channel to its versioned trained pipeline. Its brain mask is a preprocessing tissue mask, not a lesion segmentation or Task B result.

| Stage | Actual behavior | Integration risk and requirement |
|---|---|---|
| Modality/channel | MONAI load with channel-first; select `channel_index` if >1 channel, otherwise select channel 0 regardless of requested index. MSD cache explicitly selects 1; MEN/MET select native `t1n` channel 0. | No verified modality detection; negative Python indices accepted, out-of-range becomes preprocessing fallback. A 4D time series is indistinguishable from modalities. Require explicit sequence/channel metadata and nonnegative bounds; reject ambiguous 4D rather than guess. Never substitute T1ce/FLAIR or upstream lesion crops. |
| Orientation | `LoadImaged(...ensure_channel_first=True)` then `Orientationd(...axcodes='RAS')` always. | Main checks RAS, but does not implement this transform. Orientation labels alone do not prove voxel/world correspondence. Golden-test axis permutations, flips, translations, and obliquity; preserve qform/sform choice and transform trace. RAS orientation is not atlas registration. |
| Affine | Uses loader metadata during spatial transforms; returns NumPy volume without transformed affine. | No explicit affine finite/singular or qform/sform conflict gate. `inverse_transform_info` only stores crop/pad array geometry, not original affine, RAS transform, resampling transform, or physical coordinates. Cannot support claimed full native-space invertibility. Maintain a parallel provenance/geometry trace without changing tensor values. Cropped-away tissue is unrecoverable. |
| Spacing | Compares header zooms with target using `np.allclose(..., atol=.05)`; skips resampling if close, otherwise MONAI `Spacingd(mode='bilinear')`. | Approximate spacing can be reported as exact `[1,1,1]`; tolerance and relative tolerance matter. Report actual effective affine/spacing, detect oblique/sheared headers, and test boundary cases. Do not tighten tolerance or change interpolation silently: that changes model input. |
| DICOM | Stacks slices sorted by patient-position z / instance fallback; writes float32 NIfTI with identity affine. Single 2D DICOM becomes singleton 3D. Writes beside input by default. | Loses pixel spacing, patient orientation/origin and potentially slice geometry/scaling; no series grouping or robust decoder contract. pydicom undeclared. Initially reject DICOM at the integration boundary; plan a separately validated converter using real geometry, explicit series selection, and scratch outputs. Current DICOM path is not safe evidence of supported native input semantics. |
| Mask | Positive-voxel 15th percentile threshold, largest component, hole fill, one 6-connected dilation. No morphological closing despite an explanatory docstring. | Heuristic can omit cortex, dura or disconnected lesions and is not tumor localization. Preserve for the audited model; validate tissue/lesion coverage before considering a replacement. Assumptions about already skull-stripped/template-aligned training cohorts do not transfer to arbitrary native scans. |
| Intensity | Percentiles .5/99.5 of values inside the generated mask; clip then compute mask-based mean/std + `1e-8`; outside mask zero. | Not simply “all nonzero voxels”: dilation can include initially zero tissue. Mean/std computed before crop. Changing to MONAI NormalizeIntensity, adding bias correction or new skull stripping is not packaging-only. |
| Crop/pad | Center on array, not brain/lesion center; zero-pad/center-crop to 128³; mask treated identically. | Report states 8–16% peripheral brain loss on 15 cases; lesion loss unmeasured. From 240×240×155 to 128³, retained grid fraction is ~23.49%, which is not the same as retained brain/lesion fraction. Do not change FOV, centering, shape or resize without evaluating frozen weights and likely retraining/recalibration. |
| QC timing | Only after normalization and crop/pad: finite values, mask count >=1000, ndim==3, normalized bounds [-15,15]. | Invalid values outside the mask/crop may disappear before QC; raw finite/geometry checks are needed. Constant normalized zero values with a large supplied mask pass the current QC function (reproduced). “PASS” status is synthesized by inference because successful preprocessing provides `passed` but no `status` field. |
| Cache precision | Cache saves float16 `.npy`, training/calibration/evaluation read float16→float32; online pipeline uses fresh float32. | Cache rounding and CUDA inference autocast differ from cached FP32 evaluation. Quantify tensor/logit/probability drift and CPU/GPU reproducibility before claiming preprocessing identical or temperature transferable. |
| Config coupling | Runtime silently falls back to built-ins if config unavailable; built-in min mask is 500 versus YAML 1000; several YAML fields are unused. | Validate and hash config; fail clearly if required bundle data missing. `interpolation_mask`, `z_score_nonzero_only`, `allowed_dimensions`, and skull-stripping description are not general enforced transform controls. A YAML change alone may do nothing. |

For metadata-only packaging, path resolution, strict error envelopes, checksums, and sidecar provenance, frozen tensors should remain identical. New raw-data validity gates may reject previously accepted invalid cases without changing valid-case tensors. Any change to orientation interpretation, resampling, mask extraction, channel, normalization, quantization or crop/pad is a model-input change. First characterize existing behavior, compare against author-confirmed training preprocessing on controlled fixtures, measure output drift on authorized held-out data, and assess retraining/recalibration. Do not claim a shape-compatible replacement is semantically compatible.

## 6. Probability, uncertainty, and QC findings

1. **Calibration domain mismatch:** `calibrate_native_t1.py:96–119` selects logits `[1,2,3]` and applies softmax within those three. Grid search minimizes ten-bin ECE over `T=.2…4.0`; shipped `T=.4`, validation n=133, ECE .061636→.015941. `evaluate_native_t1_bootstrap.py:121–138` and `evaluate_with_abstention.py:84–95` also use only these three logits. **Inference uses ten-logit softmax**, so its scores/thresholds are not the evaluated probability distribution. A main adapter cannot label the returned ten-way scores clinically calibrated based on this evaluation.
2. **Different routing:** JSON records max-confidence threshold .9 and uncertain ID 9. Inference additionally defaults to margin <.10 or entropy >1.85; these two thresholds are not in the calibration JSON or evaluated policy. Inference defaults for missing settings also differ (.35 confidence versus .9 shipped). Validate all fields; bind calibration and routing policy to model/preprocessing artifacts.
3. **Heuristic mass rewriting:** On abstention, inference sets uncertain mass to `min(max(p9, 1-margin, .51), .95)` and proportionally rescales others. This is a policy allocation, not a fitted probability of unseen disease. Entropy and margin still describe the pre-routing distribution. A source-level synthetic probe moved `p9=.001` to approximately `.700002`; it does not indicate an actual scan's risk.
4. **Fabricated QC scores:** Failed QC returns .91 for Uncertain/Other and .01 for every other name, plus entropy .45 and margin .9. This is not a model forward pass or calibration. The entropy of that returned score vector is ~.500288, not .45 (reproduced). Tag as policy-only, unavailable diagnostic probabilities, and do not project into main's `TaskCOutput`.
5. **Unsupported prediction bypass:** No explicit support gate restricts output to IDs 1/2/3. A high-confidence unsupported `Normal` output passes thresholds and is emitted as certain (reproduced with synthetic probabilities). An argmax of ID9 may likewise yield `predicted_class='Uncertain/Other'` while `is_uncertain_category=False` if thresholds pass. Add consistency/support gates; this does not establish that unseen diseases can be recognized.
6. **No OOD guarantee:** Model has no positive training or independent tests for six unsupported diseases/Normal or an Other class. It can confidently misclassify an unseen disease as a supported tumor; max-softmax/entropy routing cannot reliably catch this. The label-map assertion that unsupported diseases “are routed” describes an intention, not an implemented or validated capability. Treat all model outputs as restricted research outputs and report the coverage gap prominently.
7. **Failure contract gaps:** Missing file/artifacts can raise, corrupted images are caught as QC, model errors can raise, and failed-QC outputs still exit successfully. Nonfinite logits/invalid temperature or indices are not checked. Broad preprocessing catch can hide programming/configuration defects as patient QC. Distinguish operational error, invalid input, QC failure, unsupported output, abstention, and successful classification; avoid printing sensitive paths in exposed error messages.

## 7. Confounding, reproducibility, and independent evaluation

The upstream report honestly documents strong source separability, but the README's geometric audit does **not** establish “zero geometric leakage.” One ridge model near chance cannot rule out other shortcuts; `run_shortcut_diagnosis_native_t1.py:138` calls a result near-chance merely when the **lower CI bound <=.40**, an insufficient criterion. Its `padding_fraction` is actually nonzero fraction; cached entries can lack mask counts and substitute a derived value. Both provenance and feature definitions need verification.

Reported source recognition is 83.89% balanced accuracy (CI 77.11–90.22%) from histogram/noise features; random forest 81.30% (CI 74.31–87.56%), versus 33.33% chance. Diagnosis and source are aligned one-to-one. These features can reflect acquisition and disease; this experiment demonstrates separability/confounding risk, not a causal quantification of scanner dependence or proof the DenseNet ignores lesions.

Reported internal results (not rerun here): 884 patients, train619/val133/test132; forced three-class balanced accuracy97.78%, macro-F1 98.39%, ECE .022765 after T=.4; selective accuracy99.19% on123 non-abstained cases; threshold-policy deployment-view balanced accuracy90.83%, macro-F1 94.88%, nine abstentions. These are different endpoints. The “deployment-view” evaluation still uses three-class probabilities and max-confidence routing, so it does not reproduce actual ten-way inference plus extra thresholds/mass rewriting. Perfect bootstrap intervals on small, perfect-result subsets are not evidence of zero population error.

Frozen splits/cached volumes are untracked and unavailable. `cache_dataset.py` appends process results in `as_completed` order before seeded shuffling; therefore seed42 alone does not fix patient membership. Existing caches are reused without code/config/content fingerprint; failed non-critical QC is not used to reject cached cases. Cache filenames use unnamespaced subject IDs, risking collisions. String-ID split-disjointness assertions do not establish biological-patient identity, deduplicate source overlaps, or prove no shared BraTS subjects across releases. Main's identifier fingerprint is likewise not a true image-content hash.

Required next evaluation design: obtain author-held frozen split manifests and exact artifact hashes, preserve original splits, then establish biological patient/session and image-content deduplication across **all A/B/C datasets and releases**, especially MSD/TCGA/BraTS overlaps. Use approved independent hospital/scanner/time-period test cohorts with multiple diagnoses and explicit verified normal controls per source; do not simply rename or resplit the same three sources. Include unsupported pathologies, QC failures, and protocol variation. Use separate development/calibration/untouched test sets, patient/site-aware intervals, external calibration and per-class/coverage/risk metrics from the exact deployed API. Evaluate false confident tumor calls on normals/unsupported diseases. A leave-source-out experiment on the current source=class data alone cannot isolate generalization because it also removes the held-out disease from training. No data acquisition or retraining was performed here.

## 8. Licensing, data handling, and security boundaries

Main README/metadata say MIT, but no standalone LICENSE file is in the inspected tree. Task C has no LICENSE, package license metadata, or GitHub-detected license. Public visibility does not itself grant redistribution rights. Obtain written code and model licensing/attribution from their owner before vendoring into main; verify rights separately for code, weights, dependencies, training sources, evaluation results, and image derivatives.

Task C source notes claim MSD CC-BY-SA4.0, MEN CC-BY4.0, and metastasis “Apache2.0 / CC-BY”; these are repository assertions, not independently confirmed grant terms for the exact downloaded cohorts. Check original authoritative dataset releases, DUAs and mirrored-source permissions, plus derived model/release obligations. Do not assume data licenses automatically license the trained artifact. Main's BraTS2024 Synapse metadata does not authorize Task C's different releases.

`DATA/README.md` and older acceptance tests still suggest hippocampus→Alzheimer/volumetric diagnosis and multiple segmentation models absent from this commit; newer coverage documents explicitly exclude hippocampus and MS. Preserve these documents as history but clearly resolve misleading active guidance in a later documentation task. Never infer AD from a cropped hippocampus segmentation or normal from a missing diagnosis. Counts/splits differ between old DATA docs and final classifier reports (e.g. MEN160/40 versus140/30/30; metastasis240 versus200); require provenance rather than infer which historical acquisition was used.

Medical-image PNGs are publicly tracked alongside the checkpoint. Their presence warrants an owner review of de-identification/consent and distribution terms; contents were not downloaded, so **no PHI exposure finding is asserted**. No automatic history rewrite or deletion is proposed in this task.

The main package must remain usable without ML/GPU imports. Avoid `sys.path` mutation and importing flat training/packaging utilities: cache code creates directories on import and ZIP packaging runs on import. Keep explicit local artifact paths, approved read/write roots, scratch outputs for conversions, atomic JSON outputs, finite strict serialization, bounded compressed-volume input size, and logs free of PHI/credentials. Record artifact SHA-256s, software versions, policy version, device/precision and transforms; a Git commit alone is not a calibrated model identity.

## 9. Proposed backward-compatible adapter and independent package

This design adds files only in a future implementation. It does not replace existing schemas or make Task C depend on A/B. **Both `TaskCOutput` and `classification.json` remain unchanged.** New status, coverage, score-origin and provenance fields belong to a separate result envelope and `classification.meta.json` sidecar; no field, label, default, numerical behavior, or failure behavior of an existing contract is silently rewritten. The bridge must gate consumption of legacy outputs rather than mutate them. See the contract-preservation requirements in [verification requirements](task-c-verification-requirements.md).

### Package and input interface

Use a separate distribution `brainagenticthon-task-c`, import namespace `brainagenticthon_task_c`, under `packages/task_c/`. Keep main optional. Use a small protocol/interface so tests can inject a fake logit backend without requiring a GPU or checkpoint. Load torch/MONAI lazily only when a real predictor is constructed. No training/cache/evaluation script import at package import time.

Proposed interface (design notation, not implemented):

```python
bundle = ArtifactBundle(
    checkpoint=local_path, checkpoint_sha256=trusted_digest,
    label_map=label_map_path, preprocessing_config=config_path,
    calibration=calibration_path, provenance=bundle_manifest_path,
)
predictor = TaskCPredictor.from_bundle(bundle, device="cpu")
result = predictor.predict(InputScan(
    path=nifti_path, case_id=case_id,
    modality="T1w_noncontrast", channel_index=explicit_index,
    source_provenance=optional_provenance,
))
legacy_json = serialize_official10(result, profile="legacy_v1")
projection = project_to_main(result)  # explicit failure/None unless eligible
```

Initial supported transport: single native non-contrast 3D NIfTI or explicitly declared modality axis/channel. Reject DICOM until conversion is validated separately. Accept a file path rather than an A/B mask or classification-ground-truth metadata. Inputs need a case ID/provenance but must not use known diagnosis to force model routing. A whole-scan classifier must not be falsely attached as independently computed evidence for every lesion in an A/B result.

### Authoritative new result envelope

Record `status` from `classified`, `abstained`, `unsupported_output`, `qc_failed`, `error`; reason codes; `case_id`; declared modality/channel; support metadata; raw logits/model scores when available; calibrated score scope; actual QC; routing policy; source/bundle/preprocessing/calibration hashes; geometry trace; precision/device/timing; and optional legacy projection. A/B outputs are absent/optional, not manufactured. Use separate runtime dataclasses/schema version from `TASK_SCHEMA_VERSION=1.0.0`.

Distinguish these concepts in separate fields:

- Ten raw model outputs: **not validated ten-disease probabilities**.
- Three-way `softmax(logits[[1,2,3]]/.4)`: an internally temperature-scaled distribution **conditional on the three supported tumor labels**, not a tumor-presence probability or a validated normal/unseen-disease probability.
- Routing decision and any ten-key compatibility allocation: decision-policy output, not clinical calibration.
- Invalid/no-model result: diagnostic model probabilities unavailable. Never create clinical posterior values to fill a required field.

Retain both raw and derived values with a versioned policy; do not silently replace original scores. Until environment, numerical parity and exact-path evaluation gates pass, set clinical/legacy projection unavailable by default and label results research-only.

### Official ten-name serialization

`classification.json` keeps **exactly the original ten probability keys, order, spelling, top-level contract fields, rounding, and sum tolerance** for the `legacy_v1` compatibility profile. Do not silently reinterpret these values as a new calibrated distribution. Preserve the upstream legacy numerical behavior where backward compatibility is required, including policy fallback scores, and write an authoritative adjacent `classification.meta.json` with score origin/scope, support flags, failure reason, true QC, and `clinical_probabilities_available=false` on QC/unsupported/legacy-unvalidated results. The consumer must read this metadata and gate reporting. A standalone JSON containing only the legacy scores is unsuitable for clinical use.

A future corrected profile would require a separately authorized, versioned contract decision; it is **not part of this documentation change or the unchanged-contract adapter**. Such a profile could use matched three-class calibration plus explicit abstention, and maintain the same ten **names** in a separate transport, with unavailable classes marked in sidecar metadata. Zero/not-scored allocation is not proof a disease is absent. If an organizer requires probabilities for every disease even on invalid scans, explain the mathematical-output-versus-clinical-probability limitation rather than invent calibration. Confirm organizer rules before adding top-level fields or choosing a new numerical encoding. For a future new adapter entry point, operational errors such as missing weights must produce a distinct structured error/nonzero CLI outcome, not a fake diagnostic success; preserve the existing upstream CLI and JSON behavior. Do not advertise this as numerical backward compatibility if a new policy changes the vector.

### Main projection

Use explicit aliases for only Glioma→glioma, Meningioma→meningioma, Metastasis→metastasis. Permit projection only for a classified result with passed input/QC, a supported winner, validated bundle/policy/score semantics, and explicit restricted-research acceptance. Include all supported differential entries with consecutive ranks; do not silently renormalize ten raw scores and call that the old posterior. Keep the conditional scope visible in the accompanying envelope and model identifier; no fabricated feature attribution/evidence contribution.

For abstention, unsupported output, QC failure, missing artifacts, or operational error, projection returns a typed unavailability/error (or None in a new envelope), **not** a default `TaskCOutput(no_finding)` or an invented `Uncertain/Other` main label. Existing `TaskCOutput` cannot carry all required semantics, so it is never the sole authoritative result. Do not create a `PipelineOutput` with dummy A/B outputs just to satisfy its required fields. Existing main callers and serialization remain unchanged until they explicitly opt into this adapter.

## 10. Proposed file tree

All existing paths remain. The following are proposed additions, not files imported into main during this audit:

```text
BrainAgenticthon-2026/
├── src/brainagenticthon/
│   ├── preprocessing/              # ALL current files and schemas retained
│   ├── classification/
│   │   ├── __init__.py              # current file retained
│   │   └── task_c_adapter.py        # optional bridge; no mandatory torch import
│   ├── detection/                  # existing A scaffold retained
│   └── spatial/                    # existing B scaffold retained
├── packages/task_c/
│   ├── pyproject.toml              # separate package; Python>=3.11, optional ML extras
│   ├── README.md
│   ├── LICENSE                     # only after owner grants/clarifies rights
│   ├── THIRD_PARTY_NOTICES.md
│   ├── provenance/upstream.json    # full commit, file/artifact digests, exceptions
│   ├── src/brainagenticthon_task_c/
│   │   ├── __init__.py              # lightweight public API; no IO/model construction
│   │   ├── __main__.py
│   │   ├── contracts.py             # independent request/result/support/error types
│   │   ├── labels.py                # explicit ID/name/support/alias crosswalk
│   │   ├── artifacts.py             # digest + safe/strict checkpoint loading
│   │   ├── preprocessing_v1.py      # audited legacy tensor semantics
│   │   ├── geometry.py              # native/model-space provenance, no invented inverse
│   │   ├── predictor.py             # injectable backend + lazy MONAI implementation
│   │   ├── calibration.py           # validated domain and versioned policy
│   │   ├── official_json.py         # ten-name serializer + metadata sidecar
│   │   └── resources/              # versioned YAML/label-map/calibration, no weights
│   └── tests/                      # unit/synthetic/golden/optional-artifact tests
├── tests/                          # all existing tests preserved
│   └── integration/task_c/         # new adapter regressions, no A/B model dependency
├── configs/task_c.example.yaml     # local artifact path/digest template; no credentials
├── docs/integration/task_c.md
└── requirements/task_c/            # vetted hashed CPU/GPU locks and build manifest

External, ignored artifact store (example only):
/workspace/brainagenticthon-storage/checkpoints/task_c/<artifact-version>/best_model.pt
```

A future separately authorized provisioning step could reference a trusted artifact already present at an approved path; this audit did not create that checkpoint location or transfer weights. Research training/evaluation scripts can later live in a separate `research/task_c/` area with archived provenance, not runtime imports. No existing file needs to be overwritten to start the new package.

## 11. Task A and Task B remaining work

**Task A:** No normal-versus-abnormal detector is implemented in main. Task C's three-tumor model cannot supply Normal/abnormal detection: it has no verified healthy-control training and tumor-type scores are conditional. Define normal/abnormal scope, obtain licensed explicit normal controls and broad abnormal cohorts, prevent missing-label→normal inference, hold out biological patients/sites and A/B/C overlapping releases, then train/validate a detection method and calibrate it independently in a separate task. Main `TaskAOutput(finding_present=True)` currently requires a bounding box; a binary scan-level detector alone cannot honestly fill that schema. Either provide independently validated localization or design a separately versioned binary result and explicit bridge. Do not invent boxes or use Task C argmax as localization.

**Task B:** No lesion segmenter, spatial refinement, atlas mapping or volumetry is implemented in main. Add true lesion segmentation/localization, original/model/atlas coordinate traces, correct label conventions for each actual release (do not blindly assume all BraTS years share labels), native-space mask reconstruction, lesion-component analysis, anatomical-region/laterality attribution and physical measurements. Compute volumes with actual affine voxel volume, e.g. `abs(det(affine[:3,:3]))/1000` ml per voxel with appropriate physical-space interpretation; use nearest interpolation for label maps, never image interpolation on discrete labels. Validate lesion-wise and voxel-wise accuracy, region mapping and known-geometry volumes. Task C's brain mask, center-crop metadata and old references to absent SegResNet weights do not fulfill Task B. No A/B artifact is required for the proposed standalone C interface.

## 12. Prioritized integration plan and release gates

| Priority/stage | Work | Exit evidence |
|---|---|---|
| P0: provenance and safety | Confirm code/model redistribution rights and official competition schema; obtain actual training environment, frozen split manifests, trusted checkpoint/config/calibration SHA-256s; agree public artifact/PNG handling with owner. Preserve all original work. | Owner-authorized licenses, artifact manifest, original split identity and authoritative contract available. |
| P0: runtime and semantics | Resolve torch/MONAI conflict in isolated supported patched environment; establish safe loading; validate label IDs/config/indices; separate policy scores from probabilities; forbid unsupported/failed-QC legacy projection. | Dependency lock resolves, no unsafe pickle fallback, source-probe defects covered by new regressions, documented restricted support. |
| P1: package extraction | Add independent runtime namespace, injectable backend, request/result/envelope/serializer, optional main bridge; keep legacy runtime outputs as explicitly versioned research compatibility only. | Package imports without IO/weights/A/B; CPU synthetic tests and all existing main tests pass; no changed legacy schema behavior. |
| P1: tensor/logit parity | Compare author-confirmed preprocessing and frozen bundle under old and candidate environments; audit float16 caches, FP32 path and CUDA autocast; record geometry without changing tensors. | Reviewed golden tensors/logits and tolerances, modality/affine/spacing/crop cases, artifact identity tied to results. Any semantic change escalated for retraining/recalibration assessment. |
| P1: full-path reproducibility | Run real trusted checkpoint offline on approved local fixtures and exact deployed API; recompute probability/abstention metrics for matched domain/policy using frozen splits. | Actual CPU offline execution; no network attempts; real weight tests and consistent metrics; failures counted explicitly, no skipped tests called passes. |
| P2: independent evaluation | Acquire separately approved independent datasets, source-balanced support, normals/unseen pathologies, cross-release deduplication and patient/site-aware testing. | Predeclared external performance/calibration/abstention/OOD-error criteria evaluated on untouched data. No claim that routing reliably detects unseen diseases without evidence. |
| P2: A/B development | Separate normal/abnormal detector and spatial-analysis milestones; keep C standalone. | Genuine detection/localization/segmentation evidence and safe optional composition; no dummy upstream outputs. |

Do not merge the repositories or enable clinical-facing diagnostics based solely on internal benchmark percentages. All 21 risks are preserved in the [risk register](task-c-risk-register.md); all 26 proposed test groups and their unrun status are in [acceptance tests](task-c-acceptance-tests.md). Use the [blocker checklist](README.md) for prioritized review and the [verification requirements](task-c-verification-requirements.md) for strict offline and independent calibration gates.

## 13. Key pinned source references

- [Task C inference](https://github.com/M0maniiR0/MRI-/blob/1c1c99730788a733484aa92f09057c144b37f8fb/inference.py#L59): artifact resolution/load, QC fallback at L126, ten-way probabilities at L160, policy rewrite at L171, CLI at L241.
- [Task C preprocessing](https://github.com/M0maniiR0/MRI-/blob/1c1c99730788a733484aa92f09057c144b37f8fb/preprocess.py#L87): DICOM geometry; mask L134; QC L165; crop L213; full pipeline L253.
- [Calibration domain](https://github.com/M0maniiR0/MRI-/blob/1c1c99730788a733484aa92f09057c144b37f8fb/calibrate_native_t1.py#L96), [evaluation domain](https://github.com/M0maniiR0/MRI-/blob/1c1c99730788a733484aa92f09057c144b37f8fb/evaluate_with_abstention.py#L84), [cache/split construction](https://github.com/M0maniiR0/MRI-/blob/1c1c99730788a733484aa92f09057c144b37f8fb/cache_dataset.py#L83).
- [Main labels](https://github.com/abdulrahmantaani24-dotcom/BrainAgenticthon-2026/blob/1d235147cca72b888dc2028a0e5e3eed1711d00c/src/brainagenticthon/preprocessing/manifest_schema.py#L104), [main C schema](https://github.com/abdulrahmantaani24-dotcom/BrainAgenticthon-2026/blob/1d235147cca72b888dc2028a0e5e3eed1711d00c/src/brainagenticthon/preprocessing/task_output_schemas.py#L297).
- Dependency metadata: `dependency-metadata.json`, retrieved from `https://pypi.org/pypi/<package>/<version>/json`; [MONAI 1.6.1 metadata](https://pypi.org/pypi/monai/1.6.1/json).
- Security evidence: `torch-load-advisory.json`; public/license evidence: `repository.json`; complete pinned inventory: `tree.json`; text hashes: `preservation.json`.
