# Task B research segmentation

Implemented profile: BraTS 2021 adult glioma training, four co-registered supplied
modalities T1/T1ce/T2/FLAIR, labels 0 background, 1 NCR/NET, 2 edema, 4 enhancing
tumor. Internal class 3 is explicitly mapped back to 4 on inference. This is a
separate release profile and does not modify the existing 2024 output contract.
Release documentation: https://www.med.upenn.edu/cbica/brats2021/data.html and
https://doi.org/10.48550/arXiv.2107.02314 . Those references describe a release;
they do not verify this user's access, files or clinical evidence. No real cohort
has been verified or trained here. Other releases/label conventions are rejected.

Private manifest format: JSON object with `release: "brats2021-glioma-training"`
and `records`: each record has that same `release`, reconciled `patient_id`, unique
`case_id`, `split` (train/validation/test), `review_status: "approved"`, `files`
and `sha256` mappings with exactly `t1`, `t1ce`, `t2`, `flair`, `seg`. Values in
`files` are relative to the private image root. Every hash binds a reviewed source
file. One case per reconciled patient is required, avoiding repeated-patient
weighting in evaluation; patient reconciliation across releases/sources is external.
No automatic splitting, label generation, access attestation or mask discovery.

Separate private release evidence requires `release`, `review_status: "approved"`,
actual JSON booleans `access_verified`, `masks_verified`, `geometry_verified`,
`identity_verified` all true and genuine nonempty `access_reference`,
`provenance_reference`, `mask_reference`, `label_reference`, `geometry_reference`,
`identity_reference`. Do not set these merely because files loaded. No approved
patient or evidence examples are published. All three frozen splits are mandatory.

Validation checks exact modality/mask keys, bytes and duplicate MRI hashes/paths,
finite 3D noncomplex data, resource limits, coded NIfTI transforms, mm units,
positive spacing, invertible affine, transform agreement, aligned shape/affine,
and integer release labels. Empty masks are allowed and never imply healthy
controls for Task A. Nonzero intensity normalization is per channel; zeros remain
background. No spatial resampling changes the source grid. Shared crops/padding
and flips apply identically to all modalities and masks. Training streams one
case at a time and uses small patches, AMP on CUDA, AdamW and CE plus soft Dice.
Patch sampling is uniform; class-balanced lesion sampling remains a future tuning
option. Default patches are 64 cubed, batch 1; measure actual GPU memory first.

Validation uses full-volume overlapping sliding windows with CPU probability
accumulation. Report WT=(1,2,4), TC=(1,4), ET=(4) Dice averaged across independent
patients. Both-empty regions are null/excluded, false-positive empty-target or
false-negative nonempty-target regions score zero. Report this convention with
results; it is not a claim to match every challenge leaderboard implementation.
Best validation weights are retained. Test is explicit and one-shot; its existence
blocks further training of that run. No test images are used during fitting.

Checkpoints are private, checksum-bound, atomic epoch generations with commit
markers; safe weights-only loading restores model, optimizer, scaler and CPU/CUDA
and augmentation RNG states. Resume requires unchanged code/data/evidence/runtime/
settings, explicit trust of your own checkpoint, and an increased total epoch
count. An interrupted partial epoch replays. Hashes detect corruption, not forgery.
CUDA results across different hardware/drivers need separate reproducibility checks.

`lesion_boxes` computes six-connected nonzero mask components, inclusive native
voxel bounds, voxel counts and explicit reference/prediction provenance. Empty
masks return no boxes. No fabricated whole-volume boxes or classifier localization.
No anatomical atlas, physical volumetrics, Task A-to-B integration or Task C
integration is claimed. No medical performance is established by synthetic tests.

## Minimum private Colab workflow

1. Build the reviewed code ZIP with `scripts/build_task_a_bundle.py --output /tmp/code.zip`;
   independently retain its SHA-256 and upload only that code ZIP to private Drive.
2. Open notebook 06 privately, disable output saving, select Python 3.12 and GPU.
   Set bundle path/hash and `CONFIG_PATH` to a private copy of the example JSON;
   set a fresh private output_run. Run all cells in order. Dependency install is
   idempotent. If stale imports are detected, Restart session and rerun from top.
3. Use `synthetic_gpu` first; an actual finite CUDA forward/backward update must pass.
4. Only after independently verifying the exact release's access/provenance/masks,
   identities and geometry, supply the private manifest/evidence and image root,
   set mode=verified and real_training_enabled=true, choose epochs and a fresh run.
   Run all cells. Real training remains blocked without those inputs.
5. To resume, retain output_run, set resume/trusted_resume=true and increase total
   epochs. For final held-out evaluation explicitly set final_test_evaluation=true.
   Keep manifests, scans, checkpoints, model predictions and logs off GitHub.

Private inference after a session restart can use `segmentation.predict.load_model`
with the same cohort/config and trusted own run, then `predict` with a hash-bound
four-modality inference row and a fresh output directory outside the inputs and
Git. It saves `prediction.nii.gz` (release labels, unchanged affine/grid) and
`boxes.json` (prediction provenance). The CLI is:

```bash
python -m brainagenticthon.segmentation.predict --config PRIVATE_CONFIG.json \
  --record PRIVATE_INFERENCE_ROW.json --output PRIVATE_NEW_OUTPUT \
  --trusted-own-checkpoint
```

The CLI revalidates the original training cohort/release evidence and binds the
checkpoint to its exact code, data and runtime. A separate inference row has the
same files/sha256 maps but excludes `seg`; its images must be independently
reviewed and authorized under that release's private data access. It does not
produce Task C outputs or diagnostic conclusions.
