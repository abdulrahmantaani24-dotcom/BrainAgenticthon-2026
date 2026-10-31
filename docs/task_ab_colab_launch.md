# Reviewed code-only Task A/B Colab launch

Both notebooks use **one identical ZIP** and the same requirements-task-a.txt
(Torch 2.6.0, NumPy 2.5.4, NiBabel 5.4.2, Python 3.12). The legacy builder filename
now covers both tasks. No scans, clinical records, private manifests/reports,
Visual QC work or checkpoints belong in the ZIP or repository.

From the exact published sprint commit in a reviewed checkout:

```bash
python scripts/build_task_a_bundle.py --output /tmp/task-ab-code.zip
sha256sum /tmp/task-ab-code.zip
```

The builder needs only Python's standard library. It prints a JSON `sha256` field.
Retain this independently of the uploaded ZIP. Compare it with `sha256sum`; on
macOS use `shasum -a 256`. Build a second ZIP to a different unused path on the
same Python/zlib runtime to verify byte-for-byte determinism. The fixed ZIP
metadata and sorted entries make builds repeatable in that environment. Different
compression implementations may change ZIP bytes; always independently hash the
actual reviewed archive you upload. Do not embed the archive's own hash in its
notebooks (that would be self-referential).

1. Upload that ZIP once to private Drive, e.g.
   `MyDrive/reviewed_code/task-ab-code.zip`. Keep the retained SHA-256 separately.
2. Open Notebook 05 or 06 from the published feature commit using Colab's GitHub
   tab. Save a private Drive copy and disable saving cell outputs. Select a GPU
   runtime with Python 3.12; the dependency cell enforces that version.
3. In the mount cell set `BUNDLE_PATH`, paste the independently retained 64-character
   lowercase hash into `BUNDLE_SHA256`, and set `CONFIG_PATH` to the private config
   for that task. Both notebooks must use the **same ZIP and hash**. The verification
   cell checks the entire ZIP before extraction, then every source-file checksum.
4. Prepare a private copy of the corresponding JSON config template. Templates are
   included in the ZIP and published under configs/. For the first engineering run
   set `mode="synthetic_gpu"`, `training.epochs=1`, a dedicated private Drive root,
   and a fresh task-specific `output_run`. Keep `real_training_enabled=false`,
   resume flags false and final test evaluation false. No patient input is needed.
5. Run all cells. Installation is idempotent; research distribution pins and loaded
   module versions are checked. A stale imported version requires **Restart session**,
   followed by rerunning from the top with your saved parameters. Local code state
   recovers CODE_ROOT automatically; a full VM reset re-extracts from Drive. Do not
   manually restore paths or switch CPU settings to CUDA.
6. `synthetic_gpu` requires CUDA allocation, a finite forward/backward pass, a
   parameter update and synchronization, followed by CUDA training of that task.
   An unavailable CUDA device stops execution; there is no CPU fallback. The CPU
   `synthetic` mode must be selected explicitly when desired.

For training resume, retain the private run directory, set resume and
trusted_resume true, and increase total epochs; all other experiment inputs,
settings, code and runtime must match. Checkpoints must be your own trusted files.
Use fresh runs to switch from CPU to CUDA or change settings. Test evaluation is
explicit and prevents continuing training that run.

Real training stays blocked until the task-specific clinical/release/access,
mask/label, geometry and patient-identity safeguards are independently satisfied.
See task_a_research_training.md and task_b_research_training.md. Synthetic success
is not real-data eligibility or clinical performance. CUDA execution and hosted
Colab operation must be reported from actual observed runs, never inferred.
