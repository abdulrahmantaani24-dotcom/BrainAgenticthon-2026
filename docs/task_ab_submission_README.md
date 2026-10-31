# Task A/B research prototype handoff

Start with [the shared execution launcher](../notebooks/07_task_ab_execution_launcher.ipynb)
and [execution guide](task_ab_execution_delivery.md). Build one reviewed code-only
bundle for all three notebooks. Use private Drive for all execution artifacts.

Available: binary single-T1 research classification, patient/source-aware metrics
and descriptive calibration, compact four-modality 3D segmentation, supplied-grid
inference, genuine mask component boxes, trusted checkpoint reload/resume,
read-only eligibility reports and a synthetic CPU/CUDA validation orchestrator.

Not complete: independently verified GPU execution, eligible real-data training,
trained Task A localization, approved BraTS2021 legacy-schema source mapping,
anatomical atlas mapping, full challenge integration or clinical validation.
The launcher refuses automatic real training. Task C and prior policies are unchanged.

Verification commands from an installed development environment:

```bash
python -m pytest -q
python scripts/check_sprint_privacy.py
python scripts/check_sprint_privacy.py --staged
python -m brainagenticthon.execution.validation --mode synthetic --tasks both --output /tmp/NEW-RUN
```

Use Ruff on changed Python/notebook files. The repository baseline contains legacy
Ruff findings; preserve them separately from new code checks. Validate notebook
schema and empty outputs with nbformat before publishing. Commit only reviewed
code/config templates/tests/docs; never private scans, manifests, reports or weights.
Evaluation and execution JSON templates under configs/ are explicitly NOT_EXECUTED
or NOT_EVALUATED and contain no fabricated results.
