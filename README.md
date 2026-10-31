# BrainAgenticthon-2026

**Offline, Reproducible 3D Brain MRI AI Challenge**

An agentic AI system that processes 3D brain MRI volumes through a
multi-stage pipeline: detection → spatial characterisation → classification →
evidence-based clinical reporting — with built-in verification at every step.

---

## Challenge Tasks

| Task | Module | Description |
|------|--------|-------------|
| **A – Detection** | `src/brainagenticthon/detection/` | Detect the presence and coarse location of brain abnormalities in 3D MRI volumes. Outputs bounding boxes and confidence scores. |
| **B – Spatial Characterisation** | `src/brainagenticthon/spatial/` | Refine detections with voxel-level segmentation masks, anatomical-region mapping, and volumetric measurements. |
| **C – Classification** | `src/brainagenticthon/classification/` | Classify each detected finding, providing differential diagnosis, class probabilities, and supporting evidence. |
| **D – Evidence-Based Clinical Reporting** | `src/brainagenticthon/reporting/` | Generate structured clinical reports that integrate and cite findings from Tasks A–C. |

### Verification Component

The **verification** module (`src/brainagenticthon/verification/`) is a
cross-cutting concern that runs after every pipeline stage:

- **Reproducibility** — deterministic seeds, pinned dependency hashes, config snapshots.
- **Metric validation** — automated metric computation against ground truth.
- **Audit trail** — immutable log of every prediction, config, and environment used.
- **Integrity checks** — SHA-256 hash verification of input data and model weights.

---

## Architecture

```
Raw NIfTI ─▶ Preprocessing ─▶ Detection (A) ─▶ Spatial (B) ─▶ Classification (C) ─▶ Reporting (D)
                                     │               │                │                    │
                                     └───────────────┴────────────────┴────────────────────┘
                                                          │
                                                    Verification
                                                   (runs at every stage)
```

An **agentic orchestration layer** (`src/brainagenticthon/agents/`) coordinates
execution across tasks, manages intermediate artifacts, and implements retry /
fallback logic.

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/data_policy.md`](docs/data_policy.md) for data-handling rules.

---

## Project Structure

```
BrainAgenticthon-2026/
├── configs/                  # Experiment YAML/JSON configs
├── data/
│   ├── raw/                  # Original MRI volumes (git-ignored)
│   ├── interim/              # Preprocessed data    (git-ignored)
│   └── processed/            # Predictions & output (git-ignored)
├── notebooks/                # Exploratory Jupyter notebooks
├── src/brainagenticthon/
│   ├── preprocessing/        # Skull-strip, resample, normalise
│   ├── detection/            # Task A
│   ├── spatial/              # Task B
│   ├── classification/       # Task C
│   ├── agents/               # Agentic orchestration
│   ├── verification/         # Reproducibility & audit
│   └── reporting/            # Task D
├── backend/                  # Backend / API services
├── tests/                    # Pytest test suite
├── scripts/                  # Helper & automation scripts
├── docs/                     # Documentation
├── artifacts/                # Generated reports & outputs
├── pyproject.toml            # PEP 621 package metadata
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python ≥ 3.10
- (Optional) A CUDA-capable GPU for model training / inference

### Install (editable, development mode)

```bash
# Clone the repository
git clone https://github.com/abdulrahmantaani24-dotcom/BrainAgenticthon-2026.git
cd BrainAgenticthon-2026

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install the package in editable mode with dev extras
pip install -e ".[dev]"
```

### Run the smoke test

```bash
pytest tests/test_smoke.py -v
```

### Task A dataset readiness

See [Task A dataset readiness](docs/task_a_dataset_readiness.md) for the read-only
manifest/image audit, verified control rules, source/site confounding checks,
and authorized Colab access. The workflow does not download data or train a model.

---

## Data Policy

Medical imaging data (NIfTI, DICOM, etc.) is **never committed** to this
repository.  Place raw volumes in `data/raw/`; they are excluded by
`.gitignore`.  See [`docs/data_policy.md`](docs/data_policy.md) for details.

---

## License

MIT — see individual modules for any additional notices.

## Task A/B private research execution

See [the research prototype handoff](docs/task_ab_submission_README.md) and
[shared Colab launcher](notebooks/07_task_ab_execution_launcher.ipynb). The launcher
executes synthetic checks only; real data requires independent eligibility review.
Task C and clinical policies remain unchanged.

For authorized real-data preparation, use [Notebook 08](notebooks/08_private_real_data_readiness_training.ipynb)
and the [private readiness/training guide](docs/task_ab_real_data_workflow.md).
Its default action is read-only inventory; real execution stays disabled until all gates pass.
