# Cloud Data Workflow

> BrainAgenticthon-2026 — Google Colab-Based Data Acquisition & Training Storage

## Overview

This project uses a **split-environment** architecture:

| Environment | Role | Persistence |
|-------------|------|-------------|
| **Local machine** (Antigravity) | Code authoring, version control, offline inference | Permanent (Git) |
| **Google Colab** | Dataset acquisition, preprocessing, training | Ephemeral runtime + Google Drive |
| **Google Drive** | Persistent storage for checkpoints, metrics, outputs | Persistent (user-owned) |
| **GitHub** | Source code, configs, documentation only | Permanent (repo) |

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Google Colab Runtime                          │
│                                                                      │
│  /content/brainagenticthon/          (ephemeral workspace)           │
│  ├── data/raw/brats/                 ← downloaded via Synapse API    │
│  ├── data/interim/                   ← preprocessed volumes          │
│  └── data/processed/                 ← predictions                   │
│                                                                      │
│  /content/drive/MyDrive/BrainAgenticthon-2026/   (persistent)        │
│  ├── checkpoints/                    ← saved model states            │
│  ├── metrics/                        ← training & eval metrics       │
│  ├── outputs/                        ← final artifacts & reports     │
│  ├── logs/                           ← experiment logs               │
│  └── configs_snapshots/              ← frozen configs per run        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  01_dataset_access.ipynb                                      │    │
│  │  ● Mount Google Drive                                         │    │
│  │  ● Configure dataset source (BraTS / local / custom)          │    │
│  │  ● Dry-run inspection                                         │    │
│  │  ● Fetch & validate data                                      │    │
│  │  ● Save checkpoints & metrics to Drive                        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
         │                                    │
         │ (never uploaded)                   │ (persistent, private)
         ▼                                    ▼
   ┌───────────┐                     ┌──────────────────┐
   │  GitHub    │                     │  Google Drive     │
   │  (code     │                     │  (checkpoints,    │
   │   only)    │                     │   metrics,        │
   └───────────┘                     │   outputs)        │
                                     └──────────────────┘
```

## What Goes Where

### ✅ GitHub (committed)
- Source code (`src/`)
- Test suite (`tests/`)
- Configuration **templates** (`configs/storage.example.yaml`)
- Documentation (`docs/`)
- Notebook **code** (`notebooks/`)

### 🔒 Google Drive (persistent, never committed)
- Model checkpoints and weights
- Training / evaluation metrics (JSON, CSV)
- Experiment outputs and reports
- Config snapshots (frozen copies per experiment run)
- Experiment logs

### ⚠️ Never Stored Anywhere Public
- Raw patient MRI data
- `.synapseConfig` or API tokens
- Patient identifiers or PHI
- De-anonymised metadata

---

## Step-by-Step Workflow

### 1. Obtain Dataset Access (One-Time, Manual)

Before using the notebook, you must obtain authorised access:

1. **BraTS Challenge Data:**
   - Create an account at [Synapse](https://www.synapse.org/)
   - Navigate to the BraTS challenge project page
   - Accept the data use agreement
   - Wait for access approval
   - Configure `~/.synapseConfig` with your credentials

2. **Other Authorised Datasets:**
   - Follow the dataset provider's access procedure
   - Store credentials securely (never in Git)

### 2. Configure Storage

```bash
# Copy the example config
cp configs/storage.example.yaml configs/storage.yaml

# Edit with your actual Synapse project ID and paths
# NEVER commit configs/storage.yaml
```

### 3. Open the Colab Notebook

Open `notebooks/01_dataset_access.ipynb` in Google Colab:

1. **Mount Google Drive** — the notebook prompts for authorisation
2. **Install dependencies** — `pip install` runs inside the Colab cell
3. **Clone the repo** — pulls the latest source code
4. **Run dry-run first** — inspect what would be downloaded
5. **Fetch data** — downloads to `/content/brainagenticthon/data/raw/`
6. **Validate** — checks NIfTI integrity, modalities, case structure

### 4. Validate Downloaded Data

The `DatasetValidator` runs automatically after download:

| Check | Description |
|-------|-------------|
| File integrity | SHA-256 hash verification |
| NIfTI header | Valid 3D shape, affine, data type |
| Orientation | RAS convention |
| Voxel spacing | Plausible range (0.1–10 mm) |
| Modality completeness | All required modalities present per case |
| Case ID format | Matches expected naming pattern |
| Label map values | Segmentation labels within expected set |
| Spatial consistency | All modalities share shape and affine |

### 5. Training & Checkpointing

During training (in subsequent notebooks):

```python
from brainagenticthon.preprocessing.dataset_registry import StorageConfig

cfg = StorageConfig.from_yaml("configs/storage.yaml")

# Save checkpoint to persistent storage (Google Drive)
checkpoint_path = cfg.resolve_path("checkpoints") / f"epoch_{epoch:03d}.pt"
torch.save(model.state_dict(), checkpoint_path)

# Save metrics
metrics_path = cfg.resolve_path("metrics") / "train_metrics.json"
```

### 6. Offline Inference (Local)

After training, transfer only the final model weights to your local machine:

```bash
# Download from Google Drive (manual or via gdown)
# Place in a local directory NOT tracked by Git
```

The inference pipeline reads from local paths and requires no network access.

---

## Dataset Registry — Pluggable Sources

The `DatasetRegistry` supports multiple dataset sources:

```python
from brainagenticthon.preprocessing.dataset_registry import (
    DatasetRegistry,
    BraTSSource,
    LocalDirectorySource,
)

# List available sources
registry = DatasetRegistry()
for name in registry.list_sources():
    print(name, registry.get_source(name).description)

# Dry-run: see what would be downloaded
source = BraTSSource(
    synapse_project_id="syn00000000",  # your actual ID
    credentials_path="~/.synapseConfig",
)
source.fetch(output_dir="/content/brainagenticthon/data/raw/brats", dry_run=True)
```

### Adding a Custom Source

```python
from brainagenticthon.preprocessing.dataset_registry import DatasetSource, register_source

@register_source
class MyHospitalDataSource(DatasetSource):
    @property
    def name(self) -> str:
        return "my_hospital"

    def fetch(self, output_dir, dry_run=False):
        if dry_run:
            self.logger.info("Would copy from internal NAS to %s", output_dir)
            return
        # ... actual copy logic ...
```

---

## Security Checklist

- [ ] `.synapseConfig` is in `.gitignore` — **verified**
- [ ] `configs/storage.yaml` is in `.gitignore` — only `storage.example.yaml` is tracked
- [ ] No patient data paths in committed code
- [ ] Model weights excluded by `.gitignore` patterns
- [ ] Colab notebook does not print or log credentials
- [ ] Google Drive folder is private (not shared publicly)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `synapseclient` not installed | Run `pip install synapseclient` in Colab |
| Synapse access denied | Verify data use agreement acceptance on synapse.org |
| Google Drive not mounted | Re-run the mount cell; authorise when prompted |
| NIfTI validation fails | Check the `ValidationReport` for specific errors |
| Hash mismatch | Re-download the affected file; check network integrity |
| Colab runtime disconnected | Re-run from the mount cell; cached data persists on Drive |
