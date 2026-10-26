# Architecture Overview

> BrainAgenticthon-2026 – Offline, Reproducible 3D Brain MRI AI Challenge

## High-Level Pipeline

```
 ┌──────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────────┐
 │ Raw Data │───▶│ Preprocessing  │───▶│  Task Modules │───▶│  Reporting  │
 │ (NIfTI)  │    │ (skull-strip,  │    │  A/B/C (below)│    │  (Task D)   │
 └──────────┘    │  resample,     │    └──────┬───────┘    └──────┬──────┘
                 │  normalise)    │           │                   │
                 └────────────────┘           ▼                   ▼
                                      ┌──────────────┐    ┌──────────────┐
                                      │ Verification │◀───│   Agents     │
                                      │ (audit,      │    │ (orchestrate │
                                      │  metrics,    │    │  multi-task) │
                                      │  repro check)│    └──────────────┘
                                      └──────────────┘
```

## Component Descriptions

### `preprocessing/`
Skull-stripping, spatial resampling, intensity normalisation, and data
augmentation utilities.  All transforms are deterministic when given a fixed
random seed.

### `detection/` — Task A
Detect the presence and coarse location of brain abnormalities in 3D MRI
volumes.  Outputs bounding-box coordinates and confidence scores.

### `spatial/` — Task B
Refine detections from Task A with voxel-level spatial characterisation:
segmentation masks, anatomical-region mapping, and volumetric measurements.

### `classification/` — Task C
Classify / provide differential diagnosis for each detected finding using
the spatial features from Task B.  Outputs class labels, probabilities, and
supporting evidence references.

### `reporting/` — Task D
Generate structured, evidence-based clinical reports that integrate outputs
from Tasks A–C.  Reports cite specific findings, locations, and
classification evidence.

### `agents/`
Agentic orchestration layer.  Coordinates the execution of Tasks A → B → C → D,
manages intermediate artifacts, and implements retry / fallback logic.

### `verification/`
Cross-cutting concern that ensures:

| Check                  | Description                                      |
|------------------------|--------------------------------------------------|
| **Reproducibility**    | Deterministic seeds, pinned dependency hashes    |
| **Metric validation**  | Automated metric computation against ground truth|
| **Audit trail**        | Immutable log of every prediction + config used  |
| **Integrity**          | Hash-based checks on input data & model weights  |

## Data Flow

1. Raw NIfTI volumes are placed in `data/raw/` (never committed).
2. Preprocessing writes standardised volumes to `data/interim/`.
3. Task modules read from `data/interim/` and write predictions to
   `data/processed/`.
4. The reporting module consumes all upstream outputs and produces final
   reports in `artifacts/`.
5. Verification hooks run after every stage.

## Reproducibility Contract

* Every experiment is fully defined by a YAML config in `configs/`.
* Random seeds are set globally and per-module.
* Model weights are referenced by SHA-256 hash.
* The complete environment is captured via `pip freeze` / lockfile.
