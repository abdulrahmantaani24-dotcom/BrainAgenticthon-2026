# Data Policy

> BrainAgenticthon-2026 – Handling of Medical Imaging Data

## Guiding Principles

1. **No patient data in version control.**  Raw, interim, and processed
   medical images are excluded from Git via `.gitignore`.
2. **Offline-first.**  The pipeline is designed to run without internet
   access once data and model weights have been provisioned locally.
3. **Minimal data retention.**  Only the data necessary for the current
   experiment should reside on the local machine.

## What Is Excluded from the Repository

| Pattern / Directory         | Reason                                    |
|-----------------------------|-------------------------------------------|
| `data/raw/*`                | Original medical images (NIfTI / DICOM)   |
| `data/interim/*`            | Preprocessed volumes                      |
| `data/processed/*`          | Model predictions & derived data          |
| `*.nii`, `*.nii.gz`         | NIfTI neuroimaging files                  |
| `*.dcm`                     | DICOM files                               |
| `*.mnc`                     | MINC format                               |
| `*.nrrd`                    | NRRD format                               |
| `*.mha`, `*.mhd`           | MetaImage format                          |
| `*.hdr`, `*.img`           | Analyze format                            |
| Model checkpoints & weights | Large binary blobs                        |
| `.env`, `secrets.*`        | Credentials, API keys                     |

## What IS Tracked

* `.gitkeep` placeholders inside data directories.
* `README.md` and documentation.
* Configuration YAML/JSON files (no secrets).
* Source code and tests.

## De-identification

If sample data is ever shared for debugging, it **must** be fully
de-identified in compliance with applicable regulations (e.g., HIPAA Safe
Harbor, GDPR pseudonymisation).  Responsibility lies with the data provider.

## Access Control Recommendations

* Store raw data on an encrypted volume.
* Use file-system permissions (`chmod 700`) or OS-level ACLs.
* Log all data access for audit purposes.
