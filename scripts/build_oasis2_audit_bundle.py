"""Build a code-only Colab snapshot outside the repository; no data directories.

Only the explicit public source/notebook/documentation allowlist is packaged.
Review the files before uploading; checksums detect corruption, not authenticity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "__init__.py",
    "agents/__init__.py",
    "classification/__init__.py",
    "detection/__init__.py",
    "reporting/__init__.py",
    "spatial/__init__.py",
    "verification/__init__.py",
    "preprocessing/__init__.py",
    "preprocessing/data_validator.py",
    "preprocessing/dataset_registry.py",
    "preprocessing/manifest_schema.py",
    "preprocessing/split_manager.py",
    "preprocessing/task_output_schemas.py",
    "preprocessing/task_a_readiness.py",
    "preprocessing/oasis2_clinical_policy.py",
    "preprocessing/oasis2_mri_quality.py",
    "preprocessing/oasis2_full_quality_audit.py",
)
BUNDLE_FILES = tuple("src/brainagenticthon/" + name for name in SOURCE_NAMES) + (
    "pyproject.toml",
    "README.md",
    "notebooks/02_oasis2_clinical_validation.ipynb",
    "notebooks/03_oasis2_full_quality_audit.ipynb",
    "docs/oasis2_full_quality_audit.md",
)


def build_bundle(output, root=SOURCE_ROOT):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.resolve().is_relative_to(root) or output.suffix != ".zip" or output.is_symlink():
        raise ValueError("bundle_must_be_a_zip_outside_the_repository")
    if output.exists() and output.stat().st_nlink != 1:
        raise ValueError("bundle_output_hardlink_rejected")
    blobs = {name: (root / name).read_bytes() for name in BUNDLE_FILES}
    manifest = {
        "schema_version": 1,
        "code_only": True,
        "baseline_commit": "37b5177a0296a6a4878f4254a27d87876272e362",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()},
    }
    blobs["code_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(blobs.items()):
            entry = zipfile.ZipInfo(name)  # Stable timestamps for a reproducible archive.
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/oasis2-full-quality-audit-code.zip")
    )
    args = parser.parse_args()
    print(json.dumps({"code_only": True, "bundle_sha256": build_bundle(args.output)}))


if __name__ == "__main__":
    main()
