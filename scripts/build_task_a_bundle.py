"""Build a deterministic allowlisted public-code bundle; no data or model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = tuple("src/brainagenticthon/" + name for name in (
    "__init__.py", "detection/__init__.py", "detection/profile.py", "detection/imaging.py",
    "detection/data.py", "detection/model.py", "detection/localization.py",
    "detection/evaluation.py", "detection/training.py", "preprocessing/__init__.py",
    "preprocessing/task_a_readiness.py", "preprocessing/data_validator.py",
    "preprocessing/dataset_registry.py", "preprocessing/manifest_schema.py",
    "preprocessing/split_manager.py", "preprocessing/task_output_schemas.py",
    "preprocessing/oasis2_mri_quality.py", "segmentation/__init__.py",
    "segmentation/model.py", "segmentation/data.py", "segmentation/inference.py",
    "segmentation/training.py", "segmentation/predict.py",
)) + ("requirements-task-a.txt", "scripts/build_task_a_bundle.py",
      "docs/task_ab_colab_launch.md", "configs/task_a_research.example.json",
      "docs/task_a_research_training.md", "configs/task_b_research.example.json",
      "docs/task_b_research_training.md", "notebooks/06_task_b_segmentation_training.ipynb",
      "notebooks/05_task_a_research_training.ipynb")


def build_bundle(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).absolute()
    if (output.resolve().is_relative_to(root) or output.suffix != ".zip"
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("bundle_requires_zip_outside_repository_without_symlinks")
    if output.exists():
        raise ValueError("bundle_output_already_exists_choose_a_new_path")
    blobs = {}
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("bundle_input_not_reviewable_public_code")
        data = path.read_bytes()
        if name.endswith(".ipynb"):
            notebook = json.loads(data)
            if any(c.get("outputs") or c.get("execution_count") is not None
                   for c in notebook["cells"] if c["cell_type"] == "code"):
                raise ValueError("clear_notebook_outputs_before_bundling")
        blobs[name] = data
    manifest = {"code_only": True, "schema_version": 1,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()}}
    blobs["code_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    # Exclusive creation preserves any existing archive, even during concurrent calls.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(blobs.items()):
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"code_only": True, "sha256": build_bundle(args.output)}))


if __name__ == "__main__":
    main()
