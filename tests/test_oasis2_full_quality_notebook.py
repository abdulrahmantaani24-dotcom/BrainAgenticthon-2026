"""Validate the code-only archive and Colab audit cell with synthetic inputs."""

from __future__ import annotations

import ast
import io
import json
import runpy
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks/03_oasis2_full_quality_audit.ipynb"


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / "raw"
    metadata = tmp_path / "metadata"
    raw.mkdir()
    metadata.mkdir()
    rows = []
    for i in range(3):
        filename = f"PRIVATE_SENTINEL_SCAN_{i}.nifti.hdr"
        data = np.arange(12**3, dtype=np.float32).reshape(12, 12, 12, 1)
        nib.save(nib.Nifti1Pair(data, np.eye(4)), raw / filename)
        rows.append(
            {
                "patient_id": f"PRIVATE_SENTINEL_SUBJECT_{i}",
                "visit_id": "MR1",
                "clinical_group": "Nondemented",
                "cdr": 0,
                "relative_path": filename,
                "dataset": "OASIS-2",
                "task_a_label": None,
                "task_c_label": None,
            }
        )
    manifest = metadata / "synthetic.json"
    manifest.write_text(json.dumps(rows))
    return manifest, raw, tmp_path / "private-reports" / "synthetic-run"


@pytest.fixture
def namespace():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if set(cell["metadata"].get("tags", [])) & {
            "configuration",
            "bundle-definitions",
            "offline-definitions",
        }:
            exec("".join(cell["source"]), namespace)
    return namespace


@pytest.fixture
def bundle(tmp_path):
    builder = runpy.run_path(str(ROOT / "scripts/build_oasis2_audit_bundle.py"))
    path = tmp_path / "oasis2-full-quality-audit-code.zip"
    builder["build_bundle"](path)
    return path


def test_notebook_structure_empty_outputs_and_private_defaults(namespace):
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert "attachments" not in cell
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    assert namespace["DRY_RUN"] and namespace["MAX_NEW_PAIRS"] == 3
    assert not namespace["AUTHORIZED_PRIVATE_ACCESS"] and not namespace["RECOVER_STALE_LOCK"]


def test_bundle_is_code_only_checksummed_reproducible_and_extracts_safely(
    namespace,
    bundle,
    tmp_path,
):
    builder = runpy.run_path(str(ROOT / "scripts/build_oasis2_audit_bundle.py"))
    repeated = tmp_path / "repeated.zip"
    builder["build_bundle"](repeated)
    assert bundle.read_bytes() == repeated.read_bytes()
    root = tmp_path / "synthetic-code"
    manifest = namespace["unpack_code_bundle"](bundle.read_bytes(), root)
    assert manifest["code_only"]
    assert "src/brainagenticthon/preprocessing/oasis2_mri_quality.py" in manifest["files"]
    assert not any(
        name.startswith(("data/", "tests/", "checkpoints/")) for name in manifest["files"]
    )
    assert all(
        Path(name).suffix in {".py", ".ipynb", ".toml", ".md", ""} for name in manifest["files"]
    )
    assert namespace["unpack_code_bundle"](bundle.read_bytes(), root) == manifest


@pytest.mark.parametrize("mode", ["traversal", "medical_file", "checksum", "duplicate"])
def test_unsafe_or_corrupt_code_bundle_is_rejected_before_extraction(
    namespace,
    bundle,
    tmp_path,
    mode,
):
    with zipfile.ZipFile(bundle) as source:
        blobs = {name: source.read(name) for name in source.namelist()}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, value in blobs.items():
            if mode == "checksum" and name.endswith("oasis2_mri_quality.py"):
                value += b"\n# synthetic corruption"
            archive.writestr(name, value)
        if mode == "traversal":
            archive.writestr("../outside.py", "synthetic")
        elif mode == "medical_file":
            archive.writestr("synthetic-private-scan.img", b"synthetic only")
        elif mode == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("README.md", "synthetic duplicate")
    destination = tmp_path / "synthetic-destination"
    with pytest.raises(ValueError):
        namespace["unpack_code_bundle"](payload.getvalue(), destination)
    assert not destination.exists() and not (tmp_path / "outside.py").exists()


def test_notebook_audit_cell_runs_only_on_synthetic_inputs(namespace, inputs, capsys):
    notebook = json.loads(NOTEBOOK.read_text())
    namespace.update(
        MANIFEST=inputs[0],
        IMAGE_ROOT=inputs[1],
        OUTPUT_DIR=inputs[2],
        DRY_RUN=False,
        MAX_NEW_PAIRS=1,
    )
    cell = next(c for c in notebook["cells"] if "audit-runtime" in c["metadata"].get("tags", []))
    exec("".join(cell["source"]), namespace)
    assert namespace["report"]["completed_pairs"] == 1
    exec("".join(cell["source"]), namespace)
    assert namespace["report"]["completed_pairs"] == 2
    output = capsys.readouterr().out
    assert "PRIVATE_SENTINEL" not in output and str(inputs[1]) not in output
    assert not namespace["report"]["visual_qc_approved"]
    assert not namespace["report"]["final_task_labels_assigned"]
