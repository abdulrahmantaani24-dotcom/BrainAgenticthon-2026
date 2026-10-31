"""Offline notebook/bundle checks; execute synthetic cells without Colab/Drive."""

import ast
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import torch

from brainagenticthon.detection.data import SyntheticCohort
from brainagenticthon.detection.training import (
    TrainConfig,
    evaluate_test,
    fit,
    private_output_directory,
)
from brainagenticthon.preprocessing.oasis2_mri_quality import private_call, private_path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/05_task_a_research_training.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text())


def code_cells():
    return {c["id"]: "".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"}


def builder():
    spec = importlib.util.spec_from_file_location(
        "task_a_builder", ROOT / "scripts/build_task_a_bundle.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_structure_outputs_and_parse():
    nb = notebook()
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["private_outputs"] is True
    assert len(nb["cells"]) == 15
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    for cell in nb["cells"]:
        assert not cell["metadata"].get("attachments")
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    combined = NOTEBOOK.read_text()
    for warning in ("uncalibrated", "not", "Group/CDR", "OASIS-2", "visual QC", "native"):
        assert warning in combined
    assert "extractall" not in combined


def test_default_configuration_cannot_start_real_training():
    cfg = json.loads((ROOT / "configs/task_a_research.example.json").read_text())
    assert cfg["mode"] == "synthetic" and cfg["real_training_enabled"] is False
    assert cfg["resume"] is False and cfg["trusted_resume"] is False
    assert cfg["final_test_evaluation"] is False
    assert cfg["training"]["device"] == "cuda"
    assert cfg["training"]["num_workers"] == 0


def test_synthetic_notebook_cells_run_without_private_data(tmp_path, capsys):
    cfg = json.loads((ROOT / "configs/task_a_research.example.json").read_text())
    cfg.update(private_drive_root=str(tmp_path / "private"), output_run="runs/synthetic")
    path = tmp_path / "private-config.json"
    path.write_text(json.dumps(cfg))
    namespace = {"Path": Path, "json": json, "CONFIG_PATH": path, "DRIVE_BASE": tmp_path,
                 "CODE_ROOT": ROOT, "torch": torch, "TrainConfig": TrainConfig,
                 "SyntheticCohort": SyntheticCohort, "private_call": private_call,
                 "private_path": private_path, "private_output_directory": private_output_directory,
                 "fit": fit, "evaluate_test": evaluate_test}
    previous_threads = torch.get_num_threads()
    try:
        for identifier in ("private-config", "eligibility-gate",
                           "train-or-resume", "final-evaluation"):
            exec(compile(code_cells()[identifier], str(NOTEBOOK), "exec"), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    assert namespace["MODE"] == "synthetic"
    assert len(namespace["RESULT"].history) == 1
    assert not (namespace["OUTPUT"] / "test_metrics.json").exists()
    printed = capsys.readouterr().out
    assert str(tmp_path) not in printed
    assert "clinical_approval" in printed


def test_code_only_bundle_is_deterministic_and_allowlisted(tmp_path):
    module = builder()
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    one = module.build_bundle(first)
    assert module.build_bundle(second) == one
    assert hashlib.sha256(first.read_bytes()).hexdigest() == one
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {*module.BUNDLE_FILES, "code_manifest.json"}
        assert all(not name.endswith((".pt", ".nii", ".img", ".hdr"))
                   for name in archive.namelist())
        assert all("visual_qc" not in name for name in archive.namelist())
        manifest = json.loads(archive.read("code_manifest.json"))
        assert manifest["code_only"] is True
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(ValueError, match="already_exists"):
        module.build_bundle(first)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == one
    with pytest.raises(ValueError):
        module.build_bundle(ROOT / "unwanted.zip")


def test_bundle_rejects_saved_notebook_outputs(tmp_path):
    module = builder()
    public = tmp_path / "public-source"
    for name in module.BUNDLE_FILES:
        target = public / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    path = public / "notebooks/05_task_a_research_training.ipynb"
    nb = json.loads(path.read_text())
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "SYNTHETIC OUTPUT"}]
    path.write_text(json.dumps(nb))
    with pytest.raises(ValueError, match="clear_notebook_outputs"):
        module.build_bundle(tmp_path / "blocked.zip", root=public)
