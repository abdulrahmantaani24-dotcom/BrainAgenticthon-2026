"""Generated MRI/DICOM/documents only; no private user data or clinical claims."""

import copy
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from brainagenticthon.execution.annotation_review import inspect_pair
from brainagenticthon.execution.contracts import classification_result, segmentation_result
from brainagenticthon.execution.disease_sources import (
    REVIEW_FLAGS,
    SOURCE_NAMES,
    dicom_header,
    inspect_sources,
    manifest_diagnostics,
    reviewed_evidence,
)
from brainagenticthon.execution.handoff import research_envelope
from brainagenticthon.execution.source_workflow import run
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file
from brainagenticthon.segmentation.lesion_metrics import lesion_metrics

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def image(path, *, mask=False, affine=None):
    values = np.random.default_rng(42).normal(size=(16, 16, 16)).astype(np.float32)
    if mask:
        values[:] = 0
        values[3:7, 4:8, 5:9] = 1
    volume = nib.Nifti1Image(values, np.eye(4) if affine is None else affine)
    volume.header.set_xyzt_units("mm")
    nib.save(volume, path)
    return {"path": path.name, "sha256": digest_file(path)}


@pytest.fixture
def four_sources(tmp_path):
    parent = tmp_path / "DiseaseAcquisition"
    parent.mkdir()
    for name in SOURCE_NAMES:
        root = parent / name
        root.mkdir()
        image(root / "SYNTHETIC-PATIENT-DO-NOT-PUBLISH.nii.gz")
        (root / "source_and_license.json").write_text(
            json.dumps(
                {
                    "license": "SYNTHETIC-PENDING-NOT-APPROVED",
                    "diagnosis": "UNVERIFIED-EXAMPLE",
                    "records": [{"patient_id": "synthetic", "split": "train"}],
                }
            )
        )
        (root / "archive.zip").write_bytes(b"not-extracted")
        (root / "preview.png").write_bytes(b"not-interpreted")
        (root / "other.SYNTHETIC-PATIENT-SECRET").write_text("not-interpreted")
    cfg = json.loads((ROOT / "configs/disease_acquisition.example.json").read_text())
    cfg.update(dataset_parent=str(parent), private_project_root=str(tmp_path / "private"))
    return cfg, parent


def test_all_four_roots_read_only_sanitized_and_never_approved(four_sources, tmp_path):
    cfg, parent = four_sources
    before = {str(p): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    summary = run(cfg, tmp_path / "run")
    assert len(summary["sources"]) == 4
    assert {r["source"] for r in summary["sources"]} == set(SOURCE_NAMES)
    assert all(r["inventory_status"] == "INVENTORIED" for r in summary["sources"])
    assert all(
        r["eligibility"]["task_a"]["training_status"] == "BLOCKED"
        and r["eligibility"]["task_b"]["training_status"] == "BLOCKED"
        for r in summary["sources"]
    )
    text = json.dumps(summary)
    assert str(parent) not in text and "SYNTHETIC-PATIENT" not in text
    assert "UNVERIFIED-EXAMPLE" not in text
    private = json.loads((tmp_path / "run/inventory.private.json").read_text())
    assert "SYNTHETIC-PATIENT" in json.dumps(private)
    assert private["BraTS_directory_search"] == "not_requested_or_performed"
    assert before == {str(p): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run(cfg, tmp_path / "run")


def test_missing_roots_do_not_invent_supplied_counts(four_sources, tmp_path):
    cfg, _ = four_sources
    cfg["dataset_parent"] = str(tmp_path / "not-mounted")
    summary = inspect_sources(cfg, tmp_path / "missing")
    assert all(not row["root_observed"] and row["coverage"] is None for row in summary["sources"])
    assert not summary["training_started"]


def test_inspection_authorization_and_limits_are_explicit(four_sources, tmp_path):
    cfg, _ = four_sources
    cfg["sources"][0]["inspection_authorized"] = False
    cfg["limits"]["max_files"] = 1
    result = inspect_sources(cfg, tmp_path / "bounded")
    assert result["sources"][0]["inventory_status"] == "BLOCKED"
    assert all(row["inventory_status"] == "BLOCKED" for row in result["sources"])
    cfg["sources"][0]["directory"] = "invented_brats"
    with pytest.raises(ValueError, match="four_named"):
        inspect_sources(cfg, tmp_path / "bad")


def test_no_output_in_dataset_or_symlink_scan(four_sources, tmp_path):
    cfg, parent = four_sources
    with pytest.raises(ValueError, match="disjoint"):
        inspect_sources(cfg, parent / "outputs")
    root = parent / SOURCE_NAMES[0]
    (root / "link.nii").symlink_to(root / "SYNTHETIC-PATIENT-DO-NOT-PUBLISH.nii.gz")
    result = inspect_sources(cfg, tmp_path / "link-test")
    assert result["sources"][0]["inventory_status"] == "BLOCKED"


def test_explicit_manifest_identity_split_and_confounding_diagnostics():
    rows = [
        {
            "patient_id": "p",
            "case_id": "c",
            "split": "train",
            "sha256": "a",
            "dataset_source": "one",
            "task_a_label": 0,
        },
        {
            "patient_id": "p",
            "case_id": "d",
            "split": "test",
            "sha256": "a",
            "dataset_source": "two",
            "task_a_label": 1,
        },
        {"patient_id": "q", "case_id": "c", "split": "validation"},
    ]
    report = manifest_diagnostics(rows)
    assert "documented_patient_split_leakage" in report["diagnostic_blockers"]
    assert "documented_case_identity_conflict" in report["diagnostic_blockers"]
    assert "documented_source_class_confounding_requires_review" in report["diagnostic_blockers"]
    assert not report["clinical_labels_verified"] and not report["aliases_reconciled"]


def test_dicom_header_only_no_pixel_decoding(tmp_path, monkeypatch):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    data = FileDataset(str(tmp_path / "generated.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    data.PatientID = "SYNTHETIC-PRIVATE-ID"
    data.StudyInstanceUID = generate_uid()
    data.SeriesInstanceUID = generate_uid()
    data.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    data.Modality = "MR"
    data.SeriesDescription = "documented T1 candidate only"
    data.Rows, data.Columns = 2, 2
    data.PixelData = b"\0" * 8
    path = tmp_path / "generated.dcm"
    data.save_as(path, enforce_file_format=True)
    original = pydicom.dcmread

    def reader(*args, **kwargs):
        assert kwargs["stop_before_pixels"] is True
        assert "PixelData" not in kwargs["specific_tags"]
        result = original(*args, **kwargs)
        assert "PixelData" not in result
        return result

    monkeypatch.setattr(pydicom, "dcmread", reader)
    report = dicom_header(path)
    assert not report["pixel_decoded"] and not report["sequence_identity_verified"]
    assert "SYNTHETIC-PRIVATE-ID" not in json.dumps(report)


def reviewed_pair(root):
    return {
        "patient_id": "synthetic-p",
        "case_id": "synthetic-c",
        "split": "train",
        "review_status": "approved",
        "modality_reference": "SYNTHETIC-ONLY",
        "mask_semantics_reference": "SYNTHETIC-ONLY",
        "identity_reference": "SYNTHETIC-ONLY",
        "conversion_required": False,
        "background_label": 0,
        "label_definitions": {"0": "background", "1": "synthetic-specific-lesion"},
        "images": {"documented_sequence": image(root / "image.nii.gz")},
        "mask": image(root / "mask.nii.gz", mask=True),
    }


def test_disease_annotation_adapter_does_not_enable_training(tmp_path):
    row = reviewed_pair(tmp_path)
    report = inspect_pair(tmp_path, row)
    assert report["observed_mask_labels"] == [0.0, 1.0]
    assert report["status"] == "TECHNICAL_PAIR_CHECK_PASSED"
    assert not report["training_eligible"] and not report["empty_mask_implies_normal"]
    broken = copy.deepcopy(row)
    broken["label_definitions"] = {"0": "background", "2": "other disease label"}
    with pytest.raises(ValueError, match="mask_values"):
        inspect_pair(tmp_path, broken)
    row["mask"] = image(
        tmp_path / "misaligned.nii.gz", mask=True, affine=np.diag([2.0, 2.0, 2.0, 1.0])
    )
    with pytest.raises(ValueError, match="geometry_mismatch"):
        inspect_pair(tmp_path, row)


def test_review_attestations_never_override_current_registry(four_sources, tmp_path):
    cfg, parent = four_sources
    root = parent / SOURCE_NAMES[0]
    review = {
        "schema": "disease-source-review-v1",
        "source_id": root.name,
        "review_status": "approved",
        "evidence": {},
        "reviewed_pairs": [reviewed_pair(root)],
    }
    for flag in REVIEW_FLAGS:
        review[flag] = True
        review["evidence"][flag] = {"reference": "SYNTHETIC-REVIEW-ONLY"}
    (root / "task_ab_review.json").write_text(json.dumps(review))
    report = reviewed_evidence(root, "task_ab_review.json")
    assert all(report["flags"].values())
    assert report["reviewed_pair_checks"][0]["status"] == "TECHNICAL_PAIR_CHECK_PASSED"
    result = inspect_sources(cfg, tmp_path / "reviewed")
    assert result["sources"][0]["eligibility"]["task_a"]["training_status"] == "BLOCKED"
    assert result["sources"][0]["eligibility"]["task_b"]["training_status"] == "BLOCKED"


def test_lesion_metrics_exact_boundaries_volume_empty_and_units():
    reference = np.zeros((8, 8, 8), dtype=np.uint8)
    prediction = reference.copy()
    reference[2, 2, 2], prediction[3, 2, 2] = 1, 1
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    report = lesion_metrics(
        prediction, reference, affine=affine, units="mm", semantics_reference="SYNTHETIC-ONLY"
    )
    assert report["dice"] == report["lesion_sensitivity"] == 0
    assert report["hausdorff95_mm"] == pytest.approx(2.0)
    assert report["average_symmetric_surface_distance_mm"] == pytest.approx(2.0)
    assert report["absolute_volume_error_ml"] == 0
    prediction[2, 2, 2] = 1
    report = lesion_metrics(
        prediction, reference, affine=affine, units="mm", semantics_reference="SYNTHETIC-ONLY"
    )
    assert report["lesion_sensitivity"] == 1
    assert report["absolute_volume_error_ml"] == pytest.approx(0.024)
    unsupported = lesion_metrics(prediction, reference, semantics_reference="SYNTHETIC-ONLY")
    assert unsupported["hausdorff95_mm"] is None
    empty = lesion_metrics(
        reference * 0,
        reference * 0,
        affine=affine,
        units="mm",
        semantics_reference="SYNTHETIC-ONLY",
    )
    assert empty["dice"] is None and empty["hausdorff95_mm"] is None
    with pytest.raises(ValueError):
        lesion_metrics(prediction, reference)


def test_backend_agent_handoff_preserves_unsupported_fields():
    a = research_envelope(classification_result(0.8), task="a", model_sha256="a" * 64)
    assert not a["TaskC_invoked"] and not a["payload"]["task_a_output_compatible"]
    b = segmentation_result(torch.zeros(8, 8, 8, dtype=torch.long), np.eye(4))
    assert not research_envelope(b, task="b", model_sha256="b" * 64)["clinical_approval"]
    a["payload"]["task_a_output_compatible"] = True
    with pytest.raises(ValueError):
        research_envelope(a["payload"], task="a", model_sha256="a" * 64)


def test_missing_task_configuration_blocks_every_real_action(four_sources, tmp_path):
    cfg, _ = four_sources
    for action in ("validate", "freeze", "smoke", "train", "resume", "evaluate"):
        report = run(cfg, tmp_path / action, action=action, task="a")
        assert report["status"] == "BLOCKED" and not report["training_started"]
    assert not list(tmp_path.rglob("*.pt"))


def test_primary_notebook_runs_sanitized_inventory_from_bundle_after_restart(
    four_sources, tmp_path
):
    import subprocess
    import sys
    import zipfile

    from tests.test_task_ab_colab_setup import cells, script

    cfg, _ = four_sources
    builder = script("build_task_a_bundle")
    archive = tmp_path / "code.zip"
    builder.build_bundle(archive)
    snapshot = tmp_path / "reviewed-code"
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) <= 64
        bundle.extractall(snapshot)  # Generated, allowlisted fixture only.
    content = tmp_path / "content"
    content.mkdir()
    (content / "brainagenticthon-code-state.json").write_text(
        json.dumps(
            {
                "code_root": str(snapshot),
                "manifest_sha256": hashlib.sha256(
                    (snapshot / "code_manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    path = tmp_path / "private-workflow.json"
    path.write_text(json.dumps(cfg))
    codes = cells("09_four_source_private_launcher.ipynb")
    codes = {key: value.replace("/content", str(content)) for key, value in codes.items()}
    payload = tmp_path / "cells.json"
    payload.write_text(json.dumps(codes))
    source = """
import json, sys
from pathlib import Path
codes = json.loads(Path(sys.argv[1]).read_text())
ns = {}
exec(codes['import-code'], ns)
ns.update(CONFIG_PATH=Path(sys.argv[2]), DRIVE_BASE=Path(sys.argv[3]))
exec(codes['settings'], ns)
exec(codes['execute'], ns)
assert ns['ACTION'] == 'inventory'
assert len(ns['SUMMARY']['sources']) == 4
assert not ns['SUMMARY']['training_started']
assert all(row['eligibility']['task_b']['training_status'] == 'BLOCKED'
           for row in ns['SUMMARY']['sources'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", source, str(payload), str(path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "SYNTHETIC-PATIENT" not in result.stdout


def test_exact_boundary_limit_returns_null_not_fabricated_distance():
    p = np.zeros((8, 8, 8), dtype=np.uint8)
    p[2:5, 2:5, 2:5] = 1
    result = lesion_metrics(
        p, p, semantics_reference="SYNTHETIC", units="mm", affine=np.eye(4), max_surface_points=1
    )
    assert result["dice"] == 1 and result["hausdorff95_mm"] is None
    assert "resource_limit" in result["boundary_metrics_reason"]
