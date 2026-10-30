"""Execute notebook helpers and audit cells only against generated synthetic files.

Public setup and Colab/Drive cells are never run. No private inputs or network
services are needed; no executed outputs are written back into the notebook.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brainagenticthon.preprocessing.oasis2_clinical_policy import (
    ClinicalImage,
    LinkageAttestation,
    audit_clinical_manifest,
)

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_oasis2_clinical_validation.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture
def helpers(notebook):
    namespace = {}
    for cell in notebook["cells"]:
        tags = cell["metadata"].get("tags", [])
        if "offline-definitions" in tags or "configuration" in tags:
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def synthetic_row(**changes):
    defaults = dict(
        subject_id="synthetic-subject-0", mri_visit="synthetic-visit-0",
        group="Nondemented", cdr=0, mmse=None, mmse_field_present=False,
        image_path="synthetic-scan.hdr", source_dataset="OASIS-2",
    )
    return ClinicalImage(**{**defaults, **changes})


def write_analyze(root, name="synthetic-scan", *, shape=(12, 13, 14), constant=False,
                  nonfinite=False, spacing=(1, 1, 1)):
    data = np.zeros(shape, dtype=np.float32) if constant else (
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    )
    if nonfinite:
        data.flat[0] = np.nan
    header = Path(root) / f"{name}.hdr"
    image = nib.AnalyzeImage(data, np.diag([*spacing, 1]))
    nib.save(image, header)
    return header


def fingerprint_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def cohort(subjects=12, visits=2):
    return [synthetic_row(
        subject_id=f"synthetic-subject-{subject}", mri_visit=f"synthetic-visit-{visit}",
        group="Converted" if subject == 0 else "Nondemented",
        cdr=0.5 if subject == 0 and visit else 0,
        image_path=f"synthetic-scan-{subject}-{visit}.hdr",
    ) for subject in range(subjects) for visit in range(visits)]


def test_notebook_json_structure_sources_and_empty_outputs(notebook):
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    ids = []
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], list) and all(isinstance(s, str) for s in cell["source"])
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cell["id"])
        ids.append(cell["id"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            assert cell["metadata"]["tags"]
            ast.parse("".join(cell["source"]))
    assert len(ids) == len(set(ids))
    headings = ["".join(c["source"]).splitlines()[0] for c in notebook["cells"]
                if c["cell_type"] == "markdown"]
    assert len(headings) == 8  # Title plus seven ordered workflow sections.


def test_no_notebook_data_writers_or_automatic_medical_fetch(notebook):
    forbidden = {"save", "to_filename", "save_json", "save_csv", "write_text", "write_bytes",
                 "download", "fetch", "fetch_data", "train", "fit", "to_file_map"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "open"):
                assert len(node.args) < 2 or node.args[1].value in {"r", "rb"}


@pytest.mark.parametrize("relative", [
    "../outside.hdr", "/outside.hdr", ".hidden/scan.hdr", "__MACOSX/scan.hdr",
    "folder/._scan.hdr", "C:\\private\\scan.hdr", "folder\\scan.hdr", "",
])
def test_private_paths_reject_metadata_and_escapes(helpers, tmp_path, relative):
    with pytest.raises(ValueError):
        helpers["private_path"](tmp_path, relative)


def test_private_paths_reject_symlink_escape(helpers, tmp_path):
    root = tmp_path / "synthetic-root"
    root.mkdir()
    (root / "synthetic-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        helpers["private_path"](root, "synthetic-link/outside.hdr")


def test_private_error_wrapper_suppresses_output_warnings_and_paths(helpers, capsys):
    sentinel = "PRIVATE_SENTINEL_SYNTHETIC_ONLY"
    previous = logging.root.manager.disable

    def bad_input():
        print(sentinel)
        logging.error(sentinel)
        warnings.warn(sentinel, stacklevel=1)
        raise ValueError(sentinel)

    with pytest.raises(RuntimeError) as error:
        helpers["private_call"](bad_input)
    assert sentinel not in str(error.value)
    assert sentinel not in capsys.readouterr().out
    assert logging.root.manager.disable == previous


def write_private_manifest(root):
    (root / "raw").mkdir()
    manifest = root / "synthetic-manifest.json"
    manifest.write_text(json.dumps([{
        "patient_id": "synthetic-subject", "visit_id": "synthetic-visit",
        "clinical_group": "Nondemented", "cdr": 0, "relative_path": "synthetic-scan.hdr",
        "dataset": "OASIS-2", "diagnosis_source": "OASIS-2 demographics",
        "task_a_label": None, "task_c_label": None,
        "label_status": "PENDING_CLINICAL_POLICY",
    }]))
    return manifest


def test_exact_array_export_missing_mmse_and_optional_linkage(helpers, tmp_path):
    manifest = write_private_manifest(tmp_path)
    rows, linkage, root, fingerprints = helpers["load_private_inputs"](
        tmp_path, "raw", manifest.name,
    )
    assert root == tmp_path / "raw" and linkage is None
    assert not rows[0].mmse_field_present and rows[0].mmse is None
    assert rows[0].linkage_status == "unknown" and rows[0].split == "unassigned"
    report = audit_clinical_manifest(rows)
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0
    attestation = tmp_path / "synthetic-attestation.json"
    attestation.write_text(json.dumps(asdict(LinkageAttestation("synthetic-reference", 1, 1))))
    rows, linkage, _, fingerprints = helpers["load_private_inputs"](
        tmp_path, "raw", manifest.name, attestation.name,
    )
    report = audit_clinical_manifest(rows, linkage_attestation=linkage)
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 1
    assert report["counts"]["review_supported_annotations"] == {}
    assert helpers["inputs_unchanged"](fingerprints)
    manifest.write_text("[]")
    assert not helpers["inputs_unchanged"](fingerprints)


def test_missing_or_inconsistent_attestation_is_not_invented(helpers, tmp_path):
    manifest = write_private_manifest(tmp_path)
    with pytest.raises(OSError):
        helpers["load_private_inputs"](tmp_path, "raw", manifest.name, "missing.json")
    report = audit_clinical_manifest(
        [synthetic_row()], linkage_attestation=LinkageAttestation("synthetic", 2, 1),
    )
    assert not report["external_linkage_attestation"]["counts_consistent"]
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0


def test_distributions_deduplicate_visits_preserve_converted_scores_and_conflicts(helpers):
    early = synthetic_row(group="Converted", cdr=0)
    late = replace(early, mri_visit="synthetic-visit-1", cdr=0.5,
                   image_path="synthetic-late.hdr")
    rows = [early, replace(early, image_path="synthetic-repeat.hdr"), late]
    distribution = helpers["clinical_distribution"](rows)
    assert distribution["totals"] == {"subjects": 1, "visits": 2, "images": 3}
    assert distribution["group_cdr"] == [
        {"group": "Converted", "cdr": "0.0", "subjects": 1, "visits": 1, "images": 2},
        {"group": "Converted", "cdr": "0.5", "subjects": 1, "visits": 1, "images": 1},
    ]
    conflict = replace(early, cdr=1)
    distribution = helpers["clinical_distribution"]([early, conflict])
    assert distribution["conflicting_visits_excluded_from_table"] == 1
    assert sum(c["visits"] for c in distribution["group_cdr"]) == 0
    assert sum(c["images"] for c in distribution["group_cdr"]) == 2


def test_unexpected_category_values_are_not_printed(helpers):
    row = synthetic_row(group="PRIVATE_SENTINEL_UNEXPECTED", cdr=99)
    result = json.dumps(helpers["clinical_distribution"]([row]))
    assert "PRIVATE_SENTINEL" not in result
    assert "<unknown_group>" in result and "<invalid_cdr>" in result


def test_reference_counts_are_compared_without_filling_data(helpers):
    row = synthetic_row(linkage_status="matched")
    report = audit_clinical_manifest([row])
    comparisons = helpers["compare_observed"](report)["comparisons"]
    assert comparisons["subjects"] == {"observed_reference": 150, "actual": 1, "matches": False}
    assert comparisons["images"]["actual"] == 1
    assert row.declared_task_a_label is None and row.declared_task_c_label is None


def test_analyze_pair_valid_native_geometry_still_requires_review(helpers, tmp_path):
    header = write_analyze(tmp_path, spacing=(1.2, 1.3, 1.4))
    before = fingerprint_tree(tmp_path)
    result = helpers["validate_analyze_pair"](header, tmp_path)
    assert result["computational_checks_passed"]
    assert result["shape"] == [12, 13, 14]
    assert result["array_axis_codes"] == ["L", "A", "S"]
    assert result["header_spacing_units_unverified"] == pytest.approx([1.2, 1.3, 1.4])
    assert result["quality_status"] == "review_required"
    assert "analyze_world_geometry_and_units_unverified" in result["quality_flags"]
    assert "orientation.match" in result["quality_flags"]
    assert result["original_pair_unchanged"] and fingerprint_tree(tmp_path) == before


@pytest.mark.parametrize("changes,flag", [
    ({"nonfinite": True}, "nonfinite_voxels"),
    ({"constant": True}, "constant_volume"),
    ({"shape": (12, 13, 14, 1)}, "image_not_exactly_3d"),
    ({"shape": (5, 13, 14)}, "invalid_spatial_dimensions"),
    ({"spacing": (20, 1, 1)}, "header.voxel_size"),
])
def test_analyze_quality_failures_are_not_silently_fixed(helpers, tmp_path, changes, flag):
    header = write_analyze(tmp_path, **changes)
    before = fingerprint_tree(tmp_path)
    result = helpers["validate_analyze_pair"](header, tmp_path)
    assert not result["computational_checks_passed"]
    assert flag in result["failures"] + result["quality_flags"]
    assert fingerprint_tree(tmp_path) == before


@pytest.mark.parametrize("mode,flag", [
    ("missing", "missing_analyze_companion"),
    ("truncated", "truncated_analyze_payload"),
    ("corrupt", "analyze_read_or_validation_error"),
    ("voxel_limit", "image_exceeds_voxel_limit"),
    ("byte_limit", "pair_exceeds_byte_limit"),
    ("sidecar", "analyze_sidecar_requires_review"),
])
def test_analyze_missing_corrupt_and_resource_limits(helpers, tmp_path, monkeypatch, mode, flag):
    header = write_analyze(tmp_path)
    kwargs = {}
    if mode == "missing":
        header.with_suffix(".img").unlink()
    elif mode == "truncated":
        header.with_suffix(".img").write_bytes(b"truncated")
    elif mode == "corrupt":
        header.write_bytes(b"not an Analyze header")
    elif mode == "sidecar":
        header.with_suffix(".mat").write_bytes(b"synthetic unreviewed sidecar")
    else:
        kwargs = {"max_voxels": 1} if mode == "voxel_limit" else {"max_pair_bytes": 1}

        def no_voxel_read(*args, **kwargs):
            raise AssertionError("Resource bounds must run before materializing voxels")

        monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__array__", no_voxel_read)
    before = fingerprint_tree(tmp_path)
    result = helpers["validate_analyze_pair"](header, tmp_path, **kwargs)
    assert not result["computational_checks_passed"] and flag in result["failures"]
    assert fingerprint_tree(tmp_path) == before


def test_analyze_payload_symlink_escape_is_not_read(helpers, tmp_path):
    root = tmp_path / "synthetic-root"
    root.mkdir()
    header = write_analyze(root)
    header.with_suffix(".img").unlink()
    outside = tmp_path / "synthetic-outside.img"
    outside.write_bytes(b"synthetic only")
    header.with_suffix(".img").symlink_to(outside)
    with pytest.raises(ValueError):
        helpers["validate_analyze_pair"](header, root)


def test_nifti_pair_is_not_silently_treated_as_analyze(helpers, tmp_path):
    path = tmp_path / "synthetic-nifti-pair.hdr"
    data = np.arange(12 * 13 * 14, dtype=np.float32).reshape(12, 13, 14)
    nib.save(nib.Nifti1Pair(data, np.eye(4)), path)
    result = helpers["validate_analyze_pair"](path, tmp_path)
    assert not result["computational_checks_passed"]
    assert "not_analyze_pair" in result["failures"]


def test_sample_deduplicates_companions_excludes_hidden_and_bounds_reads(helpers, tmp_path):
    for i in range(4):
        write_analyze(tmp_path, f"synthetic-scan-{i}")
    rows = [synthetic_row(image_path=f"synthetic-scan-{i}{suffix}",
                          mri_visit=f"synthetic-visit-{i}")
            for i in range(4) for suffix in (".hdr", ".img")]
    rows += [synthetic_row(image_path="._synthetic-scan.hdr"),
             synthetic_row(image_path="__MACOSX/synthetic-scan.hdr"),
             synthetic_row(image_path="/outside.hdr"),
             synthetic_row(image_path="synthetic-scan.nii.gz")]
    before = fingerprint_tree(tmp_path)
    report = helpers["analyze_sample"](rows, tmp_path, sample_size=2, seed=7)
    assert report["eligible_unique_pairs"] == 4 and report["sampled_pairs"] == 2
    assert report["unsampled_pairs"] == 2 and report["sample_request_fulfilled"]
    assert report["exclusions"]["hidden_metadata_image_records"] == 2
    assert report["exclusions"]["unsafe_path_image_records"] == 1
    assert report["exclusions"]["unsupported_extension_image_records"] == 1
    assert not report["cohort_quality_approved"]
    assert report == helpers["analyze_sample"](rows, tmp_path, sample_size=2, seed=7)
    serialized = json.dumps(report, allow_nan=False)
    assert "synthetic-scan" not in serialized and "synthetic-subject" not in serialized
    assert fingerprint_tree(tmp_path) == before


@pytest.mark.parametrize("size", [0, -1, 21, True, 1.5])
def test_sample_size_is_deliberately_small(helpers, tmp_path, size):
    with pytest.raises(ValueError):
        helpers["analyze_sample"]([synthetic_row()], tmp_path, sample_size=size)


def test_ambiguous_pair_is_not_sampled(helpers, tmp_path):
    row = synthetic_row()
    other = replace(row, subject_id="synthetic-other", image_path="synthetic-scan.img")
    result = helpers["analyze_sample"]([row, other], tmp_path)
    assert result["sampled_pairs"] == 0
    assert not result["sample_request_fulfilled"]
    assert result["exclusions"]["pairs_linked_to_multiple_visits"] == 1


def test_split_plan_is_deterministic_subject_safe_label_free_and_in_memory(helpers, tmp_path):
    rows = cohort()
    original = [asdict(r) for r in rows]
    proposal, summary = helpers["propose_splits"](rows, root=tmp_path)
    repeated, repeated_summary = helpers["propose_splits"](list(reversed(rows)), root=tmp_path)
    assert summary == repeated_summary and summary["all_ranked_splits_nonempty"]
    by_subject = {}
    for record in proposal.records:
        by_subject.setdefault(record.subject_id, set()).add(record.split)
        assert record.task_a_label is None and record.task_c_label is None
        assert record.original_diagnosis is None and record.image_dimension == ""
        assert record.dataset_source == "oasis2"
    assert all(len(s) == 1 for s in by_subject.values())
    assert {(r.subject_id, r.split) for r in proposal.records} == {
        (r.subject_id, r.split) for r in repeated.records
    }
    assert sum(c["subjects"] for c in summary["counts"].values()) == 12
    assert sum(c["visits"] for c in summary["counts"].values()) == 24
    assert [asdict(r) for r in rows] == original and not list(tmp_path.iterdir())
    assert "synthetic-subject" not in json.dumps(summary)


def test_fixed_split_and_holdout_propagate_without_mutating_originals(helpers, tmp_path):
    rows = cohort()
    rows[0] = replace(rows[0], split="holdout")
    rows[2] = replace(rows[2], split="val")
    proposal, _ = helpers["propose_splits"](rows, root=tmp_path)
    for subject, expected in [(rows[0].subject_id, "holdout"),
                              (rows[2].subject_id, "validation")]:
        assert {r.split for r in proposal.records if r.subject_id == subject} == {expected}
    assert rows[1].split == "unassigned" and rows[2].split == "val"


@pytest.mark.parametrize("splits", [("train", "test"), ("holdout", "train"),
                                   ("validation", "holdout")])
def test_conflicting_subject_splits_block_planning(helpers, tmp_path, splits):
    row = synthetic_row(split=splits[0])
    other = replace(row, mri_visit="synthetic-late", image_path="synthetic-late.hdr",
                    split=splits[1])
    with pytest.raises(ValueError, match="conflicting_existing_subject_splits"):
        helpers["propose_splits"]([row, other], root=tmp_path)


def test_duplicate_pair_subject_visit_aliases_block_planning(helpers, tmp_path):
    row = synthetic_row()
    for other in (replace(row, subject_id="synthetic-other"),
                  replace(row, mri_visit="synthetic-other-visit", image_path="synthetic-scan.img")):
        with pytest.raises(ValueError, match="image_pair_linked_to_multiple_visits"):
            helpers["propose_splits"]([row, other], root=tmp_path)
    with pytest.raises(ValueError):
        helpers["propose_splits"]([replace(row, image_path="../outside.hdr")], root=tmp_path)


@pytest.mark.parametrize("changes", [{"declared_task_a_label": 0},
                                     {"declared_task_c_label": "ad"}])
def test_existing_labels_are_not_erased_or_reinterpreted(helpers, tmp_path, changes):
    row = synthetic_row(**changes)
    with pytest.raises(ValueError, match="existing_labels_require_separate_review"):
        helpers["propose_splits"]([row], root=tmp_path)
    assert all(getattr(row, field) == value for field, value in changes.items())


def test_small_cohort_empty_split_is_flagged_not_called_ready(helpers, tmp_path):
    proposal, summary = helpers["propose_splits"](cohort(subjects=2), root=tmp_path)
    assert not summary["all_ranked_splits_nonempty"]
    report = helpers["summarize_readiness"](proposal, tmp_path)
    assert not report["ready"] and report["assessment"] == "not_ready"
    assert "image_checks_not_run" in report["blockers"]
    assert "source_not_task_a_approved" in report["counts"]["reasons"]
    assert "synthetic-subject" not in json.dumps(report)


def test_audit_runtime_cells_execute_synthetically_without_private_io_or_writes(
    helpers, notebook, tmp_path, capsys, monkeypatch,
):
    rows = cohort(subjects=8)
    for row in rows:
        write_analyze(tmp_path, Path(row.image_path).stem)
    metadata = tmp_path / "synthetic-metadata.json"
    metadata.write_text(json.dumps([asdict(r) for r in rows]))
    helpers.update(images=rows, linkage=LinkageAttestation("synthetic-reference", len(rows), 16),
                   raw_root=tmp_path, input_fingerprints={
                       metadata: hashlib.sha256(metadata.read_bytes()).hexdigest(),
                   })
    before = fingerprint_tree(tmp_path)
    original = [asdict(r) for r in rows]
    original_nib_load = nib.load

    def confined_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path)
        return original_nib_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", confined_load)
    started = False
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("clinical_report ="):
            started = True
        if started and "colab-runtime" in cell["metadata"].get("tags", []):
            exec(compile(source, str(NOTEBOOK), "exec"), helpers)
    output = capsys.readouterr().out
    assert "synthetic-subject" not in output and "synthetic-scan" not in output
    assert str(tmp_path) not in output and "synthetic-reference" not in output
    assert '"final_task_labels_assigned": false' in output
    assert helpers["image_sample"]["computational_pass_pairs"] == 3
    assert helpers["clinical_report"]["counts"]["review_supported_annotations"] == {}
    assert [asdict(r) for r in rows] == original
    assert fingerprint_tree(tmp_path) == before
