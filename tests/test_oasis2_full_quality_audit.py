"""Entirely synthetic MRI pairs and metadata; never mount Drive or fetch data."""

from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from brainagenticthon.preprocessing import oasis2_full_quality_audit as audit
from brainagenticthon.preprocessing import oasis2_mri_quality as quality


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / "raw"
    metadata = tmp_path / "metadata"
    raw.mkdir()
    metadata.mkdir()
    records = []
    for i in range(3):
        name = f"PRIVATE_SENTINEL_SCAN_{i}.nifti.hdr"
        data = np.arange(12 * 13 * 14, dtype=np.float32).reshape(12, 13, 14, 1)
        nib.save(nib.Nifti1Pair(data, np.eye(4)), raw / name)
        records.append(
            {
                "patient_id": f"PRIVATE_SENTINEL_SUBJECT_{i}",
                "visit_id": "MR1",
                "clinical_group": "Nondemented",
                "cdr": 0,
                "relative_path": name,
                "dataset": "OASIS-2",
                "diagnosis_source": "OASIS-2 demographics",
                "task_a_label": None,
                "task_c_label": None,
                "label_status": "PENDING_CLINICAL_POLICY",
            }
        )
    manifest = metadata / "synthetic-manifest.json"
    manifest.write_text(json.dumps(records))
    output = tmp_path / "private-reports" / "synthetic-run"
    return manifest, raw, output


def source_snapshot(inputs):
    manifest, raw, _ = inputs
    return {p: p.read_bytes() for p in [manifest, *sorted(raw.iterdir())]}


def call(inputs, **kwargs):
    return audit.audit_cohort(*inputs, **kwargs)


def test_dry_run_never_opens_voxels_or_creates_output(inputs, monkeypatch):
    before = source_snapshot(inputs)
    monkeypatch.setattr(nib, "load", lambda *args, **kwargs: pytest.fail("MRI read in dry-run"))
    report = call(inputs, max_pairs=1)
    assert report["mode"] == "dry_run_no_mri_reads_or_writes"
    assert report["inventory"]["planned_unique_pairs"] == 3
    assert report["would_process_this_invocation"] == 1
    assert report["completed_pairs"] == 0 and not report["technical_audit_complete"]
    assert not inputs[2].exists() and source_snapshot(inputs) == before


def test_complete_audit_preserves_sources_and_contains_only_aggregate_outputs(inputs):
    before = source_snapshot(inputs)
    seen = []
    report = call(inputs, dry_run=False, progress=seen.append)
    assert [r["completed_pairs"] for r in seen] == [1, 2, 3]
    assert report["technical_audit_complete"] and report["all_cohort_computational_checks_passed"]
    assert report["results"]["singleton_view_pairs"] == 3
    assert report["results"]["byte_preservation_verified_pairs"] == 3
    assert report["results"]["formats"] == {"Nifti1Pair": 3}
    assert report["results"]["shapes"] == {"[12,13,14,1]": 3}
    assert report["results"]["quality_flags"]["visual_qc_not_performed"] == 3
    assert not report["visual_qc_approved"] and not report["clinical_diagnoses_approved"]
    assert not report["task_a_or_c_release_ready"] and not report["final_task_labels_assigned"]
    assert source_snapshot(inputs) == before
    output = inputs[2]
    assert {p.name for p in output.iterdir()} == {audit.STATE_NAME, audit.REPORT_NAME}
    for path in output.iterdir():
        content = path.read_text()
        assert "PRIVATE_SENTINEL" not in content and str(inputs[1]) not in content
        assert "patient_id" not in content and "relative_path" not in content
    assert "PRIVATE_SENTINEL" not in json.dumps(seen)


def test_batched_resume_rehashes_prefix_without_revalidating_committed_pairs(inputs, monkeypatch):
    checked = []
    original = quality.validate_analyze_pair

    def spy(path, *args, **kwargs):
        checked.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(quality, "validate_analyze_pair", spy)
    first = call(inputs, dry_run=False, max_pairs=1)
    assert first["completed_pairs"] == 1 and not first["technical_audit_complete"]
    second = call(inputs, dry_run=False, max_pairs=1)
    assert second["completed_pairs"] == 2
    assert second["run_started_at_utc"] == first["run_started_at_utc"]
    assert second["last_checkpoint_at_utc"].endswith("+00:00")
    third = call(inputs, dry_run=False)
    assert third["completed_pairs"] == 3
    assert len(checked) == len(set(checked)) == 3
    assert third["results"]["computational_pass_pairs"] == 3
    assert call(inputs, dry_run=False)["completed_pairs"] == 3 and len(checked) == 3
    assert call(inputs)["completed_pairs"] == 3  # Dry-run displays cached counts explicitly.


def test_keyboard_interrupt_resumes_from_last_committed_pair(inputs, monkeypatch):
    checked = []
    original = quality.validate_analyze_pair

    def interrupt_second(path, *args, **kwargs):
        if len(checked) == 1:
            raise KeyboardInterrupt
        checked.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(quality, "validate_analyze_pair", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        call(inputs, dry_run=False)
    state = json.loads((inputs[2] / audit.STATE_NAME).read_text())
    assert state["next_index"] == 1
    assert not (inputs[2] / audit.LOCK_NAME).exists()

    def resume_spy(path, *args, **kwargs):
        assert path not in checked
        return original(path, *args, **kwargs)

    monkeypatch.setattr(quality, "validate_analyze_pair", resume_spy)
    assert call(inputs, dry_run=False)["completed_pairs"] == 3


def test_interrupt_after_checkpoint_commit_regenerates_stale_report_without_duplicate_check(
    inputs,
    monkeypatch,
):
    original_write = audit._write_report
    original_validate = quality.validate_analyze_pair

    def interrupt_report(output, report):
        if report["completed_pairs"] == 1:
            raise KeyboardInterrupt
        original_write(output, report)

    monkeypatch.setattr(audit, "_write_report", interrupt_report)
    with pytest.raises(KeyboardInterrupt):
        call(inputs, dry_run=False)
    assert json.loads((inputs[2] / audit.STATE_NAME).read_text())["next_index"] == 1
    assert json.loads((inputs[2] / audit.REPORT_NAME).read_text())["completed_pairs"] == 0
    checked = []

    def spy(path, *args, **kwargs):
        checked.append(path)
        return original_validate(path, *args, **kwargs)

    monkeypatch.setattr(quality, "validate_analyze_pair", spy)
    monkeypatch.setattr(audit, "_write_report", original_write)
    assert call(inputs, dry_run=False)["completed_pairs"] == 3
    assert len(checked) == 2


def test_interrupted_atomic_write_keeps_old_checkpoint_and_repeats_only_inflight_pair(
    inputs,
    monkeypatch,
):
    original_replace = os.replace

    def fail_next_commit(source, destination):
        if (
            Path(destination).name == audit.STATE_NAME
            and json.loads(Path(source).read_text())["next_index"] == 1
        ):
            raise OSError("PRIVATE_SENTINEL_IO_FAILURE")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_next_commit)
    with pytest.raises(audit.AuditError) as error:
        call(inputs, dry_run=False)
    assert "PRIVATE_SENTINEL" not in str(error.value)
    assert json.loads((inputs[2] / audit.STATE_NAME).read_text())["next_index"] == 0
    monkeypatch.setattr(os, "replace", original_replace)
    assert call(inputs, dry_run=False)["results"]["computational_pass_pairs"] == 3


def test_stale_lock_requires_explicit_recovery_and_normal_active_writer_is_rejected(inputs):
    call(inputs, dry_run=False, max_pairs=1)
    lock = inputs[2] / audit.LOCK_NAME
    lock.write_text("synthetic-stopped-runtime-token")
    with pytest.raises(audit.AuditError, match="writer_active_or_stale_lock"):
        call(inputs, dry_run=False)
    assert lock.exists()
    assert call(inputs, dry_run=False, recover_stale_lock=True)["completed_pairs"] == 3
    with (
        audit._writer_lock(inputs[2], False),
        pytest.raises(audit.AuditError, match="writer_active_or_stale_lock"),
    ):
        call(inputs, dry_run=False)


@pytest.mark.parametrize("change", ["manifest", "code_version", "limits", "source_bytes"])
def test_resume_refuses_changed_inputs_or_semantics_without_overwriting_checkpoint(
    inputs,
    monkeypatch,
    change,
):
    call(inputs, dry_run=False, max_pairs=1)
    checkpoint = (inputs[2] / audit.STATE_NAME).read_bytes()
    kwargs = {}
    if change == "manifest":
        inputs[0].write_text(inputs[0].read_text() + "\n")
    elif change == "code_version":
        monkeypatch.setattr(audit, "AUDIT_VERSION", "synthetic-new-version")
    elif change == "limits":
        kwargs["max_voxels"] = 100
    else:
        payload = sorted(inputs[1].glob("*.img"))[0]
        info = payload.stat()
        with payload.open("r+b") as stream:
            stream.write(b"\x01")
        os.utime(payload, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(audit.AuditError):
        call(inputs, dry_run=False, **kwargs)
    assert (inputs[2] / audit.STATE_NAME).read_bytes() == checkpoint


def test_source_change_after_validator_does_not_commit_a_false_pass(inputs, monkeypatch):
    original = quality.validate_analyze_pair

    def concurrent_writer(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        with path.with_suffix(".img").open("r+b") as stream:
            stream.write(b"\x01")
        return result

    monkeypatch.setattr(quality, "validate_analyze_pair", concurrent_writer)
    with pytest.raises(audit.AuditError, match="source_changed_during_pair_check"):
        call(inputs, dry_run=False)
    assert json.loads((inputs[2] / audit.STATE_NAME).read_text())["next_index"] == 0


@pytest.mark.parametrize(
    "malformation", ["missing", "truncated", "header", "constant", "nan", "nonsingleton", "spacing"]
)
def test_malformed_mri_is_counted_as_failed_not_clinically_approved(inputs, malformation):
    header = sorted(inputs[1].glob("*.hdr"))[0]
    if malformation == "missing":
        header.with_suffix(".img").unlink()
    elif malformation == "truncated":
        header.with_suffix(".img").write_bytes(b"synthetic invalid payload")
    elif malformation == "header":
        header.write_bytes(b"synthetic corrupted header")
    else:
        shape = (12, 13, 14, 2 if malformation == "nonsingleton" else 1)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        if malformation == "constant":
            data[:] = 1
        if malformation == "nan":
            data.flat[0] = np.nan
        affine = np.diag([20 if malformation == "spacing" else 1, 1, 1, 1])
        nib.save(nib.Nifti1Pair(data, affine), header)
    before = source_snapshot(inputs)
    report = call(inputs, dry_run=False)
    assert report["completed_pairs"] == 3 and report["technical_audit_complete"]
    assert report["results"]["computational_fail_pairs"] == 1
    assert not report["all_cohort_computational_checks_passed"]
    assert not report["visual_qc_approved"] and source_snapshot(inputs) == before
    assert call(inputs, dry_run=False)["results"] == report["results"]


def test_pair_deduplication_unsafe_paths_and_visit_conflicts_are_visible(inputs):
    manifest, _, _ = inputs
    rows = json.loads(manifest.read_text())
    rows.append({**rows[0], "relative_path": rows[0]["relative_path"].replace(".hdr", ".img")})
    rows += [
        {**rows[0], "relative_path": "._synthetic-metadata.hdr"},
        {**rows[0], "relative_path": "../PRIVATE_SENTINEL_OUTSIDE.hdr"},
        {**rows[0], "relative_path": "unsupported.nii"},
        {**rows[1], "patient_id": "PRIVATE_SENTINEL_OTHER_SUBJECT"},
    ]
    manifest.write_text(json.dumps(rows))
    report = call(inputs, dry_run=False)
    assert report["inventory"]["planned_unique_pairs"] == 2
    assert report["inventory"]["duplicate_pair_or_companion_records"] == 1
    assert report["inventory"]["exclusions"] == {
        "hidden_metadata_records": 1,
        "unsafe_path_records": 1,
        "unsupported_extension_records": 1,
        "pairs_linked_to_multiple_visits": 1,
    }
    assert not report["technical_audit_complete"]
    assert "PRIVATE_SENTINEL" not in json.dumps(report)


@pytest.mark.parametrize(
    "kind",
    ["raw_child", "manifest_parent", "raw_ancestor", "symlink", "unrelated_file", "hardlink"],
)
def test_output_safety_never_overwrites_inputs_or_unrelated_files(inputs, tmp_path, kind):
    manifest, raw, output = inputs
    before = source_snapshot(inputs)
    if kind == "raw_child":
        output = raw / "reports"
    elif kind == "manifest_parent":
        output = manifest.parent
    elif kind == "raw_ancestor":
        output = tmp_path
    elif kind == "symlink":
        output.parent.mkdir()
        output.symlink_to(raw, target_is_directory=True)
    else:
        output.mkdir(parents=True)
        if kind == "unrelated_file":
            (output / "unrelated.txt").write_text("synthetic unrelated data")
        else:
            os.link(manifest, output / audit.STATE_NAME)
    with pytest.raises(audit.AuditError):
        audit.audit_cohort(manifest, raw, output, dry_run=False)
    assert source_snapshot(inputs) == before


def test_corrupt_checkpoint_refuses_reset_and_preserves_previous_report(inputs):
    call(inputs, dry_run=False, max_pairs=1)
    checkpoint = inputs[2] / audit.STATE_NAME
    checkpoint.write_text('{"synthetic_corrupt_state":true}')
    before = {p.name: p.read_bytes() for p in inputs[2].iterdir()}
    with pytest.raises(audit.AuditError):
        call(inputs, dry_run=False)
    assert {p.name: p.read_bytes() for p in inputs[2].iterdir()} == before


def test_raw_access_uses_only_read_modes_and_one_pair_validation_at_a_time(inputs, monkeypatch):
    raw = inputs[1]
    original_builtin, original_io = builtins.open, io.open
    active = []
    original_validate = quality.validate_analyze_pair

    def readonly(opener):
        def guarded(path, mode="r", *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(raw):
                assert not any(flag in mode for flag in "wax+")
            return opener(path, mode, *args, **kwargs)

        return guarded

    def one_at_a_time(path, *args, **kwargs):
        assert not active
        active.append(path)
        try:
            return original_validate(path, *args, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(builtins, "open", readonly(original_builtin))
    monkeypatch.setattr(io, "open", readonly(original_io))
    monkeypatch.setattr(quality, "validate_analyze_pair", one_at_a_time)
    assert call(inputs, dry_run=False)["completed_pairs"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pairs": 0},
        {"max_pairs": True},
        {"max_voxels": 32_000_001},
        {"max_pair_bytes": 256 * 1024 * 1024 + 1},
    ],
)
def test_configuration_cannot_exceed_reviewed_resource_bounds(inputs, kwargs):
    with pytest.raises(audit.AuditError):
        call(inputs, dry_run=False, **kwargs)
    assert not inputs[2].exists()


def test_cli_safe_failure_and_partial_and_success_statuses(inputs, capsys):
    manifest, raw, output = inputs
    args = ["--manifest", str(manifest), "--image-root", str(raw), "--output-dir", str(output)]
    assert audit.main(args) == 0 and not output.exists()
    assert audit.main([*args, "--run", "--max-pairs", "1"]) == 1
    assert audit.main([*args, "--run"]) == 0
    assert audit.main([*args, "--max-pairs", "0"]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
    assert str(raw) not in captured.out + captured.err


def test_sample_notebook_imports_same_validator_instead_of_copying_logic():
    path = Path(__file__).parents[1] / "notebooks/02_oasis2_clinical_validation.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if set(cell["metadata"].get("tags", [])) & {"configuration", "offline-definitions"}:
            exec("".join(cell["source"]), namespace)
    assert namespace["validate_analyze_pair"] is quality.validate_analyze_pair


@pytest.mark.parametrize(
    "kwargs", [{"dry_run": "False"}, {"recover_stale_lock": "False"}, {"progress": "not callable"}]
)
def test_mode_and_lock_controls_are_strict_before_any_writes(inputs, kwargs):
    with pytest.raises(audit.AuditError):
        call(inputs, **kwargs)
    assert not inputs[2].exists()


def test_parser_never_echoes_sensitive_invalid_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        audit.main(["--max-pairs", "PRIVATE_SENTINEL_NOT_AN_INTEGER"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err


def test_source_identity_is_not_aliased_to_another_oasis_version(inputs):
    rows = json.loads(inputs[0].read_text())
    rows[0]["dataset"] = "OASIS-3"
    inputs[0].write_text(json.dumps(rows))
    with pytest.raises(audit.AuditError):
        call(inputs, dry_run=False)
    assert not inputs[2].exists()


def test_unowned_temporary_files_are_never_removed(inputs):
    inputs[2].mkdir(parents=True)
    unrelated = inputs[2] / ".audit-tmp-unrelated.json"
    unrelated.write_text('{"synthetic_unrelated_data":true}')
    with pytest.raises(audit.AuditError, match="unowned_output_without_checkpoint"):
        call(inputs, dry_run=False)
    assert unrelated.read_text() == '{"synthetic_unrelated_data":true}'
