"""Synthetic metadata/volumes only; no clinical data, downloads or model calls."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import nibabel as nib
import numpy as np
import pytest

from brainagenticthon.preprocessing.dataset_registry import APPROVED_SOURCES
from brainagenticthon.preprocessing.manifest_schema import ManifestCollection, ManifestRecord
from brainagenticthon.preprocessing.task_a_readiness import (
    RecordReview,
    audit_readiness,
    load_reviews,
    main,
    merge_manifests,
    record_key,
    source_inventory,
)


@pytest.fixture
def cohort(tmp_path):
    """Invented test cases, not claims of real healthy BraTS subjects/access."""
    records, reviews = [], {}
    for source in ("brats2024_gli", "brats2024_men"):
        for split in ("train", "validation", "test"):
            for label in (0, 1):
                index = len(records)
                case = f"synthetic-{index}"
                directory = tmp_path / source
                directory.mkdir(exist_ok=True)
                path = directory / f"{case}.nii.gz"
                data = np.random.default_rng(index).random((12, 12, 12)).astype(np.float32)
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                record = ManifestRecord(
                    subject_id=case, case_id=case, dataset_source=source, cohort="synthetic",
                    session_id="visit-1", modality="t1", image_path=str(path),
                    image_dimension="3D", original_diagnosis="Normal" if label == 0 else "glioma",
                    task_a_label=label, task_c_label=None, segmentation_path=None,
                    label_provenance="clinical_record",
                    license_status=APPROVED_SOURCES[source].license_type.value,
                    site="synthetic-site", scanner="synthetic-scanner", split=split,
                    quality_flag="ok",
                )
                records.append(record)
                reviews[record_key(record)] = RecordReview(
                    patient_id=case, evidence_reference="synthetic-test-attestation",
                    access_authorized=True, patient_identity_verified=True, label_verified=True,
                    healthy_control_verified=label == 0,
                    native_t1_verified=True, whole_brain_verified=True,
                )
    return ManifestCollection(records), reviews, tmp_path


def run(cohort, **kwargs):
    collection, reviews, root = cohort
    return audit_readiness(collection, reviews, image_root=root, **kwargs)


def change_record(cohort, index=0, **changes):
    collection, reviews, root = cohort
    records = collection.records
    old = records[index]
    records[index] = replace(old, **changes)
    reviews = dict(reviews)
    review = reviews.pop(record_key(old))
    reviews[record_key(records[index])] = review
    return ManifestCollection(records), reviews, root


def test_valid_synthetic_subset_and_no_writes(cohort):
    collection, reviews, root = cohort
    before = collection.to_json()
    file_bytes = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = run(cohort)
    assert report["ready"] and report["assessment"] == "ready"
    assert report["counts"]["normal_patients"] == 6
    assert report["counts"]["abnormal_patients"] == 6
    assert report["image_checks_completed"] == 12
    assert not any(report["perfect_confounding"].values())
    assert collection.to_json() == before
    assert file_bytes == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert all(r.segmentation_path is None for r in collection.records)
    assert all(review.evidence_reference == "synthetic-test-attestation"
               for review in reviews.values())
    json.dumps(report, allow_nan=False)


def test_virtual_merge_preserves_raw_locations_and_input(cohort):
    collection, reviews, root = cohort
    a, b = ManifestCollection(collection.records[:6]), ManifestCollection(collection.records[6:])
    merged = merge_manifests(a, b)
    assert merged.to_dict_list() == collection.to_dict_list()
    assert len(a) == len(b) == 6
    assert audit_readiness(merged, reviews, image_root=root)["ready"]
    assert {r.dataset_source for r in merged.records} == {"brats2024_gli", "brats2024_men"}


@pytest.mark.parametrize("label,blocker", [
    (1, "missing_verified_healthy_controls"), (0, "missing_verified_abnormal_cases"),
])
def test_missing_class(cohort, label, blocker):
    collection, reviews, root = cohort
    subset = ManifestCollection([r for r in collection.records if r.task_a_label == label])
    report = audit_readiness(subset, reviews, image_root=root)
    assert not report["ready"] and blocker in report["blockers"]


@pytest.mark.parametrize("diagnosis", [None, "", "  ", "unknown", "missing", "N/A"])
def test_missing_diagnosis_never_normal(cohort, diagnosis):
    report = run(change_record(cohort, original_diagnosis=diagnosis))
    row = report["records"][0]
    assert not row["eligible"]
    assert "missing_diagnosis" in row["reasons"]
    assert "missing_label_never_assume_normal" in row["warnings"]


@pytest.mark.parametrize("label", [None, True, False, 2, "0"])
def test_no_label_inference_from_no_segmentation(cohort, label):
    report = run(change_record(cohort, task_a_label=label, segmentation_path=None))
    assert "missing_or_invalid_task_a_label" in report["records"][0]["reasons"]


@pytest.mark.parametrize("provenance", ["missing", "inferred", "invented"])
def test_verified_provenance_required(cohort, provenance):
    row = run(change_record(cohort, label_provenance=provenance))["records"][0]
    assert "unverified_label_provenance" in row["reasons"]


@pytest.mark.parametrize("field", [
    "access_authorized", "patient_identity_verified", "label_verified",
    "healthy_control_verified", "native_t1_verified", "whole_brain_verified",
])
def test_review_gates(cohort, field):
    collection, reviews, _ = cohort
    key = record_key(collection.records[0])
    reviews[key] = replace(reviews[key], **{field: False})
    row = run(cohort)["records"][0]
    assert not row["eligible"] and row["reasons"]


def test_cognitive_cn_alone_not_mri_normality(cohort):
    updated = change_record(cohort, original_diagnosis="CN", task_c_label="cn")
    collection, reviews, _ = updated
    key = record_key(collection.records[0])
    reviews[key] = replace(reviews[key], healthy_control_verified=False)
    assert "healthy_control_not_verified" in run(updated)["records"][0]["reasons"]


@pytest.mark.parametrize("source", ["oasis3", "adni", "figshare_2d", "local", "unknown"])
def test_no_registry_or_schema_approval_override(cohort, source):
    row = run(change_record(cohort, dataset_source=source))["records"]
    row = next(r for r in row if r["source"] == source)
    assert not row["eligible"]
    if source == "local":
        assert APPROVED_SOURCES[source].task_a_eligible
        assert any("not eligible for Task A" in reason for reason in row["reasons"])
    else:
        assert "source_not_task_a_approved" in row["reasons"]


@pytest.mark.parametrize("modality", ["t1ce", "t1c", "flair", "t2", "seg", "image_2d", ""])
def test_invalid_modalities(cohort, modality):
    row = run(change_record(cohort, modality=modality))["records"][0]
    assert not row["eligible"] and "not_native_t1_modality" in row["reasons"]


@pytest.mark.parametrize("modality", ["t1", "T1w", "t1n"])
def test_reviewed_native_t1_aliases(cohort, modality):
    assert run(change_record(cohort, modality=modality))["ready"]


def test_bad_dimensionality_and_quality_metadata(cohort):
    row = run(change_record(cohort, image_dimension="2D", quality_flag="motion_artifact"))
    assert not row["records"][0]["eligible"]
    assert "declared_image_not_3d" in row["records"][0]["reasons"]
    assert "quality_not_reviewed_ok" in row["records"][0]["reasons"]


@pytest.mark.parametrize("image_kind", ["4d", "2d", "zero", "nan", "inf", "corrupt", "missing"])
def test_actual_image_quality(cohort, image_kind):
    collection, _, _ = cohort
    path = collection.records[0].image_path
    if image_kind == "corrupt":
        from pathlib import Path
        Path(path).write_bytes(b"synthetic invalid image")
    elif image_kind == "missing":
        from pathlib import Path
        Path(path).unlink()
    else:
        shape = {"4d": (12, 12, 12, 2), "2d": (12, 12)}.get(image_kind, (12, 12, 12))
        data = np.zeros(shape, dtype=np.float32)
        if image_kind in {"nan", "inf"}:
            data.flat[0] = np.nan if image_kind == "nan" else np.inf
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    row = run(cohort)["records"][0]
    assert not row["eligible"] and row["reasons"]


def test_raw_lps_image_is_not_reoriented(cohort):
    collection, _, _ = cohort
    path = collection.records[0].image_path
    image = nib.load(path)
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), np.diag([-1., -1., 1., 1.])), path)
    report = run(cohort)
    assert report["ready"]
    assert report["records"][0]["image"]["orientation"] == "LPS"
    assert "image:orientation.match" in report["records"][0]["warnings"]


def test_duplicate_bytes_different_paths(cohort):
    from pathlib import Path
    collection, _, _ = cohort
    Path(collection.records[1].image_path).write_bytes(Path(collection.records[0].image_path).read_bytes())
    report = run(cohort)
    assert not report["ready"] and "duplicate_image_bytes" in report["blockers"]


def test_duplicate_path_and_row(cohort):
    collection, reviews, root = cohort
    merged = merge_manifests(collection, ManifestCollection([collection.records[0]]))
    report = audit_readiness(merged, reviews, image_root=root)
    assert "duplicate_image_path" in report["blockers"]
    assert not report["ready"]


def test_patient_leakage_across_sources_and_holdout(cohort):
    collection, reviews, _ = cohort
    first, other = collection.records[0], collection.records[8]
    reviews[record_key(other)] = replace(reviews[record_key(other)], patient_id=first.subject_id)
    updated = change_record(cohort, index=8, split="holdout")
    report = run(updated)
    assert "global_patient_split_leakage" in report["blockers"]
    assert first.subject_id in report["patient_split_leakage"]


def test_existing_same_source_patient_leakage(cohort):
    report = run(change_record(cohort, index=2, subject_id="synthetic-0"))
    assert "existing_split_or_manifest_violations" in report["blockers"]
    assert "inconsistent_patient_identity" in report["blockers"]


def test_repeat_sessions_same_patient_count_once(cohort):
    collection, reviews, _ = cohort
    first, other = collection.records[0], collection.records[2]
    reviews[record_key(other)] = replace(reviews[record_key(other)], patient_id=first.subject_id)
    updated = change_record(
        cohort, index=2, subject_id=first.subject_id, session_id="visit-2", split="train",
    )
    report = run(updated)
    assert "global_patient_split_leakage" not in report["blockers"]
    assert "inconsistent_patient_identity" not in report["blockers"]
    assert report["counts"]["eligible_patients"] == 11
    # The other source still supplies a validation control; repeats are not duplicates.
    assert report["ready"]


def test_conflicting_patient_labels_flagged(cohort):
    collection, reviews, _ = cohort
    key = record_key(collection.records[1])
    reviews[key] = replace(reviews[key], patient_id="synthetic-0")
    assert "conflicting_patient_labels_require_review" in run(cohort)["blockers"]


def test_source_class_confounding_with_both_classes_in_all_splits(cohort):
    collection, reviews, root = cohort
    pure = ManifestCollection([
        r for r in collection.records
        if (r.dataset_source == "brats2024_gli") == (r.task_a_label == 0)
    ])
    report = audit_readiness(pure, reviews, image_root=root)
    assert report["counts"]["normal_patients"] == report["counts"]["abnormal_patients"] == 3
    assert "source_perfectly_predicts_class" in report["blockers"]
    assert report["perfect_confounding"]["eligible_source_class"]
    assert len(report["patient_class_crosstabs"]["eligible_source_class"]) == 2


def test_site_class_confounding_even_with_mixed_sources(cohort):
    collection, reviews, root = cohort
    pure = ManifestCollection([replace(r, site=f"site-class-{r.task_a_label}")
                               for r in collection.records])
    report = audit_readiness(pure, reviews, image_root=root)
    assert "source_perfectly_predicts_class" not in report["blockers"]
    assert "source_site_perfectly_predicts_class" in report["blockers"]


def test_missing_site_scanner_and_required_split(cohort):
    report = run(change_record(cohort, site="", scanner="", split="unassigned"))
    assert "split_not_assigned_for_evaluation" in report["records"][0]["reasons"]
    report = run(change_record(cohort, site="", scanner=""))
    assert "site_or_scanner_metadata_missing" in report["blockers"]


def test_metadata_only_does_not_read_images_or_claim_ready(cohort, monkeypatch):
    import brainagenticthon.preprocessing.task_a_readiness as module
    monkeypatch.setattr(module, "_image_checks", lambda *args: pytest.fail("No image reads"))
    report = run(cohort, check_images=False)
    assert not report["ready"] and "image_checks_not_run" in report["blockers"]
    assert report["eligibility_scope"] == "metadata_candidates_only"


def test_no_reviews_no_access_and_no_labels_assigned(cohort):
    collection, _, root = cohort
    before = collection.to_json()
    report = audit_readiness(collection, {}, image_root=root)
    assert report["counts"]["eligible_records"] == 0
    assert "access_or_license_not_verified" in report["records"][0]["reasons"]
    assert collection.to_json() == before


def test_hash_verification_and_reproducibility(cohort):
    collection, reviews, root = cohort
    first = run(cohort)
    shuffled = ManifestCollection(list(reversed(collection.records)))
    assert first == audit_readiness(shuffled, reviews, image_root=root)
    key = record_key(collection.records[0])
    reviews[key] = replace(reviews[key], expected_sha256="0" * 64)
    changed = run(cohort)
    assert "image_hash_mismatch" in changed["records"][0]["reasons"]
    assert changed["reviews_sha256"] != first["reviews_sha256"]


@pytest.mark.parametrize("license_status", ["", "unknown", "public", "OASIS-DUA"])
def test_license_requirements_cannot_be_bypassed_by_review(cohort, license_status):
    row = run(change_record(cohort, license_status=license_status))["records"][0]
    assert not row["eligible"]
    assert "license_does_not_match_source_requirement" in row["reasons"]


def test_affine_disagreement_requires_review(cohort):
    collection, _, _ = cohort
    path = collection.records[0].image_path
    image = nib.load(path)
    image.set_qform(np.eye(4), code=1)
    shifted = np.eye(4)
    shifted[0, 3] = 20
    image.set_sform(shifted, code=2)
    nib.save(image, path)
    row = run(cohort)["records"][0]
    assert not row["eligible"] and "conflicting_qform_sform" in row["reasons"]


def test_missing_review_reference_or_identity(cohort):
    collection, reviews, _ = cohort
    key = record_key(collection.records[0])
    reviews[key] = replace(reviews[key], evidence_reference=" ", patient_id="")
    row = run(cohort)["records"][0]
    assert "missing_review_reference" in row["reasons"]
    assert "patient_identity_not_verified" in row["reasons"]


def test_abnormal_label_cannot_conflict_with_control_diagnosis(cohort):
    row = run(change_record(cohort, task_a_label=1))["records"][0]
    assert not row["eligible"] and "diagnosis_label_conflict" in row["reasons"]


def test_null_optional_metadata_is_reported(cohort):
    report = run(change_record(cohort, site=None, scanner=None))
    assert not report["ready"] and "site_or_scanner_metadata_missing" in report["blockers"]
    row = run(change_record(cohort, modality=None, license_status=None))["records"][0]
    assert "not_native_t1_modality" in row["reasons"]
    assert "access_or_license_not_verified" in row["reasons"]


def test_inventory_documents_restrictions_without_access_checks():
    inventory = source_inventory()
    assert set(inventory) == set(APPROVED_SOURCES)
    assert not inventory["oasis3"]["task_a_registry_eligible"]
    assert not inventory["adni"]["task_a_registry_eligible"]
    assert inventory["figshare_2d"]["training_role"] == "pretrain_only"
    assert "placeholder" in inventory["brats2024_gli"]["access_requirements"]
    assert all(s["access_state"] == "not_verified_by_catalogue" for s in inventory.values())


def test_cli_report_and_exit_codes(cohort, capsys):
    collection, reviews, root = cohort
    manifest, evidence = root / "manifest.json", root / "reviews.json"
    collection.save_json(manifest)
    evidence.write_text(json.dumps([
        {"dataset_source": k[0], "case_id": k[1], "image_path": k[2], **asdict(v)}
        for k, v in reviews.items()
    ]))
    args = ["--manifest", str(manifest), "--reviews", str(evidence), "--image-root", str(root)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["ready"]
    assert main([*args, "--metadata-only"]) == 1
    assert not json.loads(capsys.readouterr().out)["ready"]
    evidence.write_text("{}")
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["assessment"] == "audit_error"
    assert main(["--list-sources"]) == 0
    assert "oasis3" in json.loads(capsys.readouterr().out)


def test_review_json_rejects_duplicates_and_nonboolean(cohort):
    collection, reviews, root = cohort
    key = record_key(collection.records[0])
    item = {"dataset_source": key[0], "case_id": key[1], "image_path": key[2],
            **asdict(reviews[key])}
    path = root / "reviews.json"
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_reviews(path)
    path.write_text(json.dumps([{**item, "label_verified": "false"}]))
    with pytest.raises(ValueError, match="boolean"):
        load_reviews(path)


def test_empty_manifest_not_ready(tmp_path):
    report = audit_readiness(ManifestCollection(), {}, image_root=tmp_path)
    assert report["counts"]["records"] == 0 and not report["ready"]
    assert "missing_verified_healthy_controls" in report["blockers"]
