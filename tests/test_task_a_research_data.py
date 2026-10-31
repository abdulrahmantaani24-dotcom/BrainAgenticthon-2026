"""Invented MRI arrays/labels only, never evidence of real cohort eligibility."""

import copy
import json
from dataclasses import asdict, replace

import nibabel as nib
import numpy as np
import pytest
import torch

from brainagenticthon.detection.data import (
    TrainingEvidence,
    load_training_evidence,
    prepare_cohort,
)
from brainagenticthon.detection.imaging import load_mri
from brainagenticthon.detection.profile import InputProfile
from brainagenticthon.detection.training import load_private_cohort
from brainagenticthon.preprocessing.dataset_registry import APPROVED_SOURCES
from brainagenticthon.preprocessing.manifest_schema import ManifestCollection, ManifestRecord
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file
from brainagenticthon.preprocessing.task_a_readiness import RecordReview, record_key

PROFILE = InputProfile(target_shape=(16, 16, 16))


def save_image(path, shape=(16, 16, 16), *, pair=False, seed=1, version=1):
    data = np.random.default_rng(seed).normal(10, 2, shape).astype(np.float32)
    data[:2] = 0
    classes = ((nib.Nifti1Image, nib.Nifti1Pair), (nib.Nifti2Image, nib.Nifti2Pair))
    cls = classes[version - 1][int(pair)]
    image = cls(data, np.diag([-1.0, 1.0, 1.25, 1.0]))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


@pytest.fixture
def verified_synthetic_metadata(tmp_path):
    records, reviews, evidence = [], {}, {}
    for source in ("brats2024_gli", "brats2024_men"):
        for split in ("train", "validation", "test"):
            for label in (0, 1):
                index = len(records)
                path = save_image(tmp_path / f"synthetic-{index}.nii.gz", seed=index)
                record = ManifestRecord(
                    subject_id=f"synthetic-{index}", case_id=f"synthetic-{index}",
                    dataset_source=source, cohort="SYNTHETIC_NOT_REAL_BRATS_CONTROLS",
                    session_id="visit-1", modality="t1", image_path=path.name,
                    image_dimension="3D", original_diagnosis="Normal" if label == 0 else "glioma",
                    task_a_label=label, task_c_label=None, segmentation_path=None,
                    label_provenance="clinical_record",
                    license_status=APPROVED_SOURCES[source].license_type.value,
                    site="synthetic-site", scanner="synthetic-scanner", split=split,
                    quality_flag="ok",
                )
                records.append(record)
                reviews[record_key(record)] = RecordReview(
                    patient_id=f"synthetic-global-{index}", evidence_reference="synthetic-only",
                    access_authorized=True, patient_identity_verified=True, label_verified=True,
                    healthy_control_verified=not label, native_t1_verified=True,
                    whole_brain_verified=True, expected_sha256=digest_file(path),
                )
                evidence[record_key(record)] = TrainingEvidence(
                    manual_review_status="approved", visual_qc_approved=True,
                    anatomical_geometry_verified=True, abnormal_mri_verified=bool(label),
                    mri_label_evidence_reference="synthetic-only",
                    visual_qc_evidence_reference="synthetic-only",
                    geometry_evidence_reference="synthetic-only",
                    modality_evidence_reference="synthetic-only",
                )
    return ManifestCollection(records), reviews, evidence, tmp_path


def prepare(fixture):
    collection, reviews, evidence, root = fixture
    return prepare_cohort(collection, reviews, evidence, image_root=root, profile=PROFILE)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("shape", [(16, 16, 16), (16, 16, 16, 1)])
def test_read_only_3d_singleton_and_paired_names(tmp_path, pair, shape, version):
    path = save_image(tmp_path / ("mpr-1.nifti.hdr" if pair else "volume.nii.gz"),
                      shape, pair=pair, version=version)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    tensor, fingerprints = load_mri(tmp_path, path.name, PROFILE)
    assert tensor.shape == (1, 16, 16, 16) and torch.isfinite(tensor).all()
    assert len(fingerprints) == (2 if pair else 1)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert torch.count_nonzero(tensor) > 0
    assert torch.equal(tensor, load_mri(tmp_path, path.name, PROFILE, expected=fingerprints)[0])


@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 16, 2), (16, 16, 16, 1, 1)])
def test_nonspatial_and_nonsingleton_rejected(tmp_path, shape):
    save_image(tmp_path / "bad.nii", shape)
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, "bad.nii", PROFILE)


@pytest.mark.parametrize("kind", ["constant", "nan", "inf", "unknown_units", "conflict"])
def test_invalid_intensity_or_geometry_rejected(tmp_path, kind):
    path = save_image(tmp_path / "bad.nii")
    image = nib.load(path)
    data = np.asarray(image.dataobj).copy()
    if kind == "constant":
        data[:] = 1
    elif kind in {"nan", "inf"}:
        data[5, 5, 5] = np.nan if kind == "nan" else np.inf
    modified = nib.Nifti1Image(data, image.affine)
    modified.header.set_xyzt_units("unknown" if kind == "unknown_units" else "mm")
    if kind == "conflict":
        modified.set_qform(np.eye(4), code=1)
        modified.set_sform(np.diag([2.0, 2.0, 2.0, 1.0]), code=1)
    nib.save(modified, path)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, PROFILE)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["missing", "truncated", "corrupt", "unsupported"])
def test_malformed_files_rejected(tmp_path, kind):
    path = save_image(tmp_path / "bad.nifti.hdr", pair=True)
    if kind == "missing":
        path.with_suffix(".img").unlink()
    elif kind == "truncated":
        path.with_suffix(".img").write_bytes(b"short")
    elif kind == "corrupt":
        path.write_bytes(b"corrupted")
    else:
        path = tmp_path / "bad.txt"
        path.write_text("not an MRI")
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, PROFILE)


def test_limits_traversal_and_source_change(tmp_path):
    path = save_image(tmp_path / "volume.nii")
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, replace(PROFILE, max_voxels=100))
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, replace(PROFILE, max_file_bytes=100))
    with pytest.raises((ValueError, RuntimeError)):
        load_mri(tmp_path, "../private-person/scan.nii", PROFILE)
    _, fingerprints = load_mri(tmp_path, path.name, PROFILE)
    save_image(path, seed=999)
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, PROFILE, expected=fingerprints)


def test_gate_preserves_manifest_and_reads_without_mutation(verified_synthetic_metadata):
    collection, _, _, root = verified_synthetic_metadata
    metadata_before = copy.deepcopy(collection.to_dict_list())
    before = {p: p.read_bytes() for p in root.iterdir()}
    prepared = prepare(verified_synthetic_metadata)
    assert prepared.summary["patients"] == 12
    assert not prepared.summary["clinical_approval"]
    sample = prepared.dataset("train")[0]
    assert sample["image"].shape == (1, 16, 16, 16)
    assert isinstance(sample["patient"], int)
    assert collection.to_dict_list() == metadata_before
    assert before == {p: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("change", [
    {"task_a_label": None}, {"label_provenance": "inferred"}, {"original_diagnosis": None},
    {"modality": "t1ce"}, {"quality_flag": "unknown"}, {"site": ""}, {"scanner": ""},
    {"dataset_source": "oasis3"}, {"dataset_source": "oasis2"},
    {"dataset_source": "adni"}, {"dataset_source": "local"},
])
def test_existing_policy_cannot_be_overridden(verified_synthetic_metadata, change):
    collection, reviews, evidence, root = verified_synthetic_metadata
    records = collection.records
    old = records[0]
    records[0] = replace(old, **change)
    reviews[record_key(records[0])] = reviews[record_key(old)]
    evidence[record_key(records[0])] = evidence[record_key(old)]
    with pytest.raises(ValueError, match="metadata_eligibility"):
        prepare((ManifestCollection(records), reviews, evidence, root))


@pytest.mark.parametrize("field", ["access_authorized", "patient_identity_verified",
                                   "label_verified",
                                   "healthy_control_verified", "native_t1_verified",
                                   "whole_brain_verified"])
def test_missing_review_attestation_rejected(verified_synthetic_metadata, field):
    collection, reviews, _, _ = verified_synthetic_metadata
    key = record_key(collection.records[0])
    reviews[key] = replace(reviews[key], **{field: False})
    with pytest.raises(ValueError):
        prepare(verified_synthetic_metadata)


@pytest.mark.parametrize("change", [
    {"manual_review_status": "pending"}, {"visual_qc_approved": False},
    {"anatomical_geometry_verified": False}, {"mri_label_evidence_reference": ""},
    {"visual_qc_evidence_reference": ""}, {"geometry_evidence_reference": ""},
    {"modality_evidence_reference": ""},
])
def test_additional_review_is_required(verified_synthetic_metadata, change):
    collection, _, evidence, _ = verified_synthetic_metadata
    key = record_key(collection.records[0])
    evidence[key] = replace(evidence[key], **change)
    with pytest.raises(ValueError, match="evidence_required"):
        prepare(verified_synthetic_metadata)


def test_abnormal_mri_evidence_hash_and_missing_classes(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    key = record_key(collection.records[1])
    evidence[key] = replace(evidence[key], abnormal_mri_verified=False)
    with pytest.raises(ValueError):
        prepare(verified_synthetic_metadata)
    evidence[key] = replace(evidence[key], abnormal_mri_verified=True)
    reviews[key] = replace(reviews[key], expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        prepare(verified_synthetic_metadata)
    subset = ManifestCollection([r for r in collection.records if r.task_a_label == 1])
    with pytest.raises(ValueError):
        prepare((subset, reviews, evidence, root))


def test_patient_leakage_across_sources_rejected(verified_synthetic_metadata):
    collection, reviews, _, _ = verified_synthetic_metadata
    train_key, test_key = record_key(collection.records[0]), record_key(collection.records[10])
    reviews[test_key] = replace(reviews[test_key], patient_id=reviews[train_key].patient_id)
    with pytest.raises(ValueError, match="metadata_eligibility"):
        prepare(verified_synthetic_metadata)


def test_perfect_source_class_confounding_rejected(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    records, new_reviews, new_evidence = [], {}, {}
    for record in collection.records:
        source = "brats2024_gli" if record.task_a_label == 0 else "brats2024_men"
        new = replace(record, dataset_source=source)
        records.append(new)
        new_reviews[record_key(new)] = reviews[record_key(record)]
        new_evidence[record_key(new)] = evidence[record_key(record)]
    with pytest.raises(ValueError):
        prepare((ManifestCollection(records), new_reviews, new_evidence, root))


def test_group_and_cdr_not_mri_diagnoses(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    for diagnosis in ("Demented", "Converted", "CDR=0.5"):
        records = collection.records
        records[1] = replace(records[1], original_diagnosis=diagnosis)
        with pytest.raises(ValueError, match="not_mri_diagnosis"):
            prepare((ManifestCollection(records), reviews, evidence, root))


def test_repeated_sessions_patient_weighting(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    records = collection.records
    original = records[0]
    path = save_image(root / "extra-session.nii", seed=135)
    extra = replace(original, case_id="synthetic-extra", session_id="visit-2", image_path=path.name)
    reviews[record_key(extra)] = replace(
        reviews[record_key(original)], expected_sha256=digest_file(path),
    )
    evidence[record_key(extra)] = evidence[record_key(original)]
    prepared = prepare((ManifestCollection([*records, extra]), reviews, evidence, root))
    dataset = prepared.dataset("train")
    assert sorted(dataset.patient_weights) == [0.5, 0.5, 1.0, 1.0, 1.0]
    patients = [{row[1] for row in prepared.dataset(split).rows}
                for split in ("train", "validation", "test")]
    assert not patients[0] & patients[1] and not patients[0] & patients[2]


def test_evidence_schema_and_error_privacy(tmp_path):
    with pytest.raises(ValueError):
        TrainingEvidence(visual_qc_approved="true")
    path = tmp_path / "PRIVATE-PERSON.json"
    path.write_text('[{"patient": "PRIVATE-PERSON"}]')
    with pytest.raises(RuntimeError) as error:
        load_training_evidence(path)
    assert "PRIVATE-PERSON" not in str(error.value)


def test_header_repairs_cannot_hide_bad_source_spacing(tmp_path):
    path = save_image(tmp_path / "invalid-spacing.nii")
    with path.open("r+b") as stream:
        header = nib.Nifti1Header.from_fileobj(stream, check=False)
        header["pixdim"][1] = 0
        stream.seek(0)
        stream.write(header.binaryblock)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        load_mri(tmp_path, path.name, PROFILE)
    assert path.read_bytes() == before


def test_header_ras_flip_and_intensity_support_semantics(tmp_path):
    values = 1 + np.indices((16, 16, 16))[0].astype(np.float32)
    image = nib.Nifti1Image(values, np.diag([-1., 1., 1., 1.]))
    image.header.set_xyzt_units("mm")
    path = tmp_path / "known-axis.nii"
    nib.save(image, path)
    tensor, _ = load_mri(tmp_path, path.name, PROFILE)
    assert tensor[0, :4].mean() > tensor[0, -4:].mean()
    assert abs(float(tensor.mean())) < 1e-6
    assert float(tensor.std(correction=0)) == pytest.approx(1., abs=1e-6)


def test_singleton_pair_gate_requires_both_reviewed_hashes(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    records = collection.records
    old = records[0]
    key = record_key(old)
    path = save_image(root / "paired.nifti.hdr", (16, 16, 16, 1), pair=True, seed=994)
    record = replace(old, image_path=path.name)
    records[0] = record
    new_key = record_key(record)
    reviews[new_key] = replace(reviews[key], expected_sha256=digest_file(path))
    evidence[new_key] = evidence[key]
    fixture = ManifestCollection(records), reviews, evidence, root
    with pytest.raises(ValueError, match="companion"):
        prepare(fixture)
    evidence[new_key] = replace(
        evidence[new_key], companion_sha256=digest_file(path.with_suffix(".img")),
    )
    prepared = prepare(fixture)
    assert len(prepared.dataset("train")) == 4


def test_partial_site_confounding_rejected(verified_synthetic_metadata):
    collection, reviews, evidence, root = verified_synthetic_metadata
    records = collection.records
    records[0] = replace(records[0], site="synthetic-pure-negative-site")
    # Other mixed sites mean this is not the old perfect-global-confounding case.
    with pytest.raises(ValueError, match="confounding"):
        prepare((ManifestCollection(records), reviews, evidence, root))


def test_private_json_inputs_load_without_logs_or_mutation(verified_synthetic_metadata, capsys):
    collection, reviews, evidence, root = verified_synthetic_metadata
    manifest = root / "manifest.json"
    review_file, evidence_file = root / "reviews.json", root / "training-evidence.json"
    collection.save_json(manifest)
    def keyed_rows(mapping):
        return [{**dict(zip(("dataset_source", "case_id", "image_path"), key, strict=True)),
                 **asdict(value)} for key, value in mapping.items()]
    review_file.write_text(json.dumps(keyed_rows(reviews)))
    evidence_file.write_text(json.dumps(keyed_rows(evidence)))
    before = {p: p.read_bytes() for p in root.iterdir()}
    cohort = load_private_cohort(manifest, review_file, evidence_file, root, PROFILE)
    assert cohort.summary["patients"] == 12
    assert len(cohort.source_roots) == 4
    assert before == {p: p.read_bytes() for p in root.iterdir()}
    output = capsys.readouterr()
    assert not output.out and not output.err


@pytest.mark.parametrize("changes", [{"target_shape": (8, 8, 8)}, {"max_voxels": 0},
                                     {"name": "task-c"}])
def test_profile_rejects_implicit_semantic_changes(changes):
    with pytest.raises(ValueError):
        replace(PROFILE, **changes)
