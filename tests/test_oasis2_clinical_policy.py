"""Entirely synthetic metadata; never load MRI, Excel or private Drive files."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from brainagenticthon.preprocessing.oasis2_clinical_policy import (
    ClinicalImage,
    DiagnosisEvidence,
    LinkageAttestation,
    audit_clinical_manifest,
    load_clinical_images,
    load_diagnosis_evidence,
    main,
)


def image(**changes):
    defaults = dict(
        subject_id="synthetic-subject",
        mri_visit="synthetic-visit-1",
        group="Nondemented",
        cdr=0,
        mmse=29,
        image_path="synthetic-only/not-an-actual-image.nii.gz",
        linkage_status="matched",
        split="train",
        source_dataset="OASIS-2",
    )
    return ClinicalImage(**{**defaults, **changes})


def diagnosis(**changes):
    defaults = dict(
        manual_review_status="approved",
        verified_diagnosis="Alzheimer's disease",
        diagnosis_provenance="clinical_record",
        evidence_reference="synthetic-evidence",
        reviewer_reference="synthetic-reviewer",
        task_c_label="ad",
    )
    return DiagnosisEvidence(**{**defaults, **changes})


def test_candidate_is_not_a_diagnosis_or_mri_control():
    row = image()
    report = audit_clinical_manifest([row])
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 1
    assert report["counts"]["review_supported_annotations"] == {}
    assert not report["main_integration"]["task_a_ready"]
    assert not report["main_integration"]["registered"]
    assert not report["main_integration"]["final_task_a_or_c_release_ready"]
    assert row.group == "Nondemented" and row.cdr == 0 and row.mmse == 29


@pytest.mark.parametrize(
    "group,cdr",
    [
        ("Demented", 0.5),
        ("Demented", 1),
        ("Demented", 2),
        ("Converted", 0),
        ("Converted", 0.5),
        ("Nondemented", 0.5),
    ],
)
def test_group_and_scores_never_create_ad_or_mci(group, cdr):
    report = audit_clinical_manifest([image(group=group, cdr=cdr)])
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0
    assert report["counts"]["review_supported_annotations"] == {}


def test_converted_visits_preserve_cdr_changes_and_visit_scoped_evidence():
    early = image(group="Converted", cdr=0)
    late = image(
        group="Converted",
        cdr=0.5,
        mmse=26,
        mri_visit="synthetic-visit-2",
        image_path="synthetic-only/later-image.nii.gz",
    )
    report = audit_clinical_manifest([early, late], {late.visit_key: diagnosis()})
    assert report["counts"]["converted_cdr_sessions"] == {"0.0": 1, "0.5": 1}
    assert report["counts"]["review_status_visits"]["approved"] == 1
    assert report["counts"]["review_status_visits"]["pending"] == 1
    assert report["counts"]["review_supported_annotations"] == {
        "review_supported_task_c_ad_visits": 1,
    }
    assert early.cdr == 0 and late.cdr == 0.5
    assert early.mmse == 29 and late.mmse == 26


@pytest.mark.parametrize(
    "evidence",
    [
        DiagnosisEvidence(),
        diagnosis(verified_diagnosis=None),
        diagnosis(manual_review_status="pending"),
        diagnosis(manual_review_status="rejected"),
        diagnosis(diagnosis_provenance="inferred"),
        diagnosis(evidence_reference=""),
        diagnosis(reviewer_reference=""),
        diagnosis(verified_diagnosis="Demented"),
        diagnosis(verified_diagnosis="CDR 0.5", task_c_label="mci"),
        diagnosis(verified_diagnosis="dementia"),
    ],
)
def test_missing_or_unsafe_diagnosis_evidence_does_not_produce_labels(evidence):
    row = image(group="Demented", cdr=0.5)
    report = audit_clinical_manifest([row], {row.visit_key: evidence})
    assert report["counts"]["review_supported_annotations"] == {}


def test_mci_requires_explicit_clinical_evidence_not_cdr():
    row = image(group="Demented", cdr=0.5)
    bad = audit_clinical_manifest([row], {row.visit_key: diagnosis(task_c_label="mci")})
    assert bad["counts"]["review_supported_annotations"] == {}
    good = audit_clinical_manifest(
        [row],
        {
            row.visit_key: diagnosis(
                verified_diagnosis="Mild Cognitive Impairment",
                task_c_label="mci",
            )
        },
    )
    assert good["counts"]["review_supported_annotations"] == {
        "review_supported_task_c_mci_visits": 1,
    }


def test_task_a_mri_scope_is_separate_from_cognitive_diagnosis():
    row = image()
    evidence = diagnosis(verified_diagnosis="Cognitively Normal", task_a_label=0, task_c_label="cn")
    report = audit_clinical_manifest([row], {row.visit_key: evidence})
    assert (
        "review_supported_task_a_0_visits" not in report["counts"]["review_supported_annotations"]
    )
    reviewed = replace(evidence, healthy_mri_verified=True)
    report = audit_clinical_manifest([row], {row.visit_key: reviewed})
    assert report["counts"]["review_supported_annotations"]["review_supported_task_a_0_visits"] == 1
    assert not report["main_integration"]["task_a_ready"]  # Not registry/MRI/access approval.


def test_task_a_abnormal_needs_mri_evidence():
    row = image(group="Demented", cdr=1)
    evidence = diagnosis(task_a_label=1)
    report = audit_clinical_manifest([row], {row.visit_key: evidence})
    assert (
        "review_supported_task_a_1_visits" not in report["counts"]["review_supported_annotations"]
    )
    report = audit_clinical_manifest(
        [row],
        {
            row.visit_key: replace(
                evidence,
                abnormal_mri_verified=True,
            )
        },
    )
    assert report["counts"]["review_supported_annotations"]["review_supported_task_a_1_visits"] == 1


def test_group_cdr_conflict_requires_resolution():
    row = image(cdr=0.5)
    review = diagnosis()
    report = audit_clinical_manifest([row], {row.visit_key: review})
    assert report["counts"]["warnings_and_review_reasons"]["group_cdr_conflict"] == 1
    assert not report["counts"]["review_supported_annotations"]
    resolved = audit_clinical_manifest(
        [row],
        {
            row.visit_key: replace(
                review,
                conflict_resolution_reference="synthetic-private-resolution",
            )
        },
    )
    assert (
        resolved["counts"]["review_supported_annotations"]["review_supported_task_c_ad_visits"] == 1
    )
    assert row.group == "Nondemented" and row.cdr == 0.5  # Original never rewritten.


def test_missing_mmse_vs_not_exported():
    missing = audit_clinical_manifest([image(mmse=None)])
    assert missing["counts"]["warnings_and_review_reasons"]["missing_mmse"] == 1
    assert missing["counts"]["cognitive_comparison_candidate_sessions"] == 1
    absent = audit_clinical_manifest([image(mmse=None, mmse_field_present=False)])
    assert absent["counts"]["warnings_and_review_reasons"]["mmse_not_exported"] == 1
    assert "missing_mmse" not in absent["counts"]["warnings_and_review_reasons"]


def test_many_images_count_one_visit_and_one_mmse_warning():
    rows = [image(mmse=None, image_path=f"synthetic-only/image-{i}.nii.gz") for i in range(4)]
    report = audit_clinical_manifest(rows)
    assert report["counts"]["image_records"] == 4
    assert report["counts"]["unique_subject_visit_keys"] == 1
    assert report["counts"]["warnings_and_review_reasons"]["missing_mmse"] == 1


def test_conflicting_images_of_same_visit_do_not_choose_first_values():
    rows = [image(), image(cdr=0.5, image_path="synthetic-only/other-image.nii.gz")]
    report = audit_clinical_manifest(rows)
    assert report["counts"]["unique_subject_visit_keys"] == 1
    assert report["counts"]["group_sessions"] == {}
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0
    assert report["counts"]["warnings_and_review_reasons"]["conflicting_values_within_visit"] == 1


@pytest.mark.parametrize("split", ["val", "validation", "test", "holdout"])
def test_all_subject_images_and_visits_must_remain_one_split(split):
    report = audit_clinical_manifest(
        [
            image(),
            image(
                mri_visit="synthetic-visit-2", split=split, image_path="synthetic-only/other.nii.gz"
            ),
        ]
    )
    assert not report["split_audit"]["all_subjects_single_assigned_split"]
    assert report["split_audit"]["subjects_in_multiple_splits"] == 1


def test_partial_split_is_not_assigned_automatically():
    row = image(split="unassigned")
    report = audit_clinical_manifest(
        [image(), replace(row, image_path="synthetic-only/other.nii.gz")]
    )
    assert report["split_audit"]["subjects_with_unassigned_images"] == 1
    assert row.split == "unassigned"


def test_explicit_unmatched_or_ambiguous_cannot_be_overridden():
    for status in ("unmatched", "ambiguous"):
        report = audit_clinical_manifest(
            [image(linkage_status=status)],
            linkage_attestation=LinkageAttestation("synthetic-report", 1, 1),
        )
        assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0


def private_export_fixture():
    return [
        {
            "patient_id": "PRIVATE_SENTINEL_SUBJECT",
            "visit_id": "PRIVATE_SENTINEL_VISIT",
            "clinical_group": "Nondemented",
            "cdr": 0.0,
            "relative_path": "PRIVATE_SENTINEL_PATH/mri.nii.gz",
            "dataset": "OASIS-2",
            "diagnosis_source": "OASIS-2 demographics",
            "task_a_label": None,
            "task_c_label": None,
            "label_status": "PENDING_CLINICAL_POLICY",
        }
    ]


def test_exact_private_array_schema_and_external_linkage(tmp_path):
    path = tmp_path / "synthetic-export.json"
    path.write_text(json.dumps(private_export_fixture()))
    before = path.read_bytes()
    rows, digest = load_clinical_images(path)
    assert rows[0].subject_id == "PRIVATE_SENTINEL_SUBJECT"
    assert rows[0].mri_visit == "PRIVATE_SENTINEL_VISIT"
    assert rows[0].group == "Nondemented" and rows[0].mmse is None
    assert not rows[0].mmse_field_present and rows[0].linkage_status == "unknown"
    assert rows[0].diagnosis_source == "OASIS-2 demographics"
    assert rows[0].label_status == "PENDING_CLINICAL_POLICY"
    assert rows[0].declared_task_a_label is rows[0].declared_task_c_label is None
    report = audit_clinical_manifest(
        rows, linkage_attestation=LinkageAttestation("synthetic-prior-report", 1, 1)
    )
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 1
    assert report["counts"]["linkage_status_images"] == {"unknown": 1}
    assert report["external_linkage_attestation"]["counts_consistent"]
    assert path.read_bytes() == before and len(digest) == 64


def test_stale_linkage_attestation_not_trusted():
    report = audit_clinical_manifest(
        [image(linkage_status="unknown")],
        linkage_attestation=LinkageAttestation("synthetic-report", 2, 1),
    )
    assert not report["external_linkage_attestation"]["counts_consistent"]
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0


def test_explicit_nested_field_map_and_optional_mmse(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "payload": [
                    {
                        "clinical": {
                            "subject": "synthetic-s",
                            "visit": "synthetic-v",
                            "Group": "Nondemented",
                            "CDR": 0,
                            "MMSE": None,
                            "file": "synthetic/file.nii.gz",
                            "match": "matched",
                        }
                    }
                ]
            }
        )
    )
    mapping = dict(
        subject_id="clinical.subject",
        mri_visit="clinical.visit",
        group="clinical.Group",
        cdr="clinical.CDR",
        mmse="clinical.MMSE",
        image_path="clinical.file",
        linkage_status="clinical.match",
    )
    rows, _ = load_clinical_images(path, field_map=mapping, records_key="payload")
    assert rows[0].mmse_field_present and rows[0].mmse is None
    assert rows[0].linkage_status == "matched"


def test_unsafe_existing_labels_flagged_not_overwritten(tmp_path):
    path = tmp_path / "synthetic.json"
    item = {
        **private_export_fixture()[0],
        "clinical_group": "Demented",
        "cdr": 0.5,
        "task_c_label": "ad",
    }
    path.write_text(json.dumps([item]))
    before = path.read_bytes()
    rows, _ = load_clinical_images(path)
    report = audit_clinical_manifest(rows)
    assert not report["counts"]["review_supported_annotations"]
    assert (
        report["counts"]["warnings_and_review_reasons"][
            "existing_label_lacks_matching_verified_evidence_images"
        ]
        == 1
    )
    assert rows[0].declared_task_c_label == "ad" and path.read_bytes() == before


def test_cli_no_identifier_logs_no_mri_access(tmp_path, capsys, monkeypatch):
    import nibabel

    monkeypatch.setattr(nibabel, "load", lambda *a, **k: pytest.fail("No image reads"))
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(private_export_fixture()))
    before = path.read_bytes()
    link = tmp_path / "synthetic-linkage.json"
    link.write_text(
        json.dumps(
            dict(
                evidence_reference="PRIVATE_SENTINEL_EVIDENCE",
                matched_image_records=1,
                matched_subject_visit_keys=1,
            )
        )
    )
    # Unassigned input split yields review-required exit 1, not a final-ready claim.
    assert main(["--manifest", str(path), "--linkage-attestation", str(link)]) == 1
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
    report = json.loads(captured.out)
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 1
    assert not report["main_integration"]["final_task_a_or_c_release_ready"]
    assert path.read_bytes() == before
    path.write_text('{"PRIVATE_SENTINEL_BAD_JSON":')
    assert main(["--manifest", str(path)]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err


def test_duplicate_evidence_and_bad_scores_rejected(tmp_path):
    path = tmp_path / "synthetic-evidence.json"
    item = dict(subject_id="synthetic-s", mri_visit="synthetic-v")
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError):
        load_diagnosis_evidence(path)
    for cdr in (True, float("nan"), "0.5"):
        with pytest.raises(ValueError):
            image(cdr=cdr)
    with pytest.raises(ValueError):
        DiagnosisEvidence(task_c_label="pituitary")


def test_supplied_aggregate_summary_reproduced_with_synthetic_rows_only():
    # Reproduces the supplied COUNT pattern, never private identities or diagnoses.
    rows = []
    specifications = [
        ("Nondemented", 72, [0] * 188 + [0.5] * 2),
        ("Demented", 64, [0.5] * 102 + [1] * 41 + [2] * 3),
        ("Converted", 14, [0] * 18 + [0.5] * 19),
    ]
    visit_number = 0
    for group, subjects, scores in specifications:
        for i, score in enumerate(scores):
            copies = 4 if visit_number < 249 else 3
            for j in range(copies):
                rows.append(
                    image(
                        subject_id=f"synthetic-{group}-{i % subjects}",
                        mri_visit=f"synthetic-visit-{i // subjects}",
                        group=group,
                        cdr=score,
                        mmse=None if visit_number in {5, 20} else 28,
                        image_path=f"synthetic-only/visit-{visit_number}-image-{j}.nii.gz",
                    )
                )
            visit_number += 1
    report = audit_clinical_manifest(rows)
    counts = report["counts"]
    assert counts["unique_subjects"] == 150
    assert counts["unique_subject_visit_keys"] == 373
    assert counts["image_records"] == 1368
    assert counts["group_sessions"] == {"Nondemented": 190, "Demented": 146, "Converted": 37}
    assert counts["group_subjects"] == {"Nondemented": 72, "Demented": 64, "Converted": 14}
    assert counts["cognitive_comparison_candidate_sessions"] == 188
    assert counts["cognitive_comparison_candidate_subjects"] == 72
    assert counts["warnings_and_review_reasons"]["group_cdr_conflict"] == 2
    assert counts["warnings_and_review_reasons"]["missing_mmse"] == 2
    assert counts["review_supported_annotations"] == {}


@pytest.mark.parametrize(
    "cdr,mmse,reason",
    [
        (None, 28, "missing_cdr"),
        (0.25, 28, "invalid_cdr"),
        (0, -1, "invalid_mmse"),
        (0, 31, "invalid_mmse"),
        (0, 27.5, "invalid_mmse"),
    ],
)
def test_invalid_clinical_scores_are_not_comparison_candidates(cdr, mmse, reason):
    report = audit_clinical_manifest([image(cdr=cdr, mmse=mmse)])
    assert report["counts"]["warnings_and_review_reasons"][reason] == 1
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0


def test_same_image_linked_to_multiple_subject_visits_is_ambiguous():
    report = audit_clinical_manifest([image(), image(subject_id="synthetic-other")])
    assert report["counts"]["image_paths_linked_to_multiple_visits"] == 1
    assert report["counts"]["cognitive_comparison_candidate_sessions"] == 0
    assert not report["counts"]["review_supported_annotations"]


def test_val_alias_preserves_original_split_values():
    early = image(split="val")
    late = image(
        mri_visit="synthetic-visit-2", split="validation", image_path="synthetic-only/other.nii.gz"
    )
    report = audit_clinical_manifest([early, late])
    assert report["split_audit"]["all_subjects_single_assigned_split"]
    assert report["split_audit"]["split_image_counts"] == {"validation": 2}
    assert early.split == "val" and late.split == "validation"


def test_clinical_policy_does_not_grant_source_or_modality_approval():
    report = audit_clinical_manifest([image()])
    assert report["main_integration"]["records_failing_existing_schema"] == 1
    assert "image_checks_not_run" in report["main_integration"]["task_a_blockers"]
    with pytest.raises(ValueError):
        image(source_dataset="OASIS-3")


def test_cli_zero_means_metadata_valid_not_final_label_ready(tmp_path, capsys):
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps([{**private_export_fixture()[0], "split": "train", "linkage_status": "matched"}])
    )
    assert main(["--manifest", str(path)]) == 0
    output = capsys.readouterr().out
    assert "PRIVATE_SENTINEL" not in output
    assert not json.loads(output)["main_integration"]["final_task_a_or_c_release_ready"]
