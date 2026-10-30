"""OASIS-2 clinical metadata policy; no MRI reads, diagnoses or manifest writes.

Public reports are aggregate-only. Subject/visit identifiers, clinical values and
review references remain in the in-memory input objects/private source manifest.
OASIS-2 is deliberately NOT aliased to the registered OASIS-3 source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainagenticthon.preprocessing.manifest_schema import (
    TASK_C_LABELS,
    LabelProvenance,
    ManifestCollection,
    ManifestRecord,
    validate_record,
)
from brainagenticthon.preprocessing.split_manager import SplitManager
from brainagenticthon.preprocessing.task_a_readiness import (
    NORMAL_DIAGNOSES,
    RecordReview,
    audit_readiness,
    record_key,
    source_inventory,
)

POLICY_VERSION = "1.0.0"
GROUPS = frozenset({"Nondemented", "Demented", "Converted"})
CDR_VALUES = frozenset({0, 0.5, 1, 2, 3})
VisitKey = tuple[str, str]
OASIS2_FIELD_MAP = {
    "subject_id": "patient_id",
    "mri_visit": "visit_id",
    "group": "clinical_group",
    "image_path": "relative_path",
    "declared_task_a_label": "task_a_label",
    "declared_task_c_label": "task_c_label",
    "source_dataset": "dataset",
}


@dataclass(frozen=True)
class ClinicalImage:
    """One linked image row. Clinical fields retain their original values."""

    subject_id: str
    mri_visit: str
    group: str
    cdr: float | None
    mmse: float | None
    image_path: str
    linkage_status: str = "unknown"
    split: str = "unassigned"
    declared_task_a_label: int | None = None
    declared_task_c_label: str | None = None
    source_dataset: str | None = None
    diagnosis_source: str | None = None
    label_status: str | None = None
    mmse_field_present: bool = True

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not x.strip()
            for x in (self.subject_id, self.mri_visit, self.group, self.image_path)
        ):
            raise ValueError("missing_required_text")
        if self.linkage_status not in {"matched", "unmatched", "ambiguous", "unknown"}:
            raise ValueError("invalid_linkage_status")
        if self.split not in {"train", "val", "validation", "test", "holdout", "unassigned"}:
            raise ValueError("invalid_split")
        for value in (self.cdr, self.mmse):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("clinical_scores_must_be_finite_numbers_or_null")
        if type(self.mmse_field_present) is not bool or (
            not self.mmse_field_present and self.mmse is not None
        ):
            raise ValueError("inconsistent_mmse_availability")
        if self.source_dataset is not None and self.source_dataset != "OASIS-2":
            raise ValueError("unexpected_source_dataset")

    @property
    def visit_key(self) -> VisitKey:
        return self.subject_id, self.mri_visit


@dataclass(frozen=True)
class LinkageAttestation:
    """Separate prior-match evidence. Does not fill absent manifest fields."""

    evidence_reference: str
    matched_image_records: int
    matched_subject_visit_keys: int
    unmatched_image_records: int = 0
    ambiguous_image_records: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_reference, str) or not self.evidence_reference.strip():
            raise ValueError("missing_linkage_evidence_reference")
        for value in (
            self.matched_image_records,
            self.matched_subject_visit_keys,
            self.unmatched_image_records,
            self.ambiguous_image_records,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("invalid_linkage_attestation_counts")


@dataclass(frozen=True)
class DiagnosisEvidence:
    """Separate visit-scoped manual evidence, never inferred from Group/CDR/MMSE.

    approved is a reviewer attestation, not software confirmation of a diagnosis.
    Generic dementia, Converted status and CDR values are not etiologic diagnoses.
    """

    manual_review_status: str = "pending"
    verified_diagnosis: str | None = None
    diagnosis_provenance: str = LabelProvenance.MISSING.value
    evidence_reference: str = ""
    reviewer_reference: str = ""
    conflict_resolution_reference: str = ""
    task_a_label: int | None = None
    task_c_label: str | None = None
    healthy_mri_verified: bool = False
    abnormal_mri_verified: bool = False

    def __post_init__(self) -> None:
        if self.manual_review_status not in {"pending", "needs_review", "approved", "rejected"}:
            raise ValueError("invalid_manual_review_status")
        for value in (self.healthy_mri_verified, self.abnormal_mri_verified):
            if type(value) is not bool:
                raise ValueError("verification_flags_must_be_boolean")
        if self.task_a_label is not None and (
            type(self.task_a_label) is not int or self.task_a_label not in (0, 1)
        ):
            raise ValueError("invalid_task_a_label")
        if self.task_c_label is not None and self.task_c_label not in TASK_C_LABELS:
            raise ValueError("invalid_task_c_label")
        if self.verified_diagnosis is not None and not isinstance(self.verified_diagnosis, str):
            raise ValueError("invalid_diagnosis_text")
        for value in (
            self.evidence_reference,
            self.reviewer_reference,
            self.conflict_resolution_reference,
            self.diagnosis_provenance,
        ):
            if not isinstance(value, str):
                raise ValueError("invalid_evidence_text")


def _visit_issues(row: ClinicalImage) -> list[str]:
    issues = []
    if row.source_dataset != "OASIS-2":
        issues.append("dataset_identity_not_verified")
    if row.group not in GROUPS:
        issues.append("unknown_clinical_group")
    if row.cdr is None:
        issues.append("missing_cdr")
    elif row.cdr not in CDR_VALUES:
        issues.append("invalid_cdr")
    if row.group == "Nondemented" and row.cdr is not None and row.cdr != 0:
        issues.append("group_cdr_conflict")
    if row.group == "Demented" and row.cdr == 0:
        issues.append("group_cdr_conflict")
    if not row.mmse_field_present:
        issues.append("mmse_not_exported")
    elif row.mmse is None:
        issues.append("missing_mmse")
    elif not (0 <= row.mmse <= 30 and row.mmse == int(row.mmse)):
        issues.append("invalid_mmse")
    return issues


def _evidence_labels(
    evidence: DiagnosisEvidence,
    *,
    conflicts: bool,
) -> tuple[int | None, str | None, list[str]]:
    reasons = []
    diagnosis = (evidence.verified_diagnosis or "").strip().lower()
    if evidence.manual_review_status != "approved":
        reasons.append("manual_diagnosis_review_not_approved")
    if not diagnosis or diagnosis in {"unknown", "missing", "none", "n/a"}:
        reasons.append("missing_verified_diagnosis")
    if evidence.diagnosis_provenance != LabelProvenance.CLINICAL_RECORD.value:
        reasons.append("missing_clinical_diagnosis_provenance")
    if not evidence.evidence_reference.strip() or not evidence.reviewer_reference.strip():
        reasons.append("missing_diagnosis_review_reference")
    if conflicts and not evidence.conflict_resolution_reference.strip():
        reasons.append("unresolved_clinical_conflict")
    if diagnosis in {"demented", "nondemented", "converted", "dementia"} or diagnosis.startswith(
        "cdr"
    ):
        reasons.append("group_or_score_is_not_verified_diagnosis")
    if reasons:
        return None, None, reasons

    a_label, c_label = evidence.task_a_label, evidence.task_c_label
    if a_label == 0 and (
        not evidence.healthy_mri_verified
        or diagnosis not in NORMAL_DIAGNOSES | {"cognitively normal"}
    ):
        reasons.append("task_a_normal_mri_not_verified")
        a_label = None
    if a_label == 1 and (
        not evidence.abnormal_mri_verified or diagnosis in NORMAL_DIAGNOSES | {"cognitively normal"}
    ):
        reasons.append("task_a_abnormal_mri_not_verified")
        a_label = None
    # These are explicit reviewer-requested labels, checked against diagnosis,
    # never automatic mappings from clinical_group or scores.
    expected = {
        "ad": {"ad", "alzheimer's", "alzheimer's disease", "alzheimer disease"},
        "mci": {"mci", "mild cognitive impairment"},
        "cn": {"cn", "cognitively normal"},
        "glioma": {"glioma"},
        "meningioma": {"meningioma"},
        "metastasis": {"metastasis", "brain metastasis"},
        "no_finding": {"normal", "no finding", "no pathological finding"},
    }
    if c_label is not None and diagnosis not in expected[c_label]:
        reasons.append("task_c_label_not_supported_by_verified_diagnosis")
        c_label = None
    return a_label, c_label, reasons


def _manifest_row(
    row: ClinicalImage, a_label: int | None, c_label: str | None, evidence: DiagnosisEvidence
) -> ManifestRecord:
    """Non-mutating policy view; unknown oasis2 remains blocked by main schema."""
    return ManifestRecord(
        subject_id=row.subject_id,
        case_id=json.dumps(row.visit_key),
        dataset_source="oasis2",
        cohort=row.group,
        session_id=row.mri_visit,
        modality="",
        image_path=row.image_path,
        image_dimension="",
        original_diagnosis=evidence.verified_diagnosis
        if a_label is not None or c_label is not None
        else None,
        task_a_label=a_label,
        task_c_label=c_label,
        segmentation_path=None,
        label_provenance=evidence.diagnosis_provenance,
        license_status="unknown",
        site="",
        scanner="",
        split="validation" if row.split == "val" else row.split,
    )


def audit_clinical_manifest(
    images: list[ClinicalImage],
    evidence: dict[VisitKey, DiagnosisEvidence] | None = None,
    *,
    linkage_attestation: LinkageAttestation | None = None,
) -> dict[str, Any]:
    """Return only aggregates; leave the original clinical/linkage data untouched.

    A visit with inconsistent clinical values across images is not arbitrarily
    collapsed. Its unique key is counted, its conflicting values are retained in
    the input, and candidate/final annotation counts exclude that visit.
    """
    evidence = evidence or {}
    visits: dict[VisitKey, list[ClinicalImage]] = defaultdict(list)
    image_visits: dict[str, set[VisitKey]] = defaultdict(set)
    for row in images:
        visits[row.visit_key].append(row)
        image_visits[row.image_path].add(row.visit_key)
    externally_verified = bool(
        linkage_attestation
        and (
            linkage_attestation.matched_image_records == len(images)
            and linkage_attestation.matched_subject_visit_keys == len(visits)
            and linkage_attestation.unmatched_image_records == 0
            and linkage_attestation.ambiguous_image_records == 0
        )
    )
    reasons: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    group_subjects: dict[str, set[str]] = defaultdict(set)
    crosstab: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: set[VisitKey] = set()
    annotation_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    converted_cdr: Counter[str] = Counter()
    records, reviews = [], {}
    ambiguous_paths = {p for p, keys in image_visits.items() if len(keys) > 1}
    if linkage_attestation and not externally_verified:
        reasons["external_linkage_attestation_count_mismatch"] += 1
    for key, rows in sorted(visits.items()):
        first = rows[0]
        # Repeated images do not multiply clinical visit counts or MMSE warnings.
        values = {(r.group, r.cdr, r.mmse, r.mmse_field_present) for r in rows}
        conflict = len(values) != 1
        issues = set(issue for r in rows for issue in _visit_issues(r))
        if conflict:
            issues.add("conflicting_values_within_visit")
        linked = all(
            r.linkage_status == "matched" or (r.linkage_status == "unknown" and externally_verified)
            for r in rows
        )
        if not linked:
            issues.add("linkage_not_verified_matched")
        if any(r.image_path in ambiguous_paths for r in rows):
            issues.add("image_linked_to_multiple_visits")
            linked = False
        if not conflict and first.group in GROUPS:
            groups[first.group] += 1
            group_subjects[first.group].add(first.subject_id)
            cdr_key = "missing" if first.cdr is None else str(float(first.cdr))
            crosstab[first.group][cdr_key] += 1
            if first.group == "Converted":
                converted_cdr[cdr_key] += 1
                issues.add("converted_visit_requires_longitudinal_review")
        if (
            first.group == "Nondemented"
            and first.cdr == 0
            and linked
            and not (issues - {"missing_mmse", "mmse_not_exported"})
        ):
            candidates.add(key)
        review = evidence.get(key, DiagnosisEvidence())
        review_counts[review.manual_review_status] += 1
        a_label, c_label, evidence_reasons = _evidence_labels(
            review,
            conflicts=conflict or "group_cdr_conflict" in issues,
        )
        if (
            not linked
            or conflict
            or issues & {"unknown_clinical_group", "invalid_cdr", "missing_cdr", "invalid_mmse"}
        ):
            a_label, c_label = None, None
        reasons.update(issues)
        reasons.update(evidence_reasons)
        if a_label is not None:
            annotation_counts[f"review_supported_task_a_{a_label}_visits"] += 1
        if c_label is not None:
            annotation_counts[f"review_supported_task_c_{c_label}_visits"] += 1
        for row in rows:
            # Compare rather than overwrite labels in any old linkage output.
            if (row.declared_task_a_label is not None and row.declared_task_a_label != a_label) or (
                row.declared_task_c_label is not None and row.declared_task_c_label != c_label
            ):
                reasons["existing_label_lacks_matching_verified_evidence_images"] += 1
            record = _manifest_row(row, a_label, c_label, review)
            records.append(record)
            # No access/native-T1/MRI-normality authorization is invented here.
            reviews[record_key(record)] = RecordReview()

    collection = ManifestCollection(records)
    manager = SplitManager()
    split_report = manager.check_leakage(collection)
    # Reuse all existing Task A gates in metadata-only mode; never read MRI files.
    task_a = audit_readiness(collection, reviews, image_root=Path("/"), check_images=False)
    schema_errors = sum(bool(validate_record(r, strict=False)) for r in records)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        subject_splits[record.subject_id].add(record.split)
    # Existing manager checks train/validation/test. Count unknown assignments and
    # extend its subject grouping to holdout without exposing any identifiers.
    leaking_subjects = sum(len(s - {"unassigned"}) > 1 for s in subject_splits.values())
    partial_subjects = sum("unassigned" in s for s in subject_splits.values())
    review_counts["unused_evidence_visits"] = len(set(evidence) - set(visits))
    return {
        "policy_version": POLICY_VERSION,
        "mode": "clinical_metadata_dry_run_no_mri_access",
        "counts": {
            "image_records": len(images),
            "repeated_image_path_records": len(images) - len(image_visits),
            "unique_subjects": len(subject_splits),
            "unique_subject_visit_keys": len(visits),
            "linkage_status_images": dict(
                sorted(Counter(r.linkage_status for r in images).items())
            ),
            "image_paths_linked_to_multiple_visits": len(ambiguous_paths),
            "group_sessions": dict(sorted(groups.items())),
            "group_subjects": {g: len(s) for g, s in sorted(group_subjects.items())},
            "group_cdr_sessions": {g: dict(sorted(c.items())) for g, c in sorted(crosstab.items())},
            "converted_cdr_sessions": dict(sorted(converted_cdr.items())),
            "cognitive_comparison_candidate_sessions": len(candidates),
            "cognitive_comparison_candidate_subjects": len({k[0] for k in candidates}),
            "review_supported_annotations": dict(sorted(annotation_counts.items())),
            "review_status_visits": dict(sorted(review_counts.items())),
            "warnings_and_review_reasons": dict(sorted(reasons.items())),
        },
        "external_linkage_attestation": {
            "supplied": linkage_attestation is not None,
            "counts_consistent": externally_verified if linkage_attestation else None,
            "scope": "prior report attestation, not a new linkage or clinical diagnosis",
        },
        "split_audit": {
            "all_subjects_single_assigned_split": bool(images)
            and not (leaking_subjects or partial_subjects or split_report.violations),
            "subjects_in_multiple_splits": leaking_subjects,
            "subjects_with_unassigned_images": partial_subjects,
            "existing_split_manager_violation_count": len(split_report.violations),
            "split_image_counts": dict(sorted(Counter(r.split for r in records).items())),
        },
        "main_integration": {
            "source_id": "oasis2",
            "registered": "oasis2" in source_inventory(),
            "records_failing_existing_schema": schema_errors,
            "task_a_ready": task_a["ready"],
            "task_a_blockers": task_a["blockers"],
            "final_task_a_or_c_release_ready": False,
        },
        "limitations": [
            "Cognitive comparison candidates are not verified neurologically normal MRI controls.",
            "Group, CDR and MMSE do not establish Alzheimer's or MCI diagnoses.",
            "Manual evidence is an attestation, not an independently confirmed clinical diagnosis.",
            "OASIS-2 registration, access, MRI quality and label approval remain unverified.",
        ],
    }


def _lookup(item: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = item
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_clinical_images(
    path: Path,
    *,
    field_map: dict[str, str] | None = None,
    records_key: str = "records",
) -> tuple[list[ClinicalImage], str]:
    """Load only the named JSON metadata file, never Excel or referenced images.

    Explicit dot-separated field paths support private schemas without guessing.
    Numeric string conversion, fuzzy linkage and diagnosis mapping are not done.
    """
    raw = path.read_bytes()
    data = json.loads(raw)
    rows = data if isinstance(data, list) else _lookup(data, records_key)
    if not isinstance(rows, list):
        raise ValueError("manifest_requires_record_list")
    fields = {
        name: name for name in ClinicalImage.__dataclass_fields__ if name != "mmse_field_present"
    }
    fields.update(OASIS2_FIELD_MAP)
    if field_map:
        if set(field_map) - set(fields) or any(
            not isinstance(v, str) or not v for v in field_map.values()
        ):
            raise ValueError("invalid_field_map")
        fields.update(field_map)
    images = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("manifest_row_requires_object")
        defaults = {"linkage_status": "unknown", "split": "unassigned"}
        values = {name: _lookup(item, field, defaults.get(name)) for name, field in fields.items()}
        sentinel = object()
        values["mmse_field_present"] = _lookup(item, fields["mmse"], sentinel) is not sentinel
        images.append(ClinicalImage(**values))
    return images, hashlib.sha256(raw).hexdigest()


def load_diagnosis_evidence(path: Path) -> dict[VisitKey, DiagnosisEvidence]:
    items = json.loads(path.read_bytes())
    if not isinstance(items, list):
        raise ValueError("evidence_requires_list")
    result = {}
    for item in items:
        item = dict(item)
        key = (item.pop("subject_id"), item.pop("mri_visit"))
        if any(not isinstance(k, str) or not k.strip() for k in key) or key in result:
            raise ValueError("invalid_or_duplicate_visit_evidence")
        result[key] = DiagnosisEvidence(**item)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--field-map", type=Path)
    parser.add_argument("--records-key", default="records")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--linkage-attestation", type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.field_map.read_bytes()) if args.field_map else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("invalid_field_map")
        images, digest = load_clinical_images(
            args.manifest,
            field_map=mapping,
            records_key=args.records_key,
        )
        evidence = load_diagnosis_evidence(args.evidence) if args.evidence else {}
        linkage = (
            LinkageAttestation(**json.loads(args.linkage_attestation.read_bytes()))
            if args.linkage_attestation
            else None
        )
        report = audit_clinical_manifest(images, evidence, linkage_attestation=linkage)
        report["input_sha256"] = digest
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Never echo exception text, paths, raw values, subject IDs or visit keys.
        print(json.dumps({"status": "invalid_input", "final_label_release_ready": False}))
        return 2
    return (
        0
        if report["split_audit"]["all_subjects_single_assigned_split"]
        and not report["counts"]["repeated_image_path_records"]
        and not any(
            report["counts"]["warnings_and_review_reasons"].get(code, 0)
            for code in (
                "group_cdr_conflict",
                "conflicting_values_within_visit",
                "linkage_not_verified_matched",
                "image_linked_to_multiple_visits",
                "unknown_clinical_group",
                "invalid_cdr",
                "invalid_mmse",
                "existing_label_lacks_matching_verified_evidence_images",
                "missing_cdr",
                "external_linkage_attestation_count_mismatch",
                "dataset_identity_not_verified",
            )
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
