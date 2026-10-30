"""Read-only Task A data audit. No fetching, label assignment, transforms or splits.

Reviewer attestations supplement, and never override, the registry or manifest
schema. ``ready`` applies only to the eligible development subset, not a model,
clinical use, source authorization, or statistical adequacy of a study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brainagenticthon.preprocessing.dataset_registry import APPROVED_SOURCES
from brainagenticthon.preprocessing.manifest_schema import (
    ManifestCollection,
    ManifestRecord,
    validate_record,
)
from brainagenticthon.preprocessing.split_manager import SplitManager

AUDIT_VERSION = "1.0.0"
REQUIRED_SPLITS = ("train", "validation", "test")
EVALUATION_SPLITS = (*REQUIRED_SPLITS, "holdout")
NATIVE_T1 = frozenset({"t1", "t1w", "t1n"})
NORMAL_DIAGNOSES = frozenset({"normal", "healthy control", "control", "cn", "no finding"})
MISSING_DIAGNOSES = frozenset({"", "unknown", "missing", "none", "n/a", "na", "null"})
RecordKey = tuple[str, str, str]


@dataclass(frozen=True)
class RecordReview:
    """Secure reviewer evidence, NOT an inferred diagnosis or source approval.

    patient_id is a globally reconciled pseudonym across sources and sessions.
    label_verified refers to the explicit task_a_label and its original clinical
    or expert annotation. healthy_control_verified additionally requires Task A
    normality review: cognitive CN status alone does not establish MRI normality.
    """

    patient_id: str = ""
    evidence_reference: str = ""
    access_authorized: bool = False
    patient_identity_verified: bool = False
    label_verified: bool = False
    healthy_control_verified: bool = False
    native_t1_verified: bool = False
    whole_brain_verified: bool = False
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                name.endswith("verified") or name == "access_authorized"
            ) and type(value) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        if not isinstance(self.patient_id, str) or not isinstance(self.evidence_reference, str):
            raise ValueError("patient_id and evidence_reference must be strings")
        if self.expected_sha256 is not None and (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")


def record_key(record: ManifestRecord) -> RecordKey:
    """Link evidence to an exact manifest row without rewriting its image path."""
    return record.dataset_source, record.case_id, record.image_path


def merge_manifests(*collections: ManifestCollection) -> ManifestCollection:
    """Virtual union; retain duplicates for auditing, preserve raw locations."""
    return ManifestCollection([r for collection in collections for r in collection.records])


def source_inventory() -> dict[str, Any]:
    """Catalogue state only: a registry entry does not establish granted access."""
    return {
        source: {
            "task_a_registry_eligible": meta.task_a_eligible,
            "image_dimension": meta.image_dimension.value,
            "training_role": meta.training_role.value,
            "modalities": sorted(meta.modalities),
            "license_requirement": meta.license_type.value,
            "access_url": meta.access_url,
            "access_requirements": meta.access_instructions,
            "raw_subdir": meta.raw_subdir,
            "keep_separate": meta.keep_separate,
            "access_state": "not_verified_by_catalogue",
        }
        for source, meta in sorted(APPROVED_SOURCES.items())
    }


def _image_checks(path: Path, expected: str | None) -> dict[str, Any]:
    # Lazy imports keep metadata-only auditing available without imaging extras.
    import nibabel as nib
    import numpy as np

    from brainagenticthon.preprocessing.data_validator import NIfTIValidator

    try:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            return {"reasons": ["image_not_3d"], "warnings": []}
        if any(d < 10 for d in image.shape):
            return {"reasons": ["invalid_image_shape"], "warnings": []}
        if math.prod(image.shape) > 64_000_000:
            return {"reasons": ["image_exceeds_audit_voxel_limit"], "warnings": []}
        validation = NIfTIValidator().validate_all(path)
        reasons = ["image:" + r.check_name for r in validation.errors]
        warnings = ["image:" + r.check_name for r in validation.warnings]
        # RAS mismatch is informative for native images; no reorientation occurs.
        reasons.extend(w for w in warnings if w != "image:orientation.match")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if qcode and scode and not np.allclose(qform, sform, atol=1e-4):
            reasons.append("conflicting_qform_sform")
        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(data).all():
            reasons.append("nonfinite_image")
        elif float(np.max(data)) == float(np.min(data)):
            reasons.append("constant_image")
        digest = next(
            (r.details["hash"] for r in validation.results if r.check_name == "integrity.computed"),
            None,
        )
        if expected is not None and digest != expected:
            reasons.append("image_hash_mismatch")
        return {
            "reasons": sorted(set(reasons)),
            "warnings": sorted(set(warnings)),
            "sha256": digest,
            "shape": list(image.shape),
            "spacing_mm": [float(x) if np.isfinite(x) else None
                           for x in image.header.get_zooms()],
            "orientation": "".join(nib.aff2axcodes(image.affine)) if not reasons else None,
        }
    except (OSError, ValueError, RuntimeError, TypeError, nib.filebasedimages.ImageFileError):
        return {"reasons": ["image_unreadable_or_invalid"], "warnings": []}


def _row_audit(record: ManifestRecord, review: RecordReview) -> dict[str, Any]:
    reasons = ["manifest:" + error for error in validate_record(record, strict=False)]
    warnings = []
    meta = APPROVED_SOURCES.get(record.dataset_source)
    if meta is None or not meta.task_a_eligible:
        reasons.append("source_not_task_a_approved")
    if meta is not None and meta.training_role.value == "pretrain_only":
        reasons.append("pretrain_only_source")
    if record.image_dimension != "3D":
        reasons.append("declared_image_not_3d")
    if (record.modality or "").strip().lower() not in NATIVE_T1:
        reasons.append("not_native_t1_modality")
    if not review.native_t1_verified:
        reasons.append("native_t1_not_verified")
    if not review.whole_brain_verified:
        reasons.append("whole_brain_not_verified")
    if record.quality_flag != "ok":
        reasons.append("quality_not_reviewed_ok")
    if not review.access_authorized or (record.license_status or "").lower() in {
        "", "unknown", "n/a", "missing"
    }:
        reasons.append("access_or_license_not_verified")
    if meta is not None and meta.license_type.value != "unknown" and (
        record.license_status != meta.license_type.value
    ):
        reasons.append("license_does_not_match_source_requirement")
    if not review.evidence_reference.strip():
        reasons.append("missing_review_reference")
    if not review.patient_id.strip() or not review.patient_identity_verified:
        reasons.append("patient_identity_not_verified")
    diagnosis = (record.original_diagnosis or "").strip().lower()
    if diagnosis in MISSING_DIAGNOSES:
        reasons.append("missing_diagnosis")
        warnings.append("missing_label_never_assume_normal")
    if type(record.task_a_label) is not int or record.task_a_label not in (0, 1):
        reasons.append("missing_or_invalid_task_a_label")
        warnings.append("missing_label_never_assume_normal")
    if record.label_provenance not in {"clinical_record", "challenge_annotation"}:
        reasons.append("unverified_label_provenance")
    if not review.label_verified:
        reasons.append("task_a_label_not_verified")
    if record.task_a_label == 0:
        if diagnosis not in NORMAL_DIAGNOSES or not review.healthy_control_verified:
            reasons.append("healthy_control_not_verified")
    elif record.task_a_label == 1 and diagnosis in NORMAL_DIAGNOSES:
        reasons.append("diagnosis_label_conflict")
    if record.split not in EVALUATION_SPLITS:
        reasons.append("split_not_assigned_for_evaluation")
    if not (record.site or "").strip():
        warnings.append("missing_site")
    if not (record.scanner or "").strip():
        warnings.append("missing_scanner")
    return {
        "source": record.dataset_source,
        "subject_id": record.subject_id,
        "case_id": record.case_id,
        "patient_id": review.patient_id if review.patient_identity_verified else "",
        "label": record.task_a_label,
        "split": record.split,
        "site": (record.site or "").strip(),
        "scanner": (record.scanner or "").strip(),
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
    }


def _crosstab(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    table: dict[str, dict[str, int]] = {}
    seen = set()
    for row in rows:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            continue
        group = json.dumps([row[f] or "<unknown>" for f in fields])
        patient = row["patient_id"] or (row["source"], row["subject_id"])
        identity = (group, patient, row["label"])
        if identity not in seen:
            seen.add(identity)
            table.setdefault(group, {"normal": 0, "abnormal": 0})
            table[group]["normal" if row["label"] == 0 else "abnormal"] += 1
    return dict(sorted(table.items()))


def _confounded(table: dict[str, Any]) -> bool:
    labels = {label for counts in table.values() for label, count in counts.items() if count}
    return len(labels) == 2 and all(sum(v > 0 for v in c.values()) == 1 for c in table.values())


def audit_readiness(
    collection: ManifestCollection,
    reviews: dict[RecordKey, RecordReview],
    *,
    image_root: Path,
    check_images: bool = True,
) -> dict[str, Any]:
    """Read metadata/images only; never mutate records, labels, paths or splits.

    Readiness is conservative: both classes in train/validation/test, complete
    reviewer evidence, actual image checks, no leakage/duplicates, and no perfect
    source/site-class confounding. No schema or source eligibility override exists.
    """
    records = sorted(collection.records, key=lambda r: (*record_key(r), r.modality))
    rows = []
    paths: dict[str, list[int]] = defaultdict(list)
    digests: dict[str, list[int]] = defaultdict(list)
    subject_patients: dict[tuple[str, str], set[str]] = defaultdict(set)
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_labels: dict[str, set[int]] = defaultdict(set)
    blockers = set()
    root = image_root.resolve()
    for record in records:
        review = reviews.get(record_key(record), RecordReview())
        row = _row_audit(record, review)
        if row["patient_id"]:
            patient = row["patient_id"]
            subject_patients[(record.dataset_source, record.subject_id)].add(patient)
            if record.split in EVALUATION_SPLITS:
                patient_splits[patient].add(record.split)
            if type(record.task_a_label) is int and record.task_a_label in (0, 1):
                patient_labels[patient].add(record.task_a_label)
        path = (root / record.image_path).resolve()
        if record.image_path:
            paths[str(path)].append(len(rows))
        # Only read files of otherwise eligible rows; never convert, move or fetch.
        if check_images and not row["reasons"]:
            image_result = _image_checks(path, review.expected_sha256)
            row["reasons"].extend(image_result.pop("reasons"))
            row["warnings"].extend(image_result.pop("warnings"))
            row["image"] = image_result
            if image_result.get("sha256"):
                digests[image_result["sha256"]].append(len(rows))
        row["eligible"] = not row["reasons"]
        rows.append(row)

    manager = SplitManager()
    leakage = manager.check_leakage(collection)
    if not leakage.is_clean:
        blockers.add("existing_split_or_manifest_violations")
    if any(len(splits) > 1 for splits in patient_splits.values()):
        blockers.add("global_patient_split_leakage")
    if any(len(ids) > 1 for ids in subject_patients.values()):
        blockers.add("inconsistent_patient_identity")
    if any(len(labels) > 1 for labels in patient_labels.values()):
        blockers.add("conflicting_patient_labels_require_review")
    for groups, reason in ((paths, "duplicate_image_path"), (digests, "duplicate_image_bytes")):
        for indices in groups.values():
            if len(indices) > 1:
                blockers.add(reason)
                for index in indices:
                    rows[index]["reasons"].append(reason)
                    rows[index]["eligible"] = False

    eligible = [row for row in rows if row["eligible"]]
    tables = {
        "declared_source_class": _crosstab(rows, ("source",)),
        "declared_site_class": _crosstab(rows, ("source", "site")),
        "eligible_source_class": _crosstab(eligible, ("source",)),
        "eligible_site_class": _crosstab(eligible, ("source", "site")),
        "eligible_split_class": _crosstab(eligible, ("split",)),
        "eligible_source_site_split_class": _crosstab(eligible, ("source", "site", "split")),
    }
    confounding = {name: _confounded(table) for name, table in tables.items()
                   if "source_class" in name or "site_class" in name}
    if any(value for name, value in confounding.items() if "source_class" in name):
        blockers.add("source_perfectly_predicts_class")
    if any(value for name, value in confounding.items() if "site_class" in name):
        blockers.add("source_site_perfectly_predicts_class")
    if any(not row["site"] or not row["scanner"] for row in eligible):
        blockers.add("site_or_scanner_metadata_missing")
    if not any(row["label"] == 0 for row in eligible):
        blockers.add("missing_verified_healthy_controls")
    if not any(row["label"] == 1 for row in eligible):
        blockers.add("missing_verified_abnormal_cases")
    for split in REQUIRED_SPLITS:
        if {r["label"] for r in eligible if r["split"] == split} != {0, 1}:
            blockers.add(f"both_classes_required_in_{split}")
    if not check_images:
        blockers.add("image_checks_not_run")

    fingerprint = hashlib.sha256(
        json.dumps([r.to_dict() for r in records], sort_keys=True).encode()
    ).hexdigest()
    review_fingerprint = hashlib.sha256(json.dumps(
        [{"key": key, **asdict(reviews[key])} for key in sorted(reviews)], sort_keys=True,
    ).encode()).hexdigest()
    return {
        "audit_version": AUDIT_VERSION,
        "ready": not blockers,
        "assessment": "ready" if not blockers else "not_ready",
        "scope": "eligible subset for offline development; not clinical or model readiness",
        "manifest_sha256": fingerprint,
        "reviews_sha256": review_fingerprint,
        "eligibility_scope": "image_checked" if check_images else "metadata_candidates_only",
        "image_checks_requested": check_images,
        "image_checks_completed": sum("image" in row for row in rows),
        "required_splits": list(REQUIRED_SPLITS),
        "counts": {
            "records": len(rows), "eligible_records": len(eligible),
            "ineligible_records": len(rows) - len(eligible),
            "eligible_patients": len({r["patient_id"] for r in eligible}),
            "normal_patients": len({r["patient_id"] for r in eligible if r["label"] == 0}),
            "abnormal_patients": len({r["patient_id"] for r in eligible if r["label"] == 1}),
            "reasons": dict(sorted(Counter(r for row in rows for r in row["reasons"]).items())),
            "warnings": dict(sorted(Counter(w for row in rows for w in row["warnings"]).items())),
        },
        "blockers": sorted(blockers),
        "patient_split_leakage": sorted(p for p, s in patient_splits.items() if len(s) > 1),
        "existing_leakage_violations": sorted(leakage.violations),
        "cross_source_subject_id_overlaps": manager.find_cross_source_subject_overlaps(collection),
        "site_split_record_counts": manager.site_split_statistics(collection),
        "patient_class_crosstabs": tables,
        "perfect_confounding": confounding,
        "sources": source_inventory(),
        "records": rows,
    }


def load_reviews(path: Path) -> dict[RecordKey, RecordReview]:
    """Load a separate JSON evidence list; reject ambiguous duplicate row keys."""
    result = {}
    items = json.loads(path.read_text())
    if not isinstance(items, list):
        raise ValueError("Reviews must be a JSON list")
    for item in items:
        item = dict(item)
        key = tuple(item.pop(field) for field in ("dataset_source", "case_id", "image_path"))
        if len(key) != 3 or any(not isinstance(value, str) or not value for value in key):
            raise ValueError("Review row keys must be three nonempty strings")
        if key in result:
            raise ValueError("Duplicate review row key")
        result[key] = RecordReview(**item)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.list_sources:
        print(json.dumps(source_inventory(), indent=2, sort_keys=True))
        return 0
    if not args.manifest or args.reviews is None or args.image_root is None:
        parser.error("--manifest, --reviews and --image-root are required for an audit")
    try:
        collection = merge_manifests(*(ManifestCollection.load_json(p) for p in args.manifest))
        report = audit_readiness(
            collection, load_reviews(args.reviews), image_root=args.image_root,
            check_images=not args.metadata_only,
        )
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(json.dumps({"ready": False, "assessment": "audit_error",
                          "error_type": type(error).__name__}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
