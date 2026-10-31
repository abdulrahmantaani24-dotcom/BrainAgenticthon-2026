"""Private bounded inspection of four explicitly configured sources; no eligibility overrides."""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from brainagenticthon.detection.training import private_output_directory
from brainagenticthon.execution.inventory import read_json
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_path

SOURCE_NAMES = ("hwang_meningioma_5945963", "isles2022", "oasis2_legacy", "proteas_23171182")
REVIEW_FLAGS = (
    "access_verified",
    "license_verified",
    "labels_verified",
    "masks_verified",
    "modality_verified",
    "identity_verified",
    "conversion_verified",
    "near_duplicates_reviewed",
    "visual_qc_verified",
    "native_t1_verified",
    "whole_brain_verified",
)
DICOM_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
]
EVIDENCE_KEYS = (
    "license",
    "source",
    "provenance",
    "modality",
    "sequence",
    "diagnos",
    "label",
    "mask",
    "segment",
    "patient",
    "subject",
    "case",
    "session",
    "duplicate",
    "conversion",
    "dcm2niix",
    "scanner",
    "site",
)


def _number(value):
    return float(value) if np.isfinite(value) else None


def image_header(path):
    image = nib.load(path)  # Header only, including Analyze pairs; never get_fdata/dataobj.
    affine = image.affine
    header = {
        "shape": list(image.shape),
        "spacing": [_number(v) for v in image.header.get_zooms()],
        "orientation": list(nib.aff2axcodes(affine)),
        "affine": [[_number(v) for v in row] for row in affine],
        "units": (
            image.header.get_xyzt_units()[0]
            if hasattr(image.header, "get_xyzt_units")
            else "unknown"
        ),
        "coded_nifti_geometry": False,
    }
    if hasattr(image, "get_qform"):
        q, qc = image.get_qform(coded=True)
        s, sc = image.get_sform(coded=True)
        header["coded_nifti_geometry"] = bool(qc or sc)
        header["transforms_agree"] = bool(not (qc and sc) or np.allclose(q, s, atol=1e-4))
    header["technical_geometry_valid"] = bool(
        len(image.shape) in (3, 4)
        and np.prod(image.shape) <= 32_000_000
        and np.isfinite(affine).all()
        and abs(np.linalg.det(affine[:3, :3])) > 1e-8
        and all(v is not None and v > 0 for v in header["spacing"])
    )
    return header


def dicom_header(path):
    import pydicom

    data = pydicom.dcmread(
        path, stop_before_pixels=True, specific_tags=DICOM_TAGS, defer_size=1024, force=False
    )
    # No pixel_array, decompression, conversion or DICOM writes.
    identities = {
        key: hashlib.sha256(str(data.get(key)).encode()).hexdigest()
        for key in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        if data.get(key) is not None
    }
    return {
        "identity_hashes": identities,
        "documented_modality": str(data.get("Modality", "")),
        "sequence_description_candidate": str(data.get("SeriesDescription", ""))[:256],
        "protocol_candidate": str(data.get("ProtocolName", ""))[:256],
        "rows": int(data.get("Rows", 0)),
        "columns": int(data.get("Columns", 0)),
        "pixel_spacing_as_documented": [_number(float(v)) for v in data.get("PixelSpacing", [])],
        "orientation_as_documented": [
            _number(float(v)) for v in data.get("ImageOrientationPatient", [])
        ],
        "position_as_documented": [_number(float(v)) for v in data.get("ImagePositionPatient", [])],
        "volume_affine_reconstructed": False,
        "pixel_decoded": False,
        "sequence_identity_verified": False,
    }


def manifest_diagnostics(rows):
    """Diagnose explicit documented IDs/splits only; do not infer or reconcile aliases."""
    patients, cases, paths, hashes, sources = {}, {}, {}, {}, {}
    blockers = set()
    sampled = 0
    for row in rows[:5000]:
        if not isinstance(row, dict):
            continue
        sampled += 1
        patient = row.get("patient_id") or row.get("subject_id")
        case = row.get("case_id")
        split = row.get("split")
        if not isinstance(patient, (str, int)) or not patient:
            continue
        if split in ("train", "validation", "test", "holdout"):
            patients.setdefault(str(patient), set()).add(split)
            if len(patients[str(patient)]) > 1:
                blockers.add("documented_patient_split_leakage")
        if case:
            prior = cases.setdefault(str(case), str(patient))
            if prior != str(patient):
                blockers.add("documented_case_identity_conflict")
        for key, mapping in (("image_path", paths), ("sha256", hashes)):
            value = row.get(key)
            if isinstance(value, str) and value:
                seen = mapping.setdefault(value, (str(patient), split))
                if seen != (str(patient), split):
                    blockers.add("documented_duplicate_volume_or_hash_identity_split_conflict")
        label, source = row.get("task_a_label"), row.get("dataset_source")
        if type(label) is int and label in (0, 1) and isinstance(source, str):
            sources.setdefault(source, set()).add(label)
    if len(sources) > 1 and any(len(labels) == 1 for labels in sources.values()):
        blockers.add("documented_source_class_confounding_requires_review")
    return {
        "rows_sampled": sampled,
        "diagnostic_blockers": sorted(blockers),
        "aliases_reconciled": False,
        "clinical_labels_verified": False,
        "sampling_may_omit_other_records": len(rows) > 5000,
    }


def metadata_summary(path):
    """Bounded private snippets; documents are evidence candidates, never approvals."""
    if path.stat().st_size > 4 * 1024 * 1024:
        return {"status": "SKIPPED", "reason": "metadata_byte_limit"}
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            rows = []
            for index, row in enumerate(reader):
                if index >= 5000:
                    return {
                        "fields": fields,
                        "rows_sampled": len(rows),
                        "truncated": True,
                        "rows_private": rows,
                        "manifest_diagnostics": manifest_diagnostics(rows),
                    }
                rows.append(row)
        return {
            "fields": fields,
            "rows_sampled": len(rows),
            "rows_private": rows,
            "truncated": False,
            "manifest_diagnostics": manifest_diagnostics(rows),
        }
    value = read_json(path, 4 * 1024 * 1024)
    selected, visited = [], 0

    def walk(node, field="", depth=0):
        nonlocal visited
        visited += 1
        if visited > 10000 or depth > 10 or len(selected) >= 200:
            return
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, field + "." + str(key), depth + 1)
        elif isinstance(node, list):
            for index, item in enumerate(node[:5000]):
                walk(item, field + f"[{index}]", depth + 1)
        elif any(term in field.lower() for term in EVIDENCE_KEYS):
            selected.append({"field": field, "documented_value_candidate": str(node)[:512]})

    walk(value)
    return {
        "documented_candidates_private": selected,
        "nodes_sampled": visited,
        "not_clinical_or_access_approval": True,
        "manifest_diagnostics": manifest_diagnostics(
            value
            if isinstance(value, list)
            else value.get("records", [])
            if isinstance(value, dict)
            else []
        ),
    }


def reviewed_evidence(root, filename):
    flags = {key: False for key in REVIEW_FLAGS}
    result = {"flags": flags, "references": {}, "blockers": ["independent_source_review_missing"]}
    path = private_path(root, filename)
    if not path.is_file():
        return result
    review = read_json(path, 4 * 1024 * 1024)
    if (
        review.get("schema") != "disease-source-review-v1"
        or review.get("source_id") != root.name
        or review.get("review_status") != "approved"
    ):
        result["blockers"] = ["source_review_pending_or_identity_mismatch"]
        return result
    result["blockers"] = []
    for flag in REVIEW_FLAGS:
        evidence = review.get("evidence", {}).get(flag, {})
        ref = evidence.get("reference", "")
        valid = review.get(flag) is True and isinstance(ref, str) and bool(ref.strip())
        if evidence.get("relative_path"):
            target = private_path(root, evidence["relative_path"])
            valid = valid and target.is_file() and target.stat().st_size <= 256 * 1024 * 1024
            valid = valid and digest_file(target) == evidence.get("sha256")
        flags[flag] = bool(valid)
        result["references"][flag] = ref
    result["diagnosis_semantics_private"] = review.get("diagnosis_semantics", {})
    result["mask_semantics_private"] = review.get("mask_semantics", {})
    pairs = []
    rows = review.get("reviewed_pairs", [])
    if len(rows) > 1000:
        result["blockers"].append("reviewed_pair_count_limit")
        rows = []
    if flags["masks_verified"] and flags["modality_verified"]:
        from brainagenticthon.execution.annotation_review import inspect_pair

        patients, cases, image_hashes = {}, set(), set()
        for index, row in enumerate(rows):
            try:
                checked = inspect_pair(root, row)
                reasons = []
                if row["case_id"] in cases:
                    reasons.append("duplicate_case_identity")
                cases.add(row["case_id"])
                patient = row["patient_id"]
                if row.get("split") not in {"train", "validation", "test"}:
                    reasons.append("patient_split_not_frozen")
                if patient in patients and patients[patient] != row.get("split"):
                    reasons.append("patient_split_leakage")
                patients[patient] = row.get("split")
                for image in row["images"].values():
                    if image["sha256"] in image_hashes:
                        reasons.append("duplicate_image_bytes")
                    image_hashes.add(image["sha256"])
                checked["identity_split_blockers"] = reasons
                pairs.append({"record_index": index, **checked})
            except Exception:
                pairs.append(
                    {
                        "record_index": index,
                        "status": "BLOCKED",
                        "reason": "reviewed_pair_mask_geometry_identity_or_provenance_failed",
                    }
                )
    result["reviewed_pair_checks"] = pairs
    return result


def eligibility_matrix(name, review, *, root_observed):
    flags = review["flags"]
    common = [
        flag + "_missing"
        for flag in (
            "access_verified",
            "license_verified",
            "modality_verified",
            "identity_verified",
            "near_duplicates_reviewed",
        )
        if not flags[flag]
    ]
    if not root_observed:
        common.append("configured_root_not_observed")
    return {
        "source": name,
        "task_a": {
            "classification": "excluded_with_reason",
            "research_candidate": "pending_evidence",
            "training_status": "BLOCKED",
            "blockers": [
                *common,
                "source_not_approved_by_current_Task_A_registry",
                *[
                    flag + "_missing"
                    for flag in (
                        "labels_verified",
                        "native_t1_verified",
                        "whole_brain_verified",
                        "visual_qc_verified",
                    )
                    if not flags[flag]
                ],
                "eligible_Normal_Abnormal_patient_splits_and_confounding_review_required",
            ],
        },
        "task_b": {
            "classification": "candidate_pending_masks_semantics_and_reviewed_adapter",
            "training_status": "BLOCKED",
            "blockers": [
                *common,
                *(
                    ["masks_and_disease_specific_semantics_unverified"]
                    if not flags["masks_verified"]
                    else []
                ),
                "current_training_profile_is_BraTS2021_only",
                "non_BraTS_training_adapter_not_independently_reviewed",
            ],
        },
        "auxiliary_pretraining": "not_approved_or_inferred",
        "no_patient_or_diagnosis_count_inferred_from_files": True,
    }


def scan_source(parent, source, limits):
    name = source["directory"]
    if name not in SOURCE_NAMES:
        raise ValueError("only_four_explicit_dataset_directories_supported")
    root = private_path(parent, name)
    result = {
        "source": name,
        "status": "BLOCKED",
        "root_observed": False,
        "files_private": [],
        "metadata_private": [],
        "blockers": [],
        "clinical_approval": False,
    }
    review = {"flags": {key: False for key in REVIEW_FLAGS}, "references": {}, "blockers": []}
    if source.get("inspection_authorized") is not True:
        result["blockers"] = ["local_read_only_inspection_not_authorized"]
    elif not root.is_dir():
        result["blockers"] = ["configured_root_unavailable"]
    else:
        result["root_observed"] = True
        files, formats, directory_count = 0, Counter(), 0
        header_count, dicom_count, metadata_count, hash_bytes = 0, 0, 0, 0
        hashes, sop_patients, local_patient_studies = {}, {}, {}
        try:
            for directory, folders, names in os.walk(root, followlinks=False):
                directory_count += 1
                if directory_count > limits["max_directories"]:
                    raise ValueError("directory_limit_reached")
                if any((Path(directory) / n).is_symlink() for n in (*folders, *names)):
                    raise ValueError("symlink_rejected")
                for filename in sorted(names):
                    files += 1
                    if files > limits["max_files"]:
                        raise ValueError("file_limit_reached")
                    path = Path(directory) / filename
                    ext = ".nii.gz" if filename.lower().endswith(".nii.gz") else path.suffix.lower()
                    public_ext = (
                        ext
                        if ext
                        in {
                            ".nii",
                            ".nii.gz",
                            ".hdr",
                            ".img",
                            ".dcm",
                            ".json",
                            ".csv",
                            ".zip",
                            ".png",
                            ".txt",
                            ".tsv",
                            ".bval",
                            ".bvec",
                        }
                        else ("<no_extension>" if not ext else "<other>")
                    )
                    formats[public_ext] += 1
                    item = {
                        "relative_path": str(path.relative_to(root)),
                        "format": ext,
                        "bytes": path.stat().st_size,
                    }
                    if ext in {".nii", ".nii.gz", ".hdr", ".img"}:
                        if (
                            header_count < limits["max_image_headers"]
                            and item["bytes"] <= limits["max_file_bytes"]
                        ):
                            try:
                                item["header"] = image_header(path)
                            except Exception:
                                item["header_error"] = "unreadable_or_missing_companion"
                            header_count += 1
                        else:
                            item["header_skipped"] = "resource_limit"
                        if (
                            item["bytes"] <= limits["max_file_bytes"]
                            and hash_bytes + item["bytes"] <= limits["max_hash_bytes"]
                        ):
                            value = digest_file(path)
                            item["sha256"] = value
                            hash_bytes += item["bytes"]
                            if ext != ".hdr":
                                item["duplicate_bytes_of"] = hashes.get(value)
                                hashes.setdefault(value, item["relative_path"])
                    elif ext == ".dcm":
                        if (
                            dicom_count < limits["max_dicom_headers"]
                            and item["bytes"] <= limits["max_file_bytes"]
                        ):
                            try:
                                header = dicom_header(path)
                                item["header"] = header
                                ids = header["identity_hashes"]
                                sop, patient = ids.get("SOPInstanceUID"), ids.get("PatientID")
                                if sop and patient:
                                    if sop in sop_patients and sop_patients[sop] != patient:
                                        item["identity_blocker"] = (
                                            "SOP_assigned_to_multiple_patient_ids"
                                        )
                                    sop_patients[sop] = patient
                                if patient and ids.get("StudyInstanceUID"):
                                    local_patient_studies.setdefault(patient, set()).add(
                                        ids["StudyInstanceUID"]
                                    )
                            except ImportError:
                                item["header_error"] = "optional_pydicom_not_installed"
                            except Exception:
                                item["header_error"] = "invalid_or_unsupported_DICOM_header"
                            dicom_count += 1
                        else:
                            item["header_skipped"] = "DICOM_sample_limit_no_pixel_decode"
                    elif ext in {".json", ".csv"} and metadata_count < limits["max_metadata_files"]:
                        try:
                            result["metadata_private"].append(
                                {
                                    "relative_path": item["relative_path"],
                                    "inspection": metadata_summary(path),
                                }
                            )
                        except Exception:
                            item["metadata_error"] = "unreadable_or_invalid_metadata"
                        metadata_count += 1
                    # Archives and PNGs are counted, never extracted or interpreted as MRI/masks.
                    result["files_private"].append(item)
            result["status"] = "INVENTORIED"
        except (ValueError, OSError):
            result["blockers"].append("inventory_incomplete_limit_symlink_or_IO_failure")
        result["coverage"] = {
            "files_counted": min(files, limits["max_files"]),
            "format_counts": dict(formats),
            "image_headers_sampled": header_count,
            "DICOM_headers_sampled": dicom_count,
            "metadata_files_sampled": metadata_count,
            "bytes_hashed": hash_bytes,
            "coverage_is_not_complete_clinical_or_patient_validation": True,
            "cross_source_aliases_and_near_duplicates": "not_automatically_reconciled",
        }
        try:
            review = reviewed_evidence(root, source.get("review_file", "task_ab_review.json"))
        except Exception:
            result["blockers"].append("source_review_invalid_or_unreadable")
    result["review_private"] = review
    result["eligibility"] = eligibility_matrix(name, review, root_observed=result["root_observed"])
    return result


DEFAULT_LIMITS = {
    "max_files": 100000,
    "max_directories": 100000,
    "max_file_bytes": 268435456,
    "max_image_headers": 4000,
    "max_dicom_headers": 256,
    "max_metadata_files": 64,
    "max_hash_bytes": 536870912,
}


def inspect_sources(config, output):
    parent = Path(config["dataset_parent"]).absolute()
    if any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError("symlink_parent_rejected")
    sources = config["sources"]
    if len(sources) != 4 or {row["directory"] for row in sources} != set(SOURCE_NAMES):
        raise ValueError("configure_exactly_the_four_named_sources")
    limits = {**DEFAULT_LIMITS, **config.get("limits", {})}
    if set(limits) != set(DEFAULT_LIMITS) or any(
        type(v) is not int or not 0 < v <= 2_147_483_648 for v in limits.values()
    ):
        raise ValueError("invalid_inventory_limits")
    if (
        limits["max_files"] > 200000
        or limits["max_directories"] > 200000
        or limits["max_dicom_headers"] > 4096
        or limits["max_image_headers"] > 10000
        or limits["max_metadata_files"] > 512
        or limits["max_file_bytes"] > 268435456
    ):
        raise ValueError("inventory_enumeration_limit_too_large")
    output = private_output_directory(output, (parent,))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "four-source-private-inventory-v1",
        "clinical_approval": False,
        "training_started": False,
        "BraTS_directory_search": "not_requested_or_performed",
        "sources": [],
    }
    for source in sources:
        report["sources"].append(scan_source(parent, source, limits))
    with (output / "inventory.private.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    summary = {
        "schema": report["schema"],
        "clinical_approval": False,
        "training_started": False,
        "sources": [
            {
                "source": row["source"],
                "inventory_status": row["status"],
                "root_observed": row["root_observed"],
                "coverage": row.get("coverage"),
                "eligibility": row["eligibility"],
            }
            for row in report["sources"]
        ],
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary
