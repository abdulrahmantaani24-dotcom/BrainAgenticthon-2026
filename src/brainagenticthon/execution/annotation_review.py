"""Read-only disease-specific annotation checks, not a non-BraTS training adapter."""

import nibabel as nib
import numpy as np

from brainagenticthon.execution.disease_sources import image_header
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_path


def inspect_pair(root, row):
    """Require independently documented mask values; never infer glioma subregions."""
    if (
        row.get("review_status") != "approved"
        or not row.get("patient_id")
        or not row.get("case_id")
        or not row.get("mask_semantics_reference")
        or not row.get("modality_reference")
        or not row.get("identity_reference")
    ):
        raise ValueError("reviewed_identity_modality_and_mask_semantics_required")
    if type(row.get("conversion_required")) is not bool:
        raise ValueError("explicit_conversion_applicability_required")
    if row["conversion_required"] and not row.get("conversion_reference"):
        raise ValueError("explicit_conversion_provenance_or_not_applicable_required")
    definitions = row.get("label_definitions", {})
    if (
        row.get("background_label") != 0
        or type(row.get("background_label")) is not int
        or not definitions
        or "0" not in definitions
        or any(
            not str(k).isdigit() or not isinstance(v, str) or not v.strip()
            for k, v in definitions.items()
        )
    ):
        raise ValueError("explicit_integer_label_definitions_including_background_required")
    images = row.get("images", {})
    if not images or not row.get("mask"):
        raise ValueError("reviewed_image_and_mask_mapping_required")
    affine, shape = None, None
    modalities = []
    for modality, item in [*images.items(), ("mask", row["mask"])]:
        path = private_path(root, item["path"])
        if path.stat().st_size > 256 * 1024 * 1024 or digest_file(path) != item["sha256"]:
            raise ValueError("reviewed_file_hash_or_resource_limit_failed")
        header = image_header(path)
        if (
            not header["technical_geometry_valid"]
            or len(header["shape"]) != 3
            or header["units"] != "mm"
            or not header["coded_nifti_geometry"]
            or not header.get("transforms_agree", True)
        ):
            raise ValueError("coded_3d_mm_geometry_required_for_annotation_pair")
        if shape is not None and (
            header["shape"] != shape or not np.allclose(header["affine"], affine, atol=1e-4)
        ):
            raise ValueError("image_mask_geometry_mismatch_no_implicit_registration")
        shape, affine = header["shape"], header["affine"]
        if modality == "mask":
            image = nib.load(path)
            if image.get_data_dtype().kind == "c":
                raise ValueError("complex_mask_rejected")
            values = image.get_fdata(dtype=np.float32)  # Only explicit bounded reviewed masks.
            if not np.isfinite(values).all():
                raise ValueError("nonfinite_mask")
            labels = set(np.unique(values).tolist())
            if not labels.issubset({int(key) for key in definitions}):
                raise ValueError("mask_values_do_not_match_disease_specific_definitions")
        else:
            modalities.append(modality)
        if digest_file(path) != item["sha256"]:
            raise ValueError("file_changed_during_pair_review")
    return {
        "status": "TECHNICAL_PAIR_CHECK_PASSED",
        "shape": shape,
        "modalities_as_reviewed": modalities,
        "observed_mask_labels": sorted(labels),
        "empty_mask_implies_normal": False,
        "training_eligible": False,
        "reason": "disease_specific_training_adapter_and_source_policy_review_still_required",
    }
