"""Bounded read-only MRI loading and classifier-only, in-memory transforms."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional

from brainagenticthon.detection.profile import InputProfile
from brainagenticthon.preprocessing.data_validator import NIfTIValidator
from brainagenticthon.preprocessing.oasis2_mri_quality import (
    digest_file,
    inputs_unchanged,
    private_call,
    private_path,
    validate_analyze_pair,
)


def resolve_image(root: Path, image_path: str) -> Path:
    """Accept relative or absolute paths only within the authorized image root."""
    root = Path(root).resolve(strict=True)
    candidate = Path(image_path)
    if candidate.is_absolute():
        try:
            image_path = str(candidate.relative_to(root))
        except ValueError:
            raise ValueError("image_outside_authorized_root") from None
    path = private_path(root, image_path)
    if not path.is_file():
        raise ValueError("image_not_found")
    return path


def _load(path, root, profile, expected):
    pair = path.suffix in {".hdr", ".img"}
    if pair:
        path = path.with_suffix(".hdr")
        paths = (path, resolve_image(root, str(path.with_suffix(".img"))))
    elif path.name.endswith((".nii", ".nii.gz")):
        paths = (path,)
    else:
        raise ValueError("unsupported_mri_format")
    if any(p.stat().st_size > profile.max_file_bytes for p in paths):
        raise ValueError("image_exceeds_byte_limit")
    before = {p: digest_file(p) for p in paths}
    if expected is not None and before != expected:
        raise ValueError("image_changed_since_eligibility_review")
    try:
        if pair:
            result = validate_analyze_pair(
                path, root, max_voxels=profile.max_voxels,
                max_pair_bytes=profile.max_file_bytes,
            )
            if not result["computational_checks_passed"]:
                raise ValueError("paired_image_validation_failed")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            image = nib.load(str(path), mmap="r")
        if caught:
            raise ValueError("image_header_warning")
        allowed = {nib.Nifti1Image, nib.Nifti2Image, nib.Nifti1Pair, nib.Nifti2Pair}
        if type(image) not in allowed:
            raise ValueError("training_requires_coded_nifti_geometry")
        shape = image.shape
        if len(shape) != 3 and not (len(shape) == 4 and shape[3] == 1):
            raise ValueError("requires_3d_or_singleton_4d")
        if (any(d < 10 or d > 1024 for d in shape[:3])
                or math.prod(shape) > profile.max_voxels):
            raise ValueError("image_exceeds_spatial_limits")
        # NiBabel may repair invalid pixdims in its in-memory header and log
        # instead of warning. Inspect original bytes without running check_fix.
        with nib.openers.ImageOpener(str(path), mode="rb") as stream:
            raw_header = type(image.header).from_fileobj(stream, check=False)
        raw_spacing = np.asarray(raw_header.get_zooms()[:3], dtype=float)
        if (tuple(raw_header.get_data_shape()) != shape
                or not np.isfinite(raw_spacing).all()
                or np.any(raw_spacing < 0.1) or np.any(raw_spacing > 10)
                or not np.allclose(raw_spacing, image.header.get_zooms()[:3])
                or int(raw_header["qform_code"]) not in range(6)
                or int(raw_header["sform_code"]) not in range(6)):
            raise ValueError("invalid_original_spatial_header")
        dtype = image.get_data_dtype()
        if dtype.kind not in "iuf":
            raise ValueError("requires_real_numeric_voxels")
        affine = np.asarray(image.affine)
        spacing = np.asarray(image.header.get_zooms()[:3])
        if (not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1])
                or abs(np.linalg.det(affine[:3, :3])) < 1e-10
                or not np.isfinite(spacing).all() or np.any(spacing < 0.1)
                or np.any(spacing > 10)):
            raise ValueError("invalid_geometry")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError("spatial_units_not_verified_mm_in_header")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if not qcode and not scode:
            raise ValueError("missing_coded_spatial_transform")
        if qcode and float(raw_header["pixdim"][0]) not in {-1.0, 1.0}:
            raise ValueError("invalid_original_qfac")
        if qcode and scode and not np.allclose(qform, sform, atol=1e-4, rtol=1e-5):
            raise ValueError("conflicting_qform_sform")
        if not pair:
            # validate_all includes intensity materialization, so enforce the
            # header-derived resource limits BEFORE invoking it.
            report = NIfTIValidator().validate_all(path)
            if report.errors or any(r.check_name == "header.voxel_size" for r in report.warnings):
                raise ValueError("nifti_validation_failed")
        data = np.asarray(image.dataobj, dtype=np.float32)
        volume = data[..., 0] if len(shape) == 4 else data
        if not np.isfinite(volume).all() or np.max(volume) == np.min(volume):
            raise ValueError("nonfinite_or_constant_volume")
        source_orientation = nib.orientations.io_orientation(affine)
        transform = nib.orientations.ornt_transform(
            source_orientation, nib.orientations.axcodes2ornt(("R", "A", "S")),
        )
        volume = nib.orientations.apply_orientation(volume, transform)
        # Nonzero is a numerical intensity support, NOT a brain or lesion mask.
        support = volume != 0
        values = volume[support]
        low, high = np.percentile(values, [1, 99])
        clipped = np.clip(values, low, high)
        deviation = float(clipped.std(dtype=np.float64))
        if not math.isfinite(deviation) or deviation <= 1e-8:
            raise ValueError("insufficient_intensity_variation")
        normalized = np.zeros(volume.shape, dtype=np.float32)
        normalized[support] = (clipped - clipped.mean(dtype=np.float64)) / deviation
        tensor = torch.from_numpy(normalized.copy())[None, None]
        tensor = functional.interpolate(
            tensor, size=profile.target_shape, mode="trilinear", align_corners=False,
        )[0]
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite_preprocessed_tensor")
        return tensor, before
    finally:
        if not inputs_unchanged(before):
            raise ValueError("source_bytes_changed_during_read")


def load_mri(root, image_path, profile=None, *, expected=None):
    """Return [1, X, Y, Z] and private byte fingerprints; never log paths.

    Computational success is not clinical QC. The real-training gate separately
    requires explicit visual, native-T1 and anatomical-geometry evidence.
    """
    return private_call(lambda: _load(
        resolve_image(root, image_path), Path(root), profile or InputProfile(), expected,
    ))
