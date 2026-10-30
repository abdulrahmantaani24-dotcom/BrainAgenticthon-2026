"""Shared, read-only paired MRI validator extracted from notebook 02.

Validation and singleton semantics are unchanged from commit 37b5177. Both the
sample notebook and resumable full-cohort audit import this implementation.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nibabel as nib
import numpy as np

from brainagenticthon.preprocessing.data_validator import NIfTIValidator

VALIDATOR_VERSION = "1.0.0"


def private_call(operation, *args, **kwargs):
    """Suppress library output and error details that could contain private paths."""
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
              warnings.catch_warnings()):
            warnings.simplefilter("ignore")
            return operation(*args, **kwargs)
    except Exception:
        raise RuntimeError(
            "Private validation stopped. Check inputs privately; no records were changed."
        ) from None
    finally:
        logging.disable(previous)


def hidden_metadata(path):
    return any(part.startswith(".") or part == "__MACOSX" for part in Path(path).parts)


def private_path(root, relative):
    """Reject hidden metadata, traversal, absolute paths and symlink escapes."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("unsafe_relative_path")
    candidate = Path(relative)
    if candidate.is_absolute() or hidden_metadata(candidate):
        raise ValueError("unsafe_relative_path")
    root = Path(root).resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path_outside_private_root")
    return resolved


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs_unchanged(fingerprints):
    return all(digest_file(path) == digest for path, digest in fingerprints.items())


def validate_analyze_pair(header_path, root, *, max_voxels=32_000_000,
                          max_pair_bytes=256 * 1024 * 1024):
    """Inspect supported two-file headers; singleton extraction is an array view only."""
    root = Path(root).resolve()
    entry = private_path(root, str(Path(header_path).relative_to(root)))
    flags = ["analyze_world_geometry_and_units_unverified", "native_t1_not_verified",
             "visual_qc_not_performed"]
    failures = []
    result = {"computational_checks_passed": False, "quality_status": "review_required",
              "failures": failures, "quality_flags": flags, "singleton_axis_view": False,
              "in_memory_view_operation": "not_created", "existing_validator_checks": []}
    if entry.suffix not in {".hdr", ".img"}:
        failures.append("unsupported_pair_extension")
        return result
    # Replace only the last extension: mpr-1.nifti.hdr pairs with mpr-1.nifti.img.
    header = private_path(root, str(entry.with_suffix(".hdr").relative_to(root)))
    payload = private_path(root, str(header.with_suffix(".img").relative_to(root)))
    if not header.is_file() or not payload.is_file():
        failures.append("missing_analyze_companion")
        return result
    if header.stat().st_size > 1024 * 1024 or payload.stat().st_size > max_pair_bytes:
        failures.append("pair_exceeds_byte_limit")
        return result
    if header.with_suffix(".mat").exists():
        failures.append("analyze_sidecar_requires_review")
        return result
    before = {path: digest_file(path) for path in (header, payload)}
    try:
        with warnings.catch_warnings(record=True) as header_warnings:
            warnings.simplefilter("always")
            image = nib.load(str(header), mmap="r")
        if header_warnings:
            failures.append("nibabel_header_warning_requires_review")
        supported_types = {nib.AnalyzeImage, nib.Spm99AnalyzeImage, nib.Spm2AnalyzeImage,
                           nib.Nifti1Pair, nib.Nifti2Pair}
        # Filename suffixes do not determine the header format. Single-file NIfTI
        # subclasses and unrelated images must not pass merely by inheritance.
        result["source_pair_format"] = (
            type(image).__name__ if type(image) in supported_types else "unsupported"
        )
        if type(image) not in supported_types:
            failures.append("unsupported_pair_format")
            return result
        for key, expected in (("header", header), ("image", payload)):
            mapped = Path(image.file_map[key].filename)
            if private_path(root, str(mapped.relative_to(root))) != expected:
                failures.append("unexpected_pair_file_map")
                return result
        shape = tuple(int(d) for d in image.shape)
        spatial_shape = shape[:3]
        result["shape"] = list(shape)
        result["spatial_shape"] = list(spatial_shape)
        singleton = len(shape) == 4 and shape[3] == 1
        if len(shape) == 4 and not singleton:
            failures.append("non_singleton_4d_not_supported")
        elif len(shape) != 3 and not singleton:
            failures.append("image_not_3d_or_singleton_4d")
        if len(spatial_shape) != 3 or any(d < 10 or d > 1024 for d in spatial_shape):
            failures.append("invalid_spatial_dimensions")
        if math.prod(shape) > max_voxels:
            failures.append("image_exceeds_voxel_limit")
        if failures:
            return result
        dtype = image.get_data_dtype()
        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
            failures.append("voxel_dtype_not_real_numeric")
            return result
        expected_bytes = math.prod(shape) * dtype.itemsize + image.dataobj.offset
        if payload.stat().st_size < expected_bytes:
            failures.append("truncated_analyze_payload")
            return result
        # Inspect original spacing without NiBabel's automatic in-memory header
        # fixes masking invalid source metadata. No header is written or replaced.
        with header.open("rb") as stream:
            raw_header = type(image.header).from_fileobj(stream, check=False)
        spacing = np.asarray(raw_header.get_zooms()[:3], dtype=float)
        result["header_spacing_units_unverified"] = [
            float(v) if np.isfinite(v) else None for v in spacing
        ]
        if len(spacing) != 3 or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            failures.append("invalid_source_header_spacing")
        elif np.any(spacing < 0.1) or np.any(spacing > 10):
            failures.append("source_header_spacing_out_of_range")
        if tuple(raw_header.get_data_shape()) != shape:
            failures.append("loaded_shape_differs_from_source_header")
        validator = NIfTIValidator()
        checks = (("header", validator.validate_header),
                  ("dimensions", validator.validate_dimensions),
                  ("voxel_spacing", validator.validate_voxel_spacing),
                  ("orientation", validator.validate_orientation))
        for name, check in checks:
            try:
                report = check(header)  # Original paired file, not a converted/squeezed file.
                result["existing_validator_checks"].append(name)
                failures.extend(r.check_name for r in report.errors)
                flags.extend(r.check_name for r in report.warnings)
                failures.extend(r.check_name for r in report.warnings
                                if r.check_name == "header.voxel_size")
            except Exception:
                # Invalid geometry can prevent orientation decoding. Preserve
                # prior check failures and fail this check without exposing paths.
                failures.append(f"validator_{name}_error")
        affine = np.asarray(image.affine)
        affine_valid = affine.shape == (4, 4) and np.isfinite(affine).all()
        if affine_valid:
            affine_valid = (abs(np.linalg.det(affine[:3, :3])) > 1e-10
                            and abs(np.linalg.det(affine)) > 1e-10
                            and np.allclose(affine[3], [0, 0, 0, 1]))
        if not affine_valid:
            failures.append("invalid_spatial_affine")
        else:
            result["array_axis_codes"] = list(nib.aff2axcodes(affine))
        if isinstance(image, (nib.Nifti1Pair, nib.Nifti2Pair)):
            try:
                qform, qcode = image.get_qform(coded=True)
                sform, scode = image.get_sform(coded=True)
                if qcode and scode and not np.allclose(qform, sform, atol=1e-4, rtol=1e-5):
                    failures.append("conflicting_qform_sform")
            except Exception:
                failures.append("invalid_coded_spatial_transform")
        if failures:
            return result
        data = np.asarray(image.dataobj, dtype=np.float32)
        # The fourth axis was verified above. Never squeeze any spatial axis,
        # select a time series, construct a new MRI image, or alter the affine.
        volume = data[..., 0] if singleton else data
        result["singleton_axis_view"] = singleton
        result["in_memory_view_operation"] = "data[..., 0]" if singleton else "none"
        result["in_memory_volume_shape"] = list(volume.shape)
        if singleton:
            flags.append("singleton_fourth_axis_in_memory_view_only")
        if not np.isfinite(volume).all():
            failures.append("nonfinite_voxels")
        elif np.max(volume) == np.min(volume):
            failures.append("constant_volume")
    except Exception:
        failures.append("analyze_read_or_validation_error")
    finally:
        result["original_pair_unchanged"] = inputs_unchanged(before)
        if not result["original_pair_unchanged"]:
            failures.append("original_pair_changed_during_read")
        result["failures"] = sorted(set(failures))
        result["quality_flags"] = sorted(set(flags))
        result["computational_checks_passed"] = not failures
    return result
