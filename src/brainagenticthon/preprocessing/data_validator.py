"""Comprehensive validation utilities for 3D brain MRI NIfTI data.

This module provides a layered validation framework for medical imaging AI
challenge pipelines, covering individual NIfTI files, modality naming
conventions, per-case structural consistency, and whole-dataset integrity.

The validators are designed around the BraTS (Brain Tumor Segmentation)
challenge conventions but are fully configurable for other datasets and
modalities.

Example usage::

    from brainagenticthon.preprocessing.data_validator import (
        NIfTIValidator,
        CaseValidator,
        DatasetValidator,
    )

    # Validate a single file
    nifti_val = NIfTIValidator()
    report = nifti_val.validate_all("path/to/scan.nii.gz")
    print(report)

    # Validate an entire dataset
    ds_val = DatasetValidator()
    report = ds_val.generate_report("path/to/dataset")
    if not report.is_valid:
        for err in report.errors:
            print(err)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import nibabel as nib
    import numpy as np
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "nibabel and numpy are required for NIfTI validation. "
        "Install them with:  pip install nibabel numpy"
    ) from _exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BraTS constants
# ---------------------------------------------------------------------------

BRATS_MODALITIES: Dict[str, str] = {
    "t1": "T1-weighted native",
    "t1ce": "T1-weighted contrast-enhanced (gadolinium)",
    "t2": "T2-weighted",
    "flair": "T2-weighted FLAIR",
    "seg": "Segmentation label map",
}
"""Canonical BraTS modality identifiers and their descriptions.

The keys are the short identifiers expected to appear in file names
(case-insensitive).  ``seg`` is the ground-truth segmentation label map.
"""

BRATS_LABELS: Dict[int, str] = {
    0: "Background",
    1: "Necrotic tumor core (NCR)",
    2: "Peritumoral edematous/invaded tissue (ED)",
    3: "GD-enhancing tumor (ET)",
}
"""Default BraTS 2024/2025 segmentation label set.

Mapping from integer label value to a human-readable description.

* 0 — Background (healthy tissue / non-tumor)
* 1 — NCR  – Necrotic tumor core
* 2 — ED   – Peritumoral edematous / invaded tissue
* 3 — ET   – GD-enhancing tumor
"""

# Default hash chunk size for large-file streaming (8 MB).
_DEFAULT_HASH_CHUNK_SIZE: int = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# ValidationResult & ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a single validation check.

    Attributes:
        passed: Whether the check succeeded.
        check_name: Short machine-readable name of the check
            (e.g. ``"header.exists"``).
        message: Human-readable description of the result.
        severity: One of ``"error"``, ``"warning"``, or ``"info"``.
        details: Optional dictionary with extra context (e.g. actual vs
            expected values).

    Example::

        result = ValidationResult(
            passed=False,
            check_name="header.dimensions",
            message="Expected 3D volume, got 2D image",
            severity="error",
            details={"ndim": 2},
        )
    """

    passed: bool
    check_name: str
    message: str
    severity: str = "error"
    details: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"error", "warning", "info"}
        if self.severity not in allowed:
            raise ValueError(
                f"severity must be one of {allowed}, got {self.severity!r}"
            )


class ValidationReport:
    """Aggregator for multiple :class:`ValidationResult` items.

    Example::

        report = ValidationReport()
        report.add(ValidationResult(True, "check_a", "OK", "info"))
        report.add(ValidationResult(False, "check_b", "Fail", "error"))
        print(report.is_valid)  # False
        print(report.summary)
    """

    def __init__(self) -> None:
        self._results: List[ValidationResult] = []

    # -- mutators -----------------------------------------------------------

    def add(self, result: ValidationResult) -> None:
        """Append a :class:`ValidationResult` to this report."""
        self._results.append(result)

    def merge(self, other: "ValidationReport") -> None:
        """Merge all results from *other* into this report."""
        self._results.extend(other._results)

    # -- accessors ----------------------------------------------------------

    @property
    def results(self) -> List[ValidationResult]:
        """All collected results (read-only copy)."""
        return list(self._results)

    @property
    def is_valid(self) -> bool:
        """``True`` if no result has ``severity='error'`` and ``passed=False``."""
        return all(
            r.passed for r in self._results if r.severity == "error"
        )

    @property
    def errors(self) -> List[ValidationResult]:
        """Failed results with ``severity='error'``."""
        return [r for r in self._results if r.severity == "error" and not r.passed]

    @property
    def warnings(self) -> List[ValidationResult]:
        """Failed results with ``severity='warning'``."""
        return [r for r in self._results if r.severity == "warning" and not r.passed]

    @property
    def summary(self) -> str:
        """One-line summary of the report."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        n_errors = len(self.errors)
        n_warnings = len(self.warnings)
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"{status}: {passed}/{total} checks passed, "
            f"{n_errors} error(s), {n_warnings} warning(s)"
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "is_valid": self.is_valid,
            "summary": self.summary,
            "results": [asdict(r) for r in self._results],
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise the report to a JSON string.

        Extra *kwargs* are forwarded to :func:`json.dumps`.
        """
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    # -- display ------------------------------------------------------------

    def __str__(self) -> str:  # noqa: D105
        lines: List[str] = [self.summary, "=" * len(self.summary)]
        for r in self._results:
            icon = "✓" if r.passed else "✗"
            lines.append(
                f"  [{r.severity.upper():7s}] {icon} {r.check_name}: {r.message}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return f"<ValidationReport {self.summary}>"


# ---------------------------------------------------------------------------
# NIfTIValidator
# ---------------------------------------------------------------------------

class NIfTIValidator:
    """Validates individual NIfTI files for header integrity, orientation,
    dimensions, voxel spacing, intensity range, and file-level integrity.

    All public methods accept a file path (``str`` or :class:`~pathlib.Path``)
    and return a :class:`ValidationReport`.

    Example::

        validator = NIfTIValidator()
        report = validator.validate_all("/data/BraTS/case/t1.nii.gz")
        print(report)
    """

    # -- header -------------------------------------------------------------

    def validate_header(self, path: Union[str, Path]) -> ValidationReport:
        """Run basic header checks on a NIfTI file.

        Checks performed:

        1. File exists on disk.
        2. File loads successfully with *nibabel*.
        3. Image is 3-D or 4-D.
        4. Affine matrix is valid (finite, non-singular).
        5. Voxel sizes are in a plausible range (0.1 mm – 10 mm).
        6. Data type is numeric.

        Args:
            path: Path to a ``.nii`` or ``.nii.gz`` file.

        Returns:
            A :class:`ValidationReport` with results for each sub-check.
        """
        path = Path(path)
        report = ValidationReport()

        # 1. existence
        if not path.is_file():
            report.add(ValidationResult(
                passed=False,
                check_name="header.exists",
                message=f"File not found: {path}",
                severity="error",
            ))
            return report
        report.add(ValidationResult(
            passed=True,
            check_name="header.exists",
            message="File exists",
            severity="info",
        ))

        # 2. load
        try:
            img = nib.load(str(path))
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="header.loadable",
                message=f"Failed to load NIfTI: {exc}",
                severity="error",
            ))
            return report
        report.add(ValidationResult(
            passed=True,
            check_name="header.loadable",
            message="File loaded successfully",
            severity="info",
        ))

        # 3. dimensionality
        ndim = len(img.shape)
        if ndim in (3, 4):
            report.add(ValidationResult(
                passed=True,
                check_name="header.ndim",
                message=f"Image is {ndim}-D",
                severity="info",
                details={"ndim": ndim, "shape": list(img.shape)},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="header.ndim",
                message=f"Expected 3-D or 4-D image, got {ndim}-D",
                severity="error",
                details={"ndim": ndim, "shape": list(img.shape)},
            ))

        # 4. affine validity
        affine = img.affine
        affine_finite = bool(np.all(np.isfinite(affine)))
        try:
            det = np.linalg.det(affine[:3, :3])
            affine_nonsingular = bool(abs(det) > 1e-10)
        except np.linalg.LinAlgError:
            affine_nonsingular = False

        affine_ok = affine_finite and affine_nonsingular
        if affine_ok:
            report.add(ValidationResult(
                passed=True,
                check_name="header.affine",
                message="Affine matrix is valid (finite, non-singular)",
                severity="info",
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="header.affine",
                message="Affine matrix is invalid (non-finite or singular)",
                severity="error",
                details={
                    "finite": affine_finite,
                    "nonsingular": affine_nonsingular,
                },
            ))

        # 5. voxel sizes
        voxel_sizes = np.array(img.header.get_zooms()[:3], dtype=float)
        vs_ok = bool(np.all((voxel_sizes >= 0.1) & (voxel_sizes <= 10.0)))
        if vs_ok:
            report.add(ValidationResult(
                passed=True,
                check_name="header.voxel_size",
                message=f"Voxel sizes plausible: {voxel_sizes.tolist()} mm",
                severity="info",
                details={"voxel_sizes_mm": voxel_sizes.tolist()},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="header.voxel_size",
                message=(
                    f"Voxel sizes out of plausible range (0.1–10 mm): "
                    f"{voxel_sizes.tolist()} mm"
                ),
                severity="warning",
                details={"voxel_sizes_mm": voxel_sizes.tolist()},
            ))

        # 6. numeric data type
        dtype = img.get_data_dtype()
        is_numeric = np.issubdtype(dtype, np.number)
        if is_numeric:
            report.add(ValidationResult(
                passed=True,
                check_name="header.dtype",
                message=f"Data type is numeric: {dtype}",
                severity="info",
                details={"dtype": str(dtype)},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="header.dtype",
                message=f"Data type is not numeric: {dtype}",
                severity="error",
                details={"dtype": str(dtype)},
            ))

        return report

    # -- orientation --------------------------------------------------------

    def validate_orientation(
        self,
        path: Union[str, Path],
        expected: str = "RAS",
    ) -> ValidationReport:
        """Check that the image orientation matches *expected*.

        Args:
            path: Path to a NIfTI file.
            expected: Three-character orientation code (default ``"RAS"``).

        Returns:
            :class:`ValidationReport`.
        """
        path = Path(path)
        report = ValidationReport()

        try:
            img = nib.load(str(path))
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="orientation.load",
                message=f"Cannot load file for orientation check: {exc}",
                severity="error",
            ))
            return report

        ornt_codes = nib.aff2axcodes(img.affine)
        actual = "".join(ornt_codes)
        ok = actual == expected
        report.add(ValidationResult(
            passed=ok,
            check_name="orientation.match",
            message=(
                f"Orientation {'matches' if ok else 'does not match'} "
                f"expected={expected}, actual={actual}"
            ),
            severity="info" if ok else "warning",
            details={"expected": expected, "actual": actual},
        ))
        return report

    # -- dimensions ---------------------------------------------------------

    def validate_dimensions(
        self,
        path: Union[str, Path],
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dim: int = 10,
        max_dim: int = 1024,
    ) -> ValidationReport:
        """Validate image dimensions for sanity.

        Args:
            path: Path to a NIfTI file.
            expected_shape: If given, the image shape must match exactly.
            min_dim: Minimum allowed size along any spatial axis (default 10).
            max_dim: Maximum allowed size along any spatial axis (default 1024).

        Returns:
            :class:`ValidationReport`.
        """
        path = Path(path)
        report = ValidationReport()

        try:
            img = nib.load(str(path))
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="dimensions.load",
                message=f"Cannot load file for dimension check: {exc}",
                severity="error",
            ))
            return report

        shape = img.shape
        spatial_shape = shape[:3]

        # range check
        dims_in_range = all(min_dim <= d <= max_dim for d in spatial_shape)
        if dims_in_range:
            report.add(ValidationResult(
                passed=True,
                check_name="dimensions.range",
                message=f"Spatial dimensions within [{min_dim}, {max_dim}]: {spatial_shape}",
                severity="info",
                details={"shape": list(shape), "min_dim": min_dim, "max_dim": max_dim},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="dimensions.range",
                message=(
                    f"Spatial dimensions outside [{min_dim}, {max_dim}]: "
                    f"{spatial_shape}"
                ),
                severity="error",
                details={"shape": list(shape), "min_dim": min_dim, "max_dim": max_dim},
            ))

        # exact match
        if expected_shape is not None:
            match = shape == tuple(expected_shape)
            report.add(ValidationResult(
                passed=match,
                check_name="dimensions.exact",
                message=(
                    f"Shape {'matches' if match else 'does not match'} "
                    f"expected={expected_shape}, actual={shape}"
                ),
                severity="error" if not match else "info",
                details={
                    "expected": list(expected_shape),
                    "actual": list(shape),
                },
            ))

        return report

    # -- voxel spacing ------------------------------------------------------

    def validate_voxel_spacing(
        self,
        path: Union[str, Path],
        expected_spacing: Optional[Sequence[float]] = None,
        tolerance: float = 0.01,
    ) -> ValidationReport:
        """Check voxel spacing consistency.

        Args:
            path: Path to a NIfTI file.
            expected_spacing: If given, the voxel spacing must match within
                *tolerance* (in mm).
            tolerance: Maximum allowed absolute deviation per axis (default
                0.01 mm).

        Returns:
            :class:`ValidationReport`.
        """
        path = Path(path)
        report = ValidationReport()

        try:
            img = nib.load(str(path))
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="voxel_spacing.load",
                message=f"Cannot load file for spacing check: {exc}",
                severity="error",
            ))
            return report

        zooms = np.array(img.header.get_zooms()[:3], dtype=float)

        # basic sanity — positive values
        positive = bool(np.all(zooms > 0))
        report.add(ValidationResult(
            passed=positive,
            check_name="voxel_spacing.positive",
            message=(
                f"Voxel spacing {'is' if positive else 'is not'} strictly "
                f"positive: {zooms.tolist()}"
            ),
            severity="error" if not positive else "info",
            details={"spacing_mm": zooms.tolist()},
        ))

        # expected match
        if expected_spacing is not None:
            expected = np.array(expected_spacing[:3], dtype=float)
            diff = np.abs(zooms - expected)
            within_tol = bool(np.all(diff <= tolerance))
            report.add(ValidationResult(
                passed=within_tol,
                check_name="voxel_spacing.match",
                message=(
                    f"Spacing {'matches' if within_tol else 'does not match'} "
                    f"expected={expected.tolist()} (tol={tolerance}), "
                    f"actual={zooms.tolist()}, diff={diff.tolist()}"
                ),
                severity="error" if not within_tol else "info",
                details={
                    "expected": expected.tolist(),
                    "actual": zooms.tolist(),
                    "diff": diff.tolist(),
                    "tolerance": tolerance,
                },
            ))

        return report

    # -- intensity range ----------------------------------------------------

    def validate_intensity_range(
        self,
        path: Union[str, Path],
        modality: Optional[str] = None,
    ) -> ValidationReport:
        """Check image intensities for NaN, Inf, and plausible range.

        When *modality* is ``"seg"`` the check verifies that all values are
        non-negative integers.  For other modalities a simple finite-range
        check is performed.

        Args:
            path: Path to a NIfTI file.
            modality: Optional modality hint (e.g. ``"t1"``, ``"seg"``).

        Returns:
            :class:`ValidationReport`.
        """
        path = Path(path)
        report = ValidationReport()

        try:
            img = nib.load(str(path))
            data = np.asarray(img.dataobj, dtype=float)
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="intensity.load",
                message=f"Cannot load image data: {exc}",
                severity="error",
            ))
            return report

        # NaN check
        n_nan = int(np.count_nonzero(np.isnan(data)))
        if n_nan == 0:
            report.add(ValidationResult(
                passed=True,
                check_name="intensity.nan",
                message="No NaN values found",
                severity="info",
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="intensity.nan",
                message=f"Found {n_nan} NaN value(s)",
                severity="error",
                details={"nan_count": n_nan},
            ))

        # Inf check
        n_inf = int(np.count_nonzero(np.isinf(data)))
        if n_inf == 0:
            report.add(ValidationResult(
                passed=True,
                check_name="intensity.inf",
                message="No Inf values found",
                severity="info",
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="intensity.inf",
                message=f"Found {n_inf} Inf value(s)",
                severity="error",
                details={"inf_count": n_inf},
            ))

        # Modality-specific plausibility
        finite_data = data[np.isfinite(data)]
        if finite_data.size > 0:
            vmin = float(finite_data.min())
            vmax = float(finite_data.max())
            details = {"min": vmin, "max": vmax, "modality": modality}

            if modality and modality.lower() == "seg":
                # Segmentation must be non-negative integers
                all_int = bool(np.all(np.equal(np.mod(finite_data, 1), 0)))
                all_nonneg = bool(vmin >= 0)
                ok = all_int and all_nonneg
                report.add(ValidationResult(
                    passed=ok,
                    check_name="intensity.seg_plausible",
                    message=(
                        f"Segmentation label values "
                        f"{'are' if ok else 'are NOT'} non-negative integers "
                        f"(min={vmin}, max={vmax})"
                    ),
                    severity="error" if not ok else "info",
                    details=details,
                ))
            else:
                # General MRI: very large dynamic range is suspicious
                dyn_range = vmax - vmin
                suspicious = dyn_range > 1e7
                report.add(ValidationResult(
                    passed=not suspicious,
                    check_name="intensity.range_plausible",
                    message=(
                        f"Intensity range [{vmin:.4g}, {vmax:.4g}] "
                        f"(dynamic range={dyn_range:.4g})"
                        + (" — suspiciously large" if suspicious else "")
                    ),
                    severity="warning" if suspicious else "info",
                    details=details,
                ))

        return report

    # -- file integrity -----------------------------------------------------

    def validate_file_integrity(
        self,
        path: Union[str, Path],
        expected_hash: Optional[str] = None,
        algorithm: str = "sha256",
    ) -> ValidationReport:
        """Compute a hash of the file and optionally compare it.

        The hash is computed by streaming the file in 8 MB chunks so that
        arbitrarily large files can be handled without excessive memory use.

        Args:
            path: Path to any file.
            expected_hash: If given, the computed hash must match this value
                (case-insensitive hex digest).
            algorithm: Hash algorithm name recognised by :mod:`hashlib`
                (default ``"sha256"``).

        Returns:
            :class:`ValidationReport` containing the computed hash in
            *details*.
        """
        path = Path(path)
        report = ValidationReport()

        if not path.is_file():
            report.add(ValidationResult(
                passed=False,
                check_name="integrity.exists",
                message=f"File not found: {path}",
                severity="error",
            ))
            return report

        try:
            h = hashlib.new(algorithm)
        except ValueError:
            report.add(ValidationResult(
                passed=False,
                check_name="integrity.algorithm",
                message=f"Unsupported hash algorithm: {algorithm}",
                severity="error",
            ))
            return report

        try:
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(_DEFAULT_HASH_CHUNK_SIZE)
                    if not chunk:
                        break
                    h.update(chunk)
            computed = h.hexdigest()
        except OSError as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="integrity.read",
                message=f"Cannot read file for hashing: {exc}",
                severity="error",
            ))
            return report

        report.add(ValidationResult(
            passed=True,
            check_name="integrity.computed",
            message=f"{algorithm} hash computed successfully",
            severity="info",
            details={"algorithm": algorithm, "hash": computed},
        ))

        if expected_hash is not None:
            match = computed.lower() == expected_hash.lower()
            report.add(ValidationResult(
                passed=match,
                check_name="integrity.match",
                message=(
                    f"Hash {'matches' if match else 'does NOT match'} "
                    f"expected value"
                ),
                severity="error" if not match else "info",
                details={
                    "expected": expected_hash.lower(),
                    "computed": computed,
                },
            ))

        return report

    # -- aggregate ----------------------------------------------------------

    def validate_all(self, path: Union[str, Path]) -> ValidationReport:
        """Run **all** individual checks and return a combined report.

        The following validations are performed in order:

        1. :meth:`validate_header`
        2. :meth:`validate_orientation`
        3. :meth:`validate_dimensions`
        4. :meth:`validate_voxel_spacing`
        5. :meth:`validate_intensity_range`
        6. :meth:`validate_file_integrity`

        Args:
            path: Path to a NIfTI file.

        Returns:
            :class:`ValidationReport` with results from every check.
        """
        path = Path(path)
        logger.info("Running full validation on %s", path)

        report = ValidationReport()
        report.merge(self.validate_header(path))

        # If the file could not even be loaded, skip the rest.
        if not report.is_valid:
            return report

        report.merge(self.validate_orientation(path))
        report.merge(self.validate_dimensions(path))
        report.merge(self.validate_voxel_spacing(path))
        report.merge(self.validate_intensity_range(path))
        report.merge(self.validate_file_integrity(path))

        logger.info("Validation complete for %s: %s", path, report.summary)
        return report


# ---------------------------------------------------------------------------
# ModalityValidator
# ---------------------------------------------------------------------------

class ModalityValidator:
    """Validates modality naming conventions and completeness.

    By default the validator recognises BraTS modalities (t1, t1ce, t2,
    flair, seg) but the pattern dictionary is fully configurable.

    Args:
        modality_patterns: Mapping from modality name to a compiled regex
            (or raw pattern string) used to detect that modality in a
            filename.  Defaults to BraTS patterns.

    Example::

        mv = ModalityValidator()
        report = mv.validate_modality_name("BraTS-GLI-00001-000-t1.nii.gz")
        assert report.is_valid
    """

    #: Default filename patterns for BraTS modalities.
    #: Order matters: ``t1ce`` is checked before ``t1`` so that filenames
    #: containing ``t1ce`` are not incorrectly matched as plain ``t1``.
    DEFAULT_PATTERNS: Dict[str, re.Pattern[str]] = {
        "t1ce": re.compile(r"(?i)[-_.]t1ce[-_.]|[-_.]t1ce$|[-_.]t1gd[-_.]|[-_.]t1gd$"),
        "t1": re.compile(r"(?i)(?<!ce)[-_.]t1[-_.]|(?<!ce)[-_.]t1$|(?<!gd)[-_.]t1[-_.]"),
        "t2": re.compile(r"(?i)[-_.]t2[-_.]|[-_.]t2$"),
        "flair": re.compile(r"(?i)[-_.]flair[-_.]|[-_.]flair$"),
        "seg": re.compile(r"(?i)[-_.]seg[-_.]|[-_.]seg$"),
    }

    def __init__(
        self,
        modality_patterns: Optional[Dict[str, Union[str, re.Pattern[str]]]] = None,
    ) -> None:
        if modality_patterns is not None:
            self.patterns: Dict[str, re.Pattern[str]] = {
                k: re.compile(v) if isinstance(v, str) else v
                for k, v in modality_patterns.items()
            }
        else:
            self.patterns = dict(self.DEFAULT_PATTERNS)

    def _detect_modality(self, filename: str) -> Optional[str]:
        """Return the first matching modality name, or ``None``."""
        stem = Path(filename).name
        # t1ce must be checked before t1
        for mod_name, pattern in self.patterns.items():
            if pattern.search(stem):
                return mod_name
        return None

    def validate_modality_name(self, filename: Union[str, Path]) -> ValidationReport:
        """Check that *filename* contains a recognised modality identifier.

        Args:
            filename: Filename or path to check.

        Returns:
            :class:`ValidationReport`.
        """
        filename_str = str(Path(filename).name)
        report = ValidationReport()
        detected = self._detect_modality(filename_str)
        if detected is not None:
            report.add(ValidationResult(
                passed=True,
                check_name="modality.name",
                message=f"Detected modality '{detected}' in '{filename_str}'",
                severity="info",
                details={"filename": filename_str, "modality": detected},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="modality.name",
                message=(
                    f"No recognised modality in '{filename_str}'. "
                    f"Expected one of: {list(self.patterns.keys())}"
                ),
                severity="error",
                details={"filename": filename_str},
            ))
        return report

    def validate_modality_set(
        self,
        file_list: Sequence[Union[str, Path]],
        required_modalities: Optional[Sequence[str]] = None,
    ) -> ValidationReport:
        """Check that *file_list* covers all *required_modalities*.

        Args:
            file_list: Iterable of filenames or paths.
            required_modalities: Modalities that must be present.  Defaults to
                all imaging modalities (``t1``, ``t1ce``, ``t2``, ``flair``).

        Returns:
            :class:`ValidationReport`.
        """
        if required_modalities is None:
            required_modalities = ["t1", "t1ce", "t2", "flair"]
        required_set = set(required_modalities)

        report = ValidationReport()
        found: Set[str] = set()
        for f in file_list:
            detected = self._detect_modality(str(Path(f).name))
            if detected is not None:
                found.add(detected)

        missing = required_set - found
        if not missing:
            report.add(ValidationResult(
                passed=True,
                check_name="modality.set_complete",
                message=f"All required modalities present: {sorted(required_set)}",
                severity="info",
                details={"required": sorted(required_set), "found": sorted(found)},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="modality.set_complete",
                message=f"Missing modalities: {sorted(missing)}",
                severity="error",
                details={
                    "required": sorted(required_set),
                    "found": sorted(found),
                    "missing": sorted(missing),
                },
            ))
        return report


# ---------------------------------------------------------------------------
# CaseValidator
# ---------------------------------------------------------------------------

class CaseValidator:
    """Validates per-case structure, spatial consistency, and label maps.

    A *case* is a directory containing one NIfTI file per modality, typically
    named with a BraTS case identifier.

    Args:
        nifti_validator: An optional pre-configured :class:`NIfTIValidator`
            instance.  One is created automatically if not supplied.
        modality_validator: An optional pre-configured
            :class:`ModalityValidator` instance.

    Example::

        cv = CaseValidator()
        report = cv.validate_case_directory(
            "data/BraTS-GLI-00001-000",
            required_modalities=["t1", "t1ce", "t2", "flair", "seg"],
        )
    """

    #: Default regex for BraTS-style case IDs such as
    #: ``BraTS-GLI-00001-000``.
    DEFAULT_CASE_ID_PATTERN: re.Pattern[str] = re.compile(
        r"^BraTS-[A-Z]{3}-\d{5}-\d{3}$"
    )

    def __init__(
        self,
        nifti_validator: Optional[NIfTIValidator] = None,
        modality_validator: Optional[ModalityValidator] = None,
    ) -> None:
        self.nifti_validator = nifti_validator or NIfTIValidator()
        self.modality_validator = modality_validator or ModalityValidator()

    # -- case ID ------------------------------------------------------------

    def validate_case_id(
        self,
        case_id: str,
        pattern: Optional[Union[str, re.Pattern[str]]] = None,
    ) -> ValidationReport:
        """Validate a case identifier string.

        Args:
            case_id: The case identifier to check (e.g.
                ``"BraTS-GLI-00001-000"``).
            pattern: Regex pattern (compiled or string) the ID must match.
                Defaults to :attr:`DEFAULT_CASE_ID_PATTERN`.

        Returns:
            :class:`ValidationReport`.
        """
        report = ValidationReport()
        if pattern is None:
            pat = self.DEFAULT_CASE_ID_PATTERN
        elif isinstance(pattern, str):
            pat = re.compile(pattern)
        else:
            pat = pattern

        if pat.match(case_id):
            report.add(ValidationResult(
                passed=True,
                check_name="case.id_format",
                message=f"Case ID '{case_id}' matches expected pattern",
                severity="info",
                details={"case_id": case_id, "pattern": pat.pattern},
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="case.id_format",
                message=(
                    f"Case ID '{case_id}' does not match pattern "
                    f"'{pat.pattern}'"
                ),
                severity="warning",
                details={"case_id": case_id, "pattern": pat.pattern},
            ))
        return report

    # -- case directory -----------------------------------------------------

    def validate_case_directory(
        self,
        case_dir: Union[str, Path],
        required_modalities: Optional[Sequence[str]] = None,
    ) -> ValidationReport:
        """Validate that a case directory has all expected NIfTI files.

        Each discovered NIfTI file is also run through
        :meth:`NIfTIValidator.validate_all`.

        Args:
            case_dir: Path to the case directory.
            required_modalities: List of required modality names. Defaults to
                ``["t1", "t1ce", "t2", "flair", "seg"]``.

        Returns:
            :class:`ValidationReport`.
        """
        if required_modalities is None:
            required_modalities = ["t1", "t1ce", "t2", "flair", "seg"]

        case_dir = Path(case_dir)
        report = ValidationReport()

        if not case_dir.is_dir():
            report.add(ValidationResult(
                passed=False,
                check_name="case.dir_exists",
                message=f"Case directory not found: {case_dir}",
                severity="error",
            ))
            return report

        report.add(ValidationResult(
            passed=True,
            check_name="case.dir_exists",
            message=f"Case directory exists: {case_dir.name}",
            severity="info",
        ))

        # Validate case id from directory name
        report.merge(self.validate_case_id(case_dir.name))

        # Find NIfTI files
        nifti_files = sorted(
            list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii"))
        )

        if not nifti_files:
            report.add(ValidationResult(
                passed=False,
                check_name="case.has_nifti",
                message=f"No NIfTI files found in {case_dir}",
                severity="error",
            ))
            return report

        # Modality completeness
        report.merge(
            self.modality_validator.validate_modality_set(
                [f.name for f in nifti_files],
                required_modalities=required_modalities,
            )
        )

        # Per-file validation
        for nf in nifti_files:
            file_report = self.nifti_validator.validate_all(nf)
            report.merge(file_report)

        return report

    # -- spatial consistency ------------------------------------------------

    def validate_spatial_consistency(
        self,
        case_dir: Union[str, Path],
    ) -> ValidationReport:
        """Verify all NIfTI files in a case share the same affine and shape.

        Args:
            case_dir: Path to the case directory.

        Returns:
            :class:`ValidationReport`.
        """
        case_dir = Path(case_dir)
        report = ValidationReport()

        nifti_files = sorted(
            list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii"))
        )

        if len(nifti_files) < 2:
            report.add(ValidationResult(
                passed=True,
                check_name="spatial.consistency",
                message=(
                    "Fewer than 2 NIfTI files; spatial consistency check "
                    "is trivially satisfied"
                ),
                severity="info",
            ))
            return report

        try:
            ref_img = nib.load(str(nifti_files[0]))
            ref_shape = ref_img.shape[:3]
            ref_affine = ref_img.affine
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="spatial.ref_load",
                message=f"Cannot load reference file {nifti_files[0].name}: {exc}",
                severity="error",
            ))
            return report

        all_consistent = True
        for nf in nifti_files[1:]:
            try:
                img = nib.load(str(nf))
            except Exception as exc:
                report.add(ValidationResult(
                    passed=False,
                    check_name="spatial.load",
                    message=f"Cannot load {nf.name} for comparison: {exc}",
                    severity="error",
                ))
                all_consistent = False
                continue

            shape_ok = img.shape[:3] == ref_shape
            affine_ok = bool(np.allclose(img.affine, ref_affine, atol=1e-5))

            if not shape_ok:
                report.add(ValidationResult(
                    passed=False,
                    check_name="spatial.shape_match",
                    message=(
                        f"{nf.name} shape {img.shape[:3]} != reference "
                        f"{ref_shape} ({nifti_files[0].name})"
                    ),
                    severity="error",
                    details={
                        "file": nf.name,
                        "reference": nifti_files[0].name,
                        "file_shape": list(img.shape[:3]),
                        "ref_shape": list(ref_shape),
                    },
                ))
                all_consistent = False

            if not affine_ok:
                report.add(ValidationResult(
                    passed=False,
                    check_name="spatial.affine_match",
                    message=(
                        f"{nf.name} affine does not match reference "
                        f"({nifti_files[0].name})"
                    ),
                    severity="error",
                    details={
                        "file": nf.name,
                        "reference": nifti_files[0].name,
                    },
                ))
                all_consistent = False

        if all_consistent:
            report.add(ValidationResult(
                passed=True,
                check_name="spatial.consistency",
                message=(
                    f"All {len(nifti_files)} files share the same shape "
                    f"{ref_shape} and affine"
                ),
                severity="info",
                details={"n_files": len(nifti_files), "shape": list(ref_shape)},
            ))

        return report

    # -- label map ----------------------------------------------------------

    def validate_label_map(
        self,
        seg_path: Union[str, Path],
        expected_labels: Optional[Set[int]] = None,
    ) -> ValidationReport:
        """Validate a segmentation label map.

        Checks that every unique value in the segmentation is within the
        *expected_labels* set.

        Args:
            seg_path: Path to the segmentation NIfTI file.
            expected_labels: Allowed label values.  Defaults to
                ``{0, 1, 2, 3}`` (BraTS 2024/2025 convention, see
                :data:`BRATS_LABELS`).

        Returns:
            :class:`ValidationReport`.
        """
        if expected_labels is None:
            expected_labels = set(BRATS_LABELS.keys())

        seg_path = Path(seg_path)
        report = ValidationReport()

        try:
            img = nib.load(str(seg_path))
            data = np.asarray(img.dataobj)
        except Exception as exc:
            report.add(ValidationResult(
                passed=False,
                check_name="label_map.load",
                message=f"Cannot load segmentation: {exc}",
                severity="error",
            ))
            return report

        unique_vals = set(int(v) for v in np.unique(data))
        unexpected = unique_vals - expected_labels
        missing = expected_labels - unique_vals

        if not unexpected:
            report.add(ValidationResult(
                passed=True,
                check_name="label_map.values",
                message=f"All label values are within the expected set {sorted(expected_labels)}",
                severity="info",
                details={
                    "expected": sorted(expected_labels),
                    "found": sorted(unique_vals),
                },
            ))
        else:
            report.add(ValidationResult(
                passed=False,
                check_name="label_map.values",
                message=(
                    f"Unexpected label values found: {sorted(unexpected)}. "
                    f"Expected only {sorted(expected_labels)}"
                ),
                severity="error",
                details={
                    "expected": sorted(expected_labels),
                    "found": sorted(unique_vals),
                    "unexpected": sorted(unexpected),
                },
            ))

        if missing:
            report.add(ValidationResult(
                passed=True,  # missing labels are not necessarily an error
                check_name="label_map.coverage",
                message=(
                    f"Some expected labels are absent from the segmentation: "
                    f"{sorted(missing)} — this may be normal for cases without "
                    f"certain tumor sub-regions"
                ),
                severity="warning",
                details={"missing": sorted(missing)},
            ))

        # Check that label data looks like integers
        is_integer = np.issubdtype(data.dtype, np.integer) or bool(
            np.all(np.equal(np.mod(data.astype(float), 1), 0))
        )
        report.add(ValidationResult(
            passed=is_integer,
            check_name="label_map.integer",
            message=(
                f"Label data {'has' if is_integer else 'does NOT have'} "
                f"integer values (dtype={data.dtype})"
            ),
            severity="error" if not is_integer else "info",
            details={"dtype": str(data.dtype)},
        ))

        return report


# ---------------------------------------------------------------------------
# DatasetValidator
# ---------------------------------------------------------------------------

def _validate_single_case(
    case_dir: str,
    required_modalities: List[str],
) -> Dict[str, Any]:
    """Top-level helper for :class:`DatasetValidator` parallel execution.

    This must be a module-level function so that :class:`ProcessPoolExecutor`
    can pickle it.

    Returns:
        Dictionary with ``case_id`` and serialised ``report``.
    """
    cv = CaseValidator()
    case_path = Path(case_dir)
    report = ValidationReport()
    report.merge(cv.validate_case_directory(case_path, required_modalities))
    report.merge(cv.validate_spatial_consistency(case_path))

    # If there is a seg file, validate the label map
    seg_files = list(case_path.glob("*seg*"))
    for sf in seg_files:
        if sf.suffix in (".gz",) or sf.suffixes == [".nii", ".gz"] or sf.suffix == ".nii":
            report.merge(cv.validate_label_map(sf))

    return {
        "case_id": case_path.name,
        "report": report.to_dict(),
        "report_obj": report,
    }


class DatasetValidator:
    """Validates an entire BraTS-style dataset directory.

    The expected layout is::

        dataset_dir/
        ├── BraTS-GLI-00001-000/
        │   ├── BraTS-GLI-00001-000-t1.nii.gz
        │   ├── BraTS-GLI-00001-000-t1ce.nii.gz
        │   ├── BraTS-GLI-00001-000-t2.nii.gz
        │   ├── BraTS-GLI-00001-000-flair.nii.gz
        │   └── BraTS-GLI-00001-000-seg.nii.gz
        ├── BraTS-GLI-00002-000/
        │   └── ...
        └── ...

    Args:
        max_workers: Number of parallel workers for dataset-wide validation.
            Defaults to ``1`` (sequential).  Set to ``None`` to use
            :func:`os.cpu_count`.

    Example::

        dv = DatasetValidator(max_workers=4)
        report = dv.generate_report("data/BraTS2025")
        print(report)
    """

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def _list_case_dirs(self, dataset_dir: Path) -> List[Path]:
        """Return sorted list of sub-directories that look like cases."""
        return sorted(
            d for d in dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    # -- case count ---------------------------------------------------------

    def validate_case_count(
        self,
        dataset_dir: Union[str, Path],
        expected_count: Optional[int] = None,
    ) -> ValidationReport:
        """Check the number of case directories in the dataset.

        Args:
            dataset_dir: Root dataset directory.
            expected_count: If provided, the actual count must match.

        Returns:
            :class:`ValidationReport`.
        """
        dataset_dir = Path(dataset_dir)
        report = ValidationReport()

        if not dataset_dir.is_dir():
            report.add(ValidationResult(
                passed=False,
                check_name="dataset.dir_exists",
                message=f"Dataset directory not found: {dataset_dir}",
                severity="error",
            ))
            return report

        case_dirs = self._list_case_dirs(dataset_dir)
        n_cases = len(case_dirs)

        report.add(ValidationResult(
            passed=True,
            check_name="dataset.case_count",
            message=f"Found {n_cases} case(s) in dataset",
            severity="info",
            details={"count": n_cases},
        ))

        if expected_count is not None:
            match = n_cases == expected_count
            report.add(ValidationResult(
                passed=match,
                check_name="dataset.case_count_match",
                message=(
                    f"Case count {'matches' if match else 'does NOT match'} "
                    f"expected={expected_count}, actual={n_cases}"
                ),
                severity="error" if not match else "info",
                details={"expected": expected_count, "actual": n_cases},
            ))

        return report

    # -- per-case -----------------------------------------------------------

    def validate_dataset(
        self,
        dataset_dir: Union[str, Path],
        required_modalities: Optional[Sequence[str]] = None,
        expected_count: Optional[int] = None,
    ) -> ValidationReport:
        """Validate every case in the dataset.

        Args:
            dataset_dir: Root dataset directory.
            required_modalities: Required modalities per case. Defaults to
                ``["t1", "t1ce", "t2", "flair", "seg"]``.
            expected_count: If provided, the case count must match.

        Returns:
            Aggregated :class:`ValidationReport` for the whole dataset.
        """
        if required_modalities is None:
            required_modalities = ["t1", "t1ce", "t2", "flair", "seg"]
        required_list = list(required_modalities)

        dataset_dir = Path(dataset_dir)
        report = ValidationReport()

        # Top-level checks
        report.merge(self.validate_case_count(dataset_dir, expected_count))
        if not report.is_valid:
            return report

        case_dirs = self._list_case_dirs(dataset_dir)

        if not case_dirs:
            report.add(ValidationResult(
                passed=False,
                check_name="dataset.empty",
                message="Dataset contains no case directories",
                severity="error",
            ))
            return report

        logger.info(
            "Validating %d cases with %s worker(s)",
            len(case_dirs),
            self.max_workers or "all available",
        )

        workers = self.max_workers
        if workers == 1:
            # Sequential – simpler error handling and no pickling issues
            for cd in case_dirs:
                result = _validate_single_case(str(cd), required_list)
                report.merge(result["report_obj"])
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _validate_single_case, str(cd), required_list
                    ): cd
                    for cd in case_dirs
                }
                for future in as_completed(futures):
                    cd = futures[future]
                    try:
                        result = future.result()
                        # When running in a subprocess the report_obj may
                        # have been pickled; rebuild from dict if needed.
                        sub_report = result.get("report_obj")
                        if sub_report is None:
                            # Reconstruct from serialised dict
                            sub_report = ValidationReport()
                            for r in result["report"]["results"]:
                                sub_report.add(ValidationResult(**r))
                        report.merge(sub_report)
                    except Exception as exc:
                        report.add(ValidationResult(
                            passed=False,
                            check_name="dataset.case_error",
                            message=f"Error validating case {cd.name}: {exc}",
                            severity="error",
                            details={"case": cd.name, "error": str(exc)},
                        ))

        return report

    # -- full report --------------------------------------------------------

    def generate_report(
        self,
        dataset_dir: Union[str, Path],
        required_modalities: Optional[Sequence[str]] = None,
        expected_count: Optional[int] = None,
    ) -> ValidationReport:
        """Produce a full :class:`ValidationReport` for the dataset.

        This is a convenience wrapper around :meth:`validate_dataset`.

        Args:
            dataset_dir: Root dataset directory.
            required_modalities: Required modalities per case.
            expected_count: Expected number of cases.

        Returns:
            :class:`ValidationReport`.
        """
        logger.info("Generating dataset report for %s", dataset_dir)
        report = self.validate_dataset(
            dataset_dir,
            required_modalities=required_modalities,
            expected_count=expected_count,
        )
        logger.info("Dataset report: %s", report.summary)
        return report
