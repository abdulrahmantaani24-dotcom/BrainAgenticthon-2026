"""
manifest_schema.py — Versioned Master Dataset Manifest for BrainAgenticthon-2026
==================================================================================

Defines the canonical schema for the Master Dataset Manifest (MDM): a unified,
versioned catalogue of every imaging case across all approved dataset sources.

The manifest is a *virtual* combined index — it records metadata and file paths
without copying any image files into a shared directory.  Raw datasets remain
in their separate source-specific subdirectories.

Schema version
--------------
``MANIFEST_SCHEMA_VERSION = "1.0.0"``

Fields
------
Every row in the manifest is a :class:`ManifestRecord`.  See the dataclass
docstring for full field descriptions.

Key design decisions
--------------------
* **No mixing of raw data**: ``image_path`` points to the original source
  directory; no files are copied.
* **Verified labels only**: ``original_diagnosis`` must be explicitly set from
  clinical metadata.  Missing values are represented as ``None`` / empty string
  and the ``label_provenance`` field is set to ``"missing"``.
* **Healthy controls**: only records with ``original_diagnosis == "CN"`` (the
  :data:`HEALTHY_CONTROL_LABEL`) and ``label_provenance != "missing"`` are
  treated as healthy.
* **Dimensionality guard**: ``image_dimension`` must be ``"3D"`` for records
  eligible for Tasks A and B.  Figshare 2D records are flagged separately.
* **Pituitary exclusion**: records from ``figshare_2d`` with
  ``cohort == "pituitary"`` are assigned ``task_c_label = None`` to prevent
  the pituitary class from entering the 3D ranked classification space.
* **Schema validation**: :func:`validate_record` raises :exc:`ManifestError`
  on any violated constraint.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION: str = "1.0.0"
"""Semantic version of the manifest schema.  Increment the minor version on
backward-compatible field additions; increment the major version on breaking
changes."""

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Split(str, Enum):
    """Dataset split assignment."""
    TRAIN      = "train"
    VALIDATION = "validation"
    TEST       = "test"
    HOLDOUT    = "holdout"   # external / never-seen-during-development
    PRETRAIN   = "pretrain"  # pre-training only (e.g. Figshare 2D)
    UNASSIGNED = "unassigned"


class LabelProvenance(str, Enum):
    """How the label was derived."""
    CHALLENGE_ANNOTATION = "challenge_annotation"  # BraTS expert annotation
    CLINICAL_RECORD      = "clinical_record"        # from clinical DB (OASIS/ADNI)
    INFERRED             = "inferred"               # derived / heuristic
    MISSING              = "missing"                # label absent — NEVER assume healthy


class QualityFlag(str, Enum):
    """Case-level quality flags."""
    OK               = "ok"
    MOTION_ARTIFACT  = "motion_artifact"
    INCOMPLETE_SCAN  = "incomplete_scan"
    PROTOCOL_DEVIANT = "protocol_deviant"
    DUPLICATE        = "duplicate"
    EXCLUDED         = "excluded"
    UNKNOWN          = "unknown"


# ---------------------------------------------------------------------------
# Task output label constants
# ---------------------------------------------------------------------------

# Task A — Detection: binary per-volume
TASK_A_LABELS: Dict[int, str] = {
    0: "no_finding",
    1: "finding_present",
}

# Task B — Spatial characterisation: no classification label (bounding
# box / segmentation output only).  The segmentation label set is source-
# dependent; BraTS 2024 labels are defined in dataset_registry.py.

# Task C — Classification (3D only; Figshare pituitary EXCLUDED)
TASK_C_LABELS: Dict[str, str] = {
    "glioma":      "Glioma",
    "metastasis":  "Brain Metastasis",
    "meningioma":  "Meningioma",
    "ad":          "Alzheimer's Disease",
    "mci":         "Mild Cognitive Impairment",
    "cn":          "Cognitively Normal",
    "no_finding":  "No pathological finding",
    # NOTE: "pituitary" is intentionally absent from this label space.
}

PITUITARY_EXCLUDED_FROM_TASK_C: bool = True
"""Pituitary class from Figshare 2D data MUST NOT appear in Task C labels."""


# ---------------------------------------------------------------------------
# ManifestRecord
# ---------------------------------------------------------------------------

@dataclass
class ManifestRecord:
    """One row in the Master Dataset Manifest.

    Attributes
    ----------
    subject_id : str
        Pseudonymous subject identifier (e.g. ``sub-OASIS30001``).  Must be
        unique within a source.  Never use real patient names or identifiers.
    case_id : str
        Unique case identifier within the manifest (source-scoped).
        E.g. ``BraTS-GLI-00001-000`` or ``sub-OASIS30001_ses-d0090``.
    dataset_source : str
        Stable source key from ``APPROVED_SOURCES`` (e.g. ``"brats2024_gli"``).
    cohort : str
        Sub-cohort identifier within the source (e.g. ``"GLI"``, ``"CN"``).
    session_id : str
        Session identifier for longitudinal datasets; ``""`` if not applicable.
    modality : str
        Imaging modality for this file (e.g. ``"t1"``, ``"t1ce"``, ``"seg"``).
    image_path : str
        Absolute or workspace-relative path to the image file.
        Raw data stays in its original location — no copying.
    image_dimension : str
        ``"2D"`` or ``"3D"``.
    original_diagnosis : Optional[str]
        Explicit clinical/challenge diagnosis from source metadata.
        ``None`` or ``""`` = missing; NEVER infer healthy from a missing label.
    task_a_label : Optional[int]
        Task A label: ``0`` (no finding) or ``1`` (finding present).
        ``None`` if the record is not eligible for Task A.
    task_c_label : Optional[str]
        Task C classification label from ``TASK_C_LABELS``.
        ``None`` if not eligible or if pituitary (excluded).
    segmentation_path : Optional[str]
        Path to the voxel-level segmentation file (NIfTI), if available.
    label_provenance : str
        One of the ``LabelProvenance`` enum values.
    license_status : str
        DUA / license identifier (from ``LicenseType``).
    site : str
        Imaging site or scanner centre identifier (for site-aware evaluation).
    scanner : str
        Scanner model / manufacturer string (for shortcut-learning analysis).
    split : str
        Train / validation / test / pretrain / holdout assignment.
    quality_flag : str
        Case-level quality flag from ``QualityFlag``.
    notes : str
        Free-text notes (no PHI).
    schema_version : str
        Manifest schema version at record creation time.
    """
    subject_id:         str
    case_id:            str
    dataset_source:     str
    cohort:             str
    session_id:         str
    modality:           str
    image_path:         str
    image_dimension:    str
    original_diagnosis: Optional[str]
    task_a_label:       Optional[int]
    task_c_label:       Optional[str]
    segmentation_path:  Optional[str]
    label_provenance:   str
    license_status:     str
    site:               str
    scanner:            str
    split:              str
    quality_flag:       str          = QualityFlag.UNKNOWN.value
    notes:              str          = ""
    schema_version:     str          = MANIFEST_SCHEMA_VERSION

    # Fields required by the spec but allowed to be empty strings:
    #   session_id, site, scanner, notes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ManifestError(ValueError):
    """Raised when a ManifestRecord violates a schema constraint."""


_VALID_SOURCES: frozenset = frozenset({
    "brats2024_gli", "brats2024_met", "brats2024_men",
    "figshare_2d", "oasis3", "adni", "local",
})
_VALID_DIMENSIONS: frozenset = frozenset({"2D", "3D"})
_VALID_SPLITS: frozenset  = frozenset({s.value for s in Split})
_VALID_QUALITY: frozenset = frozenset({q.value for q in QualityFlag})
_VALID_PROVENANCES: frozenset = frozenset({p.value for p in LabelProvenance})
_TASK_A_ELIGIBLE_3D_SOURCES: frozenset = frozenset({
    "brats2024_gli", "brats2024_met", "brats2024_men"
})
_TASK_C_ELIGIBLE_SOURCES: frozenset = frozenset({
    "brats2024_gli", "brats2024_met", "brats2024_men",
    "oasis3", "adni",
})
_PRETRAIN_ONLY_SOURCES: frozenset = frozenset({"figshare_2d"})


def validate_record(rec: ManifestRecord, *, strict: bool = True) -> List[str]:
    """Validate a :class:`ManifestRecord` against all schema constraints.

    Parameters
    ----------
    rec : ManifestRecord
        The record to validate.
    strict : bool
        If ``True`` (default), raise :exc:`ManifestError` on the first
        violation.  If ``False``, collect all violations and return them.

    Returns
    -------
    list of str
        Violation messages (empty list = valid).

    Raises
    ------
    ManifestError
        When ``strict=True`` and at least one violation is found.
    """
    errors: List[str] = []

    def _err(msg: str) -> None:
        if strict:
            raise ManifestError(msg)
        errors.append(msg)

    # --- required string fields ---
    if not rec.subject_id:
        _err("subject_id must not be empty.")
    if not rec.case_id:
        _err("case_id must not be empty.")

    # --- source ---
    if rec.dataset_source not in _VALID_SOURCES:
        _err(
            f"dataset_source '{rec.dataset_source}' is not in the approved "
            f"source list: {sorted(_VALID_SOURCES)}"
        )

    # --- dimensionality ---
    if rec.image_dimension not in _VALID_DIMENSIONS:
        _err(
            f"image_dimension must be '2D' or '3D', got '{rec.image_dimension}'."
        )

    # --- 2D source must not be in Task A/B eligible list ---
    if rec.image_dimension == "2D" and rec.task_a_label is not None:
        _err(
            f"2D records (source={rec.dataset_source}) must not have a "
            "task_a_label (Task A is 3D only)."
        )

    # --- pre-train only sources must use split=pretrain ---
    if rec.dataset_source in _PRETRAIN_ONLY_SOURCES:
        if rec.split not in (Split.PRETRAIN.value, Split.UNASSIGNED.value):
            _err(
                f"Source '{rec.dataset_source}' is pretrain-only; "
                f"split must be 'pretrain' or 'unassigned', got '{rec.split}'."
            )

    # --- pituitary exclusion from Task C ---
    if (
        rec.dataset_source == "figshare_2d"
        and rec.cohort.lower() == "pituitary"
        and rec.task_c_label is not None
    ):
        _err(
            "Pituitary class from figshare_2d MUST NOT have a task_c_label. "
            "Set task_c_label=None to prevent it from entering the 3D "
            "ranked classification space."
        )

    # --- Task C label must come from the approved label space ---
    if rec.task_c_label is not None and rec.task_c_label not in TASK_C_LABELS:
        _err(
            f"task_c_label '{rec.task_c_label}' is not in TASK_C_LABELS: "
            f"{list(TASK_C_LABELS)}."
        )

    # --- task_a_label must be 0 or 1 ---
    if rec.task_a_label is not None and rec.task_a_label not in (0, 1):
        _err(f"task_a_label must be 0 or 1, got {rec.task_a_label}.")

    # --- healthy-control safety rule ---
    # A record is NEVER allowed to have task_c_label='cn' if label_provenance
    # is 'missing' or original_diagnosis is absent.
    if rec.task_c_label == "cn":
        if rec.label_provenance == LabelProvenance.MISSING.value:
            _err(
                "A record with label_provenance='missing' MUST NOT be "
                "assigned task_c_label='cn' (healthy).  Missing diagnosis "
                "must be reviewed before assignment."
            )
        if not rec.original_diagnosis:
            _err(
                "A record with no original_diagnosis MUST NOT be "
                "assigned task_c_label='cn'.  Provide an explicit clinical "
                "diagnosis first."
            )

    # --- label provenance ---
    if rec.label_provenance not in _VALID_PROVENANCES:
        _err(
            f"label_provenance '{rec.label_provenance}' is not valid: "
            f"{sorted(_VALID_PROVENANCES)}"
        )

    # --- split ---
    if rec.split not in _VALID_SPLITS:
        _err(
            f"split '{rec.split}' is not valid: {sorted(_VALID_SPLITS)}"
        )

    # --- quality flag ---
    if rec.quality_flag not in _VALID_QUALITY:
        _err(
            f"quality_flag '{rec.quality_flag}' is not valid: "
            f"{sorted(_VALID_QUALITY)}"
        )

    # --- image_path must not be empty ---
    if not rec.image_path:
        _err("image_path must not be empty.")

    # --- Task A requires 3D source ---
    if (
        rec.task_a_label is not None
        and rec.dataset_source not in _TASK_A_ELIGIBLE_3D_SOURCES
    ):
        _err(
            f"Source '{rec.dataset_source}' is not eligible for Task A. "
            f"Task A sources: {sorted(_TASK_A_ELIGIBLE_3D_SOURCES)}"
        )

    return errors


# ---------------------------------------------------------------------------
# ManifestCollection — virtual combined manifest
# ---------------------------------------------------------------------------

class ManifestCollection:
    """An in-memory collection of :class:`ManifestRecord` items.

    This is the *virtual combined manifest* — it unifies records across all
    approved dataset sources without copying any files.

    Parameters
    ----------
    records : list of ManifestRecord
        Initial records (optional).
    """

    def __init__(
        self,
        records: Optional[List[ManifestRecord]] = None,
        *,
        validate: bool = False,
    ) -> None:
        self._records: List[ManifestRecord] = []
        if records:
            if validate:
                for r in records:
                    validate_record(r, strict=True)
            self._records = list(records)

    # -- mutation -----------------------------------------------------------

    def add(self, record: ManifestRecord, *, validate: bool = True) -> None:
        """Add a record, optionally validating it first."""
        if validate:
            validate_record(record, strict=True)
        self._records.append(record)

    def add_many(
        self,
        records: Iterable[ManifestRecord],
        *,
        validate: bool = True,
    ) -> None:
        for rec in records:
            self.add(rec, validate=validate)

    # -- queries ------------------------------------------------------------

    @property
    def records(self) -> List[ManifestRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def filter(
        self,
        *,
        split: Optional[str] = None,
        dataset_source: Optional[str] = None,
        image_dimension: Optional[str] = None,
        quality_flag: Optional[str] = None,
        task_eligible: Optional[str] = None,
    ) -> "ManifestCollection":
        """Return a filtered sub-collection.

        Parameters
        ----------
        split : str, optional
            Filter by split value.
        dataset_source : str, optional
            Filter by source id.
        image_dimension : str, optional
            ``"2D"`` or ``"3D"``.
        quality_flag : str, optional
            Filter by quality flag.
        task_eligible : str, optional
            ``"a"``, ``"b"``, or ``"c"`` — keep only records whose source
            is eligible for that task.
        """
        from brainagenticthon.preprocessing.dataset_registry import (
            get_task_eligible_sources,
        )

        result = self._records
        if split is not None:
            result = [r for r in result if r.split == split]
        if dataset_source is not None:
            result = [r for r in result if r.dataset_source == dataset_source]
        if image_dimension is not None:
            result = [r for r in result if r.image_dimension == image_dimension]
        if quality_flag is not None:
            result = [r for r in result if r.quality_flag == quality_flag]
        if task_eligible is not None:
            eligible = set(get_task_eligible_sources(task_eligible))
            result = [r for r in result if r.dataset_source in eligible]
        return ManifestCollection(result)

    def unique_subjects(self) -> Dict[str, List[str]]:
        """Return mapping from dataset_source → sorted list of subject_ids."""
        mapping: Dict[str, Set[str]] = {}
        for rec in self._records:
            mapping.setdefault(rec.dataset_source, set()).add(rec.subject_id)
        return {src: sorted(ids) for src, ids in mapping.items()}

    def sources_present(self) -> List[str]:
        return sorted({r.dataset_source for r in self._records})

    def split_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self._records:
            counts[rec.split] = counts.get(rec.split, 0) + 1
        return counts

    # -- I/O ----------------------------------------------------------------

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "record_count": len(self._records),
                "records": self.to_dict_list(),
            },
            **kwargs,
        )

    def save_json(self, path: Path | str) -> None:
        """Write the manifest to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        logger.info("Manifest saved to %s (%d records)", path, len(self._records))

    def save_csv(self, path: Path | str) -> None:
        """Write the manifest to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_dict_list()
        if not rows:
            logger.warning("Manifest is empty; CSV not written.")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Manifest CSV saved to %s", path)

    @classmethod
    def load_json(cls, path: Path | str) -> "ManifestCollection":
        """Load a manifest from a JSON file (no re-validation by default)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw_records = data.get("records", data) if isinstance(data, dict) else data
        records = [ManifestRecord(**r) for r in raw_records]
        return cls(records)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TASK_A_LABELS",
    "TASK_C_LABELS",
    "PITUITARY_EXCLUDED_FROM_TASK_C",
    "Split",
    "LabelProvenance",
    "QualityFlag",
    "ManifestRecord",
    "ManifestError",
    "validate_record",
    "ManifestCollection",
]
