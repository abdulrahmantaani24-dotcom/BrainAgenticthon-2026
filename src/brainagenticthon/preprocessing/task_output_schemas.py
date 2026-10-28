"""
task_output_schemas.py — Output Schema Contracts for Tasks A, B, and C
=======================================================================

Defines the canonical output data structures produced by each pipeline task.
Other components (agents, reporting, verification) import these schemas to
ensure inter-component compatibility.

Schema version
--------------
``TASK_SCHEMA_VERSION = "1.0.0"``

Task A — Detection
    Input:  3D multi-parametric MRI volume (t1, t1ce, t2, flair)
    Output: :class:`TaskAOutput` — presence flag + bounding boxes + confidence

Task B — Spatial Characterisation
    Input:  3D volume + Task A bounding boxes
    Output: :class:`TaskBOutput` — voxel-level segmentation + volumetrics +
            anatomical region mapping

Task C — Classification / Differential Diagnosis
    Input:  3D volume + Task B spatial features
    Output: :class:`TaskCOutput` — class label + probabilities + evidence

Integration notes
-----------------
* All outputs carry ``case_id``, ``dataset_source``, and ``schema_version``
  so downstream components can assert compatibility.
* ``TaskBOutput.segmentation_label_map`` uses the BraTS 2024 label convention
  (0=BG, 1=NCR, 2=ED, 3=ET) for BraTS-derived cases; other sources populate
  it as an empty dict.
* ``TaskCOutput.class_label`` must be a key in
  :data:`manifest_schema.TASK_C_LABELS` — pituitary is excluded.
* None of these schemas hold raw image data or patient identifiers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from brainagenticthon.preprocessing.manifest_schema import TASK_C_LABELS

TASK_SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

@dataclass
class _TaskOutputBase:
    """Common fields shared by all task outputs."""
    case_id:        str
    dataset_source: str
    schema_version: str = TASK_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Task A — Detection
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox3D:
    """Axis-aligned bounding box in voxel coordinates.

    Attributes
    ----------
    x_min, x_max, y_min, y_max, z_min, z_max : int
        Inclusive voxel index bounds along each spatial axis.
    confidence : float
        Detection confidence in [0.0, 1.0].
    """
    x_min:      int
    x_max:      int
    y_min:      int
    y_max:      int
    z_min:      int
    z_max:      int
    confidence: float = 0.0

    def __post_init__(self) -> None:
        for attr in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be >= 0, got {getattr(self, attr)}.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}."
            )
        for lo, hi in [("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")]:
            if getattr(self, lo) > getattr(self, hi):
                raise ValueError(
                    f"{lo} ({getattr(self, lo)}) must be <= {hi} ({getattr(self, hi)})."
                )


@dataclass
class TaskAOutput(_TaskOutputBase):
    """Task A — Detection output schema.

    Attributes
    ----------
    finding_present : bool
        True if at least one abnormal finding was detected.
    bounding_boxes : list of BoundingBox3D
        Zero or more detected regions (empty when finding_present=False).
    overall_confidence : float
        Aggregate detection confidence score in [0.0, 1.0].
    model_id : str
        Identifier of the model that produced this output.
    inference_time_s : float
        Inference wall-clock time in seconds (for benchmarking).

    Integration contract
    --------------------
    * ``finding_present=False`` → ``bounding_boxes`` must be empty.
    * ``finding_present=True``  → ``bounding_boxes`` must be non-empty.
    * ``overall_confidence`` must be in [0.0, 1.0].
    * This output is consumed by Task B for spatial refinement.
    """
    finding_present:    bool              = False
    bounding_boxes:     List[BoundingBox3D] = field(default_factory=list)
    overall_confidence: float             = 0.0
    model_id:           str               = ""
    inference_time_s:   float             = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(
                f"overall_confidence must be in [0.0, 1.0], "
                f"got {self.overall_confidence}."
            )
        if self.finding_present and not self.bounding_boxes:
            raise ValueError(
                "finding_present=True requires at least one bounding box."
            )
        if not self.finding_present and self.bounding_boxes:
            raise ValueError(
                "finding_present=False must have an empty bounding_boxes list."
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Task B — Spatial Characterisation
# ---------------------------------------------------------------------------

@dataclass
class AnatomicalRegion:
    """Anatomical region associated with a detected finding.

    Attributes
    ----------
    region_name : str
        Name of the anatomical region (e.g. ``"right_temporal_lobe"``).
    atlas : str
        Atlas used for parcellation (e.g. ``"AAL3"``, ``"Talairach"``).
    overlap_fraction : float
        Fraction of the segmented volume that overlaps this region [0, 1].
    """
    region_name:       str
    atlas:             str   = "unknown"
    overlap_fraction:  float = 0.0


@dataclass
class VolumetricStats:
    """Volumetric measurements for a segmented finding.

    Attributes
    ----------
    total_volume_ml : float
        Total segmented volume in millilitres.
    ncr_volume_ml : float
        NCR (necrotic core) sub-volume in ml (0 if not applicable).
    ed_volume_ml : float
        ED (edema) sub-volume in ml.
    et_volume_ml : float
        ET (enhancing tumor) sub-volume in ml.
    voxel_size_mm3 : float
        Volume of one voxel in cubic millimetres.
    """
    total_volume_ml:  float = 0.0
    ncr_volume_ml:    float = 0.0
    ed_volume_ml:     float = 0.0
    et_volume_ml:     float = 0.0
    voxel_size_mm3:   float = 1.0


@dataclass
class TaskBOutput(_TaskOutputBase):
    """Task B — Spatial Characterisation output schema.

    Attributes
    ----------
    segmentation_path : str
        Path to the predicted segmentation NIfTI file (workspace-relative).
        Empty string if no segmentation was produced.
    segmentation_label_map : dict
        Mapping from integer label → description (BraTS convention for BraTS
        sources; empty for other sources without voxel labels).
    refined_bounding_boxes : list of BoundingBox3D
        Bounding boxes refined from Task A detections.
    anatomical_regions : list of AnatomicalRegion
        Anatomical regions overlapping the finding.
    volumetrics : VolumetricStats
        Volumetric statistics for the primary finding.
    model_id : str
        Identifier of the model that produced this output.
    inference_time_s : float
        Inference wall-clock time in seconds.

    Integration contract
    --------------------
    * Consumes :class:`TaskAOutput` bounding boxes as input hints.
    * Produces ``segmentation_path`` consumed by Task D reporting.
    * ``segmentation_label_map`` must be consistent with the source's
      ``segmentation_labels`` in :data:`dataset_registry.APPROVED_SOURCES`.
    """
    segmentation_path:      str                  = ""
    segmentation_label_map: Dict[int, str]       = field(default_factory=dict)
    refined_bounding_boxes: List[BoundingBox3D]  = field(default_factory=list)
    anatomical_regions:     List[AnatomicalRegion] = field(default_factory=list)
    volumetrics:            VolumetricStats      = field(
        default_factory=VolumetricStats
    )
    model_id:               str                  = ""
    inference_time_s:       float                = 0.0


# ---------------------------------------------------------------------------
# Task C — Classification / Differential Diagnosis
# ---------------------------------------------------------------------------

@dataclass
class ClassificationEvidence:
    """A single piece of evidence supporting a classification decision.

    Attributes
    ----------
    description : str
        Plain-language description of the evidence.
    source : str
        Where the evidence comes from (e.g. ``"volumetrics"``,
        ``"enhancement_pattern"``, ``"clinical_metadata"``).
    confidence_contribution : float
        This evidence item's contribution to the overall confidence [0, 1].
    """
    description:             str
    source:                  str   = "model_feature"
    confidence_contribution: float = 0.0


@dataclass
class DifferentialEntry:
    """One entry in the differential diagnosis list.

    Attributes
    ----------
    class_label : str
        Must be a key in :data:`manifest_schema.TASK_C_LABELS`.
    probability : float
        Estimated posterior probability in [0.0, 1.0].
    rank : int
        Rank in the differential (1 = most likely).
    """
    class_label: str
    probability: float
    rank:        int

    def __post_init__(self) -> None:
        if self.class_label not in TASK_C_LABELS:
            raise ValueError(
                f"class_label '{self.class_label}' is not in TASK_C_LABELS. "
                f"Valid labels: {list(TASK_C_LABELS)}. "
                "Note: 'pituitary' is intentionally excluded."
            )
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"probability must be in [0.0, 1.0], got {self.probability}."
            )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}.")


@dataclass
class TaskCOutput(_TaskOutputBase):
    """Task C — Classification / Differential Diagnosis output schema.

    Attributes
    ----------
    class_label : str
        Top-1 predicted class from :data:`manifest_schema.TASK_C_LABELS`.
        Must NOT be ``'pituitary'``.
    class_probability : float
        Top-1 class posterior probability in [0.0, 1.0].
    differential : list of DifferentialEntry
        Ranked differential diagnosis list.
    evidence : list of ClassificationEvidence
        Evidence items cited for the primary class.
    model_id : str
        Identifier of the model that produced this output.
    inference_time_s : float
        Inference wall-clock time in seconds.
    source_eligible : bool
        True if the source is eligible for Task C (False for Figshare 2D).

    Integration contract
    --------------------
    * ``class_label`` must be a key in :data:`manifest_schema.TASK_C_LABELS`.
    * ``class_label = 'pituitary'`` is **never** a valid output.
    * Differential entries must be ranked 1..N without gaps.
    * This output is consumed by Task D for report generation.
    """
    class_label:       str                        = "no_finding"
    class_probability: float                      = 0.0
    differential:      List[DifferentialEntry]    = field(default_factory=list)
    evidence:          List[ClassificationEvidence] = field(default_factory=list)
    model_id:          str                        = ""
    inference_time_s:  float                      = 0.0
    source_eligible:   bool                       = True

    def __post_init__(self) -> None:
        if self.class_label not in TASK_C_LABELS:
            raise ValueError(
                f"class_label '{self.class_label}' is not in TASK_C_LABELS. "
                "Note: 'pituitary' is intentionally excluded."
            )
        if not (0.0 <= self.class_probability <= 1.0):
            raise ValueError(
                f"class_probability must be in [0.0, 1.0], "
                f"got {self.class_probability}."
            )
        # Validate differential ranking
        if self.differential:
            ranks = sorted(e.rank for e in self.differential)
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError(
                    "DifferentialEntry ranks must be consecutive integers "
                    f"starting at 1, got {ranks}."
                )


# ---------------------------------------------------------------------------
# Pipeline output bundle
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """Complete pipeline output for one case (Tasks A + B + C).

    Attributes
    ----------
    case_id : str
    dataset_source : str
    task_a : TaskAOutput
    task_b : TaskBOutput
    task_c : TaskCOutput
    schema_version : str
    """
    case_id:        str
    dataset_source: str
    task_a:         TaskAOutput
    task_b:         TaskBOutput
    task_c:         TaskCOutput
    schema_version: str = TASK_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id":        self.case_id,
            "dataset_source": self.dataset_source,
            "schema_version": self.schema_version,
            "task_a":         self.task_a.to_dict(),
            "task_b":         self.task_b.to_dict(),
            "task_c":         self.task_c.to_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "TASK_SCHEMA_VERSION",
    "BoundingBox3D",
    "TaskAOutput",
    "AnatomicalRegion",
    "VolumetricStats",
    "TaskBOutput",
    "ClassificationEvidence",
    "DifferentialEntry",
    "TaskCOutput",
    "PipelineOutput",
]
