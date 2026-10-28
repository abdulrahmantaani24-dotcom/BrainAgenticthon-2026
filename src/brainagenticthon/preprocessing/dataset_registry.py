"""
dataset_registry.py — Pluggable Dataset Source Registry for BrainAgenticthon-2026
==================================================================================

Approved dataset scope (as of 2026):

1. **BraTS 2024** — Three official sub-cohorts hosted on Synapse:
   - Glioma (GLI): multi-parametric MRI, 3D, Tasks A/B/C
   - Brain Metastasis (MET): multi-parametric MRI, 3D, Tasks A/B/C
   - Meningioma (MEN): multi-parametric MRI, 3D, Tasks A/B/C

2. **Figshare Brain Tumor Dataset** — Auxiliary 2D dataset:
   - Publicly available at https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
   - Contains 2D JPEG/PNG images (NOT NIfTI 3D volumes)
   - Includes pituitary, glioma, meningioma, no-tumor classes
   - **Training role: pre-training/transfer-learning ONLY**
   - Pituitary class MUST NOT become a ranked disease label in the 3D pipeline
   - All Figshare samples MUST remain in a separate data directory

3. **OASIS / ADNI** — Longitudinal structural MRI:
   - OASIS-3: Alzheimer's, MCI, and documented healthy controls
   - ADNI (Alzheimer's Disease Neuroimaging Initiative): AD, MCI, CN
   - Access: OASIS via https://www.oasis-brains.org/ (free registration + DUA)
   - Access: ADNI via https://adni.loni.usc.edu/ (requires IRB approval + DUA)
   - Healthy controls: ONLY cases with explicitly documented 'CN' / 'Control' diagnosis
   - Missing diagnosis → NEVER classified as healthy

Architecture
------------
* :class:`DatasetSource` — abstract base class every source implements.
* :class:`DatasetRegistry` — module-level singleton catalogue.
* :func:`register_source` — class decorator for zero-boilerplate registration.
* :class:`SourceMetadata` — structured per-source metadata (modalities,
  dimensionality, training role, sub-cohorts, label provenance, etc.)

Source identifiers (stable keys used in manifests)
---------------------------------------------------
* ``brats2024_gli``   — BraTS 2024 Glioma
* ``brats2024_met``   — BraTS 2024 Brain Metastasis
* ``brats2024_men``   — BraTS 2024 Meningioma
* ``figshare_2d``     — Figshare Brain Tumor (2D only)
* ``oasis3``          — OASIS-3 structural MRI
* ``adni``            — ADNI structural MRI
* ``local``           — Local directory (manual transfer)

Security notice
---------------
.. warning::

   **Never** commit credentials, API keys, or ``.synapseConfig`` files to
   version control.  All credential paths are resolved at runtime and
   validated with explicit warnings when they appear to reside inside a
   repository tree.

"""
from __future__ import annotations

import abc
import dataclasses
import logging
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

try:
    import synapseclient  # type: ignore[import-untyped]
    _HAS_SYNAPSE = True
except ImportError:  # pragma: no cover
    synapseclient = None  # type: ignore[assignment]
    _HAS_SYNAPSE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ImageDimension(str, Enum):
    """Spatial dimensionality of images produced by a source."""
    DIM_2D = "2D"
    DIM_3D = "3D"
    MIXED  = "mixed"  # source produces both


class TrainingRole(str, Enum):
    """Permitted training roles for a dataset source.

    Rules
    -----
    * ``PRETRAIN_ONLY`` — May only be used for pre-training / transfer
      learning.  Must not contribute ranked disease labels to the 3D
      pipeline (e.g., Figshare 2D pituitary class).
    * ``TRAIN_VAL_TEST`` — May participate in all pipeline splits.
    * ``EXTERNAL_EVAL`` — Reserved for external / holdout evaluation only.
    """
    PRETRAIN_ONLY   = "pretrain_only"
    TRAIN_VAL_TEST  = "train_val_test"
    EXTERNAL_EVAL   = "external_eval"


class LicenseType(str, Enum):
    """Data use agreement / license category."""
    CC_BY_4          = "CC-BY-4.0"
    CC_BY_NC_4       = "CC-BY-NC-4.0"
    OASIS_DUA        = "OASIS-DUA"
    ADNI_DUA         = "ADNI-DUA"
    BRATS_SYNAPSE_DUA = "BraTS-Synapse-DUA"
    CUSTOM_DUA       = "custom-DUA"
    UNKNOWN          = "unknown"


# ---------------------------------------------------------------------------
# Approved BraTS 2024 Synapse project ID placeholders
# ---------------------------------------------------------------------------
# Replace these with the actual Synapse project/entity IDs once you have
# been granted access via https://www.synapse.org/ and have accepted the
# BraTS 2024 Data Use Agreement.

SYNAPSE_PROJECT_ID: str = "syn00000000"
"""Placeholder Synapse project ID for generic BraTS data."""

BRATS_EXPECTED_MODALITIES: Set[str] = {"t1", "t1ce", "t2", "flair", "seg"}
"""Standard BraTS modality suffixes expected in every case directory."""

BRATS2024_GLI_SYNAPSE_ID: str = "syn00000001"  # BraTS 2024 Glioma
BRATS2024_MET_SYNAPSE_ID: str = "syn00000002"  # BraTS 2024 Brain Metastasis
BRATS2024_MEN_SYNAPSE_ID: str = "syn00000003"  # BraTS 2024 Meningioma
"""Placeholder Synapse entity IDs.  Obtain the real IDs after approval at:
https://www.synapse.org/  (search for "BraTS 2024")
"""

# BraTS 2024 standard 3D modalities per imaging case
BRATS2024_MODALITIES: FrozenSet[str] = frozenset({"t1", "t1ce", "t2", "flair"})
BRATS2024_SEG_MODALITY: str = "seg"  # segmentation label map

# BraTS 2024 segmentation labels (all three cohorts share this schema)
BRATS2024_LABELS: Dict[int, str] = {
    0: "Background",
    1: "Necrotic tumor core (NCR)",
    2: "Peritumoral edematous/invaded tissue (ED)",
    3: "GD-enhancing tumor (ET)",
}

# Figshare 2D class labels — pituitary is intentionally NOT a ranked
# 3D pipeline label.
FIGSHARE_2D_CLASSES: Dict[str, str] = {
    "glioma":     "Glioma (2D auxiliary, pre-train only)",
    "meningioma": "Meningioma (2D auxiliary, pre-train only)",
    "pituitary":  "Pituitary adenoma (2D, EXCLUDED from 3D ranked labels)",
    "no_tumor":   "No visible tumor (2D, pre-train only)",
}
"""Figshare 2D class labels.

.. warning::
    ``pituitary`` MUST NOT be propagated as a disease class into the
    3D detection / classification pipeline.  It is retained here only for
    completeness and pre-training feature extraction.
"""

# OASIS-3 / ADNI clinical diagnosis labels
NEURODEGEN_DIAGNOSIS_LABELS: Dict[str, str] = {
    "CN":  "Cognitively Normal (documented healthy control)",
    "MCI": "Mild Cognitive Impairment",
    "AD":  "Alzheimer's Disease",
    "LMCI":"Late Mild Cognitive Impairment",
    "EMCI":"Early Mild Cognitive Impairment",
    "SMC": "Significant Memory Concern",
}
"""Valid clinical diagnosis labels for OASIS-3 / ADNI.

.. important::
    A missing or ambiguous diagnosis field MUST NOT be inferred as 'CN'
    (healthy).  Only records with an explicit 'CN' label from the original
    clinical metadata are treated as healthy controls.
"""

HEALTHY_CONTROL_LABEL: str = "CN"
"""The only permitted healthy-control label.  Any other missing or unknown
diagnosis must be flagged with ``label_provenance = "missing"`` and excluded
from training splits until manually reviewed."""


# ---------------------------------------------------------------------------
# SourceMetadata — structured per-source descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SourceMetadata:
    """Immutable descriptor for a registered dataset source.

    Parameters
    ----------
    source_id : str
        Stable identifier used as the ``dataset_source`` field in manifests.
    display_name : str
        Human-readable name.
    description : str
        One-paragraph description.
    image_dimension : ImageDimension
        Whether the source provides 2D, 3D, or mixed images.
    training_role : TrainingRole
        Permitted role(s) in the ML pipeline.
    modalities : frozenset of str
        Expected imaging modalities per case.
    sub_cohorts : tuple of str
        Sub-cohort identifiers within this source (e.g., GLI, MET, MEN).
    has_segmentation : bool
        Whether per-voxel segmentation labels are available.
    segmentation_labels : dict
        Mapping from integer label to description (empty if no segmentation).
    task_a_eligible : bool
        Whether this source may contribute to Task A (Detection).
    task_b_eligible : bool
        Whether this source may contribute to Task B (Spatial).
    task_c_eligible : bool
        Whether this source may contribute to Task C (Classification).
    license_type : LicenseType
        Data use agreement / license governing this source.
    access_url : str
        Official URL for obtaining access (no download links).
    access_instructions : str
        Plain-language description of the access process.
    raw_subdir : str
        Recommended sub-directory under the workspace raw data root.
    keep_separate : bool
        If True, data from this source MUST NOT be mixed into a shared
        directory with other sources (enforced by the manifest builder).
    pituitary_excluded : bool
        If True, pituitary labels from this source are blocked from the
        3D ranked label space.
    """
    source_id:             str
    display_name:          str
    description:           str
    image_dimension:       ImageDimension
    training_role:         TrainingRole
    modalities:            FrozenSet[str]
    sub_cohorts:           Tuple[str, ...]
    has_segmentation:      bool
    segmentation_labels:   Dict[int, str]
    task_a_eligible:       bool
    task_b_eligible:       bool
    task_c_eligible:       bool
    license_type:          LicenseType
    access_url:            str
    access_instructions:   str
    raw_subdir:            str
    keep_separate:         bool = True
    pituitary_excluded:    bool = False


# ---------------------------------------------------------------------------
# Canonical source metadata catalogue
# ---------------------------------------------------------------------------

APPROVED_SOURCES: Dict[str, SourceMetadata] = {

    "brats2024_gli": SourceMetadata(
        source_id        = "brats2024_gli",
        display_name     = "BraTS 2024 — Glioma (GLI)",
        description      = (
            "BraTS 2024 Glioma sub-cohort. Multi-parametric 3D brain MRI "
            "with voxel-level segmentation labels for necrotic core (NCR), "
            "edema (ED) and enhancing tumor (ET). Hosted on Synapse. "
            "Access requires a free Synapse account and acceptance of the "
            "BraTS 2024 Data Use Agreement."
        ),
        image_dimension  = ImageDimension.DIM_3D,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = BRATS2024_MODALITIES | {BRATS2024_SEG_MODALITY},
        sub_cohorts      = ("GLI",),
        has_segmentation = True,
        segmentation_labels = BRATS2024_LABELS,
        task_a_eligible  = True,
        task_b_eligible  = True,
        task_c_eligible  = True,
        license_type     = LicenseType.BRATS_SYNAPSE_DUA,
        access_url       = "https://www.synapse.org/",
        access_instructions = (
            "1. Create a Synapse account at https://www.synapse.org/\n"
            "2. Search for the BraTS 2024 Glioma project (replace placeholder "
            f"ID {BRATS2024_GLI_SYNAPSE_ID} with the real Synapse ID).\n"
            "3. Click 'Request Access' and accept the Data Use Agreement.\n"
            "4. Configure ~/.synapseConfig with your credentials.\n"
            "5. Install synapseclient: pip install synapseclient\n"
            "6. Update configs/storage.yaml with the real Synapse project ID."
        ),
        raw_subdir       = "data/raw/brats2024/gli",
        keep_separate    = True,
        pituitary_excluded = False,
    ),

    "brats2024_met": SourceMetadata(
        source_id        = "brats2024_met",
        display_name     = "BraTS 2024 — Brain Metastasis (MET)",
        description      = (
            "BraTS 2024 Brain Metastasis sub-cohort. Multi-parametric 3D "
            "brain MRI with voxel-level segmentation for metastatic lesions. "
            "Same access process as the Glioma cohort."
        ),
        image_dimension  = ImageDimension.DIM_3D,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = BRATS2024_MODALITIES | {BRATS2024_SEG_MODALITY},
        sub_cohorts      = ("MET",),
        has_segmentation = True,
        segmentation_labels = BRATS2024_LABELS,
        task_a_eligible  = True,
        task_b_eligible  = True,
        task_c_eligible  = True,
        license_type     = LicenseType.BRATS_SYNAPSE_DUA,
        access_url       = "https://www.synapse.org/",
        access_instructions = (
            "Same as BraTS 2024 GLI.  Use the Brain Metastasis project ID "
            f"(replace placeholder {BRATS2024_MET_SYNAPSE_ID})."
        ),
        raw_subdir       = "data/raw/brats2024/met",
        keep_separate    = True,
        pituitary_excluded = False,
    ),

    "brats2024_men": SourceMetadata(
        source_id        = "brats2024_men",
        display_name     = "BraTS 2024 — Meningioma (MEN)",
        description      = (
            "BraTS 2024 Meningioma sub-cohort. Multi-parametric 3D brain MRI "
            "with voxel-level segmentation for meningioma lesions."
        ),
        image_dimension  = ImageDimension.DIM_3D,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = BRATS2024_MODALITIES | {BRATS2024_SEG_MODALITY},
        sub_cohorts      = ("MEN",),
        has_segmentation = True,
        segmentation_labels = BRATS2024_LABELS,
        task_a_eligible  = True,
        task_b_eligible  = True,
        task_c_eligible  = True,
        license_type     = LicenseType.BRATS_SYNAPSE_DUA,
        access_url       = "https://www.synapse.org/",
        access_instructions = (
            "Same as BraTS 2024 GLI.  Use the Meningioma project ID "
            f"(replace placeholder {BRATS2024_MEN_SYNAPSE_ID})."
        ),
        raw_subdir       = "data/raw/brats2024/men",
        keep_separate    = True,
        pituitary_excluded = False,
    ),

    "figshare_2d": SourceMetadata(
        source_id        = "figshare_2d",
        display_name     = "Figshare Brain Tumor Dataset (2D)",
        description      = (
            "Auxiliary 2D brain tumor dataset from Figshare. Contains JPEG/PNG "
            "slices in four classes: glioma, meningioma, pituitary, no_tumor. "
            "Available at https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 "
            "under CC-BY-4.0. "
            "TRAINING ROLE: pre-training / transfer-learning ONLY. "
            "This source is NOT eligible for the ranked 3D detection or "
            "classification pipeline. Pituitary class MUST NOT propagate as a "
            "disease label into the 3D pipeline."
        ),
        image_dimension  = ImageDimension.DIM_2D,
        training_role    = TrainingRole.PRETRAIN_ONLY,
        modalities       = frozenset({"image_2d"}),
        sub_cohorts      = ("glioma", "meningioma", "pituitary", "no_tumor"),
        has_segmentation = False,
        segmentation_labels = {},
        task_a_eligible  = False,   # 2D only
        task_b_eligible  = False,   # 2D only
        task_c_eligible  = False,   # pre-train only; pituitary not ranked
        license_type     = LicenseType.CC_BY_4,
        access_url       = (
            "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"
        ),
        access_instructions = (
            "1. Navigate to the Figshare URL above.\n"
            "2. Download the dataset ZIP file (no account required for CC-BY).\n"
            "3. Extract to data/raw/figshare_2d/.\n"
            "4. Expected folder structure: glioma/, meningioma/, pituitary/, "
            "no_tumor/\n"
            "⚠ Do NOT place in the same raw_subdir as BraTS 3D data."
        ),
        raw_subdir       = "data/raw/figshare_2d",
        keep_separate    = True,
        pituitary_excluded = True,  # pituitary blocked from 3D label space
    ),

    "oasis3": SourceMetadata(
        source_id        = "oasis3",
        display_name     = "OASIS-3 (Longitudinal Structural MRI)",
        description      = (
            "OASIS-3: Longitudinal Multimodal Neuroimaging, Clinical, and "
            "Cognitive Dataset for Normal Aging and Alzheimer's Disease. "
            "Contains T1w structural MRI with CDR and dx_group labels. "
            "Healthy controls = only explicitly labelled 'CN' (CDR=0). "
            "Missing or ambiguous labels MUST NOT be assumed healthy."
        ),
        image_dimension  = ImageDimension.DIM_3D,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = frozenset({"t1"}),
        sub_cohorts      = ("CN", "MCI", "AD"),
        has_segmentation = False,
        segmentation_labels = {},
        task_a_eligible  = False,   # no voxel-level segmentation
        task_b_eligible  = False,
        task_c_eligible  = True,    # whole-brain classification
        license_type     = LicenseType.OASIS_DUA,
        access_url       = "https://www.oasis-brains.org/",
        access_instructions = (
            "1. Register at https://www.oasis-brains.org/\n"
            "2. Read and sign the Data Use Agreement.\n"
            "3. Access is typically granted within 1–2 business days.\n"
            "4. Download via the OASIS data portal (no direct URLs here).\n"
            "5. Store in data/raw/oasis3/ on an encrypted volume."
        ),
        raw_subdir       = "data/raw/oasis3",
        keep_separate    = True,
        pituitary_excluded = False,
    ),

    "adni": SourceMetadata(
        source_id        = "adni",
        display_name     = "ADNI (Alzheimer's Disease Neuroimaging Initiative)",
        description      = (
            "ADNI longitudinal structural MRI with clinical diagnosis labels: "
            "CN, EMCI, MCI, LMCI, AD, SMC. Access requires a signed DUA and "
            "IRB approval. Healthy controls = only explicit 'CN' records. "
            "Missing diagnosis fields are NEVER inferred as healthy."
        ),
        image_dimension  = ImageDimension.DIM_3D,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = frozenset({"t1"}),
        sub_cohorts      = ("CN", "EMCI", "MCI", "LMCI", "AD", "SMC"),
        has_segmentation = False,
        segmentation_labels = {},
        task_a_eligible  = False,
        task_b_eligible  = False,
        task_c_eligible  = True,
        license_type     = LicenseType.ADNI_DUA,
        access_url       = "https://adni.loni.usc.edu/",
        access_instructions = (
            "1. Visit https://adni.loni.usc.edu/data-samples/access-data/\n"
            "2. Complete the Data Use Agreement AND obtain IRB approval.\n"
            "3. Submit your application via the LONI IDA portal.\n"
            "4. After approval download via the ADNI data portal.\n"
            "5. Store in data/raw/adni/ on an encrypted volume."
        ),
        raw_subdir       = "data/raw/adni",
        keep_separate    = True,
        pituitary_excluded = False,
    ),

    "local": SourceMetadata(
        source_id        = "local",
        display_name     = "Local Directory (Manual Transfer)",
        description      = (
            "Data manually transferred to the local filesystem. Must be "
            "configured with a root_dir pointing to a directory with the "
            "expected case sub-directories."
        ),
        image_dimension  = ImageDimension.MIXED,
        training_role    = TrainingRole.TRAIN_VAL_TEST,
        modalities       = frozenset(),  # user-defined
        sub_cohorts      = (),
        has_segmentation = False,
        segmentation_labels = {},
        task_a_eligible  = True,
        task_b_eligible  = True,
        task_c_eligible  = True,
        license_type     = LicenseType.UNKNOWN,
        access_url       = "",
        access_instructions = "Copy data manually to the configured root_dir.",
        raw_subdir       = "data/raw/local",
        keep_separate    = False,
        pituitary_excluded = False,
    ),
}
"""Catalogue of all approved dataset sources.

Only sources listed here may be referenced in the Master Dataset Manifest.
"""


# ---------------------------------------------------------------------------
# Helpers — source queries
# ---------------------------------------------------------------------------

def get_3d_sources() -> List[str]:
    """Return source_ids that produce 3D images."""
    return [
        sid for sid, meta in APPROVED_SOURCES.items()
        if meta.image_dimension in (ImageDimension.DIM_3D, ImageDimension.MIXED)
    ]


def get_pretrain_only_sources() -> List[str]:
    """Return source_ids restricted to pre-training only."""
    return [
        sid for sid, meta in APPROVED_SOURCES.items()
        if meta.training_role == TrainingRole.PRETRAIN_ONLY
    ]


def get_task_eligible_sources(task: str) -> List[str]:
    """Return source_ids eligible for a given task ('a', 'b', or 'c')."""
    task = task.lower()
    if task == "a":
        return [sid for sid, m in APPROVED_SOURCES.items() if m.task_a_eligible]
    if task == "b":
        return [sid for sid, m in APPROVED_SOURCES.items() if m.task_b_eligible]
    if task == "c":
        return [sid for sid, m in APPROVED_SOURCES.items() if m.task_c_eligible]
    raise ValueError(f"Unknown task {task!r}; expected 'a', 'b', or 'c'.")


def is_pituitary_blocked(source_id: str) -> bool:
    """Return True if pituitary is blocked from 3D label space for this source."""
    return APPROVED_SOURCES.get(source_id, SourceMetadata(  # noqa: use .get
        source_id="", display_name="", description="",
        image_dimension=ImageDimension.DIM_2D,
        training_role=TrainingRole.PRETRAIN_ONLY,
        modalities=frozenset(), sub_cohorts=(), has_segmentation=False,
        segmentation_labels={}, task_a_eligible=False, task_b_eligible=False,
        task_c_eligible=False, license_type=LicenseType.UNKNOWN,
        access_url="", access_instructions="", raw_subdir="",
    )).pituitary_excluded


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


@dataclasses.dataclass
class StorageConfig:
    """Workspace / storage paths loaded from a YAML config file.

    Parameters
    ----------
    colab_workspace : Path
        Root workspace when running in Colab.
    persistent_base_path : Path
        Base directory for persistent artefacts (checkpoints, metrics, …).
    checkpoints_dir : str
    metrics_dir : str
    outputs_dir : str
    logs_dir : str
    """
    colab_workspace:      Path = dataclasses.field(default_factory=lambda: Path("/content"))
    persistent_base_path: Path = dataclasses.field(default_factory=lambda: Path("."))
    checkpoints_dir:      str  = "checkpoints"
    metrics_dir:          str  = "metrics"
    outputs_dir:          str  = "outputs"
    logs_dir:             str  = "logs"

    @property
    def workspace(self) -> Path:
        return self.colab_workspace if is_colab() else self.persistent_base_path

    @property
    def checkpoints_path(self) -> Path:
        return self.persistent_base_path / self.checkpoints_dir

    @property
    def metrics_path(self) -> Path:
        return self.persistent_base_path / self.metrics_dir

    def resolve_path(self, subdir_name: str) -> Path:
        """Resolve a named sub-directory under the persistent base path."""
        mapping = {
            "checkpoints": self.checkpoints_dir,
            "metrics":     self.metrics_dir,
            "outputs":     self.outputs_dir,
            "logs":        self.logs_dir,
        }
        sub = mapping.get(subdir_name, subdir_name)
        return self.persistent_base_path / sub

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "StorageConfig":
        """Load from a YAML file."""
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML is required to load StorageConfig.  "
                "Install it with:  pip install pyyaml"
            )
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        colab  = raw.get("colab", {})
        persist = raw.get("persistent", {})
        return cls(
            colab_workspace      = Path(colab.get("workspace", "/content")),
            persistent_base_path = Path(persist.get("base_path", ".")),
            checkpoints_dir      = persist.get("checkpoints_dir", "checkpoints"),
            metrics_dir          = persist.get("metrics_dir", "metrics"),
            outputs_dir          = persist.get("outputs_dir", "outputs"),
            logs_dir             = persist.get("logs_dir", "logs"),
        )


# ---------------------------------------------------------------------------
# Abstract base — DatasetSource
# ---------------------------------------------------------------------------

class DatasetSource(abc.ABC):
    """Abstract interface every dataset source must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def description(self) -> str: ...

    @property
    def metadata(self) -> Optional[SourceMetadata]:
        """Return the :class:`SourceMetadata` for this source, if registered."""
        return APPROVED_SOURCES.get(self.name)

    @abc.abstractmethod
    def required_modalities(self) -> FrozenSet[str]: ...

    @abc.abstractmethod
    def validate_credentials(self) -> bool: ...

    @abc.abstractmethod
    def list_cases(self) -> List[str]: ...

    @abc.abstractmethod
    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]: ...

    @abc.abstractmethod
    def verify_integrity(self) -> bool: ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"{__name__}.{self.__class__.__name__}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Singleton registry that maps source names to DatasetSource instances."""

    _instance: ClassVar[Optional["DatasetRegistry"]] = None
    _sources:  Dict[str, DatasetSource]

    def __new__(cls) -> "DatasetRegistry":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._sources = {}
            cls._instance = inst
        return cls._instance

    def register(self, source: DatasetSource) -> None:
        if source.name in self._sources:
            raise ValueError(
                f"A source named '{source.name}' is already registered."
            )
        self._sources[source.name] = source
        logger.info("Registered dataset source: %s", source.name)

    def unregister(self, name: str) -> None:
        if name not in self._sources:
            raise KeyError(f"No source named '{name}'.")
        del self._sources[name]
        logger.info("Unregistered dataset source: %s", name)

    def get_source(self, name: str) -> DatasetSource:
        try:
            return self._sources[name]
        except KeyError:
            available = ", ".join(sorted(self._sources)) or "(none)"
            raise KeyError(
                f"No dataset source named '{name}'. Available sources: {available}"
            ) from None

    def list_sources(self) -> List[str]:
        return sorted(self._sources)

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def __repr__(self) -> str:
        return f"<DatasetRegistry sources=[{', '.join(sorted(self._sources))}]>"


def register_source(**init_kwargs: Any) -> Callable[[Type[DatasetSource]], Type[DatasetSource]]:
    """Class decorator: instantiate and register a DatasetSource subclass."""
    def _decorator(cls: Type[DatasetSource]) -> Type[DatasetSource]:
        DatasetRegistry().register(cls(**init_kwargs))
        return cls
    return _decorator


# ---------------------------------------------------------------------------
# Built-in source implementations
# ---------------------------------------------------------------------------

class BraTS2024Source(DatasetSource):
    """Dataset source for one BraTS 2024 sub-cohort (GLI, MET, or MEN).

    Parameters
    ----------
    cohort : str
        One of ``"GLI"``, ``"MET"``, ``"MEN"``.
    synapse_project_id : str
        Synapse entity/project ID (placeholder until access is granted).
    credentials_path : Path
        Path to ``.synapseConfig`` (NEVER commit this file).
    output_dir : Path
        Local download destination.
    storage_config : StorageConfig | None
        Optional storage config for path resolution.
    """

    _COHORT_SOURCE_ID = {
        "GLI": "brats2024_gli",
        "MET": "brats2024_met",
        "MEN": "brats2024_men",
    }
    _COHORT_PLACEHOLDER_ID = {
        "GLI": BRATS2024_GLI_SYNAPSE_ID,
        "MET": BRATS2024_MET_SYNAPSE_ID,
        "MEN": BRATS2024_MEN_SYNAPSE_ID,
    }

    def __init__(
        self,
        cohort: str = "GLI",
        synapse_project_id: Optional[str] = None,
        credentials_path: Path | str = Path.home() / ".synapseConfig",
        output_dir: Optional[Path | str] = None,
        storage_config: Optional[StorageConfig] = None,
    ) -> None:
        cohort = cohort.upper()
        if cohort not in self._COHORT_SOURCE_ID:
            raise ValueError(
                f"Unknown BraTS 2024 cohort '{cohort}'. "
                f"Choose from: {list(self._COHORT_SOURCE_ID)}"
            )
        self._cohort = cohort
        self._source_id = self._COHORT_SOURCE_ID[cohort]
        self._synapse_id = synapse_project_id or self._COHORT_PLACEHOLDER_ID[cohort]
        self._credentials_path = Path(credentials_path)
        meta = APPROVED_SOURCES[self._source_id]
        self._output_dir = Path(output_dir) if output_dir else Path(meta.raw_subdir)
        self._storage_config = storage_config

        if self._synapse_id == self._COHORT_PLACEHOLDER_ID[cohort]:
            self.logger.warning(
                "BraTS2024Source(%s) is using a placeholder Synapse ID ('%s'). "
                "Replace it with the real ID after obtaining access at "
                "https://www.synapse.org/",
                cohort, self._synapse_id,
            )
        self._warn_if_credentials_in_repo()

    def _warn_if_credentials_in_repo(self) -> None:
        try:
            cred = self._credentials_path.resolve()
            for parent in cred.parents:
                if (parent / ".git").exists():
                    self.logger.warning(
                        "Credentials file '%s' is inside a Git repo (%s). "
                        "Ensure it is in .gitignore!",
                        cred, parent,
                    )
                    break
        except Exception:  # noqa: BLE001
            pass

    @property
    def name(self) -> str:
        return self._source_id

    @property
    def description(self) -> str:
        return APPROVED_SOURCES[self._source_id].description

    def required_modalities(self) -> FrozenSet[str]:
        return APPROVED_SOURCES[self._source_id].modalities

    def validate_credentials(self) -> bool:
        if not self._credentials_path.exists():
            self.logger.error(
                "Synapse credentials not found: %s", self._credentials_path
            )
            return False
        text = self._credentials_path.read_text(encoding="utf-8")
        if "authtoken" not in text.lower() and "username" not in text.lower():
            self.logger.warning(
                "Credentials file '%s' does not look like a valid .synapseConfig.",
                self._credentials_path,
            )
            return False
        return True

    def list_cases(self) -> List[str]:
        if not _HAS_SYNAPSE:
            raise RuntimeError(
                "synapseclient is required to list BraTS cases. "
                "Install it with:  pip install synapseclient"
            )
        if not self._credentials_path.exists():
            raise FileNotFoundError(
                f"Synapse credentials not found: {self._credentials_path}\n"
                "See access_instructions in APPROVED_SOURCES['brats2024_gli']."
            )
        syn = synapseclient.Synapse(configPath=str(self._credentials_path))
        syn.login(silent=True)
        children = list(syn.getChildren(self._synapse_id, includeTypes=["folder"]))
        return sorted(c["name"] for c in children)

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        target_dir = Path(output_dir) if output_dir is not None else self._output_dir
        meta = APPROVED_SOURCES[self._source_id]
        if dry_run:
            self.logger.info(
                "[DRY RUN] Would download BraTS 2024 %s from Synapse '%s' → %s",
                self._cohort, self._synapse_id, target_dir,
            )
            self.logger.info(
                "[DRY RUN] Expected modalities per case: %s",
                sorted(meta.modalities),
            )
            return []
        if not _HAS_SYNAPSE:
            raise RuntimeError("Install synapseclient:  pip install synapseclient")
        target_dir.mkdir(parents=True, exist_ok=True)
        cases = self.list_cases()
        fetched = []
        for case in cases:
            case_dir = target_dir / case
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                syn = synapseclient.Synapse(
                    configPath=str(self._credentials_path)
                )
                syn.login(silent=True)
                syn.get(case, downloadLocation=str(case_dir), ifcollision="keep.local")
                fetched.append(case_dir)
                self.logger.info("Downloaded %s → %s", case, case_dir)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Failed to download %s: %s", case, exc)
        return fetched

    def verify_integrity(self) -> bool:
        if not self._output_dir.exists():
            self.logger.error("Output dir missing: %s", self._output_dir)
            return False
        case_dirs = [p for p in self._output_dir.iterdir() if p.is_dir()]
        if not case_dirs:
            self.logger.error("No case directories in %s", self._output_dir)
            return False
        modalities = APPROVED_SOURCES[self._source_id].modalities
        ok = True
        for cd in sorted(case_dirs):
            for mod in sorted(modalities):
                expected_hy = cd / f"{cd.name}-{mod}.nii.gz"
                expected_us = cd / f"{cd.name}_{mod}.nii.gz"
                if not expected_hy.exists() and not expected_us.exists():
                    self.logger.warning("Missing: %s (or %s)", expected_hy, expected_us)
                    ok = False
        return ok


class FigshareBrainTumorSource(DatasetSource):
    """Dataset source for the Figshare 2D Brain Tumor dataset.

    .. important::
        This source is **pre-training only**.  Pituitary images are
        blocked from the 3D ranked label space.

    Parameters
    ----------
    root_dir : Path
        Root directory containing the class sub-folders (glioma/,
        meningioma/, pituitary/, no_tumor/).
    """

    def __init__(self, root_dir: Path | str = Path("data/raw/figshare_2d")) -> None:
        self._root_dir = Path(root_dir)

    @property
    def name(self) -> str:
        return "figshare_2d"

    @property
    def description(self) -> str:
        return APPROVED_SOURCES["figshare_2d"].description

    def required_modalities(self) -> FrozenSet[str]:
        return APPROVED_SOURCES["figshare_2d"].modalities

    def validate_credentials(self) -> bool:
        # CC-BY-4.0 — no credentials needed
        self.logger.info(
            "Figshare dataset is CC-BY-4.0; no credentials required."
        )
        return True

    def list_cases(self) -> List[str]:
        if not self._root_dir.exists():
            self.logger.warning("Figshare root_dir does not exist: %s", self._root_dir)
            return []
        cases = []
        for cls_dir in sorted(self._root_dir.iterdir()):
            if cls_dir.is_dir():
                for img in sorted(cls_dir.iterdir()):
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        cases.append(f"{cls_dir.name}/{img.name}")
        return cases

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        target_dir = Path(output_dir) if output_dir is not None else self._root_dir
        meta = APPROVED_SOURCES["figshare_2d"]
        if dry_run:
            self.logger.info(
                "[DRY RUN] Figshare download: visit %s → extract to %s",
                meta.access_url, target_dir,
            )
            self.logger.info(
                "[DRY RUN] ⚠ pituitary_excluded=True — pituitary class will "
                "NOT be added to the 3D ranked label space."
            )
            return []
        self.logger.warning(
            "Figshare dataset must be downloaded manually. "
            "Visit: %s", meta.access_url,
        )
        return []

    def verify_integrity(self) -> bool:
        expected_classes = set(APPROVED_SOURCES["figshare_2d"].sub_cohorts)
        if not self._root_dir.exists():
            self.logger.error("Figshare root_dir missing: %s", self._root_dir)
            return False
        found = {d.name for d in self._root_dir.iterdir() if d.is_dir()}
        missing = expected_classes - found
        if missing:
            self.logger.warning("Missing Figshare class dirs: %s", missing)
            return False
        return True


class OASISSource(DatasetSource):
    """OASIS-3 longitudinal structural MRI source.

    .. important::
        Only cases with an explicit ``'CN'`` diagnosis label are treated as
        healthy controls.  Missing or ambiguous diagnoses are NEVER inferred
        as healthy.

    Parameters
    ----------
    root_dir : Path
        Root directory of the OASIS-3 dataset (contains subject sub-dirs).
    """

    def __init__(self, root_dir: Path | str = Path("data/raw/oasis3")) -> None:
        self._root_dir = Path(root_dir)

    @property
    def name(self) -> str:
        return "oasis3"

    @property
    def description(self) -> str:
        return APPROVED_SOURCES["oasis3"].description

    def required_modalities(self) -> FrozenSet[str]:
        return APPROVED_SOURCES["oasis3"].modalities

    def validate_credentials(self) -> bool:
        if not self._root_dir.exists():
            self.logger.error(
                "OASIS-3 root not found: %s.  "
                "Obtain access at https://www.oasis-brains.org/ and "
                "download data manually.", self._root_dir,
            )
            return False
        return True

    def list_cases(self) -> List[str]:
        if not self._root_dir.exists():
            return []
        return sorted(d.name for d in self._root_dir.iterdir() if d.is_dir())

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        target_dir = Path(output_dir) if output_dir is not None else self._root_dir
        meta = APPROVED_SOURCES["oasis3"]
        if dry_run:
            self.logger.info(
                "[DRY RUN] OASIS-3 must be downloaded manually from %s "
                "after accepting the DUA.  Would place data in %s.",
                meta.access_url, target_dir,
            )
            return []
        self.logger.warning(
            "OASIS-3 must be downloaded manually. Visit: %s", meta.access_url,
        )
        return []

    def verify_integrity(self) -> bool:
        if not self._root_dir.exists():
            self.logger.error("OASIS-3 root missing: %s", self._root_dir)
            return False
        subjects = [d for d in self._root_dir.iterdir() if d.is_dir()]
        self.logger.info("OASIS-3: found %d subject dirs.", len(subjects))
        return len(subjects) > 0


class ADNISource(DatasetSource):
    """ADNI longitudinal structural MRI source.

    .. important::
        Only cases with an explicit ``'CN'`` diagnosis label are treated as
        healthy controls.  Missing or ambiguous diagnoses are NEVER inferred
        as healthy.

    Parameters
    ----------
    root_dir : Path
        Root directory of the downloaded ADNI dataset.
    """

    def __init__(self, root_dir: Path | str = Path("data/raw/adni")) -> None:
        self._root_dir = Path(root_dir)

    @property
    def name(self) -> str:
        return "adni"

    @property
    def description(self) -> str:
        return APPROVED_SOURCES["adni"].description

    def required_modalities(self) -> FrozenSet[str]:
        return APPROVED_SOURCES["adni"].modalities

    def validate_credentials(self) -> bool:
        if not self._root_dir.exists():
            self.logger.error(
                "ADNI root not found: %s.  "
                "Obtain access at https://adni.loni.usc.edu/ "
                "(requires IRB approval + DUA).", self._root_dir,
            )
            return False
        return True

    def list_cases(self) -> List[str]:
        if not self._root_dir.exists():
            return []
        return sorted(d.name for d in self._root_dir.iterdir() if d.is_dir())

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        target_dir = Path(output_dir) if output_dir is not None else self._root_dir
        meta = APPROVED_SOURCES["adni"]
        if dry_run:
            self.logger.info(
                "[DRY RUN] ADNI must be downloaded manually from %s "
                "after IRB approval + DUA.  Would place data in %s.",
                meta.access_url, target_dir,
            )
            return []
        self.logger.warning(
            "ADNI must be downloaded manually. Visit: %s", meta.access_url,
        )
        return []

    def verify_integrity(self) -> bool:
        if not self._root_dir.exists():
            self.logger.error("ADNI root missing: %s", self._root_dir)
            return False
        subjects = [d for d in self._root_dir.iterdir() if d.is_dir()]
        self.logger.info("ADNI: found %d subject dirs.", len(subjects))
        return len(subjects) > 0


class LocalDirectorySource(DatasetSource):
    """Dataset source backed by a local directory (manual transfer)."""

    def __init__(
        self,
        root_dir: Path | str = Path("data/raw/local"),
        expected_modalities: Optional[FrozenSet[str] | Set[str]] = None,
        storage_config: Optional[StorageConfig] = None,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._modalities = set(expected_modalities) if expected_modalities else set(BRATS_EXPECTED_MODALITIES)
        self._storage_config = storage_config

    @property
    def resolved_root(self) -> Path:
        if self._storage_config is not None:
            return self._storage_config.workspace / self._root_dir
        return self._root_dir

    @property
    def name(self) -> str:
        return "local"

    @property
    def description(self) -> str:
        return APPROVED_SOURCES["local"].description

    def required_modalities(self) -> FrozenSet[str]:
        return frozenset(self._modalities)

    def validate_credentials(self) -> bool:
        return True

    def list_cases(self) -> List[str]:
        root = self.resolved_root
        if not root.exists():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        target = Path(output_dir) if output_dir is not None else self.resolved_root
        cases = self.list_cases()
        paths = [target / c for c in cases]
        if dry_run:
            for p in paths:
                self.logger.info("[DRY RUN] Would use local case: %s", p)
        return paths

    def verify_integrity(self) -> bool:
        root = self.resolved_root
        if not root.exists():
            self.logger.error("Local root missing: %s", root)
            return False
        return True


class BraTSSource(DatasetSource):
    """Dataset source for the official BraTS challenge data hosted on Synapse.

    Maintained for backward compatibility with the generic BraTS pipeline and
    existing notebook / pipeline scripts.
    """

    def __init__(
        self,
        synapse_project_id: str = SYNAPSE_PROJECT_ID,
        credentials_path: Path | str = Path.home() / ".synapseConfig",
        output_dir: Path | str = Path("data/raw/brats"),
        storage_config: Optional[StorageConfig] = None,
    ) -> None:
        self._project_id = synapse_project_id
        self._credentials_path = Path(credentials_path)
        self._output_dir = Path(output_dir)
        self._storage_config = storage_config

        if self._project_id == SYNAPSE_PROJECT_ID:
            logger.warning(
                "BraTSSource is using the placeholder Synapse project ID "
                "('%s').  Replace it with the real project ID after you have "
                "been granted access.",
                SYNAPSE_PROJECT_ID,
            )
        self._warn_if_credentials_in_repo()

    def _warn_if_credentials_in_repo(self) -> None:
        try:
            cred_resolved = self._credentials_path.resolve()
            for parent in cred_resolved.parents:
                if (parent / ".git").exists():
                    logger.warning(
                        "Credentials file '%s' appears to be inside a Git "
                        "repository (%s).  Make sure it is listed in "
                        ".gitignore to avoid accidental commits!",
                        cred_resolved,
                        parent,
                    )
                    break
        except Exception:
            pass

    def _get_synapse_client(self) -> Any:
        if not _HAS_SYNAPSE:
            raise RuntimeError(
                "The 'synapseclient' package is required to interact with "
                "Synapse.  Install it with:  pip install synapseclient"
            )
        if not self._credentials_path.exists():
            raise FileNotFoundError(
                f"Synapse credentials file not found: {self._credentials_path}\n"
                f"Create one by following the instructions at "
                f"https://python-docs.synapse.org/build/html/Credentials.html"
            )
        try:
            syn = synapseclient.Synapse(configPath=str(self._credentials_path))
            syn.login(silent=True)
            return syn
        except Exception as exc:
            raise RuntimeError(
                f"Failed to authenticate with Synapse using "
                f"'{self._credentials_path}': {exc}"
            ) from exc

    @property
    def resolved_output_dir(self) -> Path:
        if self._storage_config is not None:
            return self._storage_config.workspace / self._output_dir
        return self._output_dir

    @property
    def name(self) -> str:
        return "brats"

    @property
    def description(self) -> str:
        return (
            "Official BraTS challenge data via Synapse "
            f"(project: {self._project_id})"
        )

    def required_modalities(self) -> Set[str]:
        return set(BRATS_EXPECTED_MODALITIES)

    def validate_credentials(self) -> bool:
        if not self._credentials_path.exists():
            logger.error(
                "Synapse credentials file does not exist: %s",
                self._credentials_path,
            )
            return False
        try:
            text = self._credentials_path.read_text(encoding="utf-8")
            if "authtoken" not in text.lower() and "username" not in text.lower():
                logger.warning(
                    "Credentials file '%s' does not appear to contain an "
                    "auth token or username.",
                    self._credentials_path,
                )
                return False
        except Exception as exc:
            logger.error(
                "Failed to read credentials file '%s': %s",
                self._credentials_path,
                exc,
            )
            return False
        logger.info("Synapse credentials file looks valid: %s", self._credentials_path)
        return True

    def list_cases(self) -> List[str]:
        syn = self._get_synapse_client()
        try:
            children = list(
                syn.getChildren(self._project_id, includeTypes=["folder"])
            )
            cases = sorted(child["name"] for child in children)
            logger.info("Found %d cases in Synapse project %s", len(cases), self._project_id)
            return cases
        except Exception as exc:
            raise RuntimeError(
                f"Failed to list cases from Synapse project '{self._project_id}': {exc}"
            ) from exc

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        syn = self._get_synapse_client()
        output = Path(output_dir) if output_dir is not None else self.resolved_output_dir
        if dry_run:
            logger.info("[DRY RUN] Would create output directory: %s", output)
        else:
            output.mkdir(parents=True, exist_ok=True)
        cases = self.list_cases()
        fetched_paths: List[Path] = []
        for case_name in cases:
            case_dir = output / case_name
            fetched_paths.append(case_dir)
            if dry_run:
                logger.info("[DRY RUN] Would download case '%s' → %s", case_name, case_dir)
                for mod in sorted(self.required_modalities()):
                    logger.info("[DRY RUN]   Expected modality file: %s_%s.nii.gz", case_name, mod)
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            syn.get(case_name, downloadLocation=str(case_dir), ifcollision="keep.local")
        return fetched_paths

    def verify_integrity(self) -> bool:
        output = self.resolved_output_dir
        if not output.exists():
            logger.error("Output directory does not exist: %s", output)
            return False
        case_dirs = sorted(p for p in output.iterdir() if p.is_dir())
        if not case_dirs:
            logger.error("No case directories found in %s", output)
            return False
        all_valid = True
        for case_dir in case_dirs:
            for mod in sorted(self.required_modalities()):
                expected_us = case_dir / f"{case_dir.name}_{mod}.nii.gz"
                expected_hy = case_dir / f"{case_dir.name}-{mod}.nii.gz"
                if not expected_us.exists() and not expected_hy.exists():
                    logger.warning("Missing modality file: %s", expected_us)
                    all_valid = False
        return all_valid


class GenericAuthorizedSource(DatasetSource):
    """Template source for other authorised medical imaging datasets."""

    VALID_AUTH_METHODS: ClassVar[Set[str]] = {"api_key", "oauth", "manual"}

    def __init__(
        self,
        source_name: str = "generic_authorized",
        source_description: str = "Generic authorized medical dataset source",
        base_url: str = "https://example.com/api/v1",
        auth_method: str = "manual",
        expected_modalities: Optional[Set[str]] = None,
        file_manifest: Optional[List[str]] = None,
        credentials_path: Optional[Path | str] = None,
        output_dir: Path | str = Path("data/raw/generic"),
        storage_config: Optional[StorageConfig] = None,
    ) -> None:
        if auth_method not in self.VALID_AUTH_METHODS:
            raise ValueError(
                f"Invalid auth_method '{auth_method}'.  "
                f"Must be one of: {sorted(self.VALID_AUTH_METHODS)}"
            )

        self._source_name = source_name
        self._source_description = source_description
        self._base_url = base_url
        self._auth_method = auth_method
        self._expected_modalities = (
            expected_modalities
            if expected_modalities is not None
            else set(BRATS_EXPECTED_MODALITIES)
        )
        self._file_manifest = file_manifest or []
        self._credentials_path = (
            Path(credentials_path) if credentials_path is not None else None
        )
        self._output_dir = Path(output_dir)
        self._storage_config = storage_config

    @property
    def resolved_output_dir(self) -> Path:
        if self._storage_config is not None:
            return self._storage_config.workspace / self._output_dir
        return self._output_dir

    @property
    def name(self) -> str:
        return self._source_name

    @property
    def description(self) -> str:
        return self._source_description

    def required_modalities(self) -> Set[str]:
        return set(self._expected_modalities)

    def validate_credentials(self) -> bool:
        if self._auth_method == "manual":
            logger.info("Auth method is 'manual' — no credentials to validate.")
            return True

        if self._credentials_path is None:
            logger.error("Auth method '%s' requires a credentials_path, but none was provided.", self._auth_method)
            return False

        if not self._credentials_path.exists():
            logger.error("Credentials file not found: %s", self._credentials_path)
            return False

        return True

    def list_cases(self) -> List[str]:
        output = self.resolved_output_dir
        if not output.exists():
            return []
        return sorted(p.name for p in output.iterdir() if p.is_dir())

    def fetch(
        self,
        output_dir: Optional[Path | str] = None,
        *,
        dry_run: bool = False,
    ) -> List[Path]:
        output = Path(output_dir) if output_dir is not None else self.resolved_output_dir
        if self._auth_method == "manual":
            if dry_run:
                logger.info("[DRY RUN] Auth method is 'manual'. Would look for data in %s", output)
            cases = self.list_cases()
            return [output / c for c in cases]

        if dry_run:
            logger.info(
                "[DRY RUN] Would authenticate via '%s' using credentials at '%s' and download from %s into %s.",
                self._auth_method,
                self._credentials_path,
                self._base_url,
                output,
            )
            return []

        logger.warning(
            "GenericAuthorizedSource.fetch() with auth_method='%s' is a template.",
            self._auth_method,
        )
        return []

    def verify_integrity(self) -> bool:
        output = self.resolved_output_dir
        if not output.exists():
            logger.error("Output directory does not exist: %s", output)
            return False
        case_dirs = sorted(p for p in output.iterdir() if p.is_dir())
        if not case_dirs:
            logger.error("No case directories found in %s", output)
            return False
        all_valid = True
        for case_dir in case_dirs:
            for mod in sorted(self.required_modalities()):
                expected_us = case_dir / f"{case_dir.name}_{mod}.nii.gz"
                expected_hy = case_dir / f"{case_dir.name}-{mod}.nii.gz"
                if not expected_us.exists() and not expected_hy.exists():
                    logger.warning("Missing modality file: %s", expected_us)
                    all_valid = False
        return all_valid


# ---------------------------------------------------------------------------
# Auto-register built-in sources
# ---------------------------------------------------------------------------

def _register_builtin_sources() -> None:
    """Register all approved built-in sources with default parameters."""
    registry = DatasetRegistry()
    _pairs: List[Tuple[str, DatasetSource]] = [
        ("brats2024_gli", BraTS2024Source(cohort="GLI")),
        ("brats2024_met", BraTS2024Source(cohort="MET")),
        ("brats2024_men", BraTS2024Source(cohort="MEN")),
        ("figshare_2d",   FigshareBrainTumorSource()),
        ("oasis3",        OASISSource()),
        ("adni",          ADNISource()),
        ("local",         LocalDirectorySource()),
        ("brats",         BraTSSource()),
        ("generic_authorized", GenericAuthorizedSource()),
    ]
    for source_id, source in _pairs:
        try:
            if source_id not in registry:
                registry.register(source)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not register %s: %s", source_id, exc)


_register_builtin_sources()

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "ImageDimension",
    "TrainingRole",
    "LicenseType",
    # Constants
    "SYNAPSE_PROJECT_ID",
    "BRATS_EXPECTED_MODALITIES",
    "BRATS2024_GLI_SYNAPSE_ID",
    "BRATS2024_MET_SYNAPSE_ID",
    "BRATS2024_MEN_SYNAPSE_ID",
    "BRATS2024_MODALITIES",
    "BRATS2024_SEG_MODALITY",
    "BRATS2024_LABELS",
    "FIGSHARE_2D_CLASSES",
    "NEURODEGEN_DIAGNOSIS_LABELS",
    "HEALTHY_CONTROL_LABEL",
    "APPROVED_SOURCES",
    # Metadata
    "SourceMetadata",
    # Helpers
    "get_3d_sources",
    "get_pretrain_only_sources",
    "get_task_eligible_sources",
    "is_pituitary_blocked",
    # Storage
    "is_colab",
    "StorageConfig",
    # ABC & registry
    "DatasetSource",
    "DatasetRegistry",
    "register_source",
    # Implementations
    "BraTSSource",
    "BraTS2024Source",
    "GenericAuthorizedSource",
    "FigshareBrainTumorSource",
    "OASISSource",
    "ADNISource",
    "LocalDirectorySource",
]
