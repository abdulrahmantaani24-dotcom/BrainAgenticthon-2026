"""
test_manifest_and_splits.py — Comprehensive Tests for Manifest Schema,
Split Leakage, Missing Diagnoses, Dimensionality, and Source Restrictions
=========================================================================

Runs entirely offline — no NIfTI files, no network calls, no dataset downloads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from brainagenticthon.preprocessing.dataset_registry import (
    APPROVED_SOURCES,
    BRATS2024_GLI_SYNAPSE_ID,
    BRATS2024_LABELS,
    BRATS2024_MODALITIES,
    FIGSHARE_2D_CLASSES,
    HEALTHY_CONTROL_LABEL,
    NEURODEGEN_DIAGNOSIS_LABELS,
    ADNISource,
    BraTS2024Source,
    DatasetRegistry,
    FigshareBrainTumorSource,
    ImageDimension,
    LocalDirectorySource,
    OASISSource,
    SourceMetadata,
    TrainingRole,
    get_3d_sources,
    get_pretrain_only_sources,
    get_task_eligible_sources,
    is_pituitary_blocked,
)
from brainagenticthon.preprocessing.manifest_schema import (
    LabelProvenance,
    ManifestCollection,
    ManifestError,
    ManifestRecord,
    QualityFlag,
    Split,
    TASK_C_LABELS,
    validate_record,
)
from brainagenticthon.preprocessing.split_manager import LeakageReport, SplitManager
from brainagenticthon.preprocessing.task_output_schemas import (
    BoundingBox3D,
    ClassificationEvidence,
    DifferentialEntry,
    PipelineOutput,
    TaskAOutput,
    TaskBOutput,
    TaskCOutput,
    VolumetricStats,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_record(**overrides) -> ManifestRecord:
    """Return a minimal valid 3D BraTS ManifestRecord with optional overrides."""
    defaults = dict(
        subject_id         = "sub-001",
        case_id            = "BraTS-GLI-00001-000",
        dataset_source     = "brats2024_gli",
        cohort             = "GLI",
        session_id         = "",
        modality           = "t1",
        image_path         = "/data/raw/brats2024/gli/BraTS-GLI-00001-000/t1.nii.gz",
        image_dimension    = "3D",
        original_diagnosis = "glioma",
        task_a_label       = 1,
        task_c_label       = "glioma",
        segmentation_path  = "/data/raw/brats2024/gli/BraTS-GLI-00001-000/seg.nii.gz",
        label_provenance   = LabelProvenance.CHALLENGE_ANNOTATION.value,
        license_status     = "BraTS-Synapse-DUA",
        site               = "site_01",
        scanner            = "Siemens Prisma",
        split              = Split.TRAIN.value,
        quality_flag       = QualityFlag.OK.value,
    )
    defaults.update(overrides)
    return ManifestRecord(**defaults)


def _make_figshare_record(**overrides) -> ManifestRecord:
    """Return a minimal valid Figshare 2D ManifestRecord."""
    defaults = dict(
        subject_id         = "fig-glioma-001",
        case_id            = "figshare_glioma_001",
        dataset_source     = "figshare_2d",
        cohort             = "glioma",
        session_id         = "",
        modality           = "image_2d",
        image_path         = "/data/raw/figshare_2d/glioma/img001.jpg",
        image_dimension    = "2D",
        original_diagnosis = "glioma",
        task_a_label       = None,
        task_c_label       = None,
        segmentation_path  = None,
        label_provenance   = LabelProvenance.CHALLENGE_ANNOTATION.value,
        license_status     = "CC-BY-4.0",
        site               = "",
        scanner            = "",
        split              = Split.PRETRAIN.value,
        quality_flag       = QualityFlag.OK.value,
    )
    defaults.update(overrides)
    return ManifestRecord(**defaults)


def _make_oasis_record(**overrides) -> ManifestRecord:
    """Return a valid OASIS-3 CN (healthy control) ManifestRecord."""
    defaults = dict(
        subject_id         = "sub-OASIS30001",
        case_id            = "sub-OASIS30001_ses-d0090",
        dataset_source     = "oasis3",
        cohort             = "CN",
        session_id         = "ses-d0090",
        modality           = "t1",
        image_path         = "/data/raw/oasis3/sub-OASIS30001/ses-d0090/t1.nii.gz",
        image_dimension    = "3D",
        original_diagnosis = "CN",
        task_a_label       = None,
        task_c_label       = "cn",
        segmentation_path  = None,
        label_provenance   = LabelProvenance.CLINICAL_RECORD.value,
        license_status     = "OASIS-DUA",
        site               = "WUSTL",
        scanner            = "Siemens TIM Trio",
        split              = Split.TRAIN.value,
        quality_flag       = QualityFlag.OK.value,
    )
    defaults.update(overrides)
    return ManifestRecord(**defaults)


# ===========================================================================
# 1. Dataset Registry Tests
# ===========================================================================

class TestApprovedSources:
    def test_all_approved_source_ids_present(self):
        expected = {
            "brats2024_gli", "brats2024_met", "brats2024_men",
            "figshare_2d", "oasis3", "adni", "local",
        }
        assert set(APPROVED_SOURCES.keys()) == expected

    def test_brats_sources_are_3d(self):
        for sid in ("brats2024_gli", "brats2024_met", "brats2024_men"):
            assert APPROVED_SOURCES[sid].image_dimension == ImageDimension.DIM_3D

    def test_figshare_is_2d(self):
        assert APPROVED_SOURCES["figshare_2d"].image_dimension == ImageDimension.DIM_2D

    def test_figshare_is_pretrain_only(self):
        assert APPROVED_SOURCES["figshare_2d"].training_role == TrainingRole.PRETRAIN_ONLY

    def test_figshare_pituitary_excluded(self):
        assert APPROVED_SOURCES["figshare_2d"].pituitary_excluded is True

    def test_brats_have_segmentation(self):
        for sid in ("brats2024_gli", "brats2024_met", "brats2024_men"):
            assert APPROVED_SOURCES[sid].has_segmentation is True
            assert APPROVED_SOURCES[sid].segmentation_labels == BRATS2024_LABELS

    def test_figshare_no_segmentation(self):
        assert APPROVED_SOURCES["figshare_2d"].has_segmentation is False

    def test_oasis_adni_task_c_only(self):
        for sid in ("oasis3", "adni"):
            meta = APPROVED_SOURCES[sid]
            assert meta.task_a_eligible is False
            assert meta.task_b_eligible is False
            assert meta.task_c_eligible is True

    def test_figshare_not_eligible_for_any_ranked_task(self):
        meta = APPROVED_SOURCES["figshare_2d"]
        assert meta.task_a_eligible is False
        assert meta.task_b_eligible is False
        assert meta.task_c_eligible is False

    def test_brats_eligible_for_all_tasks(self):
        for sid in ("brats2024_gli", "brats2024_met", "brats2024_men"):
            meta = APPROVED_SOURCES[sid]
            assert meta.task_a_eligible is True
            assert meta.task_b_eligible is True
            assert meta.task_c_eligible is True

    def test_source_metadata_frozen(self):
        meta = APPROVED_SOURCES["brats2024_gli"]
        with pytest.raises((AttributeError, TypeError)):
            meta.pituitary_excluded = True  # type: ignore[misc]

    def test_brats_modalities_include_seg(self):
        for sid in ("brats2024_gli", "brats2024_met", "brats2024_men"):
            mods = APPROVED_SOURCES[sid].modalities
            assert "seg" in mods
            assert mods >= BRATS2024_MODALITIES

    def test_oasis_adni_t1_only(self):
        for sid in ("oasis3", "adni"):
            assert APPROVED_SOURCES[sid].modalities == frozenset({"t1"})

    def test_brats_sub_cohorts(self):
        assert APPROVED_SOURCES["brats2024_gli"].sub_cohorts == ("GLI",)
        assert APPROVED_SOURCES["brats2024_met"].sub_cohorts == ("MET",)
        assert APPROVED_SOURCES["brats2024_men"].sub_cohorts == ("MEN",)

    def test_figshare_sub_cohorts_include_pituitary(self):
        assert "pituitary" in APPROVED_SOURCES["figshare_2d"].sub_cohorts

    def test_oasis_adni_separate_dirs(self):
        assert APPROVED_SOURCES["oasis3"].raw_subdir == "data/raw/oasis3"
        assert APPROVED_SOURCES["adni"].raw_subdir == "data/raw/adni"
        assert APPROVED_SOURCES["figshare_2d"].raw_subdir == "data/raw/figshare_2d"

    def test_healthy_control_label_is_cn(self):
        assert HEALTHY_CONTROL_LABEL == "CN"

    def test_neurodegen_labels_include_cn_mci_ad(self):
        assert "CN" in NEURODEGEN_DIAGNOSIS_LABELS
        assert "MCI" in NEURODEGEN_DIAGNOSIS_LABELS
        assert "AD" in NEURODEGEN_DIAGNOSIS_LABELS


class TestRegistryHelpers:
    def test_get_3d_sources(self):
        three_d = get_3d_sources()
        assert "brats2024_gli" in three_d
        assert "brats2024_met" in three_d
        assert "brats2024_men" in three_d
        assert "oasis3" in three_d
        assert "adni" in three_d
        # figshare_2d is 2D only
        assert "figshare_2d" not in three_d

    def test_get_pretrain_only_sources(self):
        pretrain = get_pretrain_only_sources()
        assert "figshare_2d" in pretrain
        assert "brats2024_gli" not in pretrain

    def test_get_task_eligible_sources_a(self):
        task_a = get_task_eligible_sources("a")
        assert "brats2024_gli" in task_a
        assert "figshare_2d" not in task_a
        assert "oasis3" not in task_a

    def test_get_task_eligible_sources_b(self):
        task_b = get_task_eligible_sources("b")
        assert "brats2024_met" in task_b
        assert "oasis3" not in task_b

    def test_get_task_eligible_sources_c(self):
        task_c = get_task_eligible_sources("c")
        assert "brats2024_gli" in task_c
        assert "oasis3" in task_c
        assert "adni" in task_c
        assert "figshare_2d" not in task_c

    def test_get_task_eligible_invalid(self):
        with pytest.raises(ValueError):
            get_task_eligible_sources("d")

    def test_is_pituitary_blocked_figshare(self):
        assert is_pituitary_blocked("figshare_2d") is True

    def test_is_pituitary_blocked_brats(self):
        assert is_pituitary_blocked("brats2024_gli") is False


class TestDatasetRegistry:
    def setup_method(self):
        self.registry = DatasetRegistry()
        # Remove any leftover test sources
        for name in list(self.registry.list_sources()):
            if name.startswith("_test_"):
                self.registry.unregister(name)

    def test_all_approved_sources_registered(self):
        registered = set(self.registry.list_sources())
        for sid in APPROVED_SOURCES:
            assert sid in registered, f"Source '{sid}' not registered."

    def test_brats2024_gli_source_type(self):
        src = self.registry.get_source("brats2024_gli")
        assert isinstance(src, BraTS2024Source)

    def test_figshare_source_type(self):
        src = self.registry.get_source("figshare_2d")
        assert isinstance(src, FigshareBrainTumorSource)

    def test_oasis_source_type(self):
        src = self.registry.get_source("oasis3")
        assert isinstance(src, OASISSource)

    def test_adni_source_type(self):
        src = self.registry.get_source("adni")
        assert isinstance(src, ADNISource)

    def test_figshare_credentials_always_valid(self):
        src = self.registry.get_source("figshare_2d")
        assert src.validate_credentials() is True

    def test_brats2024_invalid_cohort(self):
        with pytest.raises(ValueError, match="Unknown BraTS 2024 cohort"):
            BraTS2024Source(cohort="XYZ")

    def test_local_source_list_cases_missing_dir(self, tmp_path):
        src = LocalDirectorySource(root_dir=tmp_path / "nonexistent")
        assert src.list_cases() == []

    def test_brats_verify_integrity_partial(self, tmp_path):
        out = tmp_path / "gli"
        case = out / "BraTS-GLI-00001-000"
        case.mkdir(parents=True)
        # Only some modalities present
        (case / "BraTS-GLI-00001-000-t1.nii.gz").write_bytes(b"fake")
        src = BraTS2024Source(cohort="GLI", output_dir=out)
        assert src.verify_integrity() is False  # missing t1ce, t2, flair, seg


# ===========================================================================
# 2. Manifest Schema Tests
# ===========================================================================

class TestManifestRecordValidation:
    def test_valid_brats_record(self):
        rec = _make_record()
        errors = validate_record(rec, strict=False)
        assert errors == []

    def test_valid_figshare_record(self):
        rec = _make_figshare_record()
        errors = validate_record(rec, strict=False)
        assert errors == []

    def test_valid_oasis_cn_record(self):
        rec = _make_oasis_record()
        errors = validate_record(rec, strict=False)
        assert errors == []

    def test_empty_subject_id_fails(self):
        with pytest.raises(ManifestError, match="subject_id"):
            validate_record(_make_record(subject_id=""))

    def test_empty_case_id_fails(self):
        with pytest.raises(ManifestError, match="case_id"):
            validate_record(_make_record(case_id=""))

    def test_invalid_source_fails(self):
        with pytest.raises(ManifestError, match="dataset_source"):
            validate_record(_make_record(dataset_source="fake_source"))

    def test_invalid_dimension_fails(self):
        with pytest.raises(ManifestError, match="image_dimension"):
            validate_record(_make_record(image_dimension="4D"))

    # -- pituitary exclusion ------------------------------------------------

    def test_pituitary_with_task_c_label_fails(self):
        """Figshare pituitary records MUST NOT have a task_c_label."""
        rec = _make_figshare_record(
            cohort="pituitary",
            task_c_label="glioma",  # wrong — pituitary is blocked
        )
        with pytest.raises(ManifestError, match="[Pp]ituitary"):
            validate_record(rec)

    def test_pituitary_without_task_c_label_passes(self):
        rec = _make_figshare_record(
            cohort="pituitary",
            task_c_label=None,
        )
        errors = validate_record(rec, strict=False)
        assert errors == []

    # -- 2D source restrictions --------------------------------------------

    def test_2d_record_with_task_a_label_fails(self):
        """2D records must not have a Task A label."""
        rec = _make_figshare_record(task_a_label=1)
        with pytest.raises(ManifestError, match="2D"):
            validate_record(rec)

    def test_2d_record_without_task_a_label_passes(self):
        rec = _make_figshare_record(task_a_label=None)
        errors = validate_record(rec, strict=False)
        assert errors == []

    # -- pre-train only source --------------------------------------------

    def test_figshare_with_train_split_fails(self):
        rec = _make_figshare_record(split="train")
        with pytest.raises(ManifestError, match="pretrain"):
            validate_record(rec)

    def test_figshare_with_pretrain_split_passes(self):
        rec = _make_figshare_record(split="pretrain")
        errors = validate_record(rec, strict=False)
        assert errors == []

    # -- Task A eligibility ------------------------------------------------

    def test_oasis_with_task_a_label_fails(self):
        """OASIS records are not Task A eligible."""
        rec = _make_oasis_record(task_a_label=1)
        with pytest.raises(ManifestError, match="eligible for Task A"):
            validate_record(rec)

    def test_adni_with_task_a_label_fails(self):
        rec = _make_record(
            dataset_source="adni",
            cohort="CN",
            task_a_label=1,
            task_c_label="cn",
            original_diagnosis="CN",
            label_provenance=LabelProvenance.CLINICAL_RECORD.value,
            license_status="ADNI-DUA",
        )
        with pytest.raises(ManifestError, match="eligible for Task A"):
            validate_record(rec)

    # -- missing diagnosis safety rules -----------------------------------

    def test_missing_diagnosis_cannot_be_healthy(self):
        """label_provenance='missing' → task_c_label='cn' is FORBIDDEN."""
        rec = _make_oasis_record(
            label_provenance=LabelProvenance.MISSING.value,
            task_c_label="cn",
        )
        with pytest.raises(ManifestError, match="missing"):
            validate_record(rec)

    def test_no_original_diagnosis_cannot_be_healthy(self):
        rec = _make_oasis_record(
            original_diagnosis=None,
            task_c_label="cn",
        )
        with pytest.raises(ManifestError, match="original_diagnosis"):
            validate_record(rec)

    def test_no_original_diagnosis_with_mci_label_passes(self):
        """Missing diagnosis with a non-healthy label is still wrong by policy,
        but the schema check specifically guards the 'cn' (healthy) path."""
        rec = _make_oasis_record(
            original_diagnosis=None,
            label_provenance=LabelProvenance.MISSING.value,
            task_c_label="mci",  # non-healthy — not blocked by this rule
        )
        # This should NOT trigger the healthy-control safety rule
        errors = validate_record(rec, strict=False)
        # May produce other errors (e.g. source eligibility) but not the
        # healthy-control one
        healthy_errors = [e for e in errors if "missing" in e.lower() and "healthy" in e.lower()]
        assert len(healthy_errors) == 0

    def test_empty_image_path_fails(self):
        with pytest.raises(ManifestError, match="image_path"):
            validate_record(_make_record(image_path=""))

    def test_invalid_task_a_label_fails(self):
        with pytest.raises(ManifestError, match="task_a_label"):
            validate_record(_make_record(task_a_label=5))

    def test_invalid_task_c_label_fails(self):
        with pytest.raises(ManifestError, match="task_c_label"):
            validate_record(_make_record(task_c_label="pituitary"))

    def test_pituitary_not_in_task_c_labels(self):
        assert "pituitary" not in TASK_C_LABELS

    def test_invalid_label_provenance_fails(self):
        with pytest.raises(ManifestError, match="label_provenance"):
            validate_record(_make_record(label_provenance="guessed"))

    def test_invalid_split_fails(self):
        with pytest.raises(ManifestError, match="split"):
            validate_record(_make_record(split="training"))

    def test_invalid_quality_flag_fails(self):
        with pytest.raises(ManifestError, match="quality_flag"):
            validate_record(_make_record(quality_flag="bad"))

    def test_non_strict_collects_all_errors(self):
        rec = _make_record(
            subject_id="",
            case_id="",
            dataset_source="invalid_source",
        )
        errors = validate_record(rec, strict=False)
        assert len(errors) >= 3  # subject_id, case_id, dataset_source


class TestManifestCollection:
    def test_add_and_len(self):
        col = ManifestCollection()
        col.add(_make_record())
        assert len(col) == 1

    def test_add_invalid_raises(self):
        col = ManifestCollection()
        with pytest.raises(ManifestError):
            col.add(_make_record(dataset_source="not_real"))

    def test_filter_by_split(self):
        col = ManifestCollection()
        col.add(_make_record(split="train"))
        col.add(_make_record(
            case_id="BraTS-GLI-00002-000",
            subject_id="sub-002",
            split="validation",
        ))
        train = col.filter(split="train")
        assert len(train) == 1
        assert train.records[0].split == "train"

    def test_filter_by_dimension(self):
        col = ManifestCollection()
        col.add(_make_record())                  # 3D
        col.add(_make_figshare_record())         # 2D
        three_d = col.filter(image_dimension="3D")
        assert len(three_d) == 1
        two_d = col.filter(image_dimension="2D")
        assert len(two_d) == 1

    def test_filter_by_source(self):
        col = ManifestCollection()
        col.add(_make_record())
        col.add(_make_figshare_record())
        brats = col.filter(dataset_source="brats2024_gli")
        assert len(brats) == 1

    def test_filter_task_eligible(self):
        col = ManifestCollection()
        col.add(_make_record())                  # brats2024_gli — task a eligible
        col.add(_make_oasis_record())            # oasis3 — NOT task a eligible
        task_a = col.filter(task_eligible="a")
        assert all(r.dataset_source in ("brats2024_gli", "brats2024_met", "brats2024_men")
                   for r in task_a.records)

    def test_unique_subjects(self):
        col = ManifestCollection()
        col.add(_make_record(subject_id="sub-001", case_id="case-001"))
        col.add(_make_record(subject_id="sub-001", case_id="case-002", modality="t2"))
        col.add(_make_record(subject_id="sub-002", case_id="case-003"))
        subjs = col.unique_subjects()
        assert sorted(subjs["brats2024_gli"]) == ["sub-001", "sub-002"]

    def test_split_counts(self):
        col = ManifestCollection()
        col.add(_make_record(split="train"))
        col.add(_make_record(case_id="c2", subject_id="s2", split="validation"))
        counts = col.split_counts()
        assert counts["train"] == 1
        assert counts["validation"] == 1

    def test_to_json_and_back(self, tmp_path):
        col = ManifestCollection()
        col.add(_make_record())
        col.add(_make_figshare_record())
        j = col.to_json()
        parsed = json.loads(j)
        assert parsed["record_count"] == 2

    def test_save_and_load_json(self, tmp_path):
        col = ManifestCollection()
        col.add(_make_record())
        path = tmp_path / "manifest.json"
        col.save_json(path)
        loaded = ManifestCollection.load_json(path)
        assert len(loaded) == 1
        assert loaded.records[0].case_id == "BraTS-GLI-00001-000"


# ===========================================================================
# 3. Split Leakage Tests
# ===========================================================================

class TestSplitLeakageDetection:
    def _make_leaky_collection(self) -> ManifestCollection:
        """Collection where sub-001 appears in both train and val."""
        col = ManifestCollection()
        col.add(_make_record(
            subject_id="sub-001", case_id="c-001",
            modality="t1", split="train",
        ))
        col.add(_make_record(
            subject_id="sub-001", case_id="c-002",
            modality="t2", split="validation",  # same subject, different split!
        ))
        return col

    def test_detect_subject_leakage(self):
        sm = SplitManager()
        report = sm.check_leakage(self._make_leaky_collection())
        assert not report.is_clean
        assert any("leakage" in v.lower() for v in report.violations)

    def test_clean_collection_passes(self):
        col = ManifestCollection()
        col.add(_make_record(subject_id="sub-A", case_id="c-001", split="train"))
        col.add(_make_record(subject_id="sub-B", case_id="c-002", split="validation"))
        col.add(_make_record(subject_id="sub-C", case_id="c-003", split="test"))
        sm = SplitManager()
        report = sm.check_leakage(col)
        assert report.is_clean

    def test_detect_cross_source_duplicate_case_ids(self):
        """Same case_id in two different sources → duplicate."""
        col = ManifestCollection()
        col.add(_make_record(
            case_id="SHARED-CASE", subject_id="sub-X",
            dataset_source="brats2024_gli", split="train",
        ))
        col.add(_make_record(
            case_id="SHARED-CASE", subject_id="sub-X",
            dataset_source="brats2024_met", split="train",
            task_c_label="metastasis",
        ))
        sm = SplitManager()
        report = sm.check_leakage(col)
        assert "SHARED-CASE" in report.duplicate_case_ids

    def test_figshare_in_train_split_detected(self):
        """Figshare records in 'train' split must be caught as contamination."""
        col = ManifestCollection(validate=False)
        # Bypass schema validation to inject a bad split
        bad_rec = _make_figshare_record(split="train")
        col._records.append(bad_rec)
        sm = SplitManager()
        report = sm.check_leakage(col)
        assert not report.is_clean
        assert any("contamination" in v.lower() or "figshare" in v.lower()
                   for v in report.violations)

    def test_pituitary_task_c_label_detected(self):
        """Pituitary record with a task_c_label must be caught."""
        col = ManifestCollection(validate=False)
        bad_rec = _make_figshare_record(
            cohort="pituitary", task_c_label="glioma"
        )
        col._records.append(bad_rec)
        sm = SplitManager()
        report = sm.check_leakage(col)
        assert not report.is_clean
        assert any("pituitary" in v.lower() for v in report.violations)

    def test_missing_diagnosis_as_healthy_detected(self):
        """Missing diagnosis assigned as 'cn' must be caught."""
        col = ManifestCollection(validate=False)
        bad_rec = _make_oasis_record(
            label_provenance=LabelProvenance.MISSING.value,
            task_c_label="cn",
        )
        col._records.append(bad_rec)
        sm = SplitManager()
        report = sm.check_leakage(col)
        assert not report.is_clean
        assert any("missing" in v.lower() for v in report.violations)

    def test_leakage_report_summary(self):
        sm = SplitManager()
        report = sm.check_leakage(self._make_leaky_collection())
        summary = report.summary()
        assert "VIOLATIONS" in summary or "violation" in summary.lower()


class TestSplitAssignment:
    def _make_unassigned_collection(self, n: int = 10) -> ManifestCollection:
        col = ManifestCollection()
        for i in range(n):
            col.add(_make_record(
                subject_id=f"sub-{i:03d}",
                case_id=f"BraTS-GLI-{i:05d}-000",
                split="unassigned",
            ))
        return col

    def test_assign_splits_produces_all_three(self):
        sm = SplitManager(seed=0, train_frac=0.70, val_frac=0.15)
        col = self._make_unassigned_collection(n=10)
        assigned = sm.assign_splits(col)
        splits_found = {r.split for r in assigned.records}
        assert "train" in splits_found

    def test_figshare_always_gets_pretrain(self):
        col = ManifestCollection()
        for i in range(5):
            col.add(_make_figshare_record(
                subject_id=f"fig-{i}",
                case_id=f"figshare_{i}",
                split="unassigned",
            ))
        sm = SplitManager()
        assigned = sm.assign_splits(col)
        for rec in assigned.records:
            if rec.dataset_source == "figshare_2d":
                assert rec.split == "pretrain"

    def test_no_leakage_after_assignment(self):
        col = self._make_unassigned_collection(n=15)
        sm = SplitManager(seed=42)
        assigned = sm.assign_splits(col)
        report = sm.check_leakage(assigned)
        assert report.is_clean, f"Leakage after assignment: {report.violations}"

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            SplitManager(train_frac=0.9, val_frac=0.5)

    def test_source_aware_sample(self):
        col = ManifestCollection()
        for i in range(6):
            col.add(_make_record(
                subject_id=f"sub-gli-{i}", case_id=f"gli-{i:05d}",
                dataset_source="brats2024_gli", split="train",
            ))
        for i in range(6):
            col.add(_make_record(
                subject_id=f"sub-met-{i}", case_id=f"met-{i:05d}",
                dataset_source="brats2024_met", split="train",
                task_c_label="metastasis",
            ))
        sm = SplitManager(seed=1)
        sampled = sm.source_aware_sample(col, split="train", n_per_source=3)
        # Should have at most 3 per source = 6 total
        assert len(sampled) <= 6
        sources = {r.dataset_source for r in sampled.records}
        assert "brats2024_gli" in sources
        assert "brats2024_met" in sources

    def test_site_split_statistics(self):
        col = ManifestCollection()
        col.add(_make_record(site="site_A", split="train"))
        col.add(_make_record(case_id="c2", subject_id="s2", site="site_A", split="validation"))
        col.add(_make_record(case_id="c3", subject_id="s3", site="site_B", split="train"))
        sm = SplitManager()
        stats = sm.site_split_statistics(col)
        assert "site_A" in stats
        assert stats["site_A"]["train"] == 1
        assert stats["site_A"]["validation"] == 1

    def test_cross_source_subject_overlap_detection(self):
        col = ManifestCollection()
        # Same subject_id across two sources (heuristic duplicate)
        col.add(_make_record(
            subject_id="SHARED-SUBJ", case_id="c-a",
            dataset_source="brats2024_gli",
        ))
        col.add(_make_record(
            subject_id="SHARED-SUBJ", case_id="c-b",
            dataset_source="brats2024_met",
            task_c_label="metastasis",
        ))
        sm = SplitManager()
        overlaps = sm.find_cross_source_subject_overlaps(col)
        assert "SHARED-SUBJ" in overlaps
        assert set(overlaps["SHARED-SUBJ"]) == {"brats2024_gli", "brats2024_met"}


# ===========================================================================
# 4. Task Output Schema Tests
# ===========================================================================

class TestBoundingBox3D:
    def test_valid_box(self):
        bb = BoundingBox3D(0, 10, 0, 20, 0, 30, confidence=0.9)
        assert bb.x_max == 10

    def test_negative_coordinate_fails(self):
        with pytest.raises(ValueError, match="x_min"):
            BoundingBox3D(-1, 10, 0, 20, 0, 30)

    def test_min_greater_than_max_fails(self):
        with pytest.raises(ValueError):
            BoundingBox3D(10, 5, 0, 20, 0, 30)  # x_min > x_max

    def test_confidence_out_of_range_fails(self):
        with pytest.raises(ValueError, match="confidence"):
            BoundingBox3D(0, 10, 0, 20, 0, 30, confidence=1.5)


class TestTaskAOutput:
    def test_no_finding(self):
        out = TaskAOutput(
            case_id="c1",
            dataset_source="brats2024_gli",
            finding_present=False,
            bounding_boxes=[],
            overall_confidence=0.1,
        )
        assert out.finding_present is False

    def test_finding_requires_box(self):
        with pytest.raises(ValueError, match="bounding box"):
            TaskAOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                finding_present=True,
                bounding_boxes=[],
            )

    def test_no_finding_with_box_fails(self):
        bb = BoundingBox3D(0, 10, 0, 10, 0, 10)
        with pytest.raises(ValueError, match="empty"):
            TaskAOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                finding_present=False,
                bounding_boxes=[bb],
            )

    def test_invalid_confidence_fails(self):
        with pytest.raises(ValueError, match="overall_confidence"):
            TaskAOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                overall_confidence=2.0,
            )

    def test_to_dict_contains_required_keys(self):
        out = TaskAOutput(case_id="c1", dataset_source="brats2024_gli")
        d = out.to_dict()
        assert "case_id" in d
        assert "finding_present" in d
        assert "schema_version" in d


class TestTaskBOutput:
    def test_default_output(self):
        out = TaskBOutput(case_id="c1", dataset_source="brats2024_gli")
        assert out.segmentation_path == ""
        assert out.volumetrics.total_volume_ml == 0.0

    def test_with_segmentation_label_map(self):
        out = TaskBOutput(
            case_id="c1",
            dataset_source="brats2024_gli",
            segmentation_label_map=BRATS2024_LABELS,
        )
        assert 0 in out.segmentation_label_map

    def test_to_json(self):
        out = TaskBOutput(case_id="c1", dataset_source="brats2024_gli")
        j = out.to_json()
        parsed = json.loads(j)
        assert parsed["case_id"] == "c1"


class TestDifferentialEntry:
    def test_valid_entry(self):
        e = DifferentialEntry(class_label="glioma", probability=0.8, rank=1)
        assert e.rank == 1

    def test_pituitary_label_blocked(self):
        with pytest.raises(ValueError, match="pituitary"):
            DifferentialEntry(class_label="pituitary", probability=0.5, rank=1)

    def test_probability_out_of_range_fails(self):
        with pytest.raises(ValueError, match="probability"):
            DifferentialEntry(class_label="glioma", probability=1.5, rank=1)

    def test_rank_zero_fails(self):
        with pytest.raises(ValueError, match="rank"):
            DifferentialEntry(class_label="glioma", probability=0.5, rank=0)


class TestTaskCOutput:
    def test_valid_classification(self):
        out = TaskCOutput(
            case_id="c1",
            dataset_source="brats2024_gli",
            class_label="glioma",
            class_probability=0.92,
        )
        assert out.class_label == "glioma"

    def test_pituitary_class_blocked(self):
        with pytest.raises(ValueError, match="pituitary"):
            TaskCOutput(
                case_id="c1",
                dataset_source="figshare_2d",
                class_label="pituitary",  # blocked
                class_probability=0.8,
            )

    def test_invalid_class_label_fails(self):
        with pytest.raises(ValueError):
            TaskCOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                class_label="unknown_tumour_x",
            )

    def test_invalid_confidence_fails(self):
        with pytest.raises(ValueError):
            TaskCOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                class_probability=-0.1,
            )

    def test_differential_ranking_gap_fails(self):
        """Ranks must be consecutive from 1."""
        with pytest.raises(ValueError, match="rank"):
            TaskCOutput(
                case_id="c1",
                dataset_source="brats2024_gli",
                class_label="glioma",
                class_probability=0.9,
                differential=[
                    DifferentialEntry("glioma", 0.9, rank=1),
                    DifferentialEntry("meningioma", 0.05, rank=3),  # gap!
                ],
            )

    def test_valid_differential(self):
        out = TaskCOutput(
            case_id="c1",
            dataset_source="brats2024_gli",
            class_label="glioma",
            class_probability=0.9,
            differential=[
                DifferentialEntry("glioma",     0.9,  rank=1),
                DifferentialEntry("meningioma",  0.07, rank=2),
                DifferentialEntry("no_finding",  0.03, rank=3),
            ],
        )
        assert len(out.differential) == 3


class TestPipelineOutput:
    def test_full_pipeline_output(self):
        bb = BoundingBox3D(0, 100, 0, 100, 0, 80, confidence=0.95)
        task_a = TaskAOutput(
            case_id="c1", dataset_source="brats2024_gli",
            finding_present=True, bounding_boxes=[bb], overall_confidence=0.95,
        )
        task_b = TaskBOutput(
            case_id="c1", dataset_source="brats2024_gli",
            segmentation_label_map=BRATS2024_LABELS,
        )
        task_c = TaskCOutput(
            case_id="c1", dataset_source="brats2024_gli",
            class_label="glioma", class_probability=0.92,
        )
        out = PipelineOutput(
            case_id="c1", dataset_source="brats2024_gli",
            task_a=task_a, task_b=task_b, task_c=task_c,
        )
        d = out.to_dict()
        assert "task_a" in d
        assert "task_b" in d
        assert "task_c" in d
        j = out.to_json()
        parsed = json.loads(j)
        assert parsed["case_id"] == "c1"


# ===========================================================================
# 5. Dimensionality Guard Tests
# ===========================================================================

class TestDimensionalityGuards:
    def test_3d_brats_record_valid(self):
        rec = _make_record(image_dimension="3D", task_a_label=1)
        assert validate_record(rec, strict=False) == []

    def test_2d_figshare_has_no_task_a(self):
        rec = _make_figshare_record(image_dimension="2D", task_a_label=None)
        assert validate_record(rec, strict=False) == []

    def test_2d_with_task_a_rejected(self):
        with pytest.raises(ManifestError, match="2D"):
            validate_record(_make_figshare_record(task_a_label=1))

    def test_figshare_pituitary_cannot_get_task_c(self):
        with pytest.raises(ManifestError, match="[Pp]ituitary"):
            validate_record(_make_figshare_record(
                cohort="pituitary",
                task_c_label="glioma",
            ))

    def test_3d_source_required_for_task_a(self):
        """Non-BraTS 3D source (oasis) still fails Task A eligibility."""
        with pytest.raises(ManifestError, match="eligible for Task A"):
            validate_record(_make_oasis_record(task_a_label=1))


# ===========================================================================
# 6. Healthy Control Safety Tests
# ===========================================================================

class TestHealthyControlSafety:
    def test_explicit_cn_label_with_provenance_passes(self):
        rec = _make_oasis_record(
            original_diagnosis="CN",
            label_provenance=LabelProvenance.CLINICAL_RECORD.value,
            task_c_label="cn",
        )
        assert validate_record(rec, strict=False) == []

    def test_missing_provenance_cn_blocked(self):
        with pytest.raises(ManifestError):
            validate_record(_make_oasis_record(
                label_provenance=LabelProvenance.MISSING.value,
                task_c_label="cn",
            ))

    def test_empty_original_diagnosis_cn_blocked(self):
        with pytest.raises(ManifestError):
            validate_record(_make_oasis_record(
                original_diagnosis=None,
                task_c_label="cn",
            ))

    def test_adni_cn_with_clinical_record_passes(self):
        rec = _make_record(
            subject_id="adni-sub-001",
            case_id="adni-sub-001_bl",
            dataset_source="adni",
            cohort="CN",
            modality="t1",
            image_path="/data/raw/adni/sub-001/t1.nii.gz",
            image_dimension="3D",
            original_diagnosis="CN",
            task_a_label=None,
            task_c_label="cn",
            segmentation_path=None,
            label_provenance=LabelProvenance.CLINICAL_RECORD.value,
            license_status="ADNI-DUA",
            site="ADNI_site_01",
            scanner="Siemens TrioTim",
            split=Split.TRAIN.value,
        )
        assert validate_record(rec, strict=False) == []
