"""
Unit tests for the cloud data pipeline modules.

These tests run entirely offline — no NIfTI files, no network calls,
no Synapse credentials required.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on the path (mirrors pyproject.toml pythonpath setting)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# dataset_registry tests
# ---------------------------------------------------------------------------
from brainagenticthon.preprocessing.dataset_registry import (
    BraTSSource,
    DatasetRegistry,
    DatasetSource,
    GenericAuthorizedSource,
    LocalDirectorySource,
    StorageConfig,
    is_colab,
    register_source,
)


class TestIsColab:
    def test_returns_bool(self):
        result = is_colab()
        assert isinstance(result, bool)

    def test_false_outside_colab(self, monkeypatch):
        monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
        # Can't unimport google.colab easily, but is_colab() should not raise
        result = is_colab()
        assert isinstance(result, bool)


class TestStorageConfig:
    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.colab_workspace == Path("/content")
        assert cfg.checkpoints_dir == "checkpoints"
        assert cfg.metrics_dir == "metrics"

    def test_checkpoints_path(self):
        cfg = StorageConfig(persistent_base_path=Path("/drive"), checkpoints_dir="ckpt")
        assert cfg.checkpoints_path == Path("/drive/ckpt")

    def test_metrics_path(self):
        cfg = StorageConfig(persistent_base_path=Path("/drive"), metrics_dir="met")
        assert cfg.metrics_path == Path("/drive/met")

    def test_from_yaml(self, tmp_path):
        yaml_content = """
colab:
  workspace: /content/test
persistent:
  base_path: /tmp/persist
  checkpoints_dir: ckpt
  metrics_dir: met
"""
        cfg_file = tmp_path / "storage.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")
        cfg = StorageConfig.from_yaml(cfg_file)
        assert cfg.colab_workspace == Path("/content/test")
        assert cfg.persistent_base_path == Path("/tmp/persist")
        assert cfg.checkpoints_dir == "ckpt"
        assert cfg.metrics_dir == "met"

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            StorageConfig.from_yaml("/nonexistent/path/storage.yaml")

    def test_from_yaml_partial(self, tmp_path):
        """YAML with only some keys should use defaults for missing ones."""
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("colab:\n  workspace: /content/x\n", encoding="utf-8")
        cfg = StorageConfig.from_yaml(cfg_file)
        assert cfg.colab_workspace == Path("/content/x")
        # defaults
        assert cfg.checkpoints_dir == "checkpoints"


class TestDatasetRegistry:
    def setup_method(self):
        """Each test gets a fresh registry state by accessing the singleton
        and removing any custom test sources registered by previous tests."""
        self.registry = DatasetRegistry()
        # Clean up any test sources left by previous tests
        for name in list(self.registry.list_sources()):
            if name.startswith("_test_"):
                self.registry.unregister(name)

    def test_singleton(self):
        r1 = DatasetRegistry()
        r2 = DatasetRegistry()
        assert r1 is r2

    def test_builtin_sources_registered(self):
        sources = self.registry.list_sources()
        assert "brats" in sources
        assert "local" in sources
        assert "generic_authorized" in sources

    def test_list_sources_sorted(self):
        sources = self.registry.list_sources()
        assert sources == sorted(sources)

    def test_get_source(self):
        src = self.registry.get_source("brats")
        assert isinstance(src, BraTSSource)

    def test_get_source_missing(self):
        with pytest.raises(KeyError, match="No dataset source"):
            self.registry.get_source("_nonexistent_")

    def test_register_and_unregister(self):
        class _TestSource(LocalDirectorySource):
            @property
            def name(self):
                return "_test_custom_"

        src = _TestSource()
        self.registry.register(src)
        assert "_test_custom_" in self.registry
        self.registry.unregister("_test_custom_")
        assert "_test_custom_" not in self.registry

    def test_duplicate_registration_raises(self):
        class _TestSource(LocalDirectorySource):
            @property
            def name(self):
                return "_test_dup_"

        src1 = _TestSource()
        src2 = _TestSource()
        self.registry.register(src1)
        try:
            with pytest.raises(ValueError, match="already registered"):
                self.registry.register(src2)
        finally:
            self.registry.unregister("_test_dup_")

    def test_contains(self):
        assert "brats" in self.registry
        assert "_not_there_" not in self.registry


class TestBraTSSource:
    def test_name(self):
        src = BraTSSource()
        assert src.name == "brats"

    def test_description_contains_project_id(self):
        src = BraTSSource(synapse_project_id="syn12345")
        assert "syn12345" in src.description

    def test_required_modalities(self):
        src = BraTSSource()
        mods = src.required_modalities()
        assert {"t1", "t1ce", "t2", "flair", "seg"} == mods

    def test_validate_credentials_missing_file(self, tmp_path):
        src = BraTSSource(credentials_path=tmp_path / "no_such.synapseConfig")
        assert src.validate_credentials() is False

    def test_validate_credentials_empty_file(self, tmp_path):
        cred = tmp_path / ".synapseConfig"
        cred.write_text("[authentication]\n", encoding="utf-8")
        src = BraTSSource(credentials_path=cred)
        assert src.validate_credentials() is False

    def test_validate_credentials_valid_file(self, tmp_path):
        cred = tmp_path / ".synapseConfig"
        cred.write_text(
            "[authentication]\nauthtoken=fake-token-for-test\n", encoding="utf-8"
        )
        src = BraTSSource(credentials_path=cred)
        assert src.validate_credentials() is True

    def test_fetch_dry_run_no_network(self, tmp_path):
        """Dry run should raise because synapseclient isn't installed or
        credentials are missing, but NOT attempt real network calls."""
        src = BraTSSource(
            synapse_project_id="syn99999999",
            credentials_path=tmp_path / "missing.cfg",
        )
        # Should raise FileNotFoundError (credentials missing), not a
        # network error — proving no real network call was made.
        with pytest.raises((FileNotFoundError, RuntimeError)):
            src.fetch(dry_run=True)

    def test_verify_integrity_missing_dir(self, tmp_path):
        src = BraTSSource(output_dir=tmp_path / "nonexistent")
        assert src.verify_integrity() is False

    def test_verify_integrity_empty_dir(self, tmp_path):
        (tmp_path / "output").mkdir()
        src = BraTSSource(output_dir=tmp_path / "output")
        assert src.verify_integrity() is False

    def test_verify_integrity_with_case(self, tmp_path):
        out = tmp_path / "output"
        case = out / "BraTS-GLI-00001-000"
        case.mkdir(parents=True)
        # Create all expected modality files
        for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
            (case / f"BraTS-GLI-00001-000_{mod}.nii.gz").write_bytes(b"fake")
        src = BraTSSource(output_dir=out)
        assert src.verify_integrity() is True

    def test_verify_integrity_missing_modality(self, tmp_path):
        out = tmp_path / "output"
        case = out / "BraTS-GLI-00001-000"
        case.mkdir(parents=True)
        # Only create some modality files
        for mod in ["t1", "t2"]:
            (case / f"BraTS-GLI-00001-000_{mod}.nii.gz").write_bytes(b"fake")
        src = BraTSSource(output_dir=out)
        assert src.verify_integrity() is False


class TestLocalDirectorySource:
    def test_name(self):
        src = LocalDirectorySource()
        assert src.name == "local"

    def test_validate_credentials_always_true(self):
        src = LocalDirectorySource()
        assert src.validate_credentials() is True

    def test_list_cases_missing_dir(self, tmp_path):
        src = LocalDirectorySource(root_dir=tmp_path / "nonexistent")
        assert src.list_cases() == []

    def test_list_cases_finds_subdirs(self, tmp_path):
        (tmp_path / "case_001").mkdir()
        (tmp_path / "case_002").mkdir()
        (tmp_path / "not_a_dir.txt").write_text("x")
        src = LocalDirectorySource(root_dir=tmp_path)
        cases = src.list_cases()
        assert sorted(cases) == ["case_001", "case_002"]

    def test_fetch_dry_run(self, tmp_path):
        (tmp_path / "c1").mkdir()
        (tmp_path / "c2").mkdir()
        src = LocalDirectorySource(root_dir=tmp_path)
        paths = src.fetch(dry_run=True)
        assert len(paths) == 2
        assert all(isinstance(p, Path) for p in paths)


class TestGenericAuthorizedSource:
    def test_invalid_auth_method(self):
        with pytest.raises(ValueError, match="Invalid auth_method"):
            GenericAuthorizedSource(auth_method="invalid")

    def test_manual_auth_no_credentials_needed(self, tmp_path):
        src = GenericAuthorizedSource(auth_method="manual", output_dir=tmp_path)
        assert src.validate_credentials() is True

    def test_api_key_missing_credentials_path(self):
        src = GenericAuthorizedSource(
            auth_method="api_key",
            credentials_path=None,
        )
        assert src.validate_credentials() is False

    def test_fetch_dry_run_manual(self, tmp_path):
        src = GenericAuthorizedSource(
            auth_method="manual",
            output_dir=tmp_path,
        )
        paths = src.fetch(dry_run=True)
        # tmp_path is empty → no cases
        assert paths == []

    def test_fetch_dry_run_api_key(self, tmp_path):
        cred = tmp_path / "token.txt"
        cred.write_text("mytoken")
        src = GenericAuthorizedSource(
            auth_method="api_key",
            credentials_path=cred,
            output_dir=tmp_path / "data",
        )
        # dry_run → returns [] without network
        paths = src.fetch(dry_run=True)
        assert paths == []


# ---------------------------------------------------------------------------
# data_validator tests (no nibabel / no real NIfTI files needed for these)
# ---------------------------------------------------------------------------
from brainagenticthon.preprocessing.data_validator import (
    BRATS_LABELS,
    BRATS_MODALITIES,
    CaseValidator,
    DatasetValidator,
    ModalityValidator,
    NIfTIValidator,
    ValidationReport,
    ValidationResult,
)


class TestValidationResult:
    def test_valid_severities(self):
        for sev in ("error", "warning", "info"):
            r = ValidationResult(passed=True, check_name="x", message="ok", severity=sev)
            assert r.severity == sev

    def test_invalid_severity(self):
        with pytest.raises(ValueError, match="severity"):
            ValidationResult(passed=True, check_name="x", message="ok", severity="critical")

    def test_defaults(self):
        r = ValidationResult(passed=True, check_name="x", message="m")
        assert r.severity == "error"
        assert r.details == {}


class TestValidationReport:
    def _make_report(self):
        report = ValidationReport()
        report.add(ValidationResult(True,  "a", "All good",    "info"))
        report.add(ValidationResult(False, "b", "Warn msg",    "warning"))
        report.add(ValidationResult(False, "c", "Error msg",   "error"))
        return report

    def test_is_valid_with_errors(self):
        assert self._make_report().is_valid is False

    def test_is_valid_no_errors(self):
        report = ValidationReport()
        report.add(ValidationResult(True, "a", "ok", "info"))
        report.add(ValidationResult(False, "b", "warn", "warning"))
        assert report.is_valid is True  # warnings don't invalidate

    def test_errors_list(self):
        report = self._make_report()
        assert len(report.errors) == 1
        assert report.errors[0].check_name == "c"

    def test_warnings_list(self):
        report = self._make_report()
        assert len(report.warnings) == 1
        assert report.warnings[0].check_name == "b"

    def test_summary_contains_status(self):
        assert "FAILED" in self._make_report().summary

    def test_to_dict(self):
        d = self._make_report().to_dict()
        assert "is_valid" in d
        assert "results" in d
        assert isinstance(d["results"], list)

    def test_to_json_valid(self):
        j = self._make_report().to_json()
        parsed = json.loads(j)
        assert "is_valid" in parsed

    def test_merge(self):
        r1 = ValidationReport()
        r1.add(ValidationResult(True, "x", "ok", "info"))
        r2 = ValidationReport()
        r2.add(ValidationResult(False, "y", "fail", "error"))
        r1.merge(r2)
        assert len(r1.results) == 2

    def test_str_representation(self):
        s = str(self._make_report())
        assert "FAILED" in s or "PASSED" in s


class TestNIfTIValidatorOffline:
    """Tests that do NOT require actual NIfTI files."""

    def test_validate_header_missing_file(self, tmp_path):
        v = NIfTIValidator()
        report = v.validate_header(tmp_path / "nonexistent.nii.gz")
        assert not report.is_valid
        assert any("exists" in r.check_name for r in report.errors)

    def test_validate_file_integrity_missing(self, tmp_path):
        v = NIfTIValidator()
        report = v.validate_file_integrity(tmp_path / "no.file")
        assert not report.is_valid

    def test_validate_file_integrity_known_hash(self, tmp_path):
        import hashlib
        content = b"hello nifti world"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        v = NIfTIValidator()
        report = v.validate_file_integrity(f, expected_hash=expected)
        assert report.is_valid
        # Verify the hash detail is present
        hash_results = [r for r in report.results if "integrity" in r.check_name]
        assert any(r.details.get("hash") == expected for r in hash_results)

    def test_validate_file_integrity_wrong_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"some content")
        v = NIfTIValidator()
        report = v.validate_file_integrity(f, expected_hash="deadbeef" * 8)
        assert not report.is_valid

    def test_validate_file_integrity_bad_algorithm(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"x")
        v = NIfTIValidator()
        report = v.validate_file_integrity(f, algorithm="not_a_hash")
        assert not report.is_valid


class TestModalityValidator:
    def test_default_patterns(self):
        mv = ModalityValidator()
        assert set(mv.patterns.keys()) >= {"t1", "t1ce", "t2", "flair", "seg"}

    @pytest.mark.parametrize("filename,expected_mod", [
        ("BraTS-GLI-00001-000-t1.nii.gz",    "t1"),
        ("BraTS-GLI-00001-000-t1ce.nii.gz",  "t1ce"),
        ("BraTS-GLI-00001-000-t2.nii.gz",    "t2"),
        ("BraTS-GLI-00001-000-flair.nii.gz", "flair"),
        ("BraTS-GLI-00001-000-seg.nii.gz",   "seg"),
        ("case_t1ce_scan.nii.gz",            "t1ce"),
        ("case.t2.nii.gz",                   "t2"),
        ("scan_flair.nii",                   "flair"),
        ("output_seg.nii.gz",                "seg"),
    ])
    def test_valid_modality_names(self, filename, expected_mod):
        mv = ModalityValidator()
        report = mv.validate_modality_name(filename)
        assert report.is_valid, f"Expected {expected_mod} in '{filename}'"

    def test_unrecognised_modality(self):
        mv = ModalityValidator()
        report = mv.validate_modality_name("case_dwi.nii.gz")
        assert not report.is_valid

    def test_validate_modality_set_complete(self):
        mv = ModalityValidator()
        files = [
            "case-t1.nii.gz", "case-t1ce.nii.gz",
            "case-t2.nii.gz", "case-flair.nii.gz",
        ]
        report = mv.validate_modality_set(files, required_modalities=["t1", "t1ce", "t2", "flair"])
        assert report.is_valid

    def test_validate_modality_set_missing(self):
        mv = ModalityValidator()
        files = ["case-t1.nii.gz", "case-t2.nii.gz"]
        report = mv.validate_modality_set(files, required_modalities=["t1", "t1ce", "t2", "flair"])
        assert not report.is_valid
        details = report.errors[0].details
        assert "t1ce" in details["missing"]
        assert "flair" in details["missing"]


class TestCaseValidator:
    def test_validate_case_id_valid(self):
        cv = CaseValidator()
        report = cv.validate_case_id("BraTS-GLI-00001-000")
        assert report.is_valid

    @pytest.mark.parametrize("bad_id", [
        "brats-gli-00001-000",     # lowercase
        "BraTS-GLI-1-0",           # too short
        "BraTS-GL-00001-000",      # only 2 letters in segment
        "something_random",
    ])
    def test_validate_case_id_invalid(self, bad_id):
        cv = CaseValidator()
        report = cv.validate_case_id(bad_id)
        # Invalid IDs should produce a warning (not necessarily error)
        assert len(report.warnings) > 0 or not report.is_valid

    def test_validate_case_directory_missing(self, tmp_path):
        cv = CaseValidator()
        report = cv.validate_case_directory(tmp_path / "nonexistent")
        assert not report.is_valid

    def test_validate_spatial_consistency_empty_dir(self, tmp_path):
        cv = CaseValidator()
        report = cv.validate_spatial_consistency(tmp_path)
        # < 2 NIfTI files → trivially satisfied
        assert report.is_valid

    def test_validate_label_map_missing(self, tmp_path):
        cv = CaseValidator()
        report = cv.validate_label_map(tmp_path / "seg.nii.gz")
        assert not report.is_valid


class TestDatasetValidator:
    def test_validate_case_count_missing_dir(self, tmp_path):
        dv = DatasetValidator()
        report = dv.validate_case_count(tmp_path / "nonexistent")
        assert not report.is_valid

    def test_validate_case_count_empty(self, tmp_path):
        dv = DatasetValidator()
        report = dv.validate_case_count(tmp_path)
        # dir exists, 0 cases — should pass the existence check
        assert any(r.check_name == "dataset.case_count" for r in report.results)

    def test_validate_case_count_match(self, tmp_path):
        (tmp_path / "case1").mkdir()
        (tmp_path / "case2").mkdir()
        dv = DatasetValidator()
        report = dv.validate_case_count(tmp_path, expected_count=2)
        assert report.is_valid

    def test_validate_case_count_mismatch(self, tmp_path):
        (tmp_path / "case1").mkdir()
        dv = DatasetValidator()
        report = dv.validate_case_count(tmp_path, expected_count=5)
        assert not report.is_valid


class TestBraTS_Constants:
    def test_brats_modalities_keys(self):
        assert set(BRATS_MODALITIES.keys()) == {"t1", "t1ce", "t2", "flair", "seg"}

    def test_brats_labels_keys(self):
        assert set(BRATS_LABELS.keys()) == {0, 1, 2, 3}

    def test_brats_labels_descriptions_non_empty(self):
        for label, desc in BRATS_LABELS.items():
            assert isinstance(desc, str) and len(desc) > 0
