"""Strict additive training gate; original readiness/schema/registry are unchanged."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from brainagenticthon.detection.imaging import load_mri, resolve_image
from brainagenticthon.detection.profile import InputProfile
from brainagenticthon.preprocessing.oasis2_mri_quality import private_call
from brainagenticthon.preprocessing.task_a_readiness import audit_readiness, record_key


@dataclass(frozen=True)
class TrainingEvidence:
    """Additional private review evidence; never derived from Group/CDR or scores."""

    manual_review_status: str = "pending"
    visual_qc_approved: bool = False
    anatomical_geometry_verified: bool = False
    abnormal_mri_verified: bool = False
    mri_label_evidence_reference: str = ""
    visual_qc_evidence_reference: str = ""
    geometry_evidence_reference: str = ""
    modality_evidence_reference: str = ""
    companion_sha256: str | None = None

    def __post_init__(self):
        for name in ("visual_qc_approved", "anatomical_geometry_verified", "abnormal_mri_verified"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("training_evidence_requires_json_booleans")
        for name in ("manual_review_status", "mri_label_evidence_reference",
                     "visual_qc_evidence_reference", "geometry_evidence_reference",
                     "modality_evidence_reference"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("training_evidence_requires_strings")
        if self.companion_sha256 is not None and not _digest_valid(self.companion_sha256):
            raise ValueError("invalid_companion_digest")


def _digest_valid(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def load_training_evidence(path):
    """Separate sidecar keyed like RecordReview; original manifests stay untouched."""
    def read():
        path_object = Path(path)
        if path_object.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("evidence_file_too_large")
        items = json.loads(path_object.read_text())
        if not isinstance(items, list):
            raise ValueError("evidence_requires_list")
        result = {}
        for row in items:
            row = dict(row)
            key = tuple(row.pop(field) for field in ("dataset_source", "case_id", "image_path"))
            if any(not isinstance(value, str) or not value for value in key) or key in result:
                raise ValueError("invalid_or_duplicate_evidence_key")
            result[key] = TrainingEvidence(**row)
        return result
    return private_call(read)


class MRIDataset(Dataset):
    """One MRI in memory at a time. Batch metadata contains integer indices only."""

    def __init__(self, cohort, split):
        self.cohort = cohort
        self.rows = [row for row in cohort._rows if row[0].split == split]
        counts = Counter(row[1] for row in self.rows)
        self.patient_weights = [1 / counts[row[1]] for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record, patient, source, site, fingerprints = self.rows[index]
        image, _ = load_mri(
            self.cohort.image_root, record.image_path, self.cohort.profile,
            expected=fingerprints,
        )
        return {"image": image, "label": float(record.task_a_label),
                "patient": patient, "source": source, "site": site}


class PreparedCohort:
    """Construct through prepare_cohort; contains private metadata in RAM only."""

    def __init__(self, rows, image_root, profile, fingerprint, summary):
        self._rows = rows
        self.image_root = image_root
        self.profile = profile
        self.fingerprint = fingerprint
        self.summary = summary
        self.source_roots = (image_root,)
        self.mode = "verified_research"

    def dataset(self, split):
        return MRIDataset(self, split)


def prepare_cohort(collection, reviews, evidence, *, image_root, profile=None):
    """Fail closed on ANY ineligible row; no filtering, inferred labels or overrides.

    Reuse every existing metadata readiness rule. Only its explicit
    image_checks_not_run marker is replaced by this profile's bounded image checks,
    which additionally support a verified singleton fourth axis. Legacy readiness
    behavior is not edited. OASIS-2 is not currently a registry/schema-approved
    Task A source and cannot pass this gate, even with reviewer attestations.
    """
    profile = profile or InputProfile()
    report = private_call(
        audit_readiness, collection, reviews, image_root=Path(image_root), check_images=False,
    )
    blockers = set(report["blockers"]) - {"image_checks_not_run"}
    if blockers or not report["records"] or any(not row["eligible"] for row in report["records"]):
        raise ValueError("task_a_metadata_eligibility_failed")
    # Reject even partial pure-class source/site groups, not just perfect global confounding.
    for table in ("eligible_source_class", "eligible_site_class"):
        if any(not counts["normal"] or not counts["abnormal"]
               for counts in report["patient_class_crosstabs"][table].values()):
            raise ValueError("source_or_site_class_confounding")
    records = copy.deepcopy(sorted(collection.records, key=record_key))
    patients = sorted({reviews[record_key(r)].patient_id for r in records})
    sources = sorted({r.dataset_source for r in records})
    sites = sorted({(r.dataset_source, r.site) for r in records})
    rows, digest_sets, fingerprint_rows = [], set(), []
    for record in records:
        key = record_key(record)
        review = reviews[key]
        extra = evidence.get(key, TrainingEvidence())
        if (extra.manual_review_status != "approved" or not extra.visual_qc_approved
                or not extra.anatomical_geometry_verified
                or not all(getattr(extra, name).strip() for name in (
                    "mri_label_evidence_reference", "visual_qc_evidence_reference",
                    "geometry_evidence_reference", "modality_evidence_reference"))
                or (record.task_a_label == 1 and not extra.abnormal_mri_verified)
                or not _digest_valid(review.expected_sha256)):
            raise ValueError("explicit_mri_review_and_hash_evidence_required")
        if (record.original_diagnosis or "").strip().lower() in {
            "nondemented", "demented", "converted", "cdr=0", "cdr=0.5", "cdr 0.5",
        }:
            raise ValueError("clinical_group_or_score_is_not_mri_diagnosis")
        path = resolve_image(image_root, record.image_path)
        _, fingerprints = load_mri(image_root, record.image_path, profile)
        if fingerprints.get(path) != review.expected_sha256:
            raise ValueError("review_image_digest_mismatch")
        if len(fingerprints) == 2:
            companion = next(d for p, d in fingerprints.items() if p != path)
            if companion != extra.companion_sha256:
                raise ValueError("review_companion_digest_required_or_mismatched")
        # Detect identical voxel payloads even if paired headers differ.
        payload_digest = next((d for p, d in fingerprints.items() if p.suffix == ".img"),
                              fingerprints[path])
        if payload_digest in digest_sets:
            raise ValueError("duplicate_image_bytes")
        digest_sets.add(payload_digest)
        rows.append((record, patients.index(review.patient_id),
                     sources.index(record.dataset_source),
                     sites.index((record.dataset_source, record.site)), fingerprints))
        fingerprint_rows.append({"record": record.to_dict(), "review": asdict(review),
                                 "evidence": asdict(extra),
                                 "images": sorted(fingerprints.values())})
    fingerprint = hashlib.sha256(json.dumps(
        {"rows": fingerprint_rows, "profile": profile.to_dict()}, sort_keys=True,
    ).encode()).hexdigest()
    summary = {"eligible_images": len(rows), "patients": len(patients),
               "sources": len(sources), "source_site_groups": len(sites),
               "clinical_approval": False, "scope": "reviewed research eligibility only"}
    return PreparedCohort(rows, Path(image_root).resolve(), profile, fingerprint, summary)


class _SyntheticDataset(Dataset):
    def __init__(self, split, seed, profile):
        offset = {"train": 0, "validation": 1, "test": 2}[split]
        generator = torch.Generator().manual_seed(seed + offset)
        self.images = torch.randn(4, 1, *profile.target_shape, generator=generator)
        self.labels = [0, 1, 0, 1]
        self.images[1::2, :, 4:12, 4:12, 4:12] += 2
        self.patient_weights = [1.0] * 4
        self.offset = {"train": 0, "validation": 4, "test": 8}[split]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"image": self.images[index], "label": float(self.labels[index]),
                "patient": self.offset + index, "source": 0, "site": 0}


class SyntheticCohort:
    """Explicit engineering-only fixture. Cannot accept files, labels or manifests."""

    def __init__(self, seed=42):
        self.profile = InputProfile(target_shape=(16, 16, 16))
        self.seed = seed
        self.mode = "synthetic_only"
        self.source_roots = ()
        self.summary = {"synthetic": True, "patients": 12, "clinical_approval": False}
        self.fingerprint = hashlib.sha256(f"synthetic-v1:{seed}".encode()).hexdigest()

    def dataset(self, split):
        return _SyntheticDataset(split, self.seed, self.profile)
