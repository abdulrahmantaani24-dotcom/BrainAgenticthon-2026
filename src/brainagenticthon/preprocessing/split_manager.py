"""
split_manager.py — Subject/Session Leakage-Free Split Management
=================================================================

Enforces subject-level and session-level data integrity across train,
validation, and test splits.  Prevents:

* The same subject appearing in more than one split (within a source).
* Duplicate cases appearing across multiple sources.
* Session-level leakage for longitudinal datasets (OASIS-3, ADNI).
* Figshare 2D records from contaminating 3D pipeline splits.
* Source-aware sampling to reduce dataset shortcut learning.
* Site-aware evaluation to expose scanner/site bias.

Usage
-----
>>> from brainagenticthon.preprocessing.split_manager import SplitManager
>>> from brainagenticthon.preprocessing.manifest_schema import ManifestCollection
>>> sm = SplitManager()
>>> report = sm.check_leakage(collection)
>>> if not report.is_clean:
...     for v in report.violations:
...         print(v)
"""
from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from brainagenticthon.preprocessing.manifest_schema import (
    ManifestCollection,
    ManifestRecord,
    QualityFlag,
    Split,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leakage report
# ---------------------------------------------------------------------------

@dataclass
class LeakageReport:
    """Result of a leakage check over a :class:`ManifestCollection`.

    Attributes
    ----------
    is_clean : bool
        True if no violations were found.
    violations : list of str
        Human-readable descriptions of each violation.
    subject_split_map : dict
        Maps ``(dataset_source, subject_id)`` → set of splits the subject
        appears in.
    duplicate_case_ids : list of str
        Case IDs that appear in more than one source (cross-source duplicates).
    site_split_map : dict
        Maps site → set of splits it appears in (for site-aware evaluation).
    """
    is_clean:            bool
    violations:          List[str] = field(default_factory=list)
    subject_split_map:   Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    duplicate_case_ids:  List[str] = field(default_factory=list)
    site_split_map:      Dict[str, Set[str]] = field(default_factory=dict)

    def summary(self) -> str:
        status = "CLEAN" if self.is_clean else "VIOLATIONS FOUND"
        return (
            f"{status}: {len(self.violations)} violation(s), "
            f"{len(self.duplicate_case_ids)} cross-source duplicate(s)"
        )


# ---------------------------------------------------------------------------
# SplitManager
# ---------------------------------------------------------------------------

class SplitManager:
    """Validates and assigns subject-safe data splits.

    Parameters
    ----------
    seed : int
        Random seed for reproducible split assignments.
    train_frac : float
        Fraction of subjects assigned to training (default 0.70).
    val_frac : float
        Fraction of subjects assigned to validation (default 0.15).
        The remainder (1 - train_frac - val_frac) goes to test.
    """

    def __init__(
        self,
        seed: int = 42,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> None:
        if not (0 < train_frac < 1 and 0 < val_frac < 1):
            raise ValueError("train_frac and val_frac must be strictly between 0 and 1.")
        if train_frac + val_frac >= 1.0:
            raise ValueError(
                f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0 "
                "to leave room for the test split."
            )
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac

    # -- split assignment ---------------------------------------------------

    def assign_splits(
        self,
        collection: ManifestCollection,
        *,
        per_source: bool = True,
    ) -> ManifestCollection:
        """Assign train / validation / test splits to a collection.

        Subject-level splits are enforced: all records belonging to the same
        subject (within a source) receive the same split.

        Figshare 2D records always receive ``split="pretrain"``.

        Parameters
        ----------
        collection : ManifestCollection
            The manifest to split.  Records with ``split != "unassigned"``
            are left unchanged.
        per_source : bool
            If True (default) split subjects within each source independently
            to ensure balanced representation.

        Returns
        -------
        ManifestCollection
            A new collection with split fields populated.
        """
        _PRETRAIN_ONLY = frozenset({"figshare_2d"})
        rng = random.Random(self.seed)

        # Group subjects by (source, subject_id)
        by_source_subject: Dict[str, Dict[str, List[ManifestRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        already_split: List[ManifestRecord] = []
        for rec in collection.records:
            if rec.split != Split.UNASSIGNED.value:
                already_split.append(rec)
            else:
                by_source_subject[rec.dataset_source][rec.subject_id].append(rec)

        updated: List[ManifestRecord] = list(already_split)

        for source_id, subject_map in by_source_subject.items():
            # Pretrain-only sources always go to pretrain
            if source_id in _PRETRAIN_ONLY:
                for recs in subject_map.values():
                    for rec in recs:
                        updated.append(self._with_split(rec, Split.PRETRAIN.value))
                continue

            subjects = sorted(subject_map.keys())
            rng.shuffle(subjects)

            n = len(subjects)
            n_train = max(1, round(n * self.train_frac))
            n_val   = max(1, round(n * self.val_frac))
            # Ensure we have at least 1 test subject
            if n_train + n_val >= n and n >= 3:
                n_val = n - n_train - 1
            if n_train + n_val >= n:
                n_train = max(1, n - 1)
                n_val   = 0

            train_subjs = set(subjects[:n_train])
            val_subjs   = set(subjects[n_train:n_train + n_val])
            test_subjs  = set(subjects[n_train + n_val:])

            logger.info(
                "Source '%s': %d subjects → train=%d, val=%d, test=%d",
                source_id, n, len(train_subjs), len(val_subjs), len(test_subjs),
            )

            for subj in subjects:
                if subj in train_subjs:
                    split_val = Split.TRAIN.value
                elif subj in val_subjs:
                    split_val = Split.VALIDATION.value
                else:
                    split_val = Split.TEST.value
                for rec in subject_map[subj]:
                    updated.append(self._with_split(rec, split_val))

        return ManifestCollection(updated)

    @staticmethod
    def _with_split(rec: ManifestRecord, split: str) -> ManifestRecord:
        """Return a copy of *rec* with the split field set."""
        d = rec.to_dict()
        d["split"] = split
        return ManifestRecord(**d)

    # -- leakage detection -------------------------------------------------

    def check_leakage(self, collection: ManifestCollection) -> LeakageReport:
        """Check for subject/session leakage and cross-source duplicates.

        Checks performed
        ----------------
        1. **Subject leakage**: a subject appears in more than one of
           {train, validation, test} within the same source.
        2. **Cross-source case ID duplicates**: the same ``case_id`` appears
           in two or more different sources.
        3. **Figshare 2D contamination**: a Figshare record has
           ``split`` other than ``pretrain`` / ``unassigned``.
        4. **Pituitary in Task C**: any Figshare record has a non-None
           ``task_c_label``.
        5. **Missing diagnosis assigned as healthy**: any record has
           ``task_c_label='cn'`` with ``label_provenance='missing'``.

        Returns
        -------
        LeakageReport
        """
        violations: List[str] = []

        # --- 1. Subject leakage per source ---------------------------------
        # Map (source, subject) → set of ranked splits
        _ranked = {Split.TRAIN.value, Split.VALIDATION.value, Split.TEST.value}
        subj_splits: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for rec in collection.records:
            if rec.split in _ranked:
                subj_splits[(rec.dataset_source, rec.subject_id)].add(rec.split)

        for (src, subj), splits in subj_splits.items():
            if len(splits) > 1:
                violations.append(
                    f"Subject leakage: subject '{subj}' from source '{src}' "
                    f"appears in multiple splits: {sorted(splits)}"
                )

        # --- 2. Cross-source case_id duplicates ----------------------------
        case_id_sources: Dict[str, Set[str]] = defaultdict(set)
        for rec in collection.records:
            case_id_sources[rec.case_id].add(rec.dataset_source)

        dup_case_ids: List[str] = [
            cid for cid, srcs in case_id_sources.items() if len(srcs) > 1
        ]
        for cid in dup_case_ids:
            violations.append(
                f"Cross-source duplicate case_id '{cid}' appears in sources: "
                f"{sorted(case_id_sources[cid])}"
            )

        # --- 3. Figshare 2D contamination ----------------------------------
        _pretrain_only = frozenset({"figshare_2d"})
        for rec in collection.records:
            if rec.dataset_source in _pretrain_only:
                if rec.split not in (Split.PRETRAIN.value, Split.UNASSIGNED.value):
                    violations.append(
                        f"Figshare 2D contamination: record '{rec.case_id}' "
                        f"has split='{rec.split}' (must be 'pretrain' or 'unassigned')."
                    )

        # --- 4. Pituitary in Task C ----------------------------------------
        for rec in collection.records:
            if (
                rec.dataset_source == "figshare_2d"
                and rec.cohort.lower() == "pituitary"
                and rec.task_c_label is not None
            ):
                violations.append(
                    f"Pituitary exclusion violated: record '{rec.case_id}' "
                    f"has task_c_label='{rec.task_c_label}'."
                )

        # --- 5. Missing diagnosis → healthy --------------------------------
        for rec in collection.records:
            if rec.task_c_label == "cn" and not rec.original_diagnosis:
                violations.append(
                    f"Missing-label-as-healthy: record '{rec.case_id}' "
                    "has task_c_label='cn' but no original_diagnosis."
                )
            if (
                rec.task_c_label == "cn"
                and rec.label_provenance == "missing"
            ):
                violations.append(
                    f"Missing-label-as-healthy: record '{rec.case_id}' "
                    "has task_c_label='cn' and label_provenance='missing'."
                )

        # --- site split map (for evaluation) --------------------------------
        site_splits: Dict[str, Set[str]] = defaultdict(set)
        for rec in collection.records:
            if rec.site:
                site_splits[rec.site].add(rec.split)

        return LeakageReport(
            is_clean            = len(violations) == 0,
            violations          = violations,
            subject_split_map   = dict(subj_splits),
            duplicate_case_ids  = dup_case_ids,
            site_split_map      = dict(site_splits),
        )

    # -- source-aware sampling ---------------------------------------------

    def source_aware_sample(
        self,
        collection: ManifestCollection,
        split: str,
        n_per_source: Optional[int] = None,
        max_total: Optional[int] = None,
    ) -> ManifestCollection:
        """Sample records from a split with source-aware balancing.

        Each source contributes at most ``n_per_source`` records (if set).
        Sampling is deterministic given the instance seed.

        Parameters
        ----------
        collection : ManifestCollection
        split : str
            Split to sample from.
        n_per_source : int, optional
            Maximum records per source.
        max_total : int, optional
            Maximum total records across all sources.

        Returns
        -------
        ManifestCollection
        """
        rng = random.Random(self.seed)
        by_source: Dict[str, List[ManifestRecord]] = defaultdict(list)
        for rec in collection.records:
            if rec.split == split:
                by_source[rec.dataset_source].append(rec)

        sampled: List[ManifestRecord] = []
        for source_id in sorted(by_source):
            recs = list(by_source[source_id])
            rng.shuffle(recs)
            if n_per_source is not None:
                recs = recs[:n_per_source]
            sampled.extend(recs)

        if max_total is not None:
            rng.shuffle(sampled)
            sampled = sampled[:max_total]

        return ManifestCollection(sampled)

    # -- site-aware statistics --------------------------------------------

    def site_split_statistics(
        self, collection: ManifestCollection
    ) -> Dict[str, Dict[str, int]]:
        """Return per-site, per-split record counts.

        Useful for detecting scanner/site confounding across splits.

        Returns
        -------
        dict
            ``{site: {split: count}}``
        """
        stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rec in collection.records:
            site = rec.site or "unknown_site"
            stats[site][rec.split] += 1
        return {s: dict(counts) for s, counts in stats.items()}

    # -- duplicate detection -----------------------------------------------

    @staticmethod
    def compute_case_fingerprint(rec: ManifestRecord) -> str:
        """Return a deterministic fingerprint for a record's image file.

        Uses ``case_id + dataset_source + modality`` as a lightweight
        structural identifier.  For true content-hash deduplication, pass
        the file through :meth:`~data_validator.NIfTIValidator.validate_file_integrity`.

        Returns
        -------
        str
            Hex digest.
        """
        key = f"{rec.dataset_source}::{rec.case_id}::{rec.modality}"
        return hashlib.sha256(key.encode()).hexdigest()

    def find_cross_source_subject_overlaps(
        self, collection: ManifestCollection
    ) -> Dict[str, List[str]]:
        """Heuristically detect subjects that may be shared across sources.

        Uses ``subject_id`` equality across sources as a weak signal.  A
        real deployment should use de-identification audit trails instead.

        Returns
        -------
        dict
            ``{subject_id: [source_id, ...]}`` for subjects in >1 source.
        """
        subj_sources: Dict[str, Set[str]] = defaultdict(set)
        for rec in collection.records:
            subj_sources[rec.subject_id].add(rec.dataset_source)
        return {
            subj: sorted(srcs)
            for subj, srcs in subj_sources.items()
            if len(srcs) > 1
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "LeakageReport",
    "SplitManager",
]
