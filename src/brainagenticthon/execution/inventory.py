"""Bounded read-only private inventory and diagnostic eligibility; never infers approvals."""
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from brainagenticthon.detection.data import TrainingEvidence, load_training_evidence
from brainagenticthon.detection.imaging import load_mri, resolve_image
from brainagenticthon.detection.profile import InputProfile
from brainagenticthon.detection.training import load_private_cohort as load_a
from brainagenticthon.detection.training import private_output_directory
from brainagenticthon.preprocessing.manifest_schema import ManifestCollection
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_path
from brainagenticthon.preprocessing.task_a_readiness import (
    RecordReview,
    audit_readiness,
    load_reviews,
    record_key,
)
from brainagenticthon.segmentation.data import MODALITIES, REFERENCES, RELEASE, load_case
from brainagenticthon.segmentation.training import load_private_cohort as load_b

SUFFIXES = ('.nii', '.nii.gz', '.hdr', '.img')


def read_json(path, limit=32 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('unsafe_or_oversized_private_json')
    return json.loads(path.read_text())


def load_config(path):
    cfg = read_json(path, 1024 * 1024)
    root = Path(cfg['private_drive_root']).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('symlink_private_root_rejected')
    root = root.resolve(strict=True)
    return cfg, root


def protected_inputs(cfg, root, config_path):
    paths = [Path(config_path).resolve()]
    for key in ('image_root', 'manifest', 'reviews', 'training_evidence', 'evidence'):
        if cfg.get(key):
            paths.append(private_path(root, cfg[key]))
    for row in cfg.get('inventory', {}).get('roots', []):
        if row.get('path'):
            paths.append(private_path(root, row['path']))
    return paths


def authorized_roots(cfg, root):
    rows = cfg.get('inventory', {}).get('roots', [])
    if not rows:
        raise ValueError('authorized_inventory_roots_required')
    result = []
    for row in rows:
        if (row.get('access_authorized') is not True
                or any(not isinstance(row.get(k), str) or not row[k].strip()
                       for k in ('access_reference', 'license_reference'))):
            raise ValueError('root_access_and_license_evidence_required')
        path = private_path(root, row['path'])
        if not path.is_dir():
            raise ValueError('authorized_dataset_root_missing')
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in result):
            raise ValueError('overlapping_inventory_roots_rejected')
        result.append(path)
    images = private_path(root, cfg['image_root'])
    if not any(images.is_relative_to(path) for path in result):
        raise ValueError('image_root_not_authorized_by_inventory')
    return result


def discover(roots, *, max_files=10000):
    """No filenames create labels, modality identities, masks or patient identities."""
    if type(max_files) is not int or not 1 <= max_files <= 100000:
        raise ValueError('invalid_inventory_file_limit')
    rows, seen, visited = [], {}, 0
    for root_index, root in enumerate(roots):
        for directory, folders, files in os.walk(root, followlinks=False):
            visited += 1
            if visited > max_files * 2:
                raise ValueError('inventory_directory_limit')
            if any((Path(directory) / name).is_symlink() for name in (*folders, *files)):
                raise ValueError('inventory_symlink_rejected')
            for name in sorted(files):
                if not name.lower().endswith(SUFFIXES):
                    continue
                if len(rows) >= max_files:
                    raise ValueError('inventory_file_limit')
                path = Path(directory) / name
                item = {'root_index': root_index, 'relative_path': str(path.relative_to(root)),
                        'bytes': path.stat().st_size, 'clinical_approval': False}
                if item['bytes'] > 256 * 1024 * 1024:
                    item['blockers'] = ['source_file_exceeds_resource_limit']
                else:
                    before = digest_file(path)
                    item['sha256'] = before
                    # Headers commonly repeat; payload duplicate checks remain authoritative below.
                    if not name.lower().endswith('.hdr'):
                        item['duplicate_bytes_of'] = seen.get(before)
                        seen.setdefault(before, len(rows))
                    try:
                        image = nib.load(path)
                        item.update(shape=list(image.shape),
                                    spacing=[float(v) if np.isfinite(v) else None
                                             for v in image.header.get_zooms()],
                                    units=image.header.get_xyzt_units()[0])
                        if (np.prod(image.shape) > 32_000_000
                                or len(image.shape) not in (3, 4)
                                or not np.isfinite(image.affine).all()):
                            item['blockers'] = ['invalid_or_oversized_header']
                    except Exception:
                        item['blockers'] = ['unreadable_image_or_missing_pair']
                    if digest_file(path) != before:
                        item['blockers'] = ['source_changed_during_inventory']
                rows.append(item)
    return rows


def _task_a(cfg, root):
    paths = [private_path(root, cfg[k]) for k in ('manifest', 'reviews', 'training_evidence')]
    collection = ManifestCollection.load_json(paths[0])
    reviews, evidence = load_reviews(paths[1]), load_training_evidence(paths[2])
    profile = InputProfile(**{**cfg['profile'],
                             'target_shape': tuple(cfg['profile']['target_shape'])})
    report = audit_readiness(collection, reviews, image_root=private_path(root, cfg['image_root']),
                             check_images=False)
    blockers = sorted(set(report['blockers']) - {'image_checks_not_run'})
    for table in ('eligible_source_class', 'eligible_site_class'):
        if any(not counts['normal'] or not counts['abnormal']
               for counts in report['patient_class_crosstabs'][table].values()):
            blockers.append('source_or_site_class_confounding')
    sorted_records = sorted(collection.records, key=lambda r: (*record_key(r), r.modality))
    audit_rows = {(*record_key(r), r.modality): row
                  for r, row in zip(sorted_records, report['records'], strict=True)}
    rows, payloads = [], set()
    for index, record in enumerate(collection.records):
        review = reviews.get(record_key(record), RecordReview())
        extra = evidence.get(record_key(record), TrainingEvidence())
        # Preserve exact image identity even when one case has several MRI records.
        reasons = list(audit_rows[(*record_key(record), record.modality)]['reasons'])
        if extra.manual_review_status != 'approved':
            reasons.append('training_manual_review_pending')
        for flag in ('visual_qc_approved', 'anatomical_geometry_verified'):
            if getattr(extra, flag) is not True:
                reasons.append(flag + '_required')
        if record.task_a_label == 1 and not extra.abnormal_mri_verified:
            reasons.append('abnormal_mri_verified_required')
        refs = {key: value for key, value in asdict(extra).items() if key.endswith('_reference')}
        refs['review_reference'] = review.evidence_reference
        reasons.extend('missing_' + key for key, value in refs.items() if not value.strip())
        if (record.original_diagnosis or '').strip().lower() in {
                'nondemented', 'demented', 'converted', 'cdr=0', 'cdr=0.5', 'cdr 0.5'}:
            reasons.append('clinical_group_or_score_is_not_mri_diagnosis')
        try:
            _, fingerprints = load_mri(private_path(root, cfg['image_root']),
                                       record.image_path, profile)
            primary = resolve_image(private_path(root, cfg['image_root']), record.image_path)
            payload = next((value for path, value in fingerprints.items()
                            if path.suffix == '.img'), fingerprints.get(primary))
            if payload in payloads:
                reasons.append('duplicate_image_payload_bytes')
            payloads.add(payload)
            if fingerprints.get(primary) != review.expected_sha256:
                reasons.append('reviewed_primary_hash_missing_or_mismatch')
            if len(fingerprints) == 2 and any(value != extra.companion_sha256
                                             for path, value in fingerprints.items()
                                             if path != primary):
                reasons.append('reviewed_companion_hash_missing_or_mismatch')
        except Exception:
            reasons.append('image_profile_geometry_intensity_or_integrity_failed')
        rows.append({'record_index': index, 'blockers': sorted(set(reasons)),
                     'evidence_references': refs})
    return blockers, rows


def _task_b(cfg, root):
    manifest = read_json(private_path(root, cfg['manifest']))
    evidence = read_json(private_path(root, cfg['evidence']))
    blockers = []
    if manifest.get('release') != RELEASE or evidence.get('release') != RELEASE:
        blockers.append('explicit_brats2021_glioma_release_required')
    if evidence.get('review_status') != 'approved':
        blockers.append('release_review_pending')
    for flag in ('access_verified', 'masks_verified', 'geometry_verified', 'identity_verified'):
        if evidence.get(flag) is not True:
            blockers.append(flag + '_required')
    for key in REFERENCES:
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            blockers.append('missing_' + key)
    inv = cfg['inventory']
    if (inv.get('modality_identity_verified') is not True
            or not isinstance(inv.get('modality_identity_reference'), str)
            or not inv['modality_identity_reference'].strip()):
        blockers.append('independent_modality_identity_review_required')
    rows, patients, splits = [], {}, set()
    cases, paths, hashes = set(), set(), set()
    for index, row in enumerate(manifest.get('records', [])):
        reasons = []
        if row.get('case_id') in cases:
            reasons.append('duplicate_case_identity')
        cases.add(row.get('case_id'))
        for key, filename in row.get('files', {}).items():
            if filename in paths:
                reasons.append('duplicate_file_path')
            paths.add(filename)
            digest = row.get('sha256', {}).get(key)
            if key != 'seg' and digest:
                if digest in hashes:
                    reasons.append('duplicate_mri_bytes')
                hashes.add(digest)
        if row.get('review_status') != 'approved':
            reasons.append('case_review_pending')
        if not row.get('patient_id') or not row.get('case_id'):
            reasons.append('reconciled_patient_and_case_identity_required')
        if row.get('patient_id') in patients:
            reasons.append('repeated_patient_or_patient_split_leakage')
        patients[row.get('patient_id')] = row.get('split')
        splits.add(row.get('split'))
        if row.get('release') != RELEASE:
            reasons.append('case_release_mismatch')
        if (set(row.get('files', {})) != {*MODALITIES, 'seg'}
                or set(row.get('sha256', {})) != {*MODALITIES, 'seg'}):
            reasons.append('exact_four_modalities_mask_and_reviewed_hashes_required')
        try:
            _, mask, _ = load_case(private_path(root, cfg['image_root']), row)
            labels = sorted(int(v) for v in mask.unique().tolist())
            labels = [4 if v == 3 else v for v in labels]
        except Exception:
            reasons.append('case_pairing_geometry_labels_intensity_or_integrity_failed')
            labels = None
        rows.append({'record_index': index, 'blockers': reasons, 'observed_release_labels': labels,
                     'evidence_references': {key: evidence.get(key, '') for key in REFERENCES}})
    if splits != {'train', 'validation', 'test'}:
        blockers.append('all_three_frozen_patient_splits_required')
    return blockers, rows


def gated_cohort(cfg, root, task):
    """Authoritative unchanged gate; inventory diagnostics cannot approve a cohort."""
    if task == 'a':
        profile = InputProfile(**{**cfg['profile'],
                                 'target_shape': tuple(cfg['profile']['target_shape'])})
        return load_a(*(private_path(root, cfg[k])
                        for k in ('manifest', 'reviews', 'training_evidence', 'image_root')),
                      profile)
    return load_b(private_path(root, cfg['manifest']), private_path(root, cfg['image_root']),
                  private_path(root, cfg['evidence']))


def inspect_private(config_path, *, task, scan=True):
    if task not in {'a', 'b'}:
        raise ValueError('explicit_task_required')
    report = {'schema': 'private-mri-readiness-v1', 'task': task, 'status': 'BLOCKED',
              'clinical_approval': False, 'training_started': False,
              'eligibility_scope': 'explicit_reviewed_manifest_only', 'blockers': [],
              'records': [], 'inventory': [], 'evidence_references': {}}
    if config_path is None or not Path(config_path).is_file():
        report['blockers'] = ['private_configuration_missing']
        return report
    try:
        cfg, root = load_config(config_path)
        roots = authorized_roots(cfg, root)
        report['evidence_references'] = {'roots': [
            {key: row[key] for key in ('access_reference', 'license_reference')}
            for row in cfg['inventory']['roots']]}
        if task == 'b':
            report['evidence_references']['modality_identity_reference'] = (
                cfg['inventory'].get('modality_identity_reference', ''))
        if scan:
            report['inventory'] = discover(
                roots, max_files=cfg['inventory'].get('max_files', 10000))
        required = ('manifest', 'reviews', 'training_evidence') if task == 'a' else (
            'manifest', 'evidence')
        missing = [key for key in required if not cfg.get(key)
                   or not private_path(root, cfg[key]).is_file()]
        report['blockers'].extend('missing_private_' + key for key in missing)
        if not missing:
            before = {key: digest_file(private_path(root, cfg[key])) for key in required}
            blockers, rows = (_task_a if task == 'a' else _task_b)(cfg, root)
            report['records'], report['blockers'] = rows, blockers
            if any(row['blockers'] for row in rows):
                report['blockers'].append('individual_records_ineligible')
            try:
                cohort = gated_cohort(cfg, root, task)
                report['cohort_fingerprint'] = cohort.fingerprint
            except Exception:
                report['blockers'].append('authoritative_training_eligibility_gate_failed')
            if any(digest_file(private_path(root, cfg[key])) != value
                   for key, value in before.items()):
                report['blockers'].append('metadata_changed_during_inspection')
        if not report['blockers']:
            report['status'] = 'PASS'
    except ValueError as error:
        # Only our static authorization/resource codes are safe to expose as root blockers.
        allowed = {'authorized_inventory_roots_required',
                   'root_access_and_license_evidence_required',
                   'authorized_dataset_root_missing', 'image_root_not_authorized_by_inventory',
                   'overlapping_inventory_roots_rejected', 'inventory_file_limit',
                   'inventory_directory_limit', 'inventory_symlink_rejected'}
        report['blockers'].append(str(error) if str(error) in allowed else 'private_input_invalid')
    except Exception:
        report['blockers'].append('private_input_missing_unreadable_or_corrupt')
    report['blockers'] = sorted(set(report['blockers']))
    report['summary'] = {'candidate_files': len(report['inventory']),
                         'records': len(report['records']),
                         'record_blocker_counts': dict(Counter(
                             code for row in report['records'] for code in row['blockers']))}
    return report


def save_private_report(config_path, output, *, task, scan=True):
    protected = []
    if config_path is not None and Path(config_path).is_file():
        cfg, root = load_config(config_path)
        protected = protected_inputs(cfg, root, config_path)
    destination = private_output_directory(output, protected)
    report = inspect_private(config_path, task=task, scan=scan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    return report


def report_digest(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
