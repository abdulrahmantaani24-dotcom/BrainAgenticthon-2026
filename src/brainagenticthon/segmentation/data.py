"""Fail-closed, hash-bound BraTS 2021 adult glioma training profile.

Native supplied grid is retained. No implicit masks, labels, splits or access approvals.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file

RELEASE = 'brats2021-glioma-training'
MODALITIES = ('t1', 't1ce', 't2', 'flair')
LABELS = {0, 1, 2, 4}
REFERENCES = ('access_reference', 'provenance_reference', 'mask_reference',
              'label_reference', 'geometry_reference', 'identity_reference')


def approved(evidence):
    if (evidence.get('release') != RELEASE or evidence.get('review_status') != 'approved'
            or any(evidence.get(k) is not True for k in
                   ('access_verified', 'masks_verified', 'geometry_verified', 'identity_verified'))
            or any(not isinstance(evidence.get(k), str) or not evidence[k].strip()
                   for k in REFERENCES)):
        raise ValueError('release_specific_reviewed_access_masks_provenance_required')


def safe_path(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in str(p):
        raise ValueError('private_relative_path_required')
    target = root / p
    if any(x.is_symlink() for x in (target, *target.parents)):
        raise ValueError('symlink_input_rejected')
    if not target.resolve().is_relative_to(root):
        raise ValueError('input_outside_private_root')
    return target


def load_case(root, row, *, require_mask=True):
    """Validate bytes, modality/mask geometry and exact release labels on each read."""
    keys = (*MODALITIES, 'seg') if require_mask else MODALITIES
    if set(row['files']) != set(keys) or set(row['sha256']) != set(keys):
        raise ValueError('exact_modalities_and_mask_required')
    volumes, affine, shape = {}, None, None
    for key in keys:
        path = safe_path(root, row['files'][key])
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError('file_resource_limit')
        expected = row['sha256'][key]
        if (not isinstance(expected, str) or len(expected) != 64
                or digest_file(path) != expected):
            raise ValueError('reviewed_digest_mismatch')
        image = nib.load(path)
        if (not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image))
                or len(image.shape) != 3 or min(image.shape) < 8
                or np.prod(image.shape) > 32_000_000
                or image.header.get_xyzt_units()[0] != 'mm'
                or not np.isfinite(image.affine).all()
                or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8
                or not np.isfinite(image.header.get_zooms()).all()
                or min(image.header.get_zooms()) <= 0):
            raise ValueError('invalid_geometry_or_volume')
        q, qc = image.get_qform(coded=True)
        s, sc = image.get_sform(coded=True)
        if not (qc or sc) or (qc and sc and not np.allclose(q, s, atol=1e-4)):
            raise ValueError('coded_geometry_required_and_must_agree')
        if shape is not None and (image.shape != shape or
                                  not np.allclose(image.affine, affine, atol=1e-4)):
            raise ValueError('modalities_and_mask_must_share_geometry')
        shape, affine = image.shape, image.affine.copy()
        if image.get_data_dtype().kind == 'c':
            raise ValueError('complex_volume_rejected')
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite_voxels')
        if digest_file(path) != expected:
            raise ValueError('input_changed_during_read')
        if key == 'seg':
            if not set(np.unique(values)).issubset(LABELS):
                raise ValueError('release_labels_must_be_0_1_2_4')
            # Empty masks remain valid segmentations; they never establish Task A normality.
            volumes[key] = values.astype(np.int64)
        else:
            support = values != 0
            if support.sum() < 8 or np.std(values[support]) < 1e-6:
                raise ValueError('empty_or_constant_modality')
            low, high = np.percentile(values[support], [1, 99])
            clipped = np.clip(values[support], low, high)
            if clipped.std() < 1e-6:
                raise ValueError('constant_intensity_support')
            values[support] = (clipped - clipped.mean()) / clipped.std()
            volumes[key] = values
    image = torch.from_numpy(np.stack([volumes[k] for k in MODALITIES]))
    mask = None
    if require_mask:
        mask = torch.from_numpy(volumes['seg'])
        mask[mask == 4] = 3  # Internal class 3 is exported back to release label 4.
    return image, mask, affine


def synchronized_patch(image, mask, size, generator, augment):
    """Shared integer crop and flips; no mask interpolation or modality misalignment."""
    if len(size) != 3 or any(type(n) is not int or not 8 <= n <= 160 for n in size):
        raise ValueError('invalid_patch_size')
    pads = []
    for n, wanted in reversed(list(zip(mask.shape, size, strict=True))):
        pads.extend((0, max(0, wanted - n)))
    image = torch.nn.functional.pad(image, pads)
    mask = torch.nn.functional.pad(mask, pads)
    starts = [int(torch.randint(n - wanted + 1, (), generator=generator))
              for n, wanted in zip(mask.shape, size, strict=True)]
    slices = tuple(slice(a, a + n) for a, n in zip(starts, size, strict=True))
    image, mask = image[(slice(None), *slices)], mask[slices]
    if augment:
        for axis in range(3):
            if torch.rand((), generator=generator) < .5:
                image, mask = image.flip(axis + 1), mask.flip(axis)
    return image.contiguous(), mask.contiguous()


class VerifiedCohort:
    mode = 'verified'

    def __init__(self, manifest, root, evidence):
        approved(evidence)
        self.root = Path(root).resolve()
        rows = manifest['records']
        if not rows or manifest.get('release') != RELEASE:
            raise ValueError('single_explicit_release_required')
        patients, cases, hashes, paths = {}, set(), set(), set()
        for row in rows:
            if (row.get('release') != RELEASE
                    or row.get('split') not in ('train', 'validation', 'test')):
                raise ValueError('invalid_release_or_frozen_split')
            if not row.get('patient_id') or not row.get('case_id') or row['case_id'] in cases:
                raise ValueError('unique_cases_and_reconciled_patient_identity_required')
            cases.add(row['case_id'])
            if row['patient_id'] in patients:
                raise ValueError('one_case_per_reconciled_patient_required')
            prior = patients.setdefault(row['patient_id'], row['split'])
            if prior != row['split']:
                raise ValueError('patient_leakage')
            if row.get('review_status') != 'approved':
                raise ValueError('case_review_required')
            for key in (*MODALITIES, 'seg'):
                path = str(safe_path(self.root, row['files'][key]))
                if path in paths:
                    raise ValueError('duplicate_file_path')
                paths.add(path)
                # Masks can legitimately repeat (e.g. empty); MRI bytes cannot.
                if key != 'seg':
                    digest = row['sha256'][key]
                    if digest in hashes:
                        raise ValueError('duplicate_mri_bytes')
                    hashes.add(digest)
            load_case(self.root, row)
        if {r['split'] for r in rows} != {'train', 'validation', 'test'}:
            raise ValueError('all_three_patient_splits_required')
        self.rows = rows
        self.source_roots = (self.root,)
        self.fingerprint = hashlib.sha256(json.dumps(
            {'manifest': manifest, 'evidence': evidence}, sort_keys=True).encode()).hexdigest()

    def iter_samples(self, split):
        for row in self.rows:
            if row['split'] == split:
                yield load_case(self.root, row)[:2]


class SyntheticCohort:
    mode = 'synthetic'
    source_roots = ()
    fingerprint = hashlib.sha256(b'brats-synthetic-v1').hexdigest()

    def iter_samples(self, split):
        offsets = {'train': 0, 'validation': 100, 'test': 200}
        for index in range(2):
            g = torch.Generator().manual_seed(42 + offsets[split] + index)
            mask = torch.zeros(16, 16, 16, dtype=torch.long)
            mask[3:13, 3:13, 3:13] = 2
            mask[5:11, 5:11, 5:11] = 1
            mask[7:10, 7:10, 7:10] = 3
            image = torch.randn(4, 16, 16, 16, generator=g) * .05
            for label in range(1, 4):
                image[label - 1] += (mask == label).float() * 2
            image[3] += (mask > 0).float()
            yield image, mask
