"""Only invented tensors/NIfTI fixtures; no clinical evidence or patient data."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from brainagenticthon.detection.training import cuda_smoke_test
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file
from brainagenticthon.segmentation.data import (
    MODALITIES,
    REFERENCES,
    RELEASE,
    SyntheticCohort,
    VerifiedCohort,
    load_case,
    synchronized_patch,
)
from brainagenticthon.segmentation.inference import (
    lesion_boxes,
    region_dice,
    release_labels,
    sliding_window,
)
from brainagenticthon.segmentation.training import TrainConfig, evaluate_test, fit


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cohort_files(tmp_path):
    evidence = {k: 'SYNTHETIC-TEST-ONLY' for k in REFERENCES}
    evidence.update(release=RELEASE, review_status='approved', access_verified=True,
                    masks_verified=True, geometry_verified=True, identity_verified=True)
    rows = []
    for index, split in enumerate(('train', 'validation', 'test')):
        row = {'release': RELEASE, 'patient_id': f'synthetic-{index}',
               'case_id': f'synthetic-{index}', 'split': split, 'review_status': 'approved',
               'files': {}, 'sha256': {}}
        for channel, key in enumerate((*MODALITIES, 'seg')):
            path = tmp_path / f'synthetic-{index}-{key}.nii'
            rng = np.random.default_rng(index * 10 + channel)
            data = rng.normal(size=(16, 16, 16)).astype(np.float32)
            if key == 'seg':
                data = np.zeros((16, 16, 16), dtype=np.int16)
                data[3:8, 4:9, 5:10] = 4
            image = nib.Nifti1Image(data, np.eye(4))
            image.header.set_xyzt_units('mm')
            nib.save(image, path)
            row['files'][key], row['sha256'][key] = path.name, digest_file(path)
        rows.append(row)
    return {'release': RELEASE, 'records': rows}, evidence, tmp_path


def test_verified_gate_loads_without_modifying_sources(cohort_files):
    manifest, evidence, root = cohort_files
    before = {p: p.read_bytes() for p in root.iterdir()}
    cohort = VerifiedCohort(manifest, root, evidence)
    x, y = next(cohort.iter_samples('train'))
    assert x.shape == (4, 16, 16, 16) and y.max() == 3
    assert 4 in np.unique(release_labels(y))
    assert before == {p: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize('change', ['access', 'release', 'split', 'patient', 'mask', 'modality',
                                   'digest', 'duplicate', 'review'])
def test_invalid_cohort_fails_closed(cohort_files, change):
    manifest, evidence, root = cohort_files
    rows = manifest['records']
    if change == 'access':
        evidence['access_verified'] = 'true'
    elif change == 'release':
        manifest['release'] = 'brats2024'
    elif change == 'split':
        rows[0]['split'] = 'test'
    elif change == 'patient':
        rows[1]['patient_id'] = rows[0]['patient_id']
    elif change == 'mask':
        rows[0]['files'].pop('seg')
    elif change == 'modality':
        rows[0]['files'].pop('t1ce')
    elif change == 'digest':
        rows[0]['sha256']['t1'] = 'a' * 64
    elif change == 'duplicate':
        rows[1]['sha256']['t1'] = rows[0]['sha256']['t1']
    else:
        rows[0]['review_status'] = 'pending'
    with pytest.raises((ValueError, KeyError)):
        VerifiedCohort(manifest, root, evidence)


@pytest.mark.parametrize('change', ['label', 'affine', 'units', 'nan', 'constant', 'shape'])
def test_invalid_voxels_and_geometry(cohort_files, change):
    manifest, _, root = cohort_files
    row = manifest['records'][0]
    key = 'seg' if change == 'label' else 't1'
    path = root / row['files'][key]
    old = nib.load(path)
    data = old.get_fdata().astype(np.float32)
    affine = np.eye(4)
    if change == 'label':
        data[0, 0, 0] = 3
    elif change == 'affine':
        affine[0, 3] = 5
    elif change == 'nan':
        data[0, 0, 0] = np.nan
    elif change == 'constant':
        data[:] = 1
    elif change == 'shape':
        data = data[:8]
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('unknown' if change == 'units' else 'mm')
    nib.save(image, path)
    row['sha256'][key] = digest_file(path)
    with pytest.raises(ValueError):
        load_case(root, row)


def test_preprocessing_is_synchronized_and_label_preserving():
    mask = torch.arange(20**3).reshape(20, 20, 20) % 4
    image = mask.float().repeat(4, 1, 1, 1)
    x, y = synchronized_patch(image, mask, (16, 16, 16),
                              torch.Generator().manual_seed(7), True)
    assert torch.equal(x[0].long(), y)
    assert all(torch.equal(channel, y.float()) for channel in x)


def test_sliding_window_odd_shapes_coverage_and_mode_restore():
    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return x
    model = Pointwise().train()
    image = torch.randn(4, 19, 21, 13)
    actual = sliding_window(model, image, patch=(8, 12, 16))
    assert torch.allclose(actual, image.softmax(0), atol=1e-6)
    assert model.training
    with pytest.raises(ValueError):
        sliding_window(model, image, overlap=1)


def test_dice_boxes_empty_masks_and_provenance():
    mask = torch.zeros(16, 16, 16, dtype=torch.long)
    assert region_dice(mask, mask) == {'WT': None, 'TC': None, 'ET': None}
    assert lesion_boxes(mask, source='reference_mask') == []
    mask[1:3, 2:5, 3:7] = 3
    mask[12, 12, 12] = 1
    assert region_dice(mask, mask) == {'WT': 1., 'TC': 1., 'ET': 1.}
    boxes = lesion_boxes(mask, source='predicted_mask')
    assert boxes[0]['bounds'] == [1, 2, 2, 4, 3, 6] and boxes[0]['voxels'] == 24
    assert boxes[1]['bounds'] == [12, 12, 12, 12, 12, 12]
    assert region_dice(torch.zeros_like(mask), mask)['WT'] == 0
    with pytest.raises(ValueError):
        lesion_boxes(mask, source='classifier_score')


def config(epochs=1):
    return TrainConfig(epochs=epochs, device='cpu', patch=(16, 16, 16), patches_per_case=1)


def test_training_resume_exact_and_final_test_separation(tmp_path):
    cohort = SyntheticCohort()
    full, history, _ = fit(cohort, config(2), tmp_path / 'full')
    _, first, _ = fit(cohort, config(), tmp_path / 'resume')
    assert len(first) == 1
    with pytest.raises(ValueError, match='trusted'):
        fit(cohort, config(2), tmp_path / 'resume', resume=True)
    resumed, actual, bound = fit(cohort, config(2), tmp_path / 'resume',
                                resume=True, trusted_resume=True)
    assert actual == history
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    with pytest.raises(ValueError):
        evaluate_test(resumed, cohort, config(2), tmp_path / 'resume', bound)
    metrics = evaluate_test(resumed, cohort, config(2), tmp_path / 'resume', bound, confirm=True)
    assert metrics['cases'] == 2
    with pytest.raises(ValueError, match='test_already'):
        fit(cohort, config(3), tmp_path / 'resume', resume=True, trusted_resume=True)


def test_resume_integrity_and_settings_binding(tmp_path):
    cohort = SyntheticCohort()
    fit(cohort, config(), tmp_path / 'run')
    with pytest.raises(RuntimeError):
        fit(cohort, replace(config(2), learning_rate=.01), tmp_path / 'run',
            resume=True, trusted_resume=True)
    (tmp_path / 'run/epoch-0001.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):
        fit(cohort, config(2), tmp_path / 'run', resume=True, trusted_resume=True)


def test_synthetic_small_overfit(tmp_path):
    model, history, _ = fit(SyntheticCohort(), replace(config(40), patches_per_case=2),
                             tmp_path / 'overfit')
    assert history[-1]['loss'] < history[0]['loss'] * .6
    assert history[-1]['validation']['dice']['WT'] > .90
    assert history[-1]['validation']['mean_dice'] > .80
    assert model.training is False


def test_cuda_claims_require_cuda(tmp_path):
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match='cuda_requested'):
            cuda_smoke_test()
        with pytest.raises(ValueError, match='cuda_requested'):
            fit(SyntheticCohort(), TrainConfig(), tmp_path / 'gpu')
    else:
        assert cuda_smoke_test()['cuda_used'] is True


def test_both_notebooks_validate_and_have_no_outputs():
    import ast
    for name in ('05_task_a_research_training.ipynb', '06_task_b_segmentation_training.ipynb'):
        notebook = json.loads((Path(__file__).parents[1] / 'notebooks' / name).read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [] and cell['execution_count'] is None
                ast.parse(''.join(cell['source']))


def test_inference_checkpoint_reload_and_private_geometry(cohort_files, tmp_path):
    from brainagenticthon.segmentation.predict import load_model, predict
    manifest, _, root = cohort_files
    cfg = config()
    cohort = SyntheticCohort()
    run = tmp_path / 'run'
    model, _, _ = fit(cohort, cfg, run)
    with pytest.raises(ValueError):
        load_model(cohort, cfg, run)
    reloaded = load_model(cohort, cfg, run, trusted=True)
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items())
    row = copy.deepcopy(manifest['records'][0])
    row['files'].pop('seg')
    row['sha256'].pop('seg')
    # Inputs occupy root, so outputs must be elsewhere, never under source root.
    outside = root.parent / (root.name + '-private-inference')
    report = predict(reloaded, row, root, outside, cfg, protected=(run,))
    prediction = nib.load(outside / 'prediction.nii.gz')
    assert prediction.shape == (16, 16, 16)
    assert np.array_equal(prediction.affine, np.eye(4))
    assert set(np.unique(prediction.get_fdata())).issubset({0, 1, 2, 4})
    assert report['clinical_approval'] is False
    with pytest.raises(ValueError):
        predict(reloaded, row, root, root / 'bad-output', cfg)


def test_task_b_notebook_synthetic_cells_execute(tmp_path):
    import brainagenticthon.segmentation.training as training
    from brainagenticthon.detection.training import private_output_directory
    from brainagenticthon.preprocessing.oasis2_mri_quality import private_call, private_path
    root = Path(__file__).parents[1]
    cfg = json.loads((root / 'configs/task_b_research.example.json').read_text())
    cfg.update(private_drive_root=str(tmp_path / 'private'), output_run='runs/synthetic')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    ns = {'Path': Path, 'json': json, 'torch': torch, 'CONFIG_PATH': path,
          'DRIVE_BASE': tmp_path, 'CODE_ROOT': root, 'private_call': private_call,
          'private_path': private_path, 'private_output_directory': private_output_directory,
          'TrainConfig': TrainConfig, 'SyntheticCohort': SyntheticCohort,
          'load_private_cohort': training.load_private_cohort, 'fit': fit,
          'evaluate_test': evaluate_test}
    nb = json.loads((root / 'notebooks/06_task_b_segmentation_training.ipynb').read_text())
    for cell in nb['cells']:
        if cell['id'] in (
                'private-config', 'eligibility-gate', 'train-or-resume', 'final-evaluation'):
            exec(''.join(cell['source']), ns)
    assert len(ns['HISTORY']) == 1
    assert not (ns['OUTPUT'] / 'test_metrics.json').exists()


def test_verified_format_pipeline_uses_only_generated_nifti(cohort_files):
    manifest, evidence, root = cohort_files
    cohort = VerifiedCohort(manifest, root, evidence)
    output = root.parent / (root.name + '-private-training')
    _, history, _ = fit(cohort, config(), output)
    assert len(history) == 1 and history[0]['validation']['cases'] == 1
    row = manifest['records'][0]
    path = root / row['files']['t1']
    path.write_bytes(b'changed-after-gate')
    with pytest.raises(ValueError, match='digest'):
        next(cohort.iter_samples('train'))


def test_interrupted_b_commit_marker_recovers_prior_epoch(tmp_path, monkeypatch):
    import brainagenticthon.segmentation.training as training
    original = training._write_json

    def interrupted(path, payload):
        if path.name == 'latest.json' and payload['checkpoint'] == 'epoch-0002.pt':
            raise KeyboardInterrupt
        original(path, payload)
    monkeypatch.setattr(training, '_write_json', interrupted)
    with pytest.raises(KeyboardInterrupt):
        fit(SyntheticCohort(), config(2), tmp_path / 'run')
    marker = json.loads((tmp_path / 'run/latest.json').read_text())
    assert marker['checkpoint'] == 'epoch-0001.pt'
    monkeypatch.setattr(training, '_write_json', original)
    _, history, _ = fit(SyntheticCohort(), config(2), tmp_path / 'run',
                         resume=True, trusted_resume=True)
    assert [r['epoch'] for r in history] == [1, 2]
