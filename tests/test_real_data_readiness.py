"""All MRI and review fixtures are generated; no real training or clinical evidence."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from brainagenticthon.detection.data import SyntheticCohort as SyntheticA
from brainagenticthon.detection.training import TrainConfig as ConfigA
from brainagenticthon.execution.inventory import (
    discover,
    inspect_private,
    save_private_report,
)
from brainagenticthon.execution.real_training import (
    _smoke,
    execute_real,
    freeze_splits,
    validate_freeze,
)
from brainagenticthon.segmentation.data import SyntheticCohort as SyntheticB
from brainagenticthon.segmentation.training import TrainConfig as ConfigB
from tests import test_task_a_research_data as fixture_a
from tests import test_task_b_segmentation as fixture_b


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config(root, task):
    cfg = json.loads((Path(__file__).parents[1] /
                      f'configs/task_{task}_readiness.example.json').read_text())
    cfg.update(private_drive_root=str(root), image_root='.', manifest='manifest.json',
               output_run='runs/fresh')
    cfg['inventory']['roots'] = [{'path': '.', 'access_authorized': True,
                                  'access_reference': 'SYNTHETIC-TEST-ONLY',
                                  'license_reference': 'SYNTHETIC-TEST-ONLY'}]
    if task == 'b':
        cfg['evidence'] = 'evidence.json'
        cfg['inventory'].update(modality_identity_verified=True,
                                modality_identity_reference='SYNTHETIC-TEST-ONLY')
    else:
        cfg.update(reviews='reviews.json', training_evidence='evidence.json')
        cfg['profile']['target_shape'] = [16, 16, 16]
    return cfg


@pytest.fixture
def private_b(tmp_path):
    manifest, evidence, root = fixture_b.cohort_files.__wrapped__(tmp_path)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'evidence.json').write_text(json.dumps(evidence))
    path = root / 'config.json'
    path.write_text(json.dumps(config(root, 'b')))
    return path, root


@pytest.fixture
def private_a(tmp_path):
    collection, reviews, evidence, root = (
        fixture_a.verified_synthetic_metadata.__wrapped__(tmp_path))
    collection.save_json(root / 'manifest.json')
    def rows(mapping):
        return [{**dict(zip(('dataset_source', 'case_id', 'image_path'), key, strict=True)),
                 **asdict(value)} for key, value in mapping.items()]
    (root / 'reviews.json').write_text(json.dumps(rows(reviews)))
    (root / 'evidence.json').write_text(json.dumps(rows(evidence)))
    path = root / 'config.json'
    path.write_text(json.dumps(config(root, 'a')))
    return path, root


@pytest.mark.parametrize('fixture,task', [('private_a', 'a'), ('private_b', 'b')])
def test_inventory_pass_is_read_only_and_freeze_preserves_exact_splits(request, fixture, task):
    path, root = request.getfixturevalue(fixture)
    before = {p: p.read_bytes() for p in root.iterdir()}
    report = inspect_private(path, task=task)
    assert report['status'] == 'PASS', report
    assert report['inventory'] and all(not row['blockers'] for row in report['records'])
    assert all(row['evidence_references'] for row in report['records'])
    assert before == {p: p.read_bytes() for p in root.iterdir()}
    freeze = root.parent / (root.name + '-frozen')
    result = freeze_splits(path, freeze, task=task)
    assert result['status'] == 'PASS' and not result['training_started']
    frozen = validate_freeze(path, freeze, task=task)
    assert set(frozen['patient_splits'].values()) == {'train', 'validation', 'test'}
    assert before == {p: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        freeze_splits(path, freeze, task=task)
    (freeze / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='snapshot_corrupt'):
        validate_freeze(path, freeze, task=task)


def test_root_authorization_required_before_scanning(private_b, monkeypatch):
    import brainagenticthon.execution.inventory as inventory
    path, root = private_b
    cfg = json.loads(path.read_text())
    cfg['inventory']['roots'][0]['access_authorized'] = False
    path.write_text(json.dumps(cfg))
    def forbidden(*args, **kwargs):
        raise AssertionError('Unauthorized scan attempted')
    monkeypatch.setattr(inventory, 'discover', forbidden)
    report = inspect_private(path, task='b')
    assert report['status'] == 'BLOCKED'
    assert report['blockers'] == ['root_access_and_license_evidence_required']
    assert report['inventory'] == []


@pytest.mark.parametrize('change', ['modality', 'mask', 'labels', 'license', 'split', 'evidence'])
def test_b_individual_blockers_references_and_existing_gate(private_b, change):
    path, root = private_b
    cfg = json.loads(path.read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    if change == 'modality':
        cfg['inventory']['modality_identity_verified'] = False
    elif change == 'license':
        cfg['inventory']['roots'][0]['license_reference'] = ''
    elif change == 'mask':
        manifest['records'][0]['files'].pop('seg')
    elif change == 'split':
        manifest['records'][0]['split'] = 'test'
    elif change == 'evidence':
        evidence = json.loads((root / 'evidence.json').read_text())
        evidence['mask_reference'] = ''
        (root / 'evidence.json').write_text(json.dumps(evidence))
    else:
        import nibabel as nib

        from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file
        row = manifest['records'][0]
        target = root / row['files']['seg']
        old = nib.load(target)
        values = old.get_fdata().astype(np.float32)
        values[0, 0, 0] = 3
        image = nib.Nifti1Image(values, old.affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, target)
        row['sha256']['seg'] = digest_file(target)
    path.write_text(json.dumps(cfg))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    report = inspect_private(path, task='b')
    assert report['status'] == 'BLOCKED'
    assert report['blockers'] or any(row['blockers'] for row in report['records'])
    with pytest.raises(ValueError, match='BLOCKED'):
        freeze_splits(path, root.parent / (root.name + '-blocked-freeze'), task='b')


def test_a_individual_clinical_qc_blockers_and_no_inferred_normal(private_a):
    path, root = private_a
    evidence = json.loads((root / 'evidence.json').read_text())
    evidence[0].update(visual_qc_approved=False, geometry_evidence_reference='')
    (root / 'evidence.json').write_text(json.dumps(evidence))
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['records'][0]['original_diagnosis'] = 'Nondemented'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    report = inspect_private(path, task='a', scan=False)
    reasons = {code for row in report['records'] for code in row['blockers']}
    assert 'visual_qc_approved_required' in reasons
    assert 'missing_geometry_evidence_reference' in reasons
    assert 'clinical_group_or_score_is_not_mri_diagnosis' in reasons
    assert report['status'] == 'BLOCKED'


def test_freeze_invalidates_changed_reviews_but_allows_epoch_change(private_b):
    path, root = private_b
    freeze = root.parent / (root.name + '-freeze')
    freeze_splits(path, freeze, task='b')
    cfg = json.loads(path.read_text())
    cfg['training']['epochs'] = 2
    path.write_text(json.dumps(cfg))
    validate_freeze(path, freeze, task='b')
    evidence = json.loads((root / 'evidence.json').read_text())
    evidence['identity_reference'] += '-changed'
    (root / 'evidence.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='frozen_inputs'):
        validate_freeze(path, freeze, task='b')


def test_blocked_and_disabled_real_execution_never_calls_fit(private_b, monkeypatch):
    import brainagenticthon.execution.real_training as training
    path, root = private_b
    def forbidden(*args, **kwargs):
        raise AssertionError('Real training must not start')
    monkeypatch.setattr(training.task_b, 'fit', forbidden)
    with pytest.raises(ValueError, match='disabled'):
        execute_real(path, root / 'no-freeze', task='b', action='train')
    cfg = json.loads(path.read_text())
    cfg['real_training_enabled'] = True
    cfg['inventory']['modality_identity_verified'] = False
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='BLOCKED'):
        execute_real(path, root / 'no-freeze', task='b', action='train')


def test_no_real_cpu_fallback_even_with_synthetic_verified_fixture(private_b, monkeypatch):
    import brainagenticthon.execution.real_training as training
    path, root = private_b
    freeze = root.parent / (root.name + '-freeze')
    freeze_splits(path, freeze, task='b')
    cfg = json.loads(path.read_text())
    cfg['real_training_enabled'] = True
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Fit must not execute without CUDA')
    monkeypatch.setattr(training.task_b, 'fit', forbidden)
    with pytest.raises(ValueError, match='actual_cuda'):
        execute_real(path, freeze, task='b', action='smoke')
    assert not (root / 'runs').exists()


@pytest.mark.parametrize('task', ['a', 'b'])
def test_bounded_smoke_helper_on_generated_tensors_only(tmp_path, task):
    cohort = SyntheticA() if task == 'a' else SyntheticB()
    config_value = (ConfigA(epochs=1, device='cpu') if task == 'a' else
                    ConfigB(epochs=1, device='cpu', patch=(16, 16, 16)))
    output = tmp_path / 'generated-smoke'
    report = _smoke(cohort, task, config_value, output)
    assert report['status'] == 'PASS' and report['checkpoint_reload_verified']
    assert not report['held_out_evaluation_performed']
    with pytest.raises(FileExistsError):
        _smoke(cohort, task, config_value, output)


def test_inventory_limits_symlinks_corruption_and_private_report_safety(private_b):
    path, root = private_b
    with pytest.raises(ValueError, match='file_limit'):
        discover([root], max_files=1)
    link = root / 'linked.nii'
    link.symlink_to(next(root.glob('*.nii')))
    with pytest.raises(ValueError, match='symlink'):
        discover([root])
    # Never delete user/untracked files; fixture link is isolated to this test.
    report = inspect_private(path, task='b')
    assert 'inventory_symlink_rejected' in report['blockers']
    with pytest.raises(ValueError, match='disjoint'):
        save_private_report(path, root / 'report.json', task='b')
    output = root.parent / (root.name + '-report.json')
    saved = save_private_report(path, output, task='b')
    assert saved['status'] == 'BLOCKED'
    with pytest.raises(FileExistsError):
        save_private_report(path, output, task='b')


def test_defaults_have_no_approved_roots_or_real_execution():
    root = Path(__file__).parents[1]
    for task in ('a', 'b'):
        cfg = json.loads((root / f'configs/task_{task}_readiness.example.json').read_text())
        assert cfg['real_training_enabled'] is False
        assert all(row['access_authorized'] is False and not row['access_reference']
                   and not row['license_reference'] for row in cfg['inventory']['roots'])
    notebook = json.loads(
        (root / 'notebooks/08_private_real_data_readiness_training.ipynb').read_text())
    settings = next(c for c in notebook['cells'] if c['id'] == 'settings')
    assert "ACTION = 'inventory'" in ''.join(settings['source'])


def test_single_entry_notebook_inventory_runs_from_shared_bundle(tmp_path):
    import hashlib
    import subprocess
    import sys
    import zipfile

    from tests.test_task_ab_colab_setup import cells, script
    builder = script('build_task_a_bundle')
    archive = tmp_path / 'code.zip'
    builder.build_bundle(archive)
    snapshot = tmp_path / 'code'
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(snapshot)  # Our generated code-only fixture.
    content = tmp_path / 'content'
    content.mkdir()
    (content / 'brainagenticthon-code-state.json').write_text(json.dumps({
        'code_root': str(snapshot), 'manifest_sha256': hashlib.sha256(
            (snapshot / 'code_manifest.json').read_bytes()).hexdigest()}))
    base = tmp_path / 'drive'
    root = base / 'private'
    (root / 'raw').mkdir(parents=True)
    cfg = config(root, 'b')
    cfg['image_root'] = 'raw'
    cfg['inventory']['roots'][0]['path'] = 'raw'
    path = root / 'config.json'
    path.write_text(json.dumps(cfg))
    codes = cells('08_private_real_data_readiness_training.ipynb')
    codes = {key: code.replace('/content', str(content)) for key, code in codes.items()}
    payload = tmp_path / 'cells.json'
    payload.write_text(json.dumps(codes))
    source = '''
import json, sys
from pathlib import Path
codes = json.loads(Path(sys.argv[1]).read_text())
ns = {}
exec(codes['import-code'], ns)
ns.update(CONFIG_PATH=Path(sys.argv[2]), DRIVE_BASE=Path(sys.argv[3]))
exec(codes['settings'], ns)
ns['TASK'] = 'b'
exec(codes['execute-action'], ns)
assert ns['ACTION'] == 'inventory'
assert ns['REPORT']['status'] == 'BLOCKED'
assert 'missing_private_manifest' in ns['REPORT']['blockers']
assert ns['REPORT_PATH'].is_file()
assert not (Path(sys.argv[3]) / 'private/runs').exists()
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', source, str(payload), str(path), str(base)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(root.rglob('*.pt'))


def test_explicit_training_dispatch_only_after_gate_freeze_and_cuda(private_b, monkeypatch):
    """Control-flow mock only; fit never executes real or synthetic CUDA training."""
    import shutil

    import brainagenticthon.execution.real_training as training
    path, root = private_b
    raw = root / 'raw'
    raw.mkdir()
    for image in root.glob('*.nii'):
        shutil.copyfile(image, raw / image.name)
    cfg = json.loads(path.read_text())
    cfg['image_root'] = 'raw'
    cfg['inventory']['roots'][0]['path'] = 'raw'
    path.write_text(json.dumps(cfg))
    frozen = root / 'frozen'
    freeze_splits(path, frozen, task='b')
    cfg['real_training_enabled'] = True
    path.write_text(json.dumps(cfg))
    calls = []
    def no_training(cohort, settings, output, **options):
        calls.append(options)
        assert settings.device == 'cuda'
        assert len(cohort.rows) == 3
        output.mkdir(parents=True)
        return None, [{'epoch': 1}], {}
    monkeypatch.setattr(training.task_b, 'fit', no_training)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    result = execute_real(path, frozen, task='b', action='train')
    assert result['completed_epochs'] == 1 and calls == [{'resume': False, 'trusted_resume': False}]
    assert not list(root.rglob('*.pt'))  # Mock dispatch produced no trained model.
