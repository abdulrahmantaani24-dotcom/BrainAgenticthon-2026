"""Generated tensors and synthetic metadata only; never clinical approval evidence."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from brainagenticthon.detection import training as training_a
from brainagenticthon.detection.data import SyntheticCohort
from brainagenticthon.detection.evaluation import calibration_metrics, patient_metrics
from brainagenticthon.detection.predict import load_model, predict_validation
from brainagenticthon.execution import validation
from brainagenticthon.execution.contracts import (
    classification_result,
    legacy_task_a_from_score,
    legacy_task_b_from_brats2021,
    segmentation_result,
)
from brainagenticthon.execution.readiness import inspect, save_inspection
from brainagenticthon.preprocessing.task_output_schemas import TaskAOutput, TaskBOutput


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_calibration_exact_and_patient_weighting():
    result = calibration_metrics([0, 1], [0., 1.])
    assert result['brier_score'] == result['ece'] == 0
    assert sum(row['count'] for row in result['bins']) == 2
    assert calibration_metrics([0, 1], [.5, .5])['brier_score'] == .25
    actual = patient_metrics([0, 0, 0, 1], [0., 0., 0., 1.], [0, 0, 0, 1])
    assert actual['calibration']['brier_score'] == 0
    assert actual['patients'] == 2
    with pytest.raises(ValueError):
        calibration_metrics([0], [float('nan')])


def test_unified_cpu_execution_checkpoint_inventory_and_no_overwrite(tmp_path):
    report = validation.run_validation(tmp_path / 'run', mode='synthetic', tasks='both')
    assert report['status'] == 'PASS' and not report['gpu_executed']
    assert set(report['tasks']) == {'a', 'b'}
    for task, row in report['tasks'].items():
        assert row['device_used'] == 'cpu' and not row['amp_verified']
        assert row['resume_verified'] and row['checkpoint_reload_verified']
        assert row['checkpoint']['completed_epoch'] == 2
        assert row['model_updated'] is True
        assert len(row['losses']) == 2
        assert (tmp_path / f'run/task_{task}/test_metrics.json').is_file()
    blob = (tmp_path / 'run/execution_report.json').read_bytes()
    assert (tmp_path / 'run/execution_report.sha256').read_text().strip() == (
        hashlib.sha256(blob).hexdigest())
    with pytest.raises(FileExistsError):
        validation.run_validation(tmp_path / 'run', mode='synthetic')
    assert (tmp_path / 'run/execution_report.json').read_bytes() == blob


@pytest.mark.parametrize('selection', ['a', 'b'])
def test_selection_runs_one_task(tmp_path, selection):
    result = validation.run_validation(tmp_path / 'run', mode='synthetic', tasks=selection)
    assert result['status'] == 'PASS' and set(result['tasks']) == {selection}
    assert not (tmp_path / ('run/task_' + ('b' if selection == 'a' else 'a'))).exists()


def test_gpu_unavailable_is_machine_readable_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = validation.run_validation(tmp_path / 'run', mode='synthetic_gpu')
    assert report['status'] == 'FAIL' and report['tasks'] == {}
    assert not report['gpu_executed'] and 'no_cpu_fallback' in report['reason']
    assert json.loads((tmp_path / 'run/execution_report.json').read_text()) == report
    assert not (tmp_path / 'run/task_a').exists()


def test_failure_does_not_prevent_other_task_and_redacts_details(tmp_path, monkeypatch):
    def failure(*_):
        raise ValueError('PRIVATE-PATH-SHOULD-NOT-APPEAR')
    monkeypatch.setattr(validation, '_task_a', failure)
    report = validation.run_validation(tmp_path / 'run', mode='synthetic')
    assert report['status'] == 'FAIL'
    assert report['tasks']['b']['status'] == 'PASS'
    assert 'PRIVATE-PATH' not in json.dumps(report)


def test_no_real_training_mode(tmp_path):
    with pytest.raises(ValueError):
        validation.run_validation(tmp_path / 'run', mode='verified')
    assert not (tmp_path / 'run').exists()


def test_readiness_missing_private_evidence_is_blocked_and_read_only(tmp_path):
    for task in ('a', 'b'):
        report = inspect(None, task=task)
        assert report['status'] == 'BLOCKED' and report['required_evidence']
        assert not report['training_started']
    root = tmp_path / 'private'
    root.mkdir()
    config = root / 'config.json'
    config.write_text(json.dumps({'private_drive_root': str(root)}))
    before = config.read_bytes()
    output = root / 'readiness.json'
    report = save_inspection(config, output, task='b')
    assert set(report['missing_inputs']) == {'manifest', 'evidence', 'image_root'}
    assert config.read_bytes() == before
    with pytest.raises(FileExistsError):
        save_inspection(config, output, task='b')
    with pytest.raises(ValueError):
        save_inspection(config, config, task='b')


def test_readiness_uses_existing_gate_and_preserves_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import brainagenticthon.execution.readiness as readiness
    root = tmp_path / 'private'
    root.mkdir()
    for name in ('manifest.json', 'evidence.json'):
        (root / name).write_text('{}')
    (root / 'images').mkdir()
    cfg = {'private_drive_root': str(root), 'manifest': 'manifest.json',
           'evidence': 'evidence.json', 'image_root': 'images'}
    path = root / 'config.json'
    path.write_text(json.dumps(cfg))
    calls = []
    def gate(*args):
        calls.append(args)
        return SimpleNamespace(rows=[{'split': split} for split in ('train', 'validation', 'test')])
    monkeypatch.setattr(readiness, 'load_b', gate)
    report = inspect(path, task='b')
    assert len(calls) == 1 and report['status'] == 'ELIGIBILITY_GATE_PASSED'
    assert not report['clinical_approval'] and not report['training_started']
    with pytest.raises(ValueError):
        save_inspection(path, root / 'images/report.json', task='b')


def test_research_contracts_and_explicit_legacy_rejection():
    score = classification_result(.8)
    assert not score['task_a_output_compatible'] and 'bounding_boxes' in score['unsupported']
    with pytest.raises(ValueError):
        TaskAOutput(case_id='synthetic', dataset_source='synthetic', finding_present=True)
    with pytest.raises(ValueError):
        legacy_task_a_from_score(.8)
    mask = torch.zeros(16, 16, 16, dtype=torch.long)
    mask[1:3, 2:4, 3:5] = 3
    result = segmentation_result(mask, np.diag([2., 3., 4., 1.]))
    assert result['boxes'][0]['bounds'] == [1, 2, 2, 3, 3, 4]
    assert result['voxel_size_mm3'] == pytest.approx(24)
    assert result['volume_ml']['ET'] == pytest.approx(.192)
    assert result['anatomical_regions'] is None and not result['task_b_output_compatible']
    with pytest.raises(ValueError):
        legacy_task_b_from_brats2021(result)
    assert TaskBOutput(case_id='synthetic', dataset_source='synthetic').anatomical_regions == []
    with pytest.raises(ValueError):
        segmentation_result(mask, np.zeros((4, 4)))


def test_classifier_reload_inference_requires_trust_and_same_settings(tmp_path):
    cohort = SyntheticCohort()
    config = training_a.TrainConfig(epochs=1, device='cpu')
    training_a.fit(cohort, config, tmp_path / 'run')
    with pytest.raises(ValueError):
        load_model(cohort, config, tmp_path / 'run')
    model = load_model(cohort, config, tmp_path / 'run', trusted=True)
    rows = predict_validation(model, cohort)
    assert len(rows) == 4 and all(0 <= row['abnormality_score'] <= 1 for row in rows)
    assert all(row['localization'] is None and not row['calibrated'] for row in rows)


def test_launcher_bundle_complete_and_restart_setup_matches():
    from tests.test_task_ab_colab_setup import cells, script
    root = Path(__file__).parents[1]
    launcher = cells('07_task_ab_execution_launcher.ipynb')
    old = cells('05_task_a_research_training.ipynb')
    for name in ('mount-drive', 'verify-bundle', 'dependencies'):
        assert launcher[name] == old[name]
    assert "MODE = 'synthetic_gpu'" in launcher['launcher-settings']
    files = script('build_task_a_bundle').BUNDLE_FILES
    for path in (root / 'src/brainagenticthon/execution').glob('*.py'):
        assert str(path.relative_to(root)) in files
    assert 'notebooks/07_task_ab_execution_launcher.ipynb' in files


def test_shared_launcher_runs_from_bundle_after_restart(tmp_path):
    from tests.test_task_ab_colab_setup import cells, script
    builder = script('build_task_a_bundle')
    archive = tmp_path / 'code.zip'
    builder.build_bundle(archive)
    import zipfile
    code_root = tmp_path / 'code'
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(code_root)  # Exclusive synthetic test fixture.
    codes = cells('07_task_ab_execution_launcher.ipynb')
    content = tmp_path / 'content'
    content.mkdir()
    state = {'code_root': str(code_root), 'manifest_sha256': hashlib.sha256(
        (code_root / 'code_manifest.json').read_bytes()).hexdigest()}
    (content / 'brainagenticthon-code-state.json').write_text(json.dumps(state))
    codes = {key: code.replace('/content', str(content)) for key, code in codes.items()}
    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps(codes))
    source = '''
import json, sys
from pathlib import Path
codes = json.loads(Path(sys.argv[1]).read_text())
ns = {}
exec(codes['import-code'], ns)
ns['DRIVE_BASE'] = Path(sys.argv[2])
exec(codes['launcher-settings'], ns)
ns.update(MODE='synthetic', TASKS='both')
exec(codes['execute'], ns)
exec(codes['readiness'], ns)
assert ns['REPORT']['status'] == 'PASS'
assert not ns['REPORT']['gpu_executed']
'''
    result = subprocess.run([sys.executable, '-I', '-c', source, str(payload), str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
