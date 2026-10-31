"""Publication checks use only code archives, generated configs and synthetic tensors."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('05_task_a_research_training.ipynb', '06_task_b_segmentation_training.ipynb')


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells(name):
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code'}


def test_both_notebooks_share_bundle_and_restart_setup():
    first, second = (cells(name) for name in NOTEBOOKS)
    for name in ('mount-drive', 'verify-bundle', 'dependencies'):
        assert first[name] == second[name]
    assert 'reviewed_code/task-ab-code.zip' in first['mount-drive']
    assert "'pip', 'check'" not in first['dependencies']
    builder = script('build_task_a_bundle')
    expected = {str(path.relative_to(ROOT))
                for folder in ('detection', 'segmentation')
                for path in (ROOT / 'src/brainagenticthon' / folder).glob('*.py')}
    assert expected.issubset(builder.BUNDLE_FILES)
    for name in ('requirements-task-a.txt', 'scripts/build_task_a_bundle.py',
                 'configs/task_a_research.example.json', 'configs/task_b_research.example.json',
                 'docs/task_ab_colab_launch.md', *(f'notebooks/{n}' for n in NOTEBOOKS)):
        assert name in builder.BUNDLE_FILES


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_standalone_bundle_setup_restart_and_no_cuda_fallback(tmp_path, name):
    builder = script('build_task_a_bundle')
    archive = tmp_path / 'task-ab-code.zip'
    digest = builder.build_bundle(archive)
    # A fresh interpreter cannot reuse the checkout's already-imported modules.
    payload = {'cells': cells(name), 'archive': str(archive), 'digest': digest,
               'content': str(tmp_path / 'content'), 'task': 'a' if name.startswith('05') else 'b'}
    data = tmp_path / 'payload.json'
    data.write_text(json.dumps(payload))
    source = r'''
import json, sys, types, subprocess
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
content = Path(payload['content'])
base = content / 'drive/MyDrive'
base.mkdir(parents=True)
module = types.ModuleType('google.colab')
module.drive = types.SimpleNamespace(mount=lambda _: None)
sys.modules['google.colab'] = module
codes = {key: code.replace('/content', str(content))
         for key, code in payload['cells'].items()}
namespace = {}
exec(codes['mount-drive'], namespace)
namespace.update(BUNDLE_PATH=Path(payload['archive']), BUNDLE_SHA256=payload['digest'])
# Verification intentionally requires the archive to reside in private Drive.
archive = base / 'task-ab-code.zip'
archive.write_bytes(Path(payload['archive']).read_bytes())
namespace['BUNDLE_PATH'] = archive
exec(codes['verify-bundle'], namespace)
exec(codes['verify-bundle'], namespace)
original = subprocess.run
calls = []
subprocess.run = lambda command, **kw: calls.append(command)
try:
    # Simulate a session restart: no in-memory CODE_ROOT or imports.
    dependency_namespace = {}
    exec(codes['dependencies'], dependency_namespace)
finally:
    subprocess.run = original
assert calls and calls[0][1:4] == ['-m', 'pip', 'install']
namespace = {}
exec(codes['import-code'], namespace)
root = namespace['CODE_ROOT']
assert namespace['brainagenticthon'].__file__.startswith(str(root))
assert root == dependency_namespace['CODE_ROOT']
template = root / ('configs/task_' + payload['task'] + '_research.example.json')
config = json.loads(template.read_text())
config.update(private_drive_root=str(base / 'private'), output_run='runs/cpu', mode='synthetic')
config['training']['epochs'] = 1
path = base / 'private-config.json'
path.write_text(json.dumps(config))
namespace.update(CONFIG_PATH=path, DRIVE_BASE=base)
for key in ('private-config', 'eligibility-gate', 'train-or-resume', 'final-evaluation'):
    exec(codes[key], namespace)
assert (namespace['OUTPUT'] / 'latest.json').is_file()
assert not (namespace['OUTPUT'] / 'test_metrics.json').exists()
config.update(mode='synthetic_gpu', output_run='runs/gpu')
path.write_text(json.dumps(config))
import torch
torch.cuda.is_available = lambda: False
try:
    exec(codes['private-config'], namespace)
except ValueError as error:
    assert 'cuda_requested_but_unavailable' in str(error)
else:
    raise AssertionError('synthetic_gpu silently fell back to CPU')
assert not (base / 'private/runs/gpu/latest.json').exists()
'''
    result = subprocess.run([sys.executable, '-I', '-c', source, str(data)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr + result.stdout
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {*builder.BUNDLE_FILES, 'code_manifest.json'}


def test_staged_privacy_reads_index_instead_of_sanitized_worktree(tmp_path, monkeypatch):
    module = script('check_sprint_privacy')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Synthetic Test',
                    '-c', 'user.email=synthetic@example.invalid', 'commit', '--allow-empty',
                    '-qm', 'synthetic baseline'], check=True)
    name = 'docs/task_ab_colab_launch.md'
    target = tmp_path / name
    target.parent.mkdir()
    target.write_text('gh' + 'p_' + 'A' * 36)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '--', name], check=True)
    target.write_text('sanitized working copy')
    assert module.check()['privacy_checks_passed'] is True
    assert module.check(staged=True)['failures'] == ['possible_secret']
