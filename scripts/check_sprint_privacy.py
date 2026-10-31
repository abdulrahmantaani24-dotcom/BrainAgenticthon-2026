"""Check sprint publication candidates without reading private/untracked data directories."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'README.md', 'docs/task_ab_execution_delivery.md',
    'docs/task_ab_submission_README.md', 'notebooks/07_task_ab_execution_launcher.ipynb',
    'configs/execution_report.template.json', 'configs/evaluation_report.template.json',
    'tests/test_task_ab_execution.py', 'src/brainagenticthon/detection/evaluation.py',
    'src/brainagenticthon/detection/predict.py',
    *('src/brainagenticthon/execution/' + name for name in (
        '__init__.py', 'contracts.py', 'validation.py', 'readiness.py')),
    '.gitignore', 'scripts/build_task_a_bundle.py', 'scripts/check_sprint_privacy.py',
    'src/brainagenticthon/detection/training.py',
    'notebooks/05_task_a_research_training.ipynb',
    'notebooks/06_task_b_segmentation_training.ipynb',
    'configs/task_b_research.example.json', 'docs/task_a_research_training.md',
    'docs/task_ab_colab_launch.md', 'tests/test_task_ab_colab_setup.py',
    'docs/task_b_research_training.md', 'tests/test_task_a_research_notebook.py',
    'tests/test_task_b_segmentation.py',
    *('src/brainagenticthon/segmentation/' + name for name in (
        '__init__.py', 'data.py', 'model.py', 'inference.py', 'training.py', 'predict.py')),
}


def check(*, staged=False):
    diff = ['git', 'diff', *(['--cached'] if staged else []), '--name-only', 'HEAD']
    tracked = subprocess.check_output(diff, cwd=ROOT)
    new = (b'' if staged else subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT))
    paths = set((tracked + new).decode().splitlines())
    failures = []
    for name in sorted(paths):
        if name not in ALLOWED:
            failures.append('unexpected_publication_candidate')
            continue
        blob = (subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT).decode()
                if staged else (ROOT / name).read_text())
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|'
                     r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', blob):
            failures.append('possible_secret')
        if name.endswith('.ipynb'):
            notebook = json.loads(blob)
            if any(c.get('outputs') or c.get('execution_count') is not None
                   for c in notebook['cells'] if c['cell_type'] == 'code'):
                failures.append('saved_notebook_state')
    protected = subprocess.check_output([
        *diff, '--',
        'src/brainagenticthon/classification',
        'src/brainagenticthon/preprocessing', 'docs/oasis2_clinical_policy.md',
        'docs/data_policy.md'], cwd=ROOT).decode().splitlines()
    if protected:
        failures.append('protected_policy_or_task_c_changed')
    return {'staged': staged, 'publication_candidates': len(paths),
            'privacy_checks_passed': not failures,
            'failures': failures,
            'scope': 'allowlist, secret patterns, empty notebook outputs, protected-file diff; '
                     'does not replace manual review or inspect private data'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    result = check(staged=args.staged)
    print(json.dumps(result))
    raise SystemExit(bool(result['failures']))
