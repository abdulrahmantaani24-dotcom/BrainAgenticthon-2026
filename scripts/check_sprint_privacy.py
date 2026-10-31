"""Check sprint publication candidates without reading private/untracked data directories."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'requirements-inventory.txt', 'configs/disease_acquisition.example.json',
    'configs/disease_source_review.template.json', 'docs/four_source_delivery.md',
    'notebooks/09_four_source_private_launcher.ipynb', 'tests/test_four_source_delivery.py',
    'src/brainagenticthon/execution/disease_sources.py',
    'src/brainagenticthon/execution/annotation_review.py',
    'src/brainagenticthon/execution/source_workflow.py',
    'src/brainagenticthon/execution/handoff.py',
    'src/brainagenticthon/segmentation/lesion_metrics.py',
    'configs/task_a_readiness.example.json', 'configs/task_b_readiness.example.json',
    'configs/private_readiness_report.template.json', 'docs/task_ab_real_data_workflow.md',
    'notebooks/08_private_real_data_readiness_training.ipynb', 'tests/test_real_data_readiness.py',
    'src/brainagenticthon/execution/inventory.py',
    'src/brainagenticthon/execution/real_training.py',
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
