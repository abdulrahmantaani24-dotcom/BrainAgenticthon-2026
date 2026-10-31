"""Executable synthetic validation; PASS requires observed execution on the requested device."""
import argparse
import hashlib
import json
import math
import platform
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import torch

from brainagenticthon.detection import training as task_a
from brainagenticthon.detection.data import SyntheticCohort as SyntheticA
from brainagenticthon.detection.predict import load_model as load_a
from brainagenticthon.detection.predict import predict_validation
from brainagenticthon.detection.training import _resume_checkpoint, private_output_directory
from brainagenticthon.execution.contracts import classification_result, segmentation_result
from brainagenticthon.segmentation import training as task_b
from brainagenticthon.segmentation.data import SyntheticCohort as SyntheticB
from brainagenticthon.segmentation.inference import sliding_window
from brainagenticthon.segmentation.predict import load_model as load_b


def runtime():
    available = torch.cuda.is_available()
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': version('numpy'), 'nibabel': version('nibabel'),
            'cuda_available': available, 'cuda_runtime': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name() if available else None}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def checkpoint_inventory(run, bound):
    state = _resume_checkpoint(run, bound)
    marker = json.loads((run / 'latest.json').read_text())
    return {'checkpoint': marker['checkpoint'], 'sha256': marker['sha256'],
            'bytes': (run / marker['checkpoint']).stat().st_size,
            'completed_epoch': state['epoch'], 'integrity_verified': True,
            'weights_only_reload': True, 'architecture': bound.get('architecture')}, state


def _task_a(output, device):
    cohort = SyntheticA()
    config = task_a.TrainConfig(epochs=1, device=device, gradient_accumulation=2)
    first = task_a.fit(cohort, config, output)
    config = replace(config, epochs=2)
    result = task_a.fit(cohort, config, output, resume=True, trusted_resume=True)
    inventory, state = checkpoint_inventory(output, result.signature)
    require(inventory['completed_epoch'] == 2, 'resume_epoch_mismatch')
    require(result.history[0] == first.history[0], 'resume_history_changed')
    require(next(result.model.parameters()).device.type == device, 'wrong_execution_device')
    require(all(row['amp_enabled'] == (device == 'cuda') for row in result.history), 'amp_mismatch')
    require(bool(state['scaler']) == (device == 'cuda'), 'scaler_state_mismatch')
    require(bool(state['cuda_rng']) == (device == 'cuda'), 'cuda_rng_state_missing')
    require(all(math.isfinite(row['training_loss']) for row in result.history), 'nonfinite_loss')
    require(not torch.equal(state['model']['head.weight'],
                            _initial_head(task_a.Tiny3DClassifier, device)), 'no_model_update')
    loaded = load_a(cohort, config, output, trusted=True)
    require(all(torch.equal(value, loaded.state_dict()[key])
                for key, value in result.model.state_dict().items()), 'reload_mismatch')
    scores = predict_validation(loaded, cohort, device=device)
    contract = classification_result(scores[0]['abnormality_score'])
    metrics = task_a.evaluate_test(result, cohort, confirm_final_evaluation=True)
    return {'status': 'PASS', 'device_used': device, 'amp_verified': device == 'cuda',
            'gradient_accumulation': config.gradient_accumulation, 'model_updated': True,
            'checkpoint_reload_verified': True, 'resume_verified': True,
            'losses': [row['training_loss'] for row in result.history],
            'checkpoint': inventory, 'evaluation': metrics['overall'], 'contract': contract}


def _initial_head(factory, device):
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        return factory().head.weight.detach().cpu()


def _task_b(output, device):
    cohort = SyntheticB()
    config = task_b.TrainConfig(epochs=1, device=device, patch=(16, 16, 16), patches_per_case=1)
    _, first, _ = task_b.fit(cohort, config, output)
    config = replace(config, epochs=2)
    model, history, bound = task_b.fit(cohort, config, output, resume=True, trusted_resume=True)
    inventory, state = checkpoint_inventory(output, bound)
    require(inventory['completed_epoch'] == 2 and history[0] == first[0], 'resume_history_mismatch')
    require(next(model.parameters()).device.type == device, 'wrong_execution_device')
    require(all(row['amp_enabled'] == (device == 'cuda') for row in history), 'amp_mismatch')
    require(bool(state['scaler']) == (device == 'cuda'), 'scaler_state_mismatch')
    require(bool(state['cuda_rng']) == (device == 'cuda'), 'cuda_rng_state_missing')
    require(all(math.isfinite(row['loss']) for row in history), 'nonfinite_loss')
    require(not torch.equal(state['model']['head.weight'],
                            _initial_head(task_b.UNet3D, device)), 'no_model_update')
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
            and any(g.abs().sum() > 0 for g in gradients), 'invalid_segmentation_gradients')
    loaded = load_b(cohort, config, output, trusted=True)
    require(all(torch.equal(value, loaded.state_dict()[key])
                for key, value in model.state_dict().items()), 'reload_mismatch')
    image, _ = next(cohort.iter_samples('validation'))
    mask = sliding_window(loaded, image, patch=config.patch, device=device).argmax(0)
    contract = segmentation_result(mask, torch.eye(4).numpy())
    metrics = task_b.evaluate_test(model, cohort, config, output, bound, confirm=True)
    return {'status': 'PASS', 'device_used': device, 'amp_verified': device == 'cuda',
            'finite_nonzero_gradients': True, 'model_updated': True,
            'checkpoint_reload_verified': True, 'resume_verified': True,
            'losses': [row['loss'] for row in history], 'checkpoint': inventory,
            'evaluation': metrics, 'contract': contract}


def run_validation(output, *, tasks='both', mode='synthetic_gpu'):
    """Never reads real data. Fresh private run only. Failures are saved without private details."""
    if tasks not in {'a', 'b', 'both'} or mode not in {'synthetic', 'synthetic_gpu'}:
        raise ValueError('explicit_synthetic_mode_and_task_selection_required')
    output = private_output_directory(output)
    output.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if mode == 'synthetic_gpu' else 'cpu'
    report = {'schema': 'task-ab-execution-v1', 'status': 'FAIL', 'scope': 'synthetic_engineering',
              'requested_mode': mode, 'requested_tasks': tasks, 'runtime': runtime(),
              'gpu_executed': False, 'clinical_approval': False, 'tasks': {}}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        pins = {'torch': '2.6.0', 'numpy': '2.5.4', 'nibabel': '5.4.2'}
        require(platform.python_version_tuple()[:2] == ('3', '12'), 'python_3_12_required')
        require(all(version(name).split('+')[0] == expected
                    for name, expected in pins.items()), 'research_dependency_pin_mismatch')
        report['package_pins_verified'] = True
        if device == 'cuda':
            if not torch.cuda.is_available():
                report['reason'] = 'cuda_unavailable; no_cpu_fallback'
                return report
            report['cuda_smoke'] = task_a.cuda_smoke_test()
            report['gpu_executed'] = True
            torch.cuda.reset_peak_memory_stats()
        for name, execute in (('a', _task_a), ('b', _task_b)):
            if tasks not in {name, 'both'}:
                continue
            try:
                report['tasks'][name] = execute(output / ('task_' + name), device)
            except Exception as error:
                report['tasks'][name] = {'status': 'FAIL', 'error_type': type(error).__name__,
                                         'reason': 'synthetic_task_validation_failed'}
        selected = ('a', 'b') if tasks == 'both' else (tasks,)
        if all(report['tasks'].get(name, {}).get('status') == 'PASS' for name in selected):
            report['status'] = 'PASS'
    except Exception as error:
        report['error_type'] = type(error).__name__
        report['reason'] = 'startup_or_cuda_smoke_failed'
    finally:
        torch.set_num_threads(previous_threads)
        if device == 'cuda' and report['gpu_executed']:
            report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
        blob = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        # Exclusive report creation prevents replacing previous evidence.
        with (output / 'execution_report.json').open('x') as stream:
            stream.write(blob)
        with (output / 'execution_report.sha256').open('x') as stream:
            stream.write(hashlib.sha256(blob.encode()).hexdigest() + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', choices=('a', 'b', 'both'), default='both')
    parser.add_argument('--mode', choices=('synthetic', 'synthetic_gpu'), default='synthetic_gpu')
    args = parser.parse_args(argv)
    report = run_validation(args.output, tasks=args.tasks, mode=args.mode)
    print(json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
