"""Explicit private freeze/smoke/train/resume/evaluate workflow; no automatic real execution."""
import copy
import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from brainagenticthon.detection import training as task_a
from brainagenticthon.detection.model import Tiny3DClassifier
from brainagenticthon.detection.predict import load_model as load_a_model
from brainagenticthon.detection.training import (
    _atomic_write,
    _resume_checkpoint,
    private_output_directory,
)
from brainagenticthon.execution.inventory import (
    authorized_roots,
    gated_cohort,
    inspect_private,
    load_config,
    protected_inputs,
    read_json,
)
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_path
from brainagenticthon.segmentation import training as task_b
from brainagenticthon.segmentation.data import synchronized_patch
from brainagenticthon.segmentation.model import UNet3D
from brainagenticthon.segmentation.predict import load_model as load_b_model


def binding(cfg, root, task):
    keys = ('manifest', 'reviews', 'training_evidence') if task == 'a' else ('manifest', 'evidence')
    inputs = {key: digest_file(private_path(root, cfg[key])) for key in keys}
    payload = {'task': task, 'image_root': str(private_path(root, cfg['image_root'])),
               'profile': cfg.get('profile'), 'inventory_authorization': cfg['inventory'],
               'inputs': inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), inputs


def freeze_splits(config_path, destination, *, task):
    """Snapshot existing verified splits/evidence; never assigns or changes labels/splits."""
    report = inspect_private(config_path, task=task, scan=False)
    if report['status'] != 'PASS':
        raise ValueError('readiness_BLOCKED_cannot_freeze')
    cfg, root = load_config(config_path)
    output = private_output_directory(destination, protected_inputs(cfg, root, config_path))
    bound, inputs = binding(cfg, root, task)
    # Gates are rerun to bind the snapshot to current images and metadata.
    cohort = gated_cohort(cfg, root, task)
    if cohort.fingerprint != report['cohort_fingerprint']:
        raise ValueError('cohort_changed_during_freeze')
    manifest = read_json(private_path(root, cfg['manifest']))
    if task == 'a':
        from brainagenticthon.preprocessing.task_a_readiness import load_reviews
        reviews = load_reviews(private_path(root, cfg['reviews']))
        records = manifest if isinstance(manifest, list) else manifest['records']
        ledger = [{'patient_id': reviews[(
            r['dataset_source'], r['case_id'], r['image_path'])].patient_id,
                   'split': r['split']} for r in records]
    else:
        ledger = [{'patient_id': r['patient_id'], 'split': r['split']} for r in manifest['records']]
    patients = {}
    for row in ledger:
        if patients.setdefault(row['patient_id'], row['split']) != row['split']:
            raise ValueError('patient_split_leakage')
    output.mkdir(parents=True, exist_ok=False)
    for key, expected in inputs.items():
        blob = private_path(root, cfg[key]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('metadata_changed_during_freeze')
        (output / (key + '.json')).write_bytes(blob)
    frozen = {'schema': 'private-patient-split-freeze-v1', 'task': task,
              'binding_sha256': bound, 'cohort_fingerprint': cohort.fingerprint,
              'input_sha256': inputs, 'patient_splits': patients,
              'clinical_approval': False, 'split_policy': 'existing_verified_assignments_only'}
    with (output / 'freeze.json').open('x') as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True)
    with (output / 'freeze.sha256').open('x') as stream:
        stream.write(digest_file(output / 'freeze.json') + '\n')
    return {'status': 'PASS', 'patients': len(patients), 'training_started': False,
            'cohort_fingerprint': cohort.fingerprint}


def validate_freeze(config_path, freeze, *, task, cohort=None):
    cfg, root = load_config(config_path)
    directory = Path(freeze).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('freeze_symlink_rejected')
    frozen = read_json(directory / 'freeze.json')
    if digest_file(directory / 'freeze.json') != (directory / 'freeze.sha256').read_text().strip():
        raise ValueError('freeze_checksum_mismatch')
    bound, inputs = binding(cfg, root, task)
    if (frozen['task'] != task or frozen['binding_sha256'] != bound
            or frozen['input_sha256'] != inputs):
        raise ValueError('frozen_inputs_or_authorization_changed')
    for key, expected in inputs.items():
        snapshot = directory / (key + '.json')
        if snapshot.is_symlink() or digest_file(snapshot) != expected:
            raise ValueError('frozen_snapshot_corrupt')
    cohort = cohort or gated_cohort(cfg, root, task)
    if cohort.fingerprint != frozen['cohort_fingerprint']:
        raise ValueError('frozen_cohort_changed')
    return frozen


def _smoke(cohort, task, config, output):
    """One generated/verified training batch only; no test images or performance claims."""
    if output.exists():
        raise FileExistsError('fresh_smoke_output_required')
    torch.manual_seed(config.seed)
    if task == 'a':
        model = Tiny3DClassifier().to(config.device)
        sample = cohort.dataset('train')[0]
        x = sample['image'][None].to(config.device)
        y = torch.tensor([sample['label']], device=config.device)
    else:
        model = UNet3D().to(config.device)
        image, mask = next(cohort.iter_samples('train'))
        image, mask = synchronized_patch(image, mask, config.patch,
                                          torch.Generator().manual_seed(config.seed), False)
        x, y = image[None].to(config.device), mask[None].to(config.device)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp = config.device == 'cuda' and config.mixed_precision
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    with torch.autocast(config.device, enabled=amp):
        logits = model(x)
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, y) if task == 'a'
                else task_b.segmentation_loss(logits, y))
    if not torch.isfinite(loss):
        raise ValueError('nonfinite_smoke_loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('nonfinite_smoke_gradients')
    scaler.step(optimizer)
    scaler.update()
    if torch.equal(before, model.head.weight):
        raise ValueError('smoke_weights_did_not_change')
    output.mkdir(parents=True, exist_ok=False)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scaler': scaler.state_dict(), 'settings': asdict(config)}
    checkpoint = output / 'smoke-checkpoint.pt'
    _atomic_write(checkpoint, lambda path: torch.save(state, path))
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not all(torch.equal(value.cpu(), loaded['model'][key])
               for key, value in model.state_dict().items()):
        raise ValueError('smoke_reload_mismatch')
    return {'status': 'PASS', 'scope': 'one_training_batch_smoke_only',
            'loss': float(loss.detach()), 'device_used': config.device, 'amp_enabled': amp,
            'checkpoint_sha256': digest_file(checkpoint), 'checkpoint_reload_verified': True,
            'held_out_evaluation_performed': False, 'clinical_approval': False}


def execute_real(config_path, freeze, *, task, action):
    """All gates/frozen bytes must pass now; explicit CUDA opt-in is mandatory."""
    if action not in {'smoke', 'train', 'resume', 'evaluate'} or task not in {'a', 'b'}:
        raise ValueError('explicit_real_action_and_task_required')
    cfg, root = load_config(config_path)
    if cfg.get('mode') != 'verified' or cfg.get('real_training_enabled') is not True:
        raise ValueError('real_training_disabled')
    report = inspect_private(config_path, task=task, scan=False)
    if report['status'] != 'PASS':
        raise ValueError('readiness_BLOCKED_no_real_execution')
    authorized_roots(cfg, root)
    cohort = gated_cohort(cfg, root, task)
    validate_freeze(config_path, freeze, task=task, cohort=cohort)
    settings = copy.deepcopy(cfg['training'])
    if task == 'b':
        settings['patch'] = tuple(settings['patch'])
    config = (task_a.TrainConfig if task == 'a' else task_b.TrainConfig)(**settings)
    if config.device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('real_execution_requires_actual_cuda_no_fallback')
    if 'REPLACE_WITH' in cfg['output_run']:
        raise ValueError('fresh_private_output_run_required')
    output = private_output_directory(private_path(root, cfg['output_run']),
                                      (*protected_inputs(cfg, root, config_path), Path(freeze)))
    if action == 'smoke':
        result = _smoke(cohort, task, config, output)
    elif action in {'train', 'resume'}:
        resume = action == 'resume'
        if resume and (cfg.get('resume') is not True or cfg.get('trusted_resume') is not True):
            raise ValueError('explicit_trusted_resume_required')
        if not resume and cfg.get('resume') is not False:
            raise ValueError('new_training_requires_resume_false')
        fitted = (task_a.fit if task == 'a' else task_b.fit)(
            cohort, config, output, resume=resume, trusted_resume=resume)
        history = fitted.history if task == 'a' else fitted[1]
        result = {'status': 'PASS', 'completed_epochs': len(history), 'action': action,
                  'device_used': 'cuda', 'clinical_approval': False,
                  'held_out_evaluation_performed': False}
    else:
        if cfg.get('final_test_evaluation') is not True or cfg.get('trusted_resume') is not True:
            raise ValueError('explicit_final_test_and_own_checkpoint_trust_required')
        if task == 'a':
            model = load_a_model(cohort, config, output, trusted=True)
            bound = task_a._signature(cohort, config)
            state = _resume_checkpoint(output, bound)
            fitted = task_a.TrainingResult(model, state['history'], output, bound,
                                           config.threshold, config.device)
            metrics = task_a.evaluate_test(fitted, cohort, confirm_final_evaluation=True)
        else:
            model = load_b_model(cohort, config, output, trusted=True)
            metrics = task_b.evaluate_test(model, cohort, config, output,
                                           task_b.signature(cohort, config), confirm=True)
        result = {'status': 'PASS', 'action': action, 'metrics': metrics,
                  'clinical_approval': False, 'held_out_evaluation_performed': True}
    with (output / ('action-' + uuid.uuid4().hex + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result
