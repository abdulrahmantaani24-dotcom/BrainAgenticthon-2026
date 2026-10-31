"""Streaming Task B training; private atomic checkpoints and strict epoch resume."""
import argparse
import copy
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as functional

from brainagenticthon.detection.training import (
    _atomic_write,
    _resume_checkpoint,
    _write_json,
    private_output_directory,
)
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_call
from brainagenticthon.segmentation.data import (
    SyntheticCohort,
    VerifiedCohort,
    synchronized_patch,
)
from brainagenticthon.segmentation.inference import region_dice, sliding_window
from brainagenticthon.segmentation.model import UNet3D


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = .001
    seed: int = 42
    device: str = 'cuda'
    mixed_precision: bool = True
    patch: tuple = (64, 64, 64)
    patches_per_case: int = 2

    def __post_init__(self):
        import math
        if (type(self.epochs) is not int or not 1 <= self.epochs <= 1000
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.patches_per_case) is not int or not 1 <= self.patches_per_case <= 64
                or type(self.mixed_precision) is not bool or self.device not in {'cpu', 'cuda'}
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1
                or len(self.patch) != 3
                or any(type(n) is not int or not 8 <= n <= 160 for n in self.patch)):
            raise ValueError('invalid_segmentation_training_config')


def segmentation_loss(logits, target):
    probabilities = logits.float().softmax(1)
    truth = functional.one_hot(target, 4).movedim(-1, 1).float()
    axes = (0, 2, 3, 4)
    numerator = 2 * (probabilities * truth).sum(axes)
    denominator = (probabilities + truth).sum(axes)
    dice = ((numerator + 1e-5) / (denominator + 1e-5))[1:].mean()
    return functional.cross_entropy(logits, target) + 1 - dice


def evaluate(model, cohort, split, config):
    rows = []
    for image, mask in cohort.iter_samples(split):
        prediction = sliding_window(model, image, patch=config.patch,
                                    device=config.device).argmax(0)
        rows.append(region_dice(prediction, mask))
    if not rows:
        raise ValueError('evaluation_split_empty')
    aggregate = {}
    for key in ('WT', 'TC', 'ET'):
        values = [r[key] for r in rows if r[key] is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    defined = [v for v in aggregate.values() if v is not None]
    return {'cases': len(rows), 'dice': aggregate,
            'mean_dice': sum(defined) / len(defined) if defined else None,
            'both_empty': 'excluded', 'empty_prediction_with_nonempty_target': 0}


def signature(cohort, config):
    settings = asdict(config)
    settings.pop('epochs')
    package = Path(__file__).parent
    code = ''.join(digest_file(p) for p in sorted(package.glob('*.py')))
    import nibabel
    import numpy
    return {'format': 1, 'architecture': 'compact-unet3d-v1', 'mode': cohort.mode,
            'dataset': cohort.fingerprint, 'settings': settings,
            'code': hashlib.sha256(code.encode()).hexdigest(),
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'cuda': torch.version.cuda, 'numpy': numpy.__version__,
                        'nibabel': nibabel.__version__}}


def fit(cohort, config, output, *, resume=False, trusted_resume=False, on_epoch=None):
    if not isinstance(cohort, (VerifiedCohort, SyntheticCohort)):
        raise ValueError('verified_or_explicit_synthetic_cohort_required')
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda_requested_but_unavailable')
    output = private_output_directory(output, cohort.source_roots)
    if resume and not trusted_resume:
        raise ValueError('trusted_own_checkpoint_required')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('new_run_requires_empty_output')
    if (output / 'test_metrics.json').exists():
        raise ValueError('test_already_evaluated')
    bound = signature(cohort, config)
    state = _resume_checkpoint(output, bound) if resume else None
    if state and state['epoch'] >= config.epochs:
        raise ValueError('increase_total_epochs')
    torch.manual_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config.seed)
    model = UNet3D().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp = config.device == 'cuda' and config.mixed_precision
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start, best, best_model = [], 0, -1, None
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['patch_rng'])
        if config.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, start = state['history'], state['epoch']
        best, best_model = state['best'], state['best_model']
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, config.epochs):
        model.train()
        losses = []
        for image, mask in cohort.iter_samples('train'):
            for _ in range(config.patches_per_case):
                x, y = synchronized_patch(image, mask, config.patch, generator, True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(config.device, enabled=amp):
                    loss = segmentation_loss(model(x[None].to(config.device)),
                                             y[None].to(config.device))
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite_training_loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
        if not losses:
            raise ValueError('training_split_empty')
        metrics = evaluate(model, cohort, 'validation', config)
        score = metrics['mean_dice']
        if score is None:
            raise ValueError('no_validation_lesion_evidence')
        if score > best:
            best = score
            best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        history.append({'epoch': epoch + 1, 'loss': sum(losses) / len(losses),
                        'validation': metrics, 'amp_enabled': amp})
        state = {'signature': bound, 'epoch': epoch + 1, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'torch_rng': torch.get_rng_state(), 'patch_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if config.device == 'cuda' else [],
                 'history': history, 'best': best, 'best_model': best_model}
        checkpoint = output / f'epoch-{epoch + 1:04d}.pt'
        _atomic_write(checkpoint, lambda p, snapshot=state: torch.save(snapshot, p))
        _write_json(output / 'latest.json', {'checkpoint': checkpoint.name,
                                            'sha256': digest_file(checkpoint)})
        _write_json(output / 'history.json', {'epochs': history, 'signature': bound})
        if on_epoch:
            on_epoch(history[-1])
    model.load_state_dict(best_model)
    model.eval()
    return model, history, bound


def evaluate_test(model, cohort, config, output, bound, *, confirm=False):
    output = private_output_directory(output, cohort.source_roots)
    if not confirm or bound != signature(cohort, config):
        raise ValueError('explicit_final_test_and_identical_experiment_required')
    if (output / 'test_metrics.json').exists():
        raise ValueError('test_already_evaluated')
    metrics = evaluate(model, cohort, 'test', config)
    _write_json(output / 'test_metrics.json', metrics)
    return metrics


def load_private_cohort(manifest, root, evidence):
    def read():
        paths = [Path(manifest).resolve(), Path(evidence).resolve()]
        if any(p.stat().st_size > 32 * 1024 * 1024 for p in paths):
            raise ValueError('metadata_resource_limit')
        before = [digest_file(p) for p in paths]
        cohort = VerifiedCohort(json.loads(paths[0].read_text()), root,
                                json.loads(paths[1].read_text()))
        if before != [digest_file(p) for p in paths]:
            raise ValueError('metadata_changed')
        cohort.source_roots = (*cohort.source_roots, *paths)
        return cohort
    return private_call(read)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    config = TrainConfig(epochs=args.epochs, device=args.device, patch=(16, 16, 16))
    model, history, bound = fit(SyntheticCohort(), config, args.output)
    metrics = evaluate_test(model, SyntheticCohort(), config, args.output, bound, confirm=True)
    print(json.dumps({'synthetic_only': True, 'device': args.device,
                      'epochs': len(history), 'test': metrics}))


if __name__ == '__main__':
    main()
