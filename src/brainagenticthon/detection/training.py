"""Colab/CPU research training, atomic epoch checkpoints and explicit test evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import uuid
import zipfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from brainagenticthon.detection.data import (
    PreparedCohort,
    SyntheticCohort,
    load_training_evidence,
    prepare_cohort,
)
from brainagenticthon.detection.evaluation import evaluate
from brainagenticthon.detection.model import Tiny3DClassifier
from brainagenticthon.preprocessing.manifest_schema import ManifestCollection
from brainagenticthon.preprocessing.oasis2_mri_quality import digest_file, private_call
from brainagenticthon.preprocessing.task_a_readiness import load_reviews


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 1
    gradient_accumulation: int = 2
    learning_rate: float = 0.0003
    seed: int = 42
    mixed_precision: bool = True
    num_workers: int = 0
    device: str = "cuda"
    threshold: float = 0.5

    def __post_init__(self):
        for name, low, high in (("epochs", 1, 1000), ("batch_size", 1, 16),
                                ("gradient_accumulation", 1, 64), ("num_workers", 0, 2),
                                ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid_training_integer")
        if (type(self.mixed_precision) is not bool or self.device not in {"cpu", "cuda"}
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1
                or not math.isfinite(self.threshold) or not 0 < self.threshold < 1):
            raise ValueError("invalid_training_configuration")


def private_output_directory(output, protected=()):
    """Disjoint from inputs and every Git worktree; reject symlink destinations."""
    candidate = Path(output).absolute()
    if any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError("output_symlink_rejected")
    candidate = candidate.resolve()
    if any((p / ".git").is_file() or (p / ".git" / "HEAD").is_file()
           for p in (candidate, *candidate.parents)):
        raise ValueError("outputs_must_remain_outside_git")
    for source in protected:
        source = Path(source).resolve()
        if candidate.is_relative_to(source) or source.is_relative_to(candidate):
            raise ValueError("outputs_must_be_disjoint_from_sources")
    return candidate


def _atomic_write(path, writer):
    temporary = path.parent / (".task-a-" + uuid.uuid4().hex + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_write(path, lambda p: p.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"))


def _implementation_digest():
    package = Path(__file__).resolve().parents[1]
    files = sorted((package / "detection").glob("*.py")) + [
        package / "preprocessing" / name for name in (
            "task_a_readiness.py", "manifest_schema.py", "dataset_registry.py",
            "split_manager.py", "data_validator.py", "oasis2_mri_quality.py",
        )
    ]
    return hashlib.sha256("".join(digest_file(p) for p in files).encode()).hexdigest()


def _signature(cohort, config):
    settings = asdict(config)
    settings.pop("epochs")  # Only total epoch count may increase on resume.
    return {"format_version": 1, "architecture": "tiny-residual-3d-v1",
            "mode": cohort.mode, "dataset_sha256": cohort.fingerprint,
            "profile": cohort.profile.to_dict(), "settings": settings,
            "implementation_sha256": _implementation_digest(),
            "runtime": {"torch": str(torch.__version__), "numpy": version("numpy"),
                        "nibabel": version("nibabel"),
                        "python": platform.python_version(), "cuda": torch.version.cuda}}


def _resume_checkpoint(output, signature):
    def read():
        marker = json.loads((output / "latest.json").read_text())
        name = marker["checkpoint"]
        if (not isinstance(name, str) or Path(name).name != name
                or not name.startswith("epoch-") or not name.endswith(".pt")):
            raise ValueError("invalid_checkpoint_marker")
        checkpoint = output / name
        if checkpoint.is_symlink() or checkpoint.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("unsafe_checkpoint")
        if digest_file(checkpoint) != marker["sha256"]:
            raise ValueError("checkpoint_integrity_failed")
        with zipfile.ZipFile(checkpoint) as archive:
            if sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
                raise ValueError("checkpoint_resource_limit")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("resume_requires_identical_data_profile_code_and_settings")
        return state
    return private_call(read)


@dataclass
class TrainingResult:
    model: nn.Module
    history: list
    output: Path
    signature: dict
    threshold: float
    device: str


def fit(cohort, config, output_dir, *, resume=False, trusted_resume=False, on_epoch=None):
    """Train on gated data or explicit synthetic tensors, never directly on a path.

    Resume repeats only an interrupted, uncheckpointed epoch. A generation file
    followed by an atomic latest.json commit marker protects the previous complete
    epoch if writing is interrupted. Test data is never loaded by this function.
    """
    if not isinstance(cohort, (PreparedCohort, SyntheticCohort)):
        raise ValueError("prepared_or_explicit_synthetic_cohort_required")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_requested_but_unavailable")
    output = private_output_directory(output_dir, cohort.source_roots)
    if resume and not trusted_resume:
        raise ValueError("only_explicitly_trusted_own_checkpoints_may_resume")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("new_run_requires_empty_private_output_directory")
    if (output / "test_metrics.json").exists():
        raise ValueError("test_already_evaluated_do_not_continue_training_this_run")
    signature = _signature(cohort, config)
    state = _resume_checkpoint(output, signature) if resume else None
    if state and state["epoch"] >= config.epochs:
        raise ValueError("resume_total_epochs_must_increase")
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config.seed)
    model = Tiny3DClassifier().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp = config.mixed_precision and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_data = cohort.dataset("train")
    sampler = WeightedRandomSampler(
        train_data.patient_weights, len(train_data), replacement=True, generator=generator,
    )
    loader = DataLoader(train_data, batch_size=config.batch_size, sampler=sampler,
                        num_workers=config.num_workers, generator=generator)
    validation = DataLoader(cohort.dataset("validation"), batch_size=config.batch_size,
                            num_workers=config.num_workers)
    history, best, best_state, start = [], -1.0, None, 0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["sampler_rng"])
        if amp or config.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, best, best_state, start = (state["history"], state["best_auroc"],
                                            state["best_model"], state["epoch"])
    output.mkdir(parents=True, exist_ok=True)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(start, config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, examples = 0.0, 0
        for step, batch in enumerate(loader):
            window_start = (step // config.gradient_accumulation) * config.gradient_accumulation
            window_size = min(config.gradient_accumulation, len(loader) - window_start)
            with torch.autocast(device_type=config.device, enabled=amp):
                logits = model(batch["image"].to(config.device))
                labels = batch["label"].to(config.device, dtype=torch.float32)
                loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            scaler.scale(loss / window_size).backward()
            if (step + 1) % config.gradient_accumulation == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(labels)
            examples += len(labels)
        metrics = evaluate(model, validation, config.device, config.threshold)
        auroc = metrics["overall"]["auroc"]
        if auroc is None:
            raise ValueError("validation_requires_both_patient_classes")
        if auroc > best:
            best = auroc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        history.append({"epoch": epoch + 1, "training_loss": total_loss / examples,
                        "validation": metrics, "amp_enabled": amp})
        state = {"signature": signature, "epoch": epoch + 1,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                 "sampler_rng": generator.get_state(), "history": history,
                 "best_auroc": best, "best_model": best_state}
        checkpoint = output / f"epoch-{epoch + 1:04d}.pt"
        _atomic_write(checkpoint, lambda p, snapshot=state: torch.save(snapshot, p))
        _write_json(output / "latest.json", {"checkpoint": checkpoint.name,
                                           "sha256": digest_file(checkpoint)})
        _write_json(output / "history.json", {"signature": signature, "epochs": history})
        if on_epoch is not None:
            on_epoch(copy.deepcopy(history[-1]))
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, history, output, signature, config.threshold, config.device)


def evaluate_test(result, cohort, *, confirm_final_evaluation=False):
    """Explicit final test step; never selects weights or threshold using test data."""
    if not confirm_final_evaluation:
        raise ValueError("explicit_final_test_confirmation_required")
    if result.signature["dataset_sha256"] != cohort.fingerprint:
        raise ValueError("test_cohort_mismatch")
    if (result.output / "test_metrics.json").exists():
        raise ValueError("test_already_evaluated")
    loader = DataLoader(cohort.dataset("test"), batch_size=1, num_workers=0)
    metrics = evaluate(result.model, loader, result.device, result.threshold)
    _write_json(result.output / "test_metrics.json", metrics)
    return metrics


def load_private_cohort(manifest, reviews, evidence, image_root, profile=None):
    """No input records or exception details are printed; no inputs are modified."""
    def read():
        inputs = [Path(p).resolve() for p in (manifest, reviews, evidence)]
        if any(p.stat().st_size > 32 * 1024 * 1024 for p in inputs):
            raise ValueError("private_metadata_exceeds_limit")
        before = {p: digest_file(p) for p in inputs}
        cohort = prepare_cohort(
            ManifestCollection.load_json(inputs[0]), load_reviews(inputs[1]),
            load_training_evidence(inputs[2]), image_root=image_root, profile=profile,
        )
        if any(digest_file(p) != digest for p, digest in before.items()):
            raise ValueError("input_metadata_changed_during_gate")
        cohort.source_roots = (*cohort.source_roots, *inputs)
        return cohort
    return private_call(read)


def cuda_smoke_test():
    """Require actual CUDA forward/backward, finite gradients and a weight update."""
    if not torch.cuda.is_available():
        raise ValueError("cuda_requested_but_unavailable")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = Tiny3DClassifier().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.head.weight.detach().clone()
        image = torch.randn(2, 1, 16, 16, 16, device="cuda")
        labels = torch.tensor([0., 1.], device="cuda")
        with torch.autocast("cuda"):
            loss = nn.functional.binary_cross_entropy_with_logits(model(image), labels)
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError("cuda_smoke_nonfinite")
        optimizer.step()
        torch.cuda.synchronize()
        if torch.equal(before, model.head.weight):
            raise RuntimeError("cuda_smoke_no_update")
        return {"cuda_used": True, "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "finite_loss": float(loss.detach()), "weight_update": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.synthetic:
        parser.error("CLI smoke mode requires --synthetic; verified-data workflow is notebook 05")
    # Paths are deliberately excluded from stdout and from caught error details.
    try:
        torch.set_num_threads(1)
        if args.device == "cuda":
            print(json.dumps(cuda_smoke_test()))
        result = fit(SyntheticCohort(), TrainConfig(epochs=args.epochs, device=args.device),
                     args.output)
        metrics = evaluate_test(result, SyntheticCohort(), confirm_final_evaluation=True)
        print(json.dumps({"synthetic_only": True, "epochs": len(result.history),
                          "test": metrics["overall"]}, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({"completed": False, "reason": "synthetic_smoke_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
