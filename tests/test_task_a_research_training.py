"""CPU-only synthetic optimization/resume/metrics; no real-patient training."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from brainagenticthon.detection.data import SyntheticCohort
from brainagenticthon.detection.evaluation import binary_metrics, patient_metrics
from brainagenticthon.detection.localization import SpatialAnnotation, localization_target
from brainagenticthon.detection.model import Tiny3DClassifier, predict_scores
from brainagenticthon.detection.training import (
    TrainConfig,
    evaluate_test,
    fit,
    main,
    private_output_directory,
)


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cpu_config(epochs=1, **kwargs):
    return TrainConfig(epochs=epochs, device="cpu", **kwargs)


def test_model_logits_gradient_and_research_inference():
    model = Tiny3DClassifier()
    image = torch.randn(2, 1, 16, 16, 16)
    logits = model(image)
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert model.head.weight.grad.abs().sum() > 0
    scores = predict_scores(model, image)
    assert scores.shape == (2,) and torch.all((scores >= 0) & (scores <= 1))
    assert model.training  # Inference restores prior mode.
    assert not hasattr(model, "bounding_boxes")
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 16, 16, 16))


@pytest.mark.parametrize("scores,expected", [
    ([0.1, 0.9, 0.2, 0.8], 1.0), ([0.5, 0.5, 0.5, 0.5], 0.5),
    ([0.9, 0.1, 0.8, 0.2], 0.0),
])
def test_auroc_exact_and_ties(scores, expected):
    assert binary_metrics([0, 1, 0, 1], scores)["auroc"] == expected


def test_metrics_confusion_matrix_and_undefined_values():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["sensitivity"] == result["specificity"] == 0.5
    assert result["precision"] == result["recall"] == 0.5
    single = binary_metrics([0, 0], [0.1, 0.2])
    assert single["auroc"] is None and single["sensitivity"] is None
    assert single["precision"] is None
    json.dumps(single, allow_nan=False)


def test_patient_aggregation_prevents_image_level_pseudoreplication():
    result = patient_metrics([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], [0, 0, 0, 1])
    assert result["patients"] == 2 and result["confusion_matrix"] == [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="conflicting_patient"):
        patient_metrics([0, 1], [0.1, 0.9], [1, 1])


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 2], [0.1, 0.8]),
                                         ([0, 1], [0.1, float("nan")]), ([0], [2.0])])
def test_invalid_evaluation_rejected(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


def test_cpu_smoke_weights_change_checkpoint_and_test_separation(tmp_path):
    data = SyntheticCohort()
    torch.manual_seed(42)
    initial = Tiny3DClassifier().head.weight.detach().clone()
    result = fit(data, cpu_config(), tmp_path / "private-run")
    assert not torch.equal(result.model.head.weight.detach(), initial)
    assert len(result.history) == 1 and not result.history[0]["amp_enabled"]
    assert result.history[0]["validation"]["overall"]["patients"] == 4
    assert not (result.output / "test_metrics.json").exists()
    with pytest.raises(ValueError):
        evaluate_test(result, data)
    report = evaluate_test(result, data, confirm_final_evaluation=True)
    assert report["overall"]["patients"] == 4
    assert report["by_source"][0]["patients"] == 4
    with pytest.raises(ValueError, match="already_evaluated"):
        evaluate_test(result, data, confirm_final_evaluation=True)
    with pytest.raises(ValueError, match="test_already"):
        fit(data, cpu_config(2), result.output, resume=True, trusted_resume=True)


def test_interruption_resume_matches_uninterrupted_training(tmp_path):
    data = SyntheticCohort()
    config = cpu_config(2, batch_size=1, gradient_accumulation=3)
    full = fit(data, config, tmp_path / "full")

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(data, config, tmp_path / "resumed", on_epoch=interrupt)
    marker = json.loads((tmp_path / "resumed/latest.json").read_text())
    assert marker["checkpoint"] == "epoch-0001.pt"
    resumed = fit(data, config, tmp_path / "resumed", resume=True, trusted_resume=True)
    assert [epoch["epoch"] for epoch in resumed.history] == [1, 2]
    assert resumed.history == full.history
    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])


def test_interrupted_checkpoint_marker_preserves_previous_epoch(tmp_path, monkeypatch):
    import brainagenticthon.detection.training as training

    data = SyntheticCohort()
    config = cpu_config(2)
    original = training._write_json

    def interrupt_marker(path, value):
        if path.name == "latest.json" and value["checkpoint"] == "epoch-0002.pt":
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(training, "_write_json", interrupt_marker)
    with pytest.raises(KeyboardInterrupt):
        fit(data, config, tmp_path / "run")
    assert json.loads((tmp_path / "run/latest.json").read_text())["checkpoint"] == "epoch-0001.pt"
    monkeypatch.setattr(training, "_write_json", original)
    result = fit(data, config, tmp_path / "run", resume=True, trusted_resume=True)
    assert [h["epoch"] for h in result.history] == [1, 2]


def test_resume_requires_trust_integrity_and_same_configuration(tmp_path):
    data = SyntheticCohort()
    output = tmp_path / "run"
    fit(data, cpu_config(), output)
    with pytest.raises(ValueError, match="trusted"):
        fit(data, cpu_config(2), output, resume=True)
    with pytest.raises(RuntimeError):
        fit(data, cpu_config(2, batch_size=2), output, resume=True, trusted_resume=True)
    with pytest.raises(RuntimeError):
        fit(SyntheticCohort(seed=50), cpu_config(2), output, resume=True, trusted_resume=True)
    (output / "epoch-0001.pt").write_bytes(b"corrupt checkpoint")
    with pytest.raises(RuntimeError):
        fit(data, cpu_config(2), output, resume=True, trusted_resume=True)


def test_output_safety_and_no_implicit_gpu_fallback(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / ".git/HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError, match="outside_git"):
        private_output_directory(repository / "runs")
    raw = tmp_path / "raw"
    raw.mkdir()
    for output in (raw, raw / "checkpoints", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            private_output_directory(output, (raw,))
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        private_output_directory(link / "runs")
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        fit(SyntheticCohort(), cpu_config(), output)
    assert (output / "keep.txt").read_text() == "preserve"
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="cuda_requested"):
            fit(SyntheticCohort(), TrainConfig(epochs=1), tmp_path / "gpu")


@pytest.mark.parametrize("changes", [{"batch_size": 0}, {"epochs": -1},
                                     {"num_workers": 3}, {"mixed_precision": "yes"},
                                     {"learning_rate": float("nan")}, {"threshold": 1.0}])
def test_invalid_training_config(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def annotation():
    return SpatialAnnotation(
        case_id="synthetic", dataset_source="brats2024_gli", image_sha256="a" * 64,
        native_shape=(16, 16, 16), boxes=((1, 4, 2, 6, 3, 7),),
        provenance="expert_annotation", evidence_reference="synthetic-only",
        reviewer_reference="synthetic-only", review_status="approved",
    )


def test_localization_requires_real_annotations_and_returns_targets_only():
    record = SimpleNamespace(case_id="synthetic", dataset_source="brats2024_gli", task_a_label=1)
    target = localization_target(
        record, annotation(), image_sha256="a" * 64, native_shape=(16, 16, 16),
    )
    assert target == ((1, 4, 2, 6, 3, 7),)
    with pytest.raises(ValueError, match="genuine_spatial"):
        localization_target(record, None, image_sha256="a" * 64, native_shape=(16, 16, 16))
    for change in ({"review_status": "pending"}, {"boxes": ()},
                   {"boxes": ((0, 16, 0, 5, 0, 5),)}, {"coordinate_space": "resized_tensor"},
                   {"case_id": "other"}, {"provenance": "inferred"}):
        with pytest.raises(ValueError):
            localization_target(
                record, replace(annotation(), **change), image_sha256="a" * 64,
                native_shape=(16, 16, 16),
            )


def test_synthetic_cli_aggregate_only(tmp_path, capsys):
    assert main(["--synthetic", "--output", str(tmp_path / "smoke")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["synthetic_only"] and result["epochs"] == 1
    assert str(tmp_path) not in json.dumps(result)
