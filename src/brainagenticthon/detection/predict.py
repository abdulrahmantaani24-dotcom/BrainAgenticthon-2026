"""Reload trusted classifier checkpoints and score already-gated research cohorts."""
import torch

from brainagenticthon.detection.data import PreparedCohort, SyntheticCohort
from brainagenticthon.detection.model import Tiny3DClassifier, predict_scores
from brainagenticthon.detection.training import (
    _resume_checkpoint,
    _signature,
    private_output_directory,
)


def load_model(cohort, config, run, *, trusted=False):
    if not trusted or not isinstance(cohort, (PreparedCohort, SyntheticCohort)):
        raise ValueError('trusted_own_checkpoint_and_gated_cohort_required')
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda_requested_but_unavailable')
    run = private_output_directory(run, cohort.source_roots)
    state = _resume_checkpoint(run, _signature(cohort, config))
    model = Tiny3DClassifier().to(config.device)
    model.load_state_dict(state['best_model'])
    return model.eval()


def predict_validation(model, cohort, *, device='cpu'):
    """Only uncalibrated scores, no diagnosis/normality/localization export.

    Held-out test inference remains the explicit final evaluate_test workflow.
    No identifiers or image data are returned; rows are local validation indices.
    """
    if not isinstance(cohort, (PreparedCohort, SyntheticCohort)):
        raise ValueError('gated_or_synthetic_cohort_required')
    data = cohort.dataset('validation')
    scores = []
    for index in range(len(data)):
        image = data[index]['image'][None].to(device)
        score = float(predict_scores(model, image)[0].cpu())
        scores.append({'validation_index': index, 'abnormality_score': score,
                       'calibrated': False, 'localization': None})
    return scores
