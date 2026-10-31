"""Explicit research contracts without inventing legacy detection/anatomy fields."""
import numpy as np
import torch

from brainagenticthon.segmentation.inference import lesion_boxes, release_labels


def classification_result(score):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('finite_research_score_required')
    return {'schema': 'task-a-research-score-v1', 'abnormality_score': float(score),
            'calibrated': False, 'clinical_approval': False,
            'task_a_output_compatible': False,
            'unsupported': ['finding_present', 'bounding_boxes'],
            'reason': 'classifier_scores_do_not_establish_lesion_localization'}


def segmentation_result(mask, affine):
    """Supplied-grid predicted-mask measurements; no atlas or registry relabeling."""
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 3
            or not set(torch.unique(mask).tolist()).issubset({0, 1, 2, 3})):
        raise ValueError('internal_3d_segmentation_mask_required')
    affine = np.asarray(affine, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-8):
        raise ValueError('verified_millimetre_geometry_required')
    voxel = float(abs(np.linalg.det(affine[:3, :3])))
    return {'schema': 'task-b-brats2021-research-v1',
            'release': 'brats2021-glioma-training', 'clinical_approval': False,
            'label_map': {0: 'background', 1: 'NCR/NET', 2: 'edema', 4: 'enhancing_tumor'},
            'boxes': lesion_boxes(release_labels(mask), source='predicted_mask'),
            'voxel_size_mm3': voxel,
            'volume_ml': {name: int((mask == label).sum()) * voxel / 1000
                          for name, label in (('NCR_NET', 1), ('ED', 2), ('ET', 3))},
            'coordinate_space': 'supplied_native_grid',
            'anatomical_regions': None, 'task_b_output_compatible': False,
            'reason': 'BraTS2021_source_not_in_existing_approved_registry; '
                      'no_2024_label_or_atlas_mapping_is_assumed'}


def legacy_task_a_from_score(*args, **kwargs):
    raise ValueError('genuine_localizer_predictions_required_for_TaskAOutput')


def legacy_task_b_from_brats2021(*args, **kwargs):
    raise ValueError('reviewed_source_policy_extension_required_for_TaskBOutput')
