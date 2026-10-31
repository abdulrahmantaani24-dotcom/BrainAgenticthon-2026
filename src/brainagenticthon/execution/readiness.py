"""Read-only private eligibility inspection through existing gates; never starts training."""
import argparse
import json
from pathlib import Path

from brainagenticthon.detection.profile import InputProfile
from brainagenticthon.detection.training import load_private_cohort as load_a
from brainagenticthon.detection.training import private_output_directory
from brainagenticthon.preprocessing.oasis2_mri_quality import private_path
from brainagenticthon.segmentation.training import load_private_cohort as load_b

REQUIRED_EVIDENCE = {
    'a': ['authorized_source_and_identity_review', 'eligible_true_Normal_and_Abnormal_MRI_labels',
          'healthy_control_verification_for_normal', 'abnormal_MRI_evidence_for_abnormal',
          'native_T1_and_whole_brain_coverage', 'independent_anatomical_geometry_and_laterality',
          'qualified_visual_QC', 'reviewed_file_and_companion_hashes',
          'frozen_patient_level_splits_with_both_classes', 'no_source_or_site_class_confounding'],
    'b': ['authorized_BraTS2021_glioma_access', 'release_specific_mask_and_provenance_references',
          'T1_T1ce_T2_FLAIR_and_segmentation_files_with_reviewed_hashes',
          'aligned_coded_mm_geometry', 'release_labels_0_1_2_4',
          'globally_reconciled_patient_identity', 'frozen_train_validation_test_patient_splits'],
}


def inspect(config_path, *, task):
    """Return aggregates only. Input files/images are never modified or uploaded."""
    if task not in REQUIRED_EVIDENCE:
        raise ValueError('explicit_task_required')
    report = {'task': task, 'status': 'BLOCKED', 'training_started': False,
              'clinical_approval': False, 'required_evidence': REQUIRED_EVIDENCE[task],
              'technical_readiness_is_not_clinical_approval': True}
    if config_path is None or not Path(config_path).is_file():
        report['reason'] = 'private_configuration_and_Drive_inputs_unavailable'
        return report
    try:
        path = Path(config_path)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('config_size_limit')
        cfg = json.loads(path.read_text())
        root = Path(cfg['private_drive_root']).resolve(strict=True)
        keys = ('manifest', 'reviews', 'training_evidence', 'image_root') if task == 'a' else (
            'manifest', 'evidence', 'image_root')
        missing = [key for key in keys if not cfg.get(key)
                   or not private_path(root, cfg[key]).exists()]
        if missing:
            report.update(reason='required_private_inputs_missing', missing_inputs=missing)
            return report
        if task == 'a':
            profile = InputProfile(**{**cfg['profile'],
                                     'target_shape': tuple(cfg['profile']['target_shape'])})
            cohort = load_a(private_path(root, cfg['manifest']), private_path(root, cfg['reviews']),
                            private_path(root, cfg['training_evidence']),
                            private_path(root, cfg['image_root']), profile)
            report['aggregate'] = cohort.summary
        else:
            cohort = load_b(private_path(root, cfg['manifest']),
                            private_path(root, cfg['image_root']),
                            private_path(root, cfg['evidence']))
            report['aggregate'] = {'cases': len(cohort.rows),
                                   'release': 'brats2021-glioma-training',
                                   'split_counts': {key: sum(row['split'] == key
                                                            for row in cohort.rows)
                                                   for key in ('train', 'validation', 'test')}}
        report.update(status='ELIGIBILITY_GATE_PASSED', reason='no_training_was_started')
    except Exception as error:
        report.update(reason='private_eligibility_gate_rejected_inputs',
                      error_type=type(error).__name__)
    return report


def save_inspection(config_path, output, *, task):
    """Persist aggregate report only outside Git and disjoint from configured private inputs."""
    protected = []
    if config_path is not None:
        path = Path(config_path).resolve()
        protected.append(path)
        if path.is_file() and path.stat().st_size <= 1024 * 1024:
            cfg = json.loads(path.read_text())
            root = Path(cfg['private_drive_root']).resolve()
            for key in ('manifest', 'reviews', 'training_evidence', 'image_root', 'evidence'):
                if cfg.get(key):
                    protected.append(private_path(root, cfg[key]))
    output = private_output_directory(output, protected)
    report = inspect(config_path, task=task)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('a', 'b'), required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = save_inspection(args.config, args.output, task=args.task)
    print(json.dumps(report))
    return 0 if report['status'] == 'ELIGIBILITY_GATE_PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
