"""Private inference from an explicitly trusted, bound research checkpoint."""
import argparse
import json
from pathlib import Path

import nibabel as nib

from brainagenticthon.detection.training import _resume_checkpoint, private_output_directory
from brainagenticthon.preprocessing.oasis2_mri_quality import private_call
from brainagenticthon.segmentation.data import load_case
from brainagenticthon.segmentation.inference import lesion_boxes, release_labels, sliding_window
from brainagenticthon.segmentation.model import UNet3D
from brainagenticthon.segmentation.training import TrainConfig, load_private_cohort, signature


def load_model(cohort, config, run, *, trusted=False):
    if not trusted:
        raise ValueError('only_explicitly_trusted_own_checkpoint')
    run = private_output_directory(run, cohort.source_roots)
    state = _resume_checkpoint(run, signature(cohort, config))
    model = UNet3D().to(config.device)
    model.load_state_dict(state['best_model'])
    return model.eval()


def predict(model, row, root, output, config, *, protected=()):
    """Write release labels and provenance-bearing boxes only to a fresh private directory."""
    root = Path(root).resolve()
    output = private_output_directory(output, (root, *protected))
    if output.exists() and any(output.iterdir()):
        raise ValueError('fresh_private_inference_directory_required')
    image, _, affine = load_case(root, row, require_mask=False)
    mask = sliding_window(model, image, patch=config.patch, device=config.device).argmax(0)
    labels = release_labels(mask)
    boxes = lesion_boxes(labels, source='predicted_mask')
    output.mkdir(parents=True, exist_ok=True)
    prediction = nib.Nifti1Image(labels, affine)
    prediction.header.set_xyzt_units('mm')
    prediction.set_qform(affine, code=0)
    prediction.set_sform(affine, code=1)
    nib.save(prediction, output / 'prediction.nii.gz')
    (output / 'boxes.json').write_text(json.dumps(
        {'release': 'brats2021-glioma-training', 'clinical_approval': False,
         'boxes': boxes}, indent=2) + '\n')
    return {'components': len(boxes), 'clinical_approval': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-own-checkpoint', action='store_true', required=True)
    args = parser.parse_args()

    def execute():
        from brainagenticthon.preprocessing.oasis2_mri_quality import private_path
        cfg = json.loads(args.config.read_text())
        if cfg.get('mode') != 'verified' or cfg.get('real_training_enabled') is not True:
            raise ValueError('verified_private_configuration_required')
        root = Path(cfg['private_drive_root']).resolve()
        cohort = load_private_cohort(private_path(root, cfg['manifest']),
                                      private_path(root, cfg['image_root']),
                                      private_path(root, cfg['evidence']))
        settings = {**cfg['training'], 'patch': tuple(cfg['training']['patch'])}
        config = TrainConfig(**settings)
        run = private_path(root, cfg['output_run'])
        model = load_model(cohort, config, run, trusted=args.trusted_own_checkpoint)
        if args.record.stat().st_size > 1024 * 1024:
            raise ValueError('inference_record_resource_limit')
        row = json.loads(args.record.read_text())
        return predict(model, row, private_path(root, cfg['image_root']), args.output, config,
                       protected=(*cohort.source_roots, run, args.config, args.record))
    print(json.dumps(private_call(execute)))


if __name__ == '__main__':
    main()
