"""Full supplied-grid inference, BraTS region Dice and actual connected-component boxes."""
from collections import deque
from itertools import product

import numpy as np
import torch
from torch.nn import functional as functional


def sliding_window(model, image, patch=(64, 64, 64), overlap=.5, device='cpu'):
    if (image.ndim != 4 or image.shape[0] != 4 or not torch.isfinite(image).all()
            or len(patch) != 3 or any(type(n) is not int or not 8 <= n <= 160 for n in patch)
            or not 0 <= overlap < 1):
        raise ValueError('invalid_sliding_window_input')
    shape = image.shape[1:]
    padding = [v for n, p in reversed(list(zip(shape, patch, strict=True)))
               for v in (0, max(0, p - n))]
    padded = functional.pad(image.cpu(), padding)
    starts = []
    for n, p in zip(padded.shape[1:], patch, strict=True):
        stride = max(1, int(p * (1 - overlap)))
        starts.append(sorted({*range(0, n - p + 1, stride), n - p}))
    votes = torch.zeros(4, *padded.shape[1:], dtype=torch.float32)
    counts = torch.zeros(padded.shape[1:], dtype=torch.float32)
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for position in product(*starts):
                slices = tuple(slice(a, a + p) for a, p in zip(position, patch, strict=True))
                logits = model(padded[(slice(None), *slices)].unsqueeze(0).to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite_inference')
                votes[(slice(None), *slices)] += logits[0].float().softmax(0).cpu()
                counts[slices] += 1
    finally:
        model.train(previous)
    if not torch.all(counts > 0):
        raise RuntimeError('sliding_window_coverage_failed')
    slices = tuple(slice(0, n) for n in shape)
    return (votes / counts)[(slice(None), *slices)]


def region_dice(prediction, target):
    if (prediction.shape != target.shape or prediction.ndim != 3
            or any(not set(torch.unique(x).tolist()).issubset({0, 1, 2, 3})
                   for x in (prediction, target))):
        raise ValueError('aligned_internal_label_masks_required')
    regions = {'WT': (1, 2, 3), 'TC': (1, 3), 'ET': (3,)}
    report = {}
    for name, labels in regions.items():
        p = sum((prediction == n).int() for n in labels).bool()
        t = sum((target == n).int() for n in labels).bool()
        denominator = int(p.sum() + t.sum())
        # Both empty has no lesion overlap evidence; exclude from macro average.
        report[name] = 2 * int((p & t).sum()) / denominator if denominator else None
    return report


def lesion_boxes(mask, *, source):
    """Inclusive native-grid bounds from genuine mask components, never classifier scores.

    Six-connectivity; caller must distinguish reference annotations from model predictions.
    """
    if source not in {'reference_mask', 'predicted_mask'}:
        raise ValueError('explicit_mask_provenance_required')
    values = np.asarray(mask)
    if values.ndim != 3 or not set(np.unique(values)).issubset({0, 1, 2, 3, 4}):
        raise ValueError('valid_label_mask_required')
    remaining = values > 0
    boxes = []
    for seed in np.argwhere(remaining):
        seed = tuple(int(v) for v in seed)
        if not remaining[seed]:
            continue
        queue = deque([seed])
        remaining[seed] = False
        low, high, count = list(seed), list(seed), 0
        while queue:
            point = queue.popleft()
            count += 1
            for axis in range(3):
                low[axis], high[axis] = min(low[axis], point[axis]), max(high[axis], point[axis])
                for delta in (-1, 1):
                    neighbor = list(point)
                    neighbor[axis] += delta
                    q = tuple(neighbor)
                    if 0 <= q[axis] < remaining.shape[axis] and remaining[q]:
                        remaining[q] = False
                        queue.append(q)
        boxes.append({'bounds': [low[0], high[0], low[1], high[1], low[2], high[2]],
                      'voxels': count, 'source': source,
                      'coordinate_space': 'supplied_native_grid'})
    return boxes


def release_labels(mask):
    result = mask.detach().cpu().numpy().astype(np.uint8)
    result[result == 3] = 4
    return result
