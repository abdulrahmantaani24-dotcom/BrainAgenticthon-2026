"""Bounded generic foreground metrics; explicit units and disease semantics, no atlas mapping."""

from collections import deque

import numpy as np
import torch


def _components(mask):
    pending = mask.copy()
    groups = []
    for seed in np.argwhere(pending):
        seed = tuple(int(v) for v in seed)
        if not pending[seed]:
            continue
        queue, group = deque([seed]), set()
        pending[seed] = False
        while queue:
            point = queue.popleft()
            group.add(int(np.ravel_multi_index(point, mask.shape)))
            for axis in range(3):
                for delta in (-1, 1):
                    other = list(point)
                    other[axis] += delta
                    other = tuple(other)
                    if 0 <= other[axis] < mask.shape[axis] and pending[other]:
                        pending[other] = False
                        queue.append(other)
        groups.append(group)
    return groups


def _surface(mask):
    inside = mask.copy()
    padded = np.pad(mask, 1)
    for axis in range(3):
        for delta in (-1, 1):
            slices = [slice(1, n + 1) for n in mask.shape]
            slices[axis] = slice(1 + delta, mask.shape[axis] + 1 + delta)
            inside &= padded[tuple(slices)]
    return np.argwhere(mask & ~inside)


def _directed_distances(first, second):
    result = []
    target = torch.as_tensor(second, dtype=torch.float64)
    for start in range(0, len(first), 128):
        chunk = torch.as_tensor(first[start : start + 128], dtype=torch.float64)
        result.extend(torch.cdist(chunk, target).min(1).values.tolist())
    return result


def lesion_metrics(
    prediction,
    reference,
    *,
    affine=None,
    units=None,
    semantics_reference="",
    component_iou=0.1,
    max_surface_points=4096,
):
    p, t = np.asarray(prediction), np.asarray(reference)
    if (
        p.ndim != 3
        or p.shape != t.shape
        or p.size > 32_000_000
        or not np.isfinite(p).all()
        or not np.isfinite(t).all()
        or not semantics_reference.strip()
        or not 0 < component_iou <= 1
        or type(max_surface_points) is not int
        or not 1 <= max_surface_points <= 4096
    ):
        raise ValueError("aligned_bounded_masks_and_reviewed_foreground_semantics_required")
    if not np.isin(p, (0, 1)).all() or not np.isin(t, (0, 1)).all():
        raise ValueError("explicit_binary_foreground_masks_required_no_implicit_label_mapping")
    p, t = p.astype(bool), t.astype(bool)
    if int(p.sum() + t.sum()) > 2_000_000:
        raise ValueError("foreground_component_resource_limit")
    predicted, truth = _components(p), _components(t)
    if len(predicted) * len(truth) > 1_000_000:
        raise ValueError("component_matching_resource_limit")
    candidates = []
    for i, a in enumerate(predicted):
        for j, b in enumerate(truth):
            intersection = len(a & b)
            iou = intersection / (len(a) + len(b) - intersection)
            if iou >= component_iou:
                candidates.append((iou, i, j))
    matched_p, matched_t = set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matched_p and j not in matched_t:
            matched_p.add(i)
            matched_t.add(j)
    total = int(p.sum() + t.sum())
    report = {
        "dice": 2 * int((p & t).sum()) / total if total else None,
        "reference_components": len(truth),
        "predicted_components": len(predicted),
        "lesion_sensitivity": len(matched_t) / len(truth) if truth else None,
        "false_positive_components": len(predicted) - len(matched_p),
        "matching": "greedy_one_to_one_component_IoU",
        "IoU_threshold": component_iou,
        "hausdorff95_mm": None,
        "average_symmetric_surface_distance_mm": None,
        "absolute_volume_error_ml": None,
        "relative_volume_error": None,
        "clinical_approval": False,
        "anatomical_regions": None,
    }
    if int(t.sum()):
        report["relative_volume_error"] = abs(int(p.sum()) - int(t.sum())) / int(t.sum())
    if units != "mm" or affine is None:
        report["physical_metrics_reason"] = "verified_mm_geometry_not_supplied"
        return report
    affine = np.asarray(affine, dtype=float)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or abs(np.linalg.det(affine[:3, :3])) < 1e-8
    ):
        raise ValueError("invalid_verified_affine")
    voxel = abs(float(np.linalg.det(affine[:3, :3])))
    report["absolute_volume_error_ml"] = abs(int(p.sum()) - int(t.sum())) * voxel / 1000
    a, b = _surface(p), _surface(t)
    if not len(a) or not len(b):
        report["boundary_metrics_reason"] = "empty_surface_no_invented_distance"
    elif max(len(a), len(b)) > max_surface_points:
        report["boundary_metrics_reason"] = "exact_boundary_metric_resource_limit_no_subsampling"
    else:
        a = a @ affine[:3, :3].T + affine[:3, 3]
        b = b @ affine[:3, :3].T + affine[:3, 3]
        forward, backward = _directed_distances(a, b), _directed_distances(b, a)
        report["hausdorff95_mm"] = float(np.percentile(forward + backward, 95))
        report["average_symmetric_surface_distance_mm"] = float(np.mean(forward + backward))
    return report
