"""Patient-level research metrics. Undefined values remain null, not invented zeros."""

from collections import defaultdict

import numpy as np
import torch


def binary_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or not labels.size
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)
            or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("invalid_binary_evaluation_inputs")
    labels = labels.astype(int)
    predictions = scores >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    auroc = None
    if positives and negatives:
        # Mann-Whitney rank sum with average ranks for exact ties; O(n log n).
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=float)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and scores[order[stop]] == scores[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2
            start = stop
        auroc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                      / (positives * negatives))
    return {"patients": len(labels), "threshold": threshold, "auroc": auroc,
            "sensitivity": tp / positives if positives else None,
            "specificity": tn / negatives if negatives else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positives if positives else None,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "matrix_order": "rows=true [normal,abnormal]; columns=predicted [normal,abnormal]",
            "score_status": "uncalibrated_research_sigmoid",
            "auroc_undefined_reason": "single_class" if auroc is None else None}


def patient_metrics(labels, scores, patients, threshold=0.5):
    if not (len(labels) == len(scores) == len(patients)):
        raise ValueError("evaluation_lengths_differ")
    grouped = defaultdict(list)
    for label, score, patient in zip(labels, scores, patients, strict=True):
        grouped[patient].append((label, score))
    final_labels, final_scores = [], []
    for values in grouped.values():
        if len({value[0] for value in values}) != 1:
            raise ValueError("conflicting_patient_labels_require_visit_policy")
        final_labels.append(values[0][0])
        final_scores.append(float(np.mean([value[1] for value in values])))
    result = binary_metrics(final_labels, final_scores, threshold)
    result["aggregation"] = "mean image score per globally reconciled patient"
    return result


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    labels, scores, patients, sources, sites = [], [], [], [], []
    for batch in loader:
        score = torch.sigmoid(model(batch["image"].to(device))).cpu().tolist()
        scores.extend(score)
        labels.extend(batch["label"].tolist())
        patients.extend(batch["patient"].tolist())
        sources.extend(batch["source"].tolist())
        sites.extend(batch["site"].tolist())
    result = {"overall": patient_metrics(labels, scores, patients, threshold)}
    # Anonymous group indices; no patient IDs, site names or raw paths in reports.
    for name, groups in (("by_source", sources), ("by_source_site", sites)):
        result[name] = []
        for group in sorted(set(groups)):
            indices = [i for i, value in enumerate(groups) if value == group]
            metrics = patient_metrics([labels[i] for i in indices], [scores[i] for i in indices],
                                      [patients[i] for i in indices], threshold)
            result[name].append({"group_index": group, **metrics})
    return result
