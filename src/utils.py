"""
Evaluation metrics and utilities for GNO.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_alignment_metrics(K: torch.Tensor, Y: torch.Tensor,
                              threshold: float = 0.5) -> dict:
    """
    Compute alignment evaluation metrics for one sample.

    Args:
        K : (n_src, n_tgt) predicted alignment scores
        Y : (n_src, n_tgt) ground truth
    """
    K_np = K.detach().cpu().numpy().flatten()
    Y_np = Y.detach().cpu().numpy().flatten()

    # Binary predictions
    pred_binary = (K_np >= threshold).astype(int)
    true_binary = (Y_np > 0).astype(int)

    metrics = {}

    # Precision, Recall, F1
    if true_binary.sum() > 0 or pred_binary.sum() > 0:
        p, r, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average="binary", zero_division=0)
        metrics["precision"] = float(p)
        metrics["recall"] = float(r)
        metrics["f1"] = float(f1)
    else:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    # AUC (only if both classes present)
    if len(np.unique(true_binary)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(true_binary, K_np))
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    # Exact match accuracy (for identical entities)
    if Y_np.max() >= 1.0:
        exact_mask = Y_np >= 1.0
        if exact_mask.sum() > 0:
            exact_pred = pred_binary[exact_mask]
            metrics["exact_match_acc"] = float(exact_pred.mean())
        else:
            metrics["exact_match_acc"] = 0.0
    else:
        metrics["exact_match_acc"] = 0.0

    # Sparsity of predictions
    metrics["pred_sparsity"] = float(1.0 - pred_binary.mean())

    return metrics


def aggregate_metrics(metrics_list: list) -> dict:
    """Average a list of metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        agg[k] = float(np.mean(vals)) if vals else 0.0
    return agg


def hungarian_matching(K: torch.Tensor) -> torch.Tensor:
    """
    Solve optimal 1-to-1 assignment using Hungarian algorithm.
    Used as a baseline comparison.
    """
    from scipy.optimize import linear_sum_assignment
    cost = 1.0 - K.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    result = torch.zeros_like(K)
    for r, c in zip(row_ind, col_ind):
        result[r, c] = 1.0
    return result
