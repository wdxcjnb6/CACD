"""
utils/metrics.py
================
Unified metrics module with three parts:

1. Forecasting error metrics
   - MAE / MSE / RMSE / MAPE / MSPE / RSE / CORR

2. Causal discovery metrics
   - Lag-free evaluation:  pred [C, C]            vs gt [C, C]
   - Lag-aware evaluation: pred [C, C, max_lag+1] vs gt [C, C, max_lag+1]
   - AUROC / AUPRC / F1 / Precision / Recall / SHD
   - build_pred_matrix_from_triplets()
   - binarize_by_cumulative_ratio()
   - evaluate_causal_graph()
   - evaluate_causal_graph_with_lag()
   - print_metrics()

3. Sign-modulation metrics
   - evaluate_sign_modulation()  — assess excitatory/inhibitory direction accuracy on TP edges
   - save_three_causal_matrices() — export causal / lag / modulation direction matrices as CSVs

Ground-truth causal graph format (CSV with header):
    src,tgt,lag[,sign]
    0,1,0          <- variable 0 -> variable 1, no lag (lag=0)
    0,2,1,+1       <- variable 0 -> variable 2, lag=1, excitatory
    1,3,2,-1       <- variable 1 -> variable 3, lag=2, inhibitory
  src/tgt: 0-indexed; lag=0 means no lag; sign is optional (+1=excitatory, -1=inhibitory)
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Part 1: Forecasting error metrics
# ============================================================

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-12)


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(
        ((true - true.mean(0)) ** 2).sum(0) *
        ((pred - pred.mean(0)) ** 2).sum(0)
    ) + 1e-12
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mask = np.abs(true) > 1e-8
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))


def MSPE(pred, true):
    mask = np.abs(true) > 1e-8
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask]))


def metric(pred, true):
    mae  = MAE(pred, true)
    mse  = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse  = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr


# ============================================================
# Part 2: Causal discovery metrics
# ============================================================

def build_pred_matrix_from_triplets(triplets, d_in, max_lag=None, score_col=4):
    """
    Convert model-output triplets into a prediction-strength matrix.

    Parameters
    ----------
    triplets  : list, each row is (src, tgt, lag, direction, causal_strength, sign_stability)
                  col 0: src
                  col 1: tgt
                  col 2: lag
                  col 3: direction       — ∂y_tgt/∂x_src[lag], signed (positive=excitatory)
                  col 4: causal_strength — unsigned causal strength (default score column)
                  col 5: sign_stability  — directional reliability
    d_in      : int, number of channels C
    max_lag   : int or None
        None -> [C, C]  (take max score across all lags for each src/tgt pair)
        int  -> [C, C, max_lag+1]
    score_col : int, column index of the score (default 4)

    Returns
    -------
    np.ndarray of shape [C, C] or [C, C, max_lag+1]
    """
    if max_lag is None:
        mat = np.zeros((d_in, d_in), dtype=np.float32)
        for row in triplets:
            src, tgt = int(row[0]), int(row[1])
            score    = float(row[score_col])
            if 0 <= src < d_in and 0 <= tgt < d_in:
                mat[src, tgt] = max(mat[src, tgt], score)
    else:
        mat = np.zeros((d_in, d_in, max_lag + 1), dtype=np.float32)
        for row in triplets:
            src, tgt, lag = int(row[0]), int(row[1]), int(row[2])
            score         = float(row[score_col])
            if 0 <= src < d_in and 0 <= tgt < d_in and 0 <= lag <= max_lag:
                mat[src, tgt, lag] = score
    return mat


def binarize_by_cumulative_ratio(triplets, d_in, cumulative_ratio=0.95):
    """
    Binarize edges using a per-target cumulative-strength threshold.

    For each target channel, edges are sorted by causal_strength (col 4) in
    descending order and accumulated until the cumulative fraction of that
    target's total incoming strength reaches cumulative_ratio. Self-loops
    (src == tgt) are always skipped.

    Parameters
    ----------
    triplets         : list, each row is (src, tgt, lag, direction, causal_strength, ...)
    d_in             : int, number of channels C
    cumulative_ratio : float, cumulative-strength threshold (default 0.95)

    Returns
    -------
    pred_bin              : np.ndarray [C, C], binary adjacency matrix [src, tgt]
    strength_threshold_eff: float, minimum causal_strength among all selected edges
    lag_map               : dict, (src, tgt) -> lag
    direction_map         : dict, (src, tgt) -> +1 or -1
    """
    C        = d_in
    pred_bin = np.zeros((C, C), dtype=np.float32)
    lag_map  = {}
    selected_strengths = []

    if not triplets:
        return pred_bin, 0.0, lag_map, {}

    tgt_groups = defaultdict(list)
    for row in triplets:
        src, tgt = int(row[0]), int(row[1])
        if src == tgt:
            continue
        tgt_groups[tgt].append(row)

    direction_map = {}

    for tgt, rows in tgt_groups.items():
        rows_sorted = sorted(rows, key=lambda r: float(r[4]), reverse=True)
        col_total   = sum(float(r[4]) for r in rows_sorted)
        if col_total <= 0:
            continue

        cumsum = 0.0
        for row in rows_sorted:
            src       = int(row[0])
            lag       = int(row[2])
            direction = float(row[3])
            strength  = float(row[4])
            cumsum   += strength
            pred_bin[src, tgt]        = 1.0
            lag_map[(src, tgt)]       = lag
            direction_map[(src, tgt)] = 1 if direction > 0 else -1
            selected_strengths.append(strength)
            if cumsum / col_total >= cumulative_ratio:
                break

    strength_threshold_eff = min(selected_strengths) if selected_strengths else 0.0
    return pred_bin, strength_threshold_eff, lag_map, direction_map


def _remove_diag(mat):
    """Flatten [C, C] to [C*(C-1)] by removing diagonal entries."""
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    C = mat.shape[0]
    return mat[~np.eye(C, dtype=bool)]


def _find_best_f1_threshold(scores, labels, n_thresholds=100):
    """Deprecated (data leakage). Stub retained for backward compatibility."""
    raise RuntimeError(
        "_find_best_f1_threshold is deprecated (data leakage). "
        "Compute strength_threshold_eff via cumulative_ratio and pass it "
        "explicitly as the threshold argument."
    )


def evaluate_causal_graph(pred_matrix, gt_matrix, threshold=None, pred_bin_ext=None):
    """
    Lag-free causal graph evaluation: pred [C, C] vs gt [C, C].

    Parameters
    ----------
    pred_matrix  : np.ndarray [C, C]  prediction strengths (used for AUROC/AUPRC)
    gt_matrix    : np.ndarray [C, C]  ground-truth binary adjacency matrix
    threshold    : float, recorded for logging purposes
    pred_bin_ext : np.ndarray [C, C] or None
                   External binary matrix (from binarize_by_cumulative_ratio).
                   If provided, used directly for F1/Precision/Recall/SHD;
                   otherwise pred_matrix is thresholded at `threshold`.

    Returns
    -------
    dict: n_positive_edges, n_negative_edges, auroc, auprc,
          threshold, f1, precision, recall, shd
    """
    assert pred_matrix.shape == gt_matrix.shape, \
        f"Shape mismatch: pred={pred_matrix.shape}, gt={gt_matrix.shape}"

    scores = _remove_diag(pred_matrix.astype(np.float32))
    labels = _remove_diag(gt_matrix.astype(np.float32)).astype(int)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    results = {
        'n_positive_edges': n_pos,
        'n_negative_edges': n_neg,
    }

    if n_pos > 0 and n_neg > 0:
        results['auroc'] = float(roc_auc_score(labels, scores))
        results['auprc'] = float(average_precision_score(labels, scores))
    else:
        results['auroc'] = float('nan')
        results['auprc'] = float('nan')

    if threshold is None:
        raise ValueError(
            "threshold cannot be None. Compute strength_threshold_eff via "
            "cumulative_ratio and pass it explicitly."
        )
    results['threshold'] = float(threshold)

    if pred_bin_ext is not None:
        pred_bin_flat = _remove_diag(pred_bin_ext.astype(np.float32)).astype(int)
    else:
        pred_bin_flat = (scores >= threshold).astype(int)

    results['f1']        = float(f1_score(labels, pred_bin_flat, zero_division=0))
    results['precision'] = float(precision_score(labels, pred_bin_flat, zero_division=0))
    results['recall']    = float(recall_score(labels, pred_bin_flat, zero_division=0))
    results['shd']       = int(np.sum(pred_bin_flat != labels))

    return results


def evaluate_causal_graph_with_lag(pred_matrix_lag, gt_matrix_lag,
                                    threshold=None, pred_bin_ext=None,
                                    lag_map=None):
    """
    Lag-aware causal graph evaluation: pred [C, C, max_lag+1] vs gt [C, C, max_lag+1].

    Two groups of metrics:
      edge_* : graph-level metrics after collapsing the lag dimension
      lag_*  : lag-prediction accuracy evaluated only on true-positive edges

    Parameters
    ----------
    pred_matrix_lag : np.ndarray [C, C, max_lag+1]
    gt_matrix_lag   : np.ndarray [C, C, max_lag+1]
    threshold       : float, recorded for logging
    pred_bin_ext    : np.ndarray [C, C], external binary adjacency matrix
    lag_map         : dict (src, tgt) -> predicted_lag

    Returns
    -------
    dict containing edge_* and lag_* metric groups
    """
    assert pred_matrix_lag.shape == gt_matrix_lag.shape, \
        f"Shape mismatch: pred={pred_matrix_lag.shape}, gt={gt_matrix_lag.shape}"

    C = pred_matrix_lag.shape[0]

    pred_collapsed = pred_matrix_lag.max(axis=-1)
    gt_collapsed   = (gt_matrix_lag.sum(axis=-1) > 0).astype(np.float32)
    edge_metrics   = {
        f'edge_{k}': v
        for k, v in evaluate_causal_graph(
            pred_collapsed, gt_collapsed, threshold, pred_bin_ext=pred_bin_ext
        ).items()
    }

    if pred_bin_ext is None or lag_map is None:
        return edge_metrics

    gt_bin = (gt_matrix_lag.sum(axis=-1) > 0).astype(int)
    p = pred_bin_ext.astype(int).copy()
    np.fill_diagonal(p,      0)
    np.fill_diagonal(gt_bin, 0)

    n_tp          = int(np.sum((p == 1) & (gt_bin == 1)))
    n_lag_correct = 0

    for src in range(C):
        for tgt in range(C):
            if src == tgt:
                continue
            if p[src, tgt] == 1 and gt_bin[src, tgt] == 1:  # true-positive edge
                pred_lag = lag_map.get((src, tgt), -1)
                gt_lags  = np.where(gt_matrix_lag[src, tgt] > 0)[0]
                if len(gt_lags) > 0 and pred_lag == int(gt_lags[0]):
                    n_lag_correct += 1

    lag_metrics = {
        'lag_n_tp':      n_tp,
        'lag_n_correct': n_lag_correct,
        'lag_accuracy':  float(n_lag_correct / n_tp) if n_tp > 0 else float('nan'),
    }

    return {**edge_metrics, **lag_metrics}


def print_metrics(metrics, title="Causal Discovery Metrics"):
    """Pretty-print a causal metrics dictionary."""
    print(f"\n{'=' * 54}")
    print(f"  {title}")
    print(f"{'=' * 54}")
    print(f"  {'Metric':<32} {'Value':>10}")
    print(f"  {'-' * 44}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<32} {'NaN':>10}" if np.isnan(v) else f"  {k:<32} {v:>10.4f}")
        else:
            print(f"  {k:<32} {v:>10}")
    print(f"{'=' * 54}\n")


# ============================================================
# Part 3: Sign-modulation metrics + three-matrix export
# ============================================================

def evaluate_sign_modulation(triplets, pred_bin, lag_map, causal_gt_sign, d_in):
    """
    Evaluate whether predicted modulation direction (excitatory/inhibitory)
    matches the ground-truth sign on true-positive edges.
    Aligned with the lag_* metric structure: n_tp / n_correct / accuracy.

    Parameters
    ----------
    triplets        : list, each row is (src, tgt, lag, direction, causal_strength, ...)
    pred_bin        : np.ndarray [C, C], binary causal adjacency matrix
    lag_map         : dict (src, tgt) -> lag
    causal_gt_sign  : np.ndarray [C, C] or None (+1=excitatory, -1=inhibitory, 0=unknown)
    d_in            : int

    Returns
    -------
    dict:
        sign_n_tp       : number of TP edges with a known GT sign
        sign_n_correct  : edges where predicted direction matches GT sign
        sign_accuracy   : sign_n_correct / sign_n_tp, or NaN if no TP edges
    """
    nan = float('nan')

    if causal_gt_sign is None:
        return {
            'sign_n_tp':      nan,
            'sign_n_correct': nan,
            'sign_accuracy':  nan,
        }

    C = d_in

    # Build direction map for the edge at the predicted lag
    direction_map = {}
    for row in triplets:
        src, tgt  = int(row[0]), int(row[1])
        lag_pred  = int(row[2])
        direction = float(row[3])
        if lag_map.get((src, tgt), -1) == lag_pred:
            direction_map[(src, tgt)] = direction

    # Collapse GT sign to [C, C] (compatible with both [C,C] and [C,C,lag] inputs)
    if causal_gt_sign.ndim == 3:
        gt_sign_2d = np.zeros((C, C), dtype=np.float32)
        for (src, tgt), lag in lag_map.items():
            if 0 <= src < C and 0 <= tgt < C and lag < causal_gt_sign.shape[2]:
                gt_sign_2d[src, tgt] = causal_gt_sign[src, tgt, lag]
    else:
        gt_sign_2d = causal_gt_sign.astype(np.float32)

    gt_bin = (np.abs(gt_sign_2d) > 0).astype(int)
    p      = pred_bin.astype(int).copy()
    np.fill_diagonal(p,      0)
    np.fill_diagonal(gt_bin, 0)

    n_tp      = 0
    n_correct = 0

    for src in range(C):
        for tgt in range(C):
            if src == tgt:
                continue
            if p[src, tgt] == 1 and gt_bin[src, tgt] == 1:  # TP edge
                gt_sign_val = float(gt_sign_2d[src, tgt])
                if gt_sign_val == 0:
                    continue  # GT sign unknown; skip
                n_tp += 1
                pred_dir  = direction_map.get((src, tgt), 0.0)
                pred_sign = 1 if pred_dir >= 0 else -1
                gt_sign   = 1 if gt_sign_val > 0 else -1
                if pred_sign == gt_sign:
                    n_correct += 1

    sign_accuracy = float(n_correct / n_tp) if n_tp > 0 else nan

    return {
        'sign_n_tp':      n_tp,
        'sign_n_correct': n_correct,
        'sign_accuracy':  sign_accuracy,
    }


def save_three_causal_matrices(triplets, pred_bin, lag_map, d_in, out_dir,
                                roi_labels=None):
    """
    Save causal discovery results as five CSV matrices; no ground truth required.

    Files produced (rows = src, columns = tgt, labels = ROI abbreviations):
      causal_matrix.csv        — information flow (binary 0/1)
      lag_matrix.csv           — lag in samples (integer; 0 = edge absent)
      modulation_matrix.csv    — modulation direction (+1=excitatory, -1=inhibitory, 0=absent)
      strength_matrix.csv      — raw causal strength at present edges
      strength_matrix_norm.csv — globally normalized strength (divided by matrix max) in [0, 1]

    Parameters
    ----------
    triplets   : list, each row is (src, tgt, lag, direction, causal_strength, ...)
                   col 3: direction       — ∂y_tgt/∂x_src[τ̂], signed
                   col 4: causal_strength — unsigned strength
    pred_bin   : np.ndarray [C, C], binary causal adjacency matrix
    lag_map    : dict (src, tgt) -> lag
    d_in       : int, number of channels C
    out_dir    : str, output directory
    roi_labels : list of str or None
                 If None, labels default to ch0, ch1, ...
    """
    C = d_in
    if roi_labels is None or len(roi_labels) != C:
        roi_labels = [f'ch{i}' for i in range(C)]

    # Modulation direction matrix: +1 / -1 / 0
    # Use the sign of the direction at the highest-strength lag
    mod_mat = np.zeros((C, C), dtype=np.int8)
    for row in triplets:
        src, tgt  = int(row[0]), int(row[1])
        direction = float(row[3])
        strength  = float(row[4])
        if 0 <= src < C and 0 <= tgt < C:
            if strength > abs(mod_mat[src, tgt]):
                mod_mat[src, tgt] = np.int8(np.sign(direction))

    # Lag matrix (only populated where pred_bin == 1)
    lag_mat = np.zeros((C, C), dtype=np.int32)
    for (src, tgt), lag in lag_map.items():
        if 0 <= src < C and 0 <= tgt < C:
            lag_mat[src, tgt] = int(lag)

    # Strength matrix: raw causal_strength (col 4) per edge
    strength_mat = np.zeros((C, C), dtype=np.float64)
    for row in triplets:
        src, tgt = int(row[0]), int(row[1])
        strength = float(row[4])
        if 0 <= src < C and 0 <= tgt < C and pred_bin[src, tgt] > 0:
            strength_mat[src, tgt] = max(strength_mat[src, tgt], strength)
    np.fill_diagonal(strength_mat, 0.0)

    # Global normalization: divide by the maximum off-diagonal value
    score_no_diag = strength_mat.copy()
    np.fill_diagonal(score_no_diag, 0.0)
    global_max = float(np.abs(score_no_diag).max())
    if global_max > 0:
        strength_norm = strength_mat / global_max
    else:
        strength_norm = np.zeros_like(strength_mat)
    strength_norm[pred_bin == 0] = 0.0
    np.fill_diagonal(strength_norm, 0.0)

    df_causal   = pd.DataFrame(pred_bin.astype(np.int32),        index=roi_labels, columns=roi_labels)
    df_lag      = pd.DataFrame(lag_mat,                           index=roi_labels, columns=roi_labels)
    df_mod      = pd.DataFrame(mod_mat,                           index=roi_labels, columns=roi_labels)
    df_strength = pd.DataFrame(strength_mat.astype(np.float64),  index=roi_labels, columns=roi_labels)
    df_str_norm = pd.DataFrame(strength_norm.astype(np.float64), index=roi_labels, columns=roi_labels)

    os.makedirs(out_dir, exist_ok=True)
    df_causal.to_csv(  os.path.join(out_dir, 'causal_matrix.csv'),        index_label='src\\tgt')
    df_lag.to_csv(     os.path.join(out_dir, 'lag_matrix.csv'),            index_label='src\\tgt')
    df_mod.to_csv(     os.path.join(out_dir, 'modulation_matrix.csv'),     index_label='src\\tgt')
    df_strength.to_csv(os.path.join(out_dir, 'strength_matrix.csv'),       index_label='src\\tgt')
    df_str_norm.to_csv(os.path.join(out_dir, 'strength_matrix_norm.csv'),  index_label='src\\tgt')

    n_edges = int(pred_bin.sum())
    print(f"[File] Matrices saved to: {out_dir}")
    print(f"  causal_matrix.csv        — {n_edges} edges (information flow, binary)")
    print(f"  lag_matrix.csv           — lag per edge (in samples)")
    print(f"  modulation_matrix.csv    — modulation direction (+1=excitatory, -1=inhibitory, 0=absent)")
    print(f"  strength_matrix.csv      — raw causal strength at present edges")
    print(f"  strength_matrix_norm.csv — globally normalized strength (divided by matrix max) [0, 1]")
